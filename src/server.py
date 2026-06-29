"""FastAPI server exposing the ARExplorer agent over REST + serving the UI.

Run from the project root::

    uvicorn src.server:app --port 8000 --reload

Then open http://127.0.0.1:8000/ — the left panel chats with the agent and the
main panel renders the returned graph with d3.js (force / radial layouts).

The agent replies with a structured `AgentResponse` (see `src/schema.py`); the
`/api/chat` endpoint returns that JSON straight to the browser.
"""

import json
import uuid
from pathlib import Path

from dotenv import load_dotenv

# Load the project-root .env BEFORE importing the agent, so GOOGLE_API_KEY,
# REPLICATE_API_TOKEN and the NER_*/RELATION_* settings the agent reads via
# os.environ are populated. `adk web` does this through its CLI; this custom
# server must do it explicitly. Pre-existing environment variables win.
load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)

from fastapi import FastAPI  # noqa: E402
from fastapi.responses import FileResponse, StreamingResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from google.adk.artifacts import InMemoryArtifactService  # noqa: E402
from google.adk.runners import Runner  # noqa: E402
from google.adk.sessions import InMemorySessionService  # noqa: E402
from google.genai import types  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from src.agent import root_agent  # noqa: E402
from src.schema import AgentResponse  # noqa: E402

APP_NAME = "arexplorer"
STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="ARExplorer UI")

_session_service = InMemorySessionService()
_runner = Runner(
    agent=root_agent,
    app_name=APP_NAME,
    session_service=_session_service,
    artifact_service=InMemoryArtifactService(),
)


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


async def _ensure_session(session_id: str) -> None:
    existing = await _session_service.get_session(
        app_name=APP_NAME, user_id=APP_NAME, session_id=session_id
    )
    if existing is None:
        await _session_service.create_session(
            app_name=APP_NAME, user_id=APP_NAME, session_id=session_id
        )


def _parse_final(text: str) -> AgentResponse:
    """Coerce the agent's final text into an AgentResponse.

    With `output_schema` the final response is a JSON string; fall back to a
    plain chat message if anything unexpected comes back.
    """
    try:
        return AgentResponse.model_validate_json(text)
    except Exception:
        try:
            return AgentResponse.model_validate(json.loads(text))
        except Exception:
            return AgentResponse(message=text)


# Human-readable verbs for the tools the agent can call, used to turn raw
# event traffic into a friendly activity log streamed to the UI.
_TOOL_LABELS = {
    "extract_named_entities": "Extracting named entities",
    "form_entity_pairs": "Forming entity pairs",
    "classify_relations": "Classifying relations",
    "graph_operation": "Updating the relation graph",
    "load_artifacts": "Loading intermediate data",
    "set_model_response": "Composing the final answer",
}

# Summary keys (produced by the offload callback) worth surfacing in the log,
# mapped to short display labels.
_COUNT_LABELS = {
    "document_count": "documents",
    "entity_count": "entities",
    "pair_count": "pairs",
    "relation_count": "relations",
    "node_count": "nodes",
    "edge_count": "edges",
}


def _sse(payload: dict) -> str:
    """Serialize a payload as one Server-Sent Events message."""
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _brief_args(args: dict) -> str:
    """Render a short, payload-free description of a tool call's arguments."""
    if not args:
        return ""
    bits = []
    for key, value in args.items():
        if isinstance(value, list):
            bits.append(f"{key}: {len(value)} item(s)")
        elif isinstance(value, dict):
            bits.append(f"{key}: {{…}}")
        elif isinstance(value, str):
            shown = value if len(value) <= 40 else value[:39] + "…"
            bits.append(f"{key}: {shown!r}")
        else:
            bits.append(f"{key}: {value}")
    return " (" + ", ".join(bits) + ")"


def _brief_response(response: object) -> str:
    """Render a short summary of a tool response (status + counts)."""
    if not isinstance(response, dict):
        return "done"
    bits = []
    status = response.get("status")
    if status and status != "success":
        bits.append(str(status))
    for key, label in _COUNT_LABELS.items():
        if key in response:
            bits.append(f"{response[key]} {label}")
    label_counts = response.get("label_counts")
    if isinstance(label_counts, dict) and label_counts:
        bits.append(
            ", ".join(f"{k}: {v}" for k, v in label_counts.items())
        )
    if response.get("status") == "error" and response.get("error"):
        bits.append(str(response["error"]))
    return "done" + (" — " + "; ".join(bits) if bits else "")


def _event_logs(event) -> list[str]:
    """Turn one ADK event into zero or more human-readable log lines."""
    logs: list[str] = []

    for call in event.get_function_calls():
        label = _TOOL_LABELS.get(call.name, call.name)
        logs.append(f"{label}{_brief_args(call.args or {})}…")

    for resp in event.get_function_responses():
        label = _TOOL_LABELS.get(resp.name, resp.name)
        logs.append(f"{label} → {_brief_response(resp.response)}")

    # Surface any intermediate (non-final) natural-language text the model
    # emits while reasoning, e.g. before it starts calling tools.
    if (
        not event.is_final_response()
        and event.content
        and event.content.parts
        and not event.get_function_calls()
        and not event.get_function_responses()
    ):
        for part in event.content.parts:
            text = (getattr(part, "text", None) or "").strip()
            if text:
                logs.append(text if len(text) <= 200 else text[:199] + "…")

    return logs


@app.post("/api/chat")
async def chat(req: ChatRequest) -> dict:
    session_id = req.session_id or uuid.uuid4().hex
    await _ensure_session(session_id)

    message = types.Content(role="user", parts=[types.Part.from_text(text=req.message)])

    final_text = ""
    async for event in _runner.run_async(
        user_id=APP_NAME, session_id=session_id, new_message=message
    ):
        if event.is_final_response() and event.content and event.content.parts:
            final_text = event.content.parts[0].text or ""

    response = _parse_final(final_text)
    return {"session_id": session_id, "response": response.model_dump()}


@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest) -> StreamingResponse:
    """Run the agent and stream server-side activity as Server-Sent Events.

    Emits one `log` message per tool call / tool result (and any intermediate
    model text) as the agent works, then a single `final` message carrying the
    parsed `AgentResponse`. Errors are surfaced as an `error` message so the UI
    can stop the thinking indicator gracefully.
    """
    session_id = req.session_id or uuid.uuid4().hex

    async def event_stream():
        await _ensure_session(session_id)
        yield _sse({"type": "session", "session_id": session_id})

        message = types.Content(
            role="user", parts=[types.Part.from_text(text=req.message)]
        )

        final_text = ""
        try:
            async for event in _runner.run_async(
                user_id=APP_NAME, session_id=session_id, new_message=message
            ):
                for line in _event_logs(event):
                    yield _sse({"type": "log", "text": line})

                if (
                    event.is_final_response()
                    and event.content
                    and event.content.parts
                ):
                    final_text = event.content.parts[0].text or ""
        except Exception as exc:  # noqa: BLE001 - report any agent failure to UI
            yield _sse({"type": "error", "error": str(exc)})
            return

        response = _parse_final(final_text)
        yield _sse(
            {
                "type": "final",
                "session_id": session_id,
                "response": response.model_dump(),
            }
        )

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
