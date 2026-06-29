"""FastAPI server exposing the ARExplorer agent over REST + serving the UI.

Run from the project root::

    uvicorn src.server:app --port 8000 --reload

Then open http://127.0.0.1:8000/ — the left panel chats with the agent and the
main panel renders the returned graph with d3.js (force / radial layouts).

The agent replies with a structured `AgentResponse` (see `src/schema.py`); the
`/api/chat` endpoint returns that JSON straight to the browser.
"""

import asyncio
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

from src import progress  # noqa: E402
from src.agent import OUTPUT_ARTIFACT_NAME, root_agent  # noqa: E402
from src.schema import AgentResponse  # noqa: E402

APP_NAME = "arexplorer"
STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="ARExplorer UI")

_session_service = InMemorySessionService()
_artifact_service = InMemoryArtifactService()
_runner = Runner(
    agent=root_agent,
    app_name=APP_NAME,
    session_service=_session_service,
    artifact_service=_artifact_service,
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


def _output_artifact_ref(event) -> dict | None:
    """Return the artifact pointer if this event is an `output` tool response.

    The `output` tool builds the graph deterministically and persists the
    finished `AgentResponse` as a versioned artifact, returning
    ``{"artifact": {"name", "version"}}``. We capture that pointer so the
    server can load the authoritative response for THIS turn's tool call
    (loading by version keeps it turn-safe — a later conversational turn that
    doesn't call `output` won't resurrect a stale graph).
    """
    for resp in event.get_function_responses():
        if resp.name != "output":
            continue
        payload = resp.response
        if isinstance(payload, dict) and isinstance(payload.get("artifact"), dict):
            artifact = payload["artifact"]
            if artifact.get("name"):
                return artifact
    return None


async def _load_output_response(
    session_id: str, ref: dict
) -> AgentResponse | None:
    """Load + validate the AgentResponse the `output` tool saved this turn."""
    try:
        part = await _artifact_service.load_artifact(
            app_name=APP_NAME,
            user_id=APP_NAME,
            session_id=session_id,
            filename=ref["name"],
            version=ref.get("version"),
        )
    except Exception:  # noqa: BLE001 - any artifact-service hiccup falls back
        return None
    if part is None or part.inline_data is None or part.inline_data.data is None:
        return None

    data = part.inline_data.data
    text = data.decode("utf-8") if isinstance(data, bytes) else data
    try:
        return AgentResponse.model_validate_json(text)
    except Exception:  # noqa: BLE001
        return None


# Shown when the model ends its turn without producing any usable response
# (e.g. an empty `finishReason: STOP` turn with no parts and no tool call).
_EMPTY_TURN_FALLBACK = (
    "The model ended its turn without returning a response — this happens "
    "intermittently with the lightweight model. Please try sending your "
    "request again."
)

# Shown when the structured output carries a payload but the model left the
# natural-language `message` empty. Intentionally task-agnostic: it makes no
# assumption about what the schema's payload fields represent.
_RESULT_READY_FALLBACK = "Here is the result for your request."


def _carries_payload(response: AgentResponse) -> bool:
    """True if the structured output has content beyond the chat `message`.

    Stays task-agnostic by comparing every non-`message` field against a
    pristine, defaults-only instance of the same model rather than inspecting
    any specific field (e.g. `graph`). If `AgentResponse` later gains or
    changes payload fields, this keeps working without edits.
    """
    current = response.model_dump()
    try:
        baseline = type(response)(message=current.get("message", "")).model_dump()
    except Exception:  # noqa: BLE001 - schema without a trivially-empty instance
        baseline = {}
    for name, value in current.items():
        if name == "message":
            continue
        if value and value != baseline.get(name):
            return True
    return False


def _ensure_message(response: AgentResponse) -> AgentResponse:
    """Guarantee the user always sees something on termination.

    The model occasionally returns an empty turn (`finishReason: STOP` with no
    content and no `set_model_response` call), which would otherwise surface as
    a blank chat bubble. If the structured output still carries a payload we
    acknowledge it generically; otherwise we explain the empty turn and invite
    a retry. Stays decoupled from what the workflow actually produces.
    """
    if response.message and response.message.strip():
        return response

    response.message = (
        _RESULT_READY_FALLBACK
        if _carries_payload(response)
        else _EMPTY_TURN_FALLBACK
    )
    return response


async def _finalize(
    session_id: str, final_text: str, output_ref: dict | None
) -> AgentResponse:
    """Pick the authoritative final response for a turn.

    Prefers the graph the `output` tool built (so the model never has to
    hand-write a long graph into `set_model_response`); otherwise falls back to
    parsing the model's structured final text. The result always carries a
    non-empty `message` (see `_ensure_message`) so an empty model turn never
    reaches the UI as a blank bubble.
    """
    if output_ref is not None:
        built = await _load_output_response(session_id, output_ref)
        if built is not None:
            return _ensure_message(built)
    return _ensure_message(_parse_final(final_text))


# Human-readable verbs for the tools the agent can call, used to turn raw
# event traffic into a friendly activity log streamed to the UI.
_TOOL_LABELS = {
    "extract_named_entities": "Extracting named entities",
    "form_entity_pairs": "Forming entity pairs",
    "classify_relations": "Classifying relations",
    "graph_operation": "Updating the relation graph",
    "output": "Building the output graph",
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
    output_ref = None
    async for event in _runner.run_async(
        user_id=APP_NAME, session_id=session_id, new_message=message
    ):
        output_ref = _output_artifact_ref(event) or output_ref
        if event.is_final_response() and event.content and event.content.parts:
            final_text = event.content.parts[0].text or ""

    response = await _finalize(session_id, final_text, output_ref)
    return {"session_id": session_id, "response": response.model_dump()}


@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest) -> StreamingResponse:
    """Run the agent and stream server-side activity as Server-Sent Events.

    Emits one `log` message per tool call / tool result (and any intermediate
    model text) as the agent works, `progress` messages while a long-running
    tool reports incremental status (see `src.progress`), then a single `final`
    message carrying the parsed `AgentResponse`. Errors are surfaced as an
    `error` message so the UI can stop the thinking indicator gracefully.

    The agent run is driven in a background task that feeds a queue, while
    long-running tools publish progress onto the SAME queue from a worker
    thread. This lets us interleave progress with events even though a single
    tool call blocks the `run_async` iteration until it returns.
    """
    session_id = req.session_id or uuid.uuid4().hex

    async def event_stream():
        await _ensure_session(session_id)
        yield _sse({"type": "session", "session_id": session_id})

        message = types.Content(
            role="user", parts=[types.Part.from_text(text=req.message)]
        )

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def publish(item: dict) -> None:
            # Called from a tool's worker thread; hop back onto the loop safely.
            loop.call_soon_threadsafe(queue.put_nowait, ("progress", item))

        async def run_agent() -> None:
            try:
                async for event in _runner.run_async(
                    user_id=APP_NAME, session_id=session_id, new_message=message
                ):
                    await queue.put(("event", event))
            except Exception as exc:  # noqa: BLE001 - surfaced to the UI below
                await queue.put(("error", exc))
            finally:
                await queue.put(("done", None))

        # Install the publisher BEFORE spawning the task so the copied context
        # the task runs in (and thus the tool wrappers) sees it.
        token = progress.set_publisher(publish)
        task = asyncio.create_task(run_agent())

        final_text = ""
        output_ref = None
        try:
            while True:
                kind, payload = await queue.get()
                if kind == "progress":
                    yield _sse(payload)
                elif kind == "event":
                    event = payload
                    for line in _event_logs(event):
                        yield _sse({"type": "log", "text": line})
                    output_ref = _output_artifact_ref(event) or output_ref
                    if (
                        event.is_final_response()
                        and event.content
                        and event.content.parts
                    ):
                        final_text = event.content.parts[0].text or ""
                elif kind == "error":
                    yield _sse({"type": "error", "error": str(payload)})
                    return
                elif kind == "done":
                    break
        finally:
            progress.reset_publisher(token)
            if not task.done():
                task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass

        response = await _finalize(session_id, final_text, output_ref)
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
