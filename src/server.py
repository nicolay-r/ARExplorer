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
from fastapi.responses import FileResponse  # noqa: E402
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


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
