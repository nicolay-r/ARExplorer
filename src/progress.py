"""Out-of-band progress reporting for long-running tools.

Some tools (notably `classify_relations`, which drives `bulk-chain` over many
entity pairs) run for a long time inside a single ADK tool call. The normal ADK
event stream only yields a tool's *result* once it finishes, so the UI would
otherwise sit frozen with no feedback.

This module provides a tiny, transport-agnostic publish channel that lets a
tool emit incremental progress updates while it runs. The active publisher is
held in a `contextvars.ContextVar` so it is scoped to the current request/task
(and copied into any task spawned from it via `asyncio.create_task`):

  - The server installs a publisher for the duration of an agent run (see
    `src.server`); the publisher is a plain callable that forwards the update
    to the SSE client in a thread-safe way.
  - A tool wrapper reads the publisher with `get_publisher()` and hands a bound
    callback down to the blocking worker, which calls it from a worker thread.

When no publisher is installed (e.g. `adk web`, the non-streaming `/api/chat`
endpoint, or unit tests) `get_publisher()` returns ``None`` and progress
reporting is a no-op.
"""

import contextvars
from typing import Callable, Optional

# A publisher receives a single progress event dict, e.g.
# ``{"type": "progress", "tool": "classify_relations", "done": 30, "total": 124}``.
ProgressPublisher = Callable[[dict], None]

_publisher: contextvars.ContextVar[Optional[ProgressPublisher]] = (
    contextvars.ContextVar("are_progress_publisher", default=None)
)


def set_publisher(publisher: ProgressPublisher) -> contextvars.Token:
    """Install the active progress publisher; returns a token for `reset`."""
    return _publisher.set(publisher)


def reset_publisher(token: contextvars.Token) -> None:
    """Restore the publisher to its previous value (best-effort)."""
    try:
        _publisher.reset(token)
    except (ValueError, LookupError):
        # Token created in a different context (e.g. reset from another task);
        # safe to ignore since the contextvar is request/task scoped anyway.
        pass


def get_publisher() -> Optional[ProgressPublisher]:
    """Return the active publisher, or ``None`` when progress is not collected."""
    return _publisher.get()
