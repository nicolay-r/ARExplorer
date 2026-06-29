import json
import logging
from typing import Optional

from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmResponse
from google.genai import types as genai_types

logger = logging.getLogger(__name__)

# Shown when the model ends a turn with no content and no tool call.
_EMPTY_TURN_MESSAGE = (
    "I couldn't produce a response on that attempt — the model returned an "
    "empty turn. Please try sending your request again."
)

# finish_reason values that are not themselves a sign of trouble worth showing.
_BENIGN_FINISH_REASONS = {"STOP", "FINISH_REASON_UNSPECIFIED", "NONE", ""}


def _is_empty_response(llm_response: LlmResponse) -> bool:
    """True when the model produced neither text nor a function call."""
    content = llm_response.content
    if content is None or not content.parts:
        return True
    for part in content.parts:
        if getattr(part, "function_call", None):
            return False
        if (getattr(part, "text", None) or "").strip():
            return False
    return True


def ensure_nonempty_response(
    *,
    callback_context: CallbackContext,
    llm_response: LlmResponse,
) -> Optional[LlmResponse]:
    """Substitute a readable fallback when the model returns an empty turn.

    Returns ``None`` (no override) for streaming partials, error responses, and
    any response that already carries text or a function call; otherwise it
    returns a replacement `LlmResponse` carrying a short, honest message.
    """
    if llm_response is None or getattr(llm_response, "partial", False):
        return None
    if llm_response.error_code or llm_response.error_message:
        return None
    if not _is_empty_response(llm_response):
        return None

    # `finish_reason` may be a plain string or a genai enum (whose str() is
    # e.g. "FinishReason.STOP"); normalise to the bare name for comparison.
    reason = getattr(llm_response, "finish_reason", None)
    reason_name = (getattr(reason, "name", None) or str(reason or "")).upper()
    message = _EMPTY_TURN_MESSAGE
    if reason is not None and reason_name not in _BENIGN_FINISH_REASONS:
        message = (
            "I couldn't produce a response on that attempt "
            f"(finish reason: {reason_name}). Please try sending your request "
            "again."
        )

    logger.warning(
        "ensure_nonempty_response: empty model turn (finish_reason=%s); "
        "substituting fallback message.",
        reason,
    )

    return LlmResponse(
        content=genai_types.Content(
            role="model",
            parts=[genai_types.Part(text=message)],
        )
    )
