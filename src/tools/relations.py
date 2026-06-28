"""Tool #2 — Relation / attitude classification between entities (bulk-chain).

Given pairs of entities in textual context, classify the attitude (relation)
that the source entity expresses towards the target entity. By default this
registers a *sentiment* relation with three facets: positive, negative, neutral.

Uses the `bulk-chain` batching framework (LLM-as-a-service) with a
Chain-of-Thought schema parameterized by the TYPE OF RELATION. See the
`bulk-chain` skill for the underlying API.
"""

import asyncio
import concurrent.futures
from functools import lru_cache

VALID_LABELS = {"positive", "negative", "neutral"}


def _run_in_isolated_loop(fn):
    """Run a blocking callable in a worker thread that owns a fresh event loop.

    `bulk-chain` (async_mode) drives its batches via `loop.run_until_complete()`.
    When this tool is invoked from inside an already-running event loop — e.g.
    the FastAPI/ADK server thread — that call raises "Cannot run the event loop
    while another loop is running". Executing it in a dedicated thread gives
    bulk-chain a clean, loop-free context.
    """

    def _worker():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return fn()
        finally:
            asyncio.set_event_loop(None)
            loop.close()

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(_worker).result()


# TODO. Generalize to schema (as sengiment is a type of relation)
def _sentiment_schema(relation_type: str) -> list[dict]:
    """Build a Chain-of-Thought schema for a given relation type.

    The relation type is the parameter of the schema (e.g. "sentiment"),
    matching the design where the schema's parameter is the TYPE OF RELATION.
    """
    return [
        {
            "prompt": (
                "Context: {text}\n"
                f"Analyse the {relation_type} attitude expressed by the source "
                "entity '{source}' towards the target entity '{target}'. "
                "Reason briefly step by step."
            ),
            "out": "reasoning",
        },
        {
            "prompt": (
                "Reasoning: {reasoning}\n"
                "Classify the attitude strictly as exactly one of: "
                "positive, negative, neutral. Respond with a single lowercase word."
            ),
            "out": "label",
        },
    ]


@lru_cache(maxsize=None)
def _get_llm(provider_filepath, model_name, api_token):
    """Lazily build and cache the bulk-chain LLM adapter for a configuration."""
    from bulk_chain.core.utils import dynamic_init

    kwargs = {"model_name": model_name}
    if api_token:
        kwargs["api_token"] = api_token
    return dynamic_init(class_filepath=provider_filepath)(**kwargs)


def _normalize_label(raw: str) -> str:
    """Map a raw model answer to one of the three valid facets."""
    text = (raw or "").strip().lower()
    for label in VALID_LABELS:
        if label in text:
            return label
    return "neutral"


def classify_relations(
    pairs: list[dict],
    relation_type: str = "sentiment",
    batch_size: int = 10,
    provider_filepath: str | None = None,
    # TODO: remove hardcoded values.
    model_name: str = "meta/meta-llama-3-70b-instruct",
    api_token: str = "",
) -> dict:
    """Classify the attitude/relation between pairs of entities.

    For every pair, determine the attitude the source entity holds towards the
    target entity within the given text, classified into one of three facets:
    positive, negative, or neutral.

    Args:
        pairs: Each item must contain:
            - text: the context sentence/document mentioning both entities.
            - source: the source entity (holder of the attitude).
            - target: the target entity (object of the attitude).
        relation_type: The type of relation to assess (default "sentiment").
            This is the parameter of the underlying schema.
        batch_size: How many pairs to query per LLM batch.
        provider_filepath: Path to the bulk-chain provider adapter script.
        model_name: Model identifier passed to the provider.
        api_token: API token for the provider (empty to rely on provider default).

    Returns:
        A dict with:
        - status: "success" or "error".
        - relations: list of {source, target, label, reasoning} where label is
          one of positive/negative/neutral.
        - error: present only when status is "error".
    """
    if not pairs:
        return {"status": "success", "relations": []}

    missing = [
        i
        for i, p in enumerate(pairs)
        if not all(k in p for k in ("text", "source", "target"))
    ]
    if missing:
        return {
            "status": "error",
            "error": f"Pairs at indices {missing} must each have 'text', 'source', 'target'.",
        }

    try:
        llm = _get_llm(provider_filepath, model_name, api_token)
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "error",
            "error": f"Failed to initialize bulk-chain LLM provider: {exc}",
        }

    from bulk_chain.api import iter_content

    def _collect() -> list:
        # The generator must be both created AND consumed inside the worker
        # thread, since iterating it is what drives bulk-chain's event loop.
        content_it = iter_content(
            schema=_sentiment_schema(relation_type),
            llm=llm,
            stream=False,
            async_mode=True,
            batch_size=batch_size,
            input_dicts_it=list(pairs),
        )
        out = []
        for batch in content_it:
            for entry in batch:
                out.append(
                    {
                        "source": entry.get("source"),
                        "target": entry.get("target"),
                        "label": _normalize_label(entry.get("label", "")),
                        "reasoning": entry.get("reasoning", ""),
                    }
                )
        return out

    try:
        relations = _run_in_isolated_loop(_collect)
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "error": f"Relation classification failed: {exc}"}

    return {"status": "success", "relations": relations}
