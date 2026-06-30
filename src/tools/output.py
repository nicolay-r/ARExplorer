"""Tool #5 — Build the final visualization graph from classified relations.

The agent's structured `AgentResponse` carries a `graph` that can be large.
Having the LLM hand-write that graph into `set_model_response` is expensive and
error-prone (e.g. emitting `label` instead of `relation`, or truncating long
edge lists). This module deterministically converts the output of
`classify_relations` (a list of ``{source, target, label}`` dicts, typically
loaded from a session artifact) into the canonical graph the UI schema expects.

The conversion is delegated to the shared `Graph` model so that graphs built
here are byte-for-byte consistent with those produced by `graph_operation`.

Pure / framework-agnostic on purpose (mirrors the other `src.tools.*`
functions); the ADK-aware wrapper that persists the result as an artifact lives
in `src.agent`.
"""

from .graph_model import Graph


def build_output_graph(relations: list[dict]) -> dict:
    """Convert classified relations into a UI-ready, canonical graph.

    Args:
        relations: list of ``{source, target, label}`` dicts (the
            `classify_relations` output; `reasoning` and extra keys are
            ignored). `label` may also arrive as `relation` / `sentiment`.

    Returns:
        ``{"nodes": [{"id", "weight"}], "edges": [{"source", "target",
        "relation", "weight"}]}`` — see `Graph.from_relations` for the weight
        semantics. Duplicate edges are merged; relations missing a source or
        target are skipped.
    """
    return Graph.from_relations(relations).to_dict()
