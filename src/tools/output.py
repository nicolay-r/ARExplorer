"""Tool #5 — Build the final visualization graph from classified relations.

The agent's structured `AgentResponse` carries a `graph` that can be large.
Having the LLM hand-write that graph into `set_model_response` is expensive and
error-prone (e.g. emitting `label` instead of `relation`, or truncating long
edge lists). This module deterministically converts the output of
`classify_relations` (a list of ``{source, target, label}`` dicts, typically
loaded from a session artifact) into the ``{"nodes": [...], "edges": [...]}``
graph the UI schema expects.

Pure / framework-agnostic on purpose (mirrors the other `src.tools.*`
functions); the ADK-aware wrapper that persists the result as an artifact lives
in `src.agent`.
"""

from collections import Counter

VALID_LABELS = {"positive", "negative", "neutral"}


def _relation_label(rel: dict) -> str:
    """Pick the attitude label from a relation record, normalised.

    Accepts `label` (the `classify_relations` key), `relation` (the UI key),
    or `sentiment`, and snaps the value onto one of the valid facets.
    """
    raw = rel.get("label") or rel.get("relation") or rel.get("sentiment") or ""
    text = str(raw).strip().lower()
    for label in VALID_LABELS:
        if label in text:
            return label
    return "neutral"


def build_output_graph(relations: list[dict]) -> dict:
    """Convert classified relations into a UI-ready graph.

    Args:
        relations: list of ``{source, target, label}`` dicts (the
            `classify_relations` output; `reasoning` and extra keys are
            ignored). `label` may also arrive as `relation` / `sentiment`.

    Returns:
        ``{"nodes": [{"id", "weight"}], "edges": [{"source", "target",
        "relation", "weight"}]}`` where:
          - node `weight` is how many relations the entity participates in
            (drives node opacity / leaf size in the UI), and
          - edge `weight` is the per-(source, target, relation) occurrence
            count normalised into ``(0, 1]`` (drives edge width).
        Duplicate edges are merged; relations missing a source or target are
        skipped.
    """
    node_freq: Counter[str] = Counter()
    edge_freq: Counter[tuple[str, str, str]] = Counter()

    for rel in relations or []:
        if not isinstance(rel, dict):
            continue
        source = (rel.get("source") or "").strip()
        target = (rel.get("target") or "").strip()
        if not source or not target:
            continue
        label = _relation_label(rel)
        node_freq[source] += 1
        node_freq[target] += 1
        edge_freq[(source, target, label)] += 1

    max_edge = max(edge_freq.values(), default=1)

    nodes = [
        {"id": name, "weight": float(count)}
        for name, count in node_freq.items()
    ]
    edges = [
        {
            "source": source,
            "target": target,
            "relation": label,
            "weight": round(count / max_edge, 3),
        }
        for (source, target, label), count in edge_freq.items()
    ]

    return {"nodes": nodes, "edges": edges}
