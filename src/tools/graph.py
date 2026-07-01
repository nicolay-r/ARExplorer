"""Tool #3 — Graph set operations (union / intersection).

A self-contained tool (no third-party resources, hence no MCP needed) that
combines two attitude/relation graphs. Useful for search-and-filter over a
massive document database expressed as graphs.

Both inputs are parsed through the canonical `Graph` model (see
`core.graph.Graph`), which tolerates either historical graph shape, and the
result is always serialised to the canonical, schema-compatible form::

    {
        "nodes": [{"id": "Anna", "weight": 1.0}, ...],
        "edges": [{"source": "Anna", "target": "Pierre",
                   "relation": "positive", "weight": 1.0}, ...],
    }

This guarantees a `graph_operation` result can be fed straight into the agent's
final `AgentResponse` graph (and to the `output` tool) without key/shape
mismatches.
"""

from ..core.graph import Graph


def graph_operation(
    operation: str,
    graph_a: dict | None = None,
    graph_b: dict | None = None,
) -> dict:
    """Combine two graphs with a set operation.

    Args:
        operation: Either "union" or "intersection".
            - union: all nodes and edges present in either graph (deduplicated).
            - intersection: only nodes and edges present in both graphs.
        graph_a: First graph (any accepted shape; see `Graph.from_dict`). May be
            ``None`` when supplied via an artifact reference — the agent-level
            `inflate_artifact_inputs` before_tool_callback fills it in before
            the function runs.
        graph_b: Second graph. Same artifact semantics as `graph_a`.

    Returns:
        A dict with:
        - status: "success" or "error".
        - graph: the resulting canonical
          ``{"nodes": [{"id", "weight"}], "edges": [{"source", "target",
          "relation", "weight"}]}``.
        - error: present only when status is "error".
    """
    op = (operation or "").strip().lower()
    if op not in {"union", "intersection"}:
        return {
            "status": "error",
            "error": f"Unknown operation '{operation}'. Use 'union' or 'intersection'.",
        }

    missing = [
        name
        for name, value in (("graph_a", graph_a), ("graph_b", graph_b))
        if value is None
    ]
    if missing:
        return {
            "status": "error",
            "error": (
                f"graph_operation: missing {', '.join(missing)}. Provide each "
                "graph inline or supply a `<name>_artifact` filename so the "
                "before_tool_callback can inflate it."
            ),
        }

    ga = Graph.from_dict(graph_a)
    gb = Graph.from_dict(graph_b)
    result = ga.union(gb) if op == "union" else ga.intersection(gb)

    return {"status": "success", "graph": result.to_dict()}
