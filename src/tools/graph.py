"""Tool #3 — Graph set operations (union / intersection).

A self-contained tool (no third-party resources, hence no MCP needed) that
combines two attitude/relation graphs. Useful for search-and-filter over a
massive document database expressed as graphs.

A graph is represented as::

    {
        "nodes": ["Anna", "Pierre", ...],
        "edges": [{"source": "Anna", "target": "Pierre", "label": "positive"}, ...],
    }
"""


def _node_key(node) -> str:
    return node if isinstance(node, str) else str(node)


def _edge_key(edge: dict) -> tuple:
    return (
        edge.get("source"),
        edge.get("target"),
        edge.get("label"),
    )


def _dedup_nodes(nodes) -> list:
    seen = set()
    out = []
    for n in nodes or []:
        k = _node_key(n)
        if k not in seen:
            seen.add(k)
            out.append(n)
    return out


def _dedup_edges(edges) -> list:
    seen = set()
    out = []
    for e in edges or []:
        k = _edge_key(e)
        if k not in seen:
            seen.add(k)
            out.append(e)
    return out


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
        graph_a: First graph with "nodes" and "edges" lists. May be ``None``
            when the caller intends to supply it via an artifact reference —
            the agent-level `inflate_artifact_inputs` before_tool_callback
            fills it in before the function runs.
        graph_b: Second graph with "nodes" and "edges" lists. Same artifact
            semantics as `graph_a`.

    Returns:
        A dict with:
        - status: "success" or "error".
        - graph: the resulting {"nodes": [...], "edges": [...]}.
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

    nodes_a = graph_a.get("nodes", []) if isinstance(graph_a, dict) else []
    nodes_b = graph_b.get("nodes", []) if isinstance(graph_b, dict) else []
    edges_a = graph_a.get("edges", []) if isinstance(graph_a, dict) else []
    edges_b = graph_b.get("edges", []) if isinstance(graph_b, dict) else []

    if op == "union":
        nodes = _dedup_nodes(list(nodes_a) + list(nodes_b))
        edges = _dedup_edges(list(edges_a) + list(edges_b))
    else:  # intersection
        b_node_keys = {_node_key(n) for n in nodes_b}
        nodes = _dedup_nodes(
            [n for n in nodes_a if _node_key(n) in b_node_keys]
        )
        b_edge_keys = {_edge_key(e) for e in edges_b}
        edges = _dedup_edges(
            [e for e in edges_a if _edge_key(e) in b_edge_keys]
        )

    return {"status": "success", "graph": {"nodes": nodes, "edges": edges}}
