"""Canonical in-memory graph model shared across the graph tools.

Historically two graph shapes coexisted in the codebase and drifted apart:

  - `graph_operation` used ``{"nodes": ["A", ...],
    "edges": [{"source", "target", "label"}, ...]}`` — bare-string nodes and a
    ``label`` attitude key.
  - `output` / the UI schema (`src.schema`) used
    ``{"nodes": [{"id", "weight"}], "edges": [{"source", "target", "relation",
    "weight"}]}`` — object nodes and a ``relation`` key.

Because the attitude key (``label`` vs ``relation``) and node shape
(string vs object) differed, a graph produced by one path could not be fed into
the other — e.g. loading a `graph_operation` artifact and returning it as the
agent's final graph failed schema validation.

`Graph` is the single source of truth. It parses EITHER historical shape and
always serialises to the canonical, schema-compatible form:

    {"nodes":  [{"id", "weight"}],
     "edges":  [{"source", "target", "relation", "weight"}]}

Pure / framework-agnostic on purpose (mirrors the other `src.tools.*`
functions): plain dict in, plain dict out, no pydantic / ADK dependency.
"""

from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True)
class Node:
    """A graph node identified by `id`, with a relative `weight`."""

    id: str
    weight: float = 1.0


@dataclass(frozen=True)
class Edge:
    """A directed `source -> target` edge carrying a `relation` attitude."""

    source: str
    target: str
    relation: str = "neutral"
    weight: float = 1.0


def _norm_relation(value: object) -> str:
    """Normalise an attitude value to a clean, lowercase label.

    Empty / missing values default to ``"neutral"``. Kept relation-type
    agnostic (no hard-coded facet whitelist) so the model works for any
    relation, with sentiment being just one configuration.
    """
    text = str(value if value is not None else "").strip().lower()
    return text or "neutral"


def _relation_of(record: dict) -> str:
    """Read the attitude from a record under any of the accepted keys."""
    return _norm_relation(
        record.get("relation")
        if record.get("relation") is not None
        else record.get("label")
        if record.get("label") is not None
        else record.get("sentiment")
    )


def _as_float(value: object, default: float = 1.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


class Graph:
    """A relation graph with a single, canonical serialization.

    Nodes are de-duplicated by `id` and edges by ``(source, target,
    relation)``, keeping the first occurrence's weight.
    """

    def __init__(self, nodes=None, edges=None) -> None:
        self._nodes: dict[str, Node] = {}
        self._edges: dict[tuple[str, str, str], Edge] = {}
        for node in nodes or []:
            self.add_node(node)
        for edge in edges or []:
            self.add_edge(edge)

    # ------------------------------------------------------------------ #
    # Mutation
    # ------------------------------------------------------------------ #
    def add_node(self, node: Node) -> None:
        if node.id and node.id not in self._nodes:
            self._nodes[node.id] = node

    def add_edge(self, edge: Edge) -> None:
        if not edge.source or not edge.target:
            return
        # Endpoints must exist as nodes (no-op if already present, so explicit
        # node weights from `from_relations` / `from_dict` are preserved).
        self.add_node(Node(edge.source))
        self.add_node(Node(edge.target))
        key = (edge.source, edge.target, edge.relation)
        if key not in self._edges:
            self._edges[key] = edge

    # ------------------------------------------------------------------ #
    # Constructors
    # ------------------------------------------------------------------ #
    @classmethod
    def from_relations(cls, relations: list[dict]) -> "Graph":
        """Build a graph from `classify_relations` output.

        Each relation is a ``{source, target, label}`` dict (``relation`` /
        ``sentiment`` keys are also accepted). Node `weight` is how many
        relations the entity participates in; edge `weight` is the
        per-``(source, target, relation)`` occurrence count normalised into
        ``(0, 1]``. Relations missing a source or target are skipped.
        """
        node_freq: Counter[str] = Counter()
        edge_freq: Counter[tuple[str, str, str]] = Counter()

        for record in relations or []:
            if not isinstance(record, dict):
                continue
            source = str(record.get("source") or "").strip()
            target = str(record.get("target") or "").strip()
            if not source or not target:
                continue
            relation = _relation_of(record)
            node_freq[source] += 1
            node_freq[target] += 1
            edge_freq[(source, target, relation)] += 1

        max_edge = max(edge_freq.values(), default=1)
        graph = cls()
        for name, count in node_freq.items():
            graph.add_node(Node(name, float(count)))
        for (source, target, relation), count in edge_freq.items():
            graph.add_edge(
                Edge(source, target, relation, round(count / max_edge, 3))
            )
        return graph

    @classmethod
    def from_dict(cls, data: object) -> "Graph":
        """Build a graph from a graph dict in EITHER historical shape.

        Tolerates:
          - a wrapper ``{"graph": {...}}`` (the offloaded `graph_operation`
            artifact shape),
          - bare-string nodes or ``{"id"|"name", "weight"}`` node objects,
          - edges keyed by ``relation`` (canonical), ``label`` (legacy
            `graph_operation`), or ``sentiment``.
        Anything unrecognised is skipped rather than raising.
        """
        if not isinstance(data, dict):
            return cls()
        if isinstance(data.get("graph"), dict):
            data = data["graph"]

        nodes: list[Node] = []
        for raw in data.get("nodes") or []:
            if isinstance(raw, str):
                if raw:
                    nodes.append(Node(raw))
            elif isinstance(raw, dict):
                node_id = raw.get("id") or raw.get("name")
                if node_id:
                    nodes.append(Node(str(node_id), _as_float(raw.get("weight"))))

        edges: list[Edge] = []
        for raw in data.get("edges") or []:
            if not isinstance(raw, dict):
                continue
            source = str(raw.get("source") or "").strip()
            target = str(raw.get("target") or "").strip()
            if not source or not target:
                continue
            edges.append(
                Edge(source, target, _relation_of(raw), _as_float(raw.get("weight")))
            )

        return cls(nodes, edges)

    # ------------------------------------------------------------------ #
    # Serializer
    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict:
        """Serialise to the canonical, schema-compatible graph dict."""
        return {
            "nodes": [
                {"id": node.id, "weight": node.weight}
                for node in self._nodes.values()
            ],
            "edges": [
                {
                    "source": edge.source,
                    "target": edge.target,
                    "relation": edge.relation,
                    "weight": edge.weight,
                }
                for edge in self._edges.values()
            ],
        }

    # ------------------------------------------------------------------ #
    # Set operations
    # ------------------------------------------------------------------ #
    def union(self, other: "Graph") -> "Graph":
        """All nodes and edges present in either graph (deduplicated)."""
        result = Graph(
            list(self._nodes.values()), list(self._edges.values())
        )
        for node in other._nodes.values():
            result.add_node(node)
        for edge in other._edges.values():
            result.add_edge(edge)
        return result

    def intersection(self, other: "Graph") -> "Graph":
        """Only nodes and edges present in BOTH graphs."""
        result = Graph()
        for node_id, node in self._nodes.items():
            if node_id in other._nodes:
                result.add_node(node)
        for key, edge in self._edges.items():
            if key in other._edges:
                result.add_edge(edge)
        return result
