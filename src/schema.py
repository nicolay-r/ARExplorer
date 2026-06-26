"""Structured output schema for the ARExplorer agent.

The agent answers with a single JSON object (ADK 2.0 `output_schema`) that the
UI splits into two parts:

  - `message` -> rendered in the left chat panel.
  - `graph`   -> rendered in the main d3.js visualization panel.

See the `adk-structured-output` skill for how `output_schema` is combined with
`tools` via the auto-injected `set_model_response` tool.
"""

from typing import List

from pydantic import BaseModel, Field


class GraphNode(BaseModel):
    id: str = Field(description="Unique entity name, used as the node label.")
    weight: float = Field(
        default=1.0,
        description="Relative importance / frequency of the entity (>= 0). "
        "Drives node opacity (force) and leaf size (radial).",
    )


class GraphEdge(BaseModel):
    source: str = Field(description="Source node id (the entity holding the attitude).")
    target: str = Field(description="Target node id (the entity the attitude is about).")
    relation: str = Field(
        description="Relation/attitude label, e.g. 'positive', 'negative', "
        "'neutral'. Drives edge color.",
    )
    weight: float = Field(
        default=1.0,
        description="Relation strength / confidence in [0, 1]. Drives edge width.",
    )


class GraphData(BaseModel):
    nodes: List[GraphNode] = Field(
        default_factory=list,
        description="Entities to display as graph nodes.",
    )
    edges: List[GraphEdge] = Field(
        default_factory=list,
        description="Attitudes/relations between entities, as directed edges.",
    )


class AgentResponse(BaseModel):
    """Top-level structured response returned by the agent."""

    message: str = Field(
        description="Natural-language reply for the chat panel. Summarize what "
        "was found and how the graph should be read. Never leave empty.",
    )
    layout: str = Field(
        default="force",
        description="Suggested visualization for the graph: 'force' or 'radial'.",
    )
    graph: GraphData = Field(
        default_factory=GraphData,
        description="Graph to visualize. Leave nodes/edges empty when the reply "
        "is purely conversational and has nothing to plot.",
    )
