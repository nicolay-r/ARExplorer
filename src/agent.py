"""ARExplorer agent — extracts Attitudes and Relations from documents.

ADK 2.0 root agent exposing three tools (wrapped over their Python APIs):
  #1 extract_named_entities  — bulk-ner annotation over massive text collections
  #2 classify_relations      — bulk-chain attitude/relation classification
  #3 graph_operation         — union / intersection over attitude graphs
"""

import os

from google.adk.agents import Agent
from google.adk.models import Gemini
from google.genai import types

from src.tools import (
    extract_named_entities as _extract_named_entities,
    classify_relations as _classify_relations,
    graph_operation,
)


def extract_named_entities(texts: list[str], batch_size: int = 10) -> dict:
    return _extract_named_entities(
        texts,
        batch_size=batch_size,
        src_dir=os.environ.get("NER_SRC_DIR"),
        class_filepath=os.environ.get("NER_CLASS_FILEPATH"),
        class_name=os.environ.get("NER_CLASS_NAME"),
        model=os.environ.get("NER_MODEL"),
    )


def classify_relations(
    pairs: list[dict],
    relation_type: str = "sentiment",
    batch_size: int = 10,
) -> dict:
    return _classify_relations(
        pairs,
        relation_type=relation_type,
        batch_size=batch_size,
        provider_filepath=os.environ.get("RELATION_PROVIDER_FILEPATH"),
        model_name=os.environ.get("RELATION_MODEL"),
        api_token=os.environ.get("REPLICATE_API_TOKEN", ""),
    )


INSTRUCTION = """\
You are ARExplorer, an assistant that extracts and explores Attitudes and
Relations between named entities found in documents.

Your typical workflow:
1. Use `extract_named_entities` to annotate named entities across the provided
   texts. This handles large volumes by chunking, so prefer it over reasoning
   about entities yourself.
2. Form candidate entity pairs from the extracted entities and use
   `classify_relations` to determine the attitude (positive, negative, or
   neutral) the source entity expresses towards the target entity.
3. When the user wants to combine or compare result sets, build graphs of the
   form {"nodes": [...], "edges": [{"source", "target", "label"}]} and use
   `graph_operation` with "union" or "intersection".

Be transparent about tool errors and ask for missing inputs (e.g. text context
for a relation) rather than guessing.
"""

root_agent = Agent(
    name="arexplorer_agent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=types.HttpRetryOptions(attempts=3, initial_delay=1.0),
    ),
    instruction=INSTRUCTION,
    tools=[
        extract_named_entities,
        classify_relations,
        graph_operation,
    ],
)
