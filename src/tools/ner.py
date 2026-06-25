"""Tool #1 — Named Entity Recognition over large text collections (bulk-ner).

Wraps the `bulk-ner` Python API as a plain function so it can be exposed both as
an ADK tool and, later, through an MCP server. Solves the LM context-limit
problem by annotating entities over a massive volume of texts and returning a
readable, LLM-friendly representation.

See the `bulk-ner` skill for the underlying API.
"""

import os
from functools import lru_cache

# Model configuration (overridable via environment). Defaults mirror the
# bulk-ner skill example using a DeepPavlov OntoNotes BERT model.
NER_SRC_DIR = os.environ.get("BULK_NER_SRC_DIR", "models")
NER_CLASS_FILEPATH = os.environ.get("BULK_NER_CLASS_FILEPATH", "dp_130.py")
NER_CLASS_NAME = os.environ.get("BULK_NER_CLASS_NAME", "DeepPavlovNER")
NER_MODEL = os.environ.get("BULK_NER_MODEL", "ner_ontonotes_bert")
NER_CHUNK_LIMIT = int(os.environ.get("BULK_NER_CHUNK_LIMIT", "128"))


@lru_cache(maxsize=1)
def _get_annotator():
    """Lazily build and cache the NER annotator.

    Imports happen here so the module can be imported without the heavy
    `bulk-ner` dependency (and its model) being installed.
    """
    from bulk_ner.api import NERAnnotator
    from bulk_ner.src.service_dynamic import dynamic_init

    ner_model = dynamic_init(
        src_dir=NER_SRC_DIR,
        class_filepath=NER_CLASS_FILEPATH,
        class_name=NER_CLASS_NAME,
    )(model=NER_MODEL)

    return NERAnnotator(
        ner_model=ner_model,
        entity_func=lambda t: [t.Value, t.Type, t.ID],
        chunk_limit=NER_CHUNK_LIMIT,
    )


def _collect_entities(result: list) -> list:
    """Pull the entity tuples out of a bulk-ner `result` list.

    A result mixes plain string spans with `[value, type, id]` entity lists.
    """
    entities = []
    for item in result:
        if isinstance(item, list) and len(item) == 3:
            value, etype, eid = item
            entities.append({"value": value, "type": etype, "id": eid})
    return entities


def extract_named_entities(texts: list[str], batch_size: int = 10) -> dict:
    """Extract named entities from a collection of texts.

    Annotates each text with named entities (people, dates, locations,
    organizations, etc.) using bulk-ner. Designed for large volumes of text:
    it chunks long inputs so the language model context limit is never exceeded.

    Args:
        texts: The texts to annotate.
        batch_size: How many texts to process per batch.

    Returns:
        A dict with:
        - status: "success" or "error".
        - documents: list of per-text results, each containing the original
          `text`, the raw bulk-ner `annotation` (mixed spans + entities), and a
          flattened `entities` list of {value, type, id}.
        - error: present only when status is "error".
    """
    if not texts:
        return {"status": "success", "documents": []}

    try:
        annotator = _get_annotator()
    except Exception as exc:  # noqa: BLE001 - surface init/import failures to the LLM
        return {
            "status": "error",
            "error": f"Failed to initialize bulk-ner model: {exc}",
        }

    def data_dict_it():
        for t in texts:
            yield {"text": t}

    try:
        documents = []
        annotated_it = annotator.iter_annotated_data(
            data_dict_it=data_dict_it(),
            schema={"text": "{text}"},
            batch_size=batch_size,
        )
        for original, data in zip(texts, annotated_it):
            annotation = data["text"]
            documents.append(
                {
                    "text": original,
                    "annotation": annotation,
                    "entities": _collect_entities(annotation),
                }
            )
        return {"status": "success", "documents": documents}
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "error": f"NER annotation failed: {exc}"}
