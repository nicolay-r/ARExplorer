"""Tool #1 — Named Entity Recognition over large text collections (bulk-ner).

Wraps the `bulk-ner` Python API as a plain function so it can be exposed both as
an ADK tool and, later, through an MCP server. Solves the LM context-limit
problem by annotating entities over a massive volume of texts and returning a
readable, LLM-friendly representation.

See the `bulk-ner` skill for the underlying API.
"""

from functools import lru_cache


@lru_cache(maxsize=None)
def _get_annotator(src_dir, class_filepath, class_name, model, chunk_limit):
    """Lazily build and cache the NER annotator for a given configuration.

    Imports happen here so the module can be imported without the heavy
    `bulk-ner` dependency (and its model) being installed.
    """
    from bulk_ner.api import NERAnnotator
    from bulk_ner.src.service_dynamic import dynamic_init

    ner_model = dynamic_init(
        src_dir=src_dir,
        # TODO: remove hardcoded values.
        class_filepath=class_filepath or "dp_130.py",
        class_name=class_name or "DeepPavlovNER",
    )(model=model or "ner_ontonotes_bert")

    return NERAnnotator(
        ner_model=ner_model,
        entity_func=lambda t: [t.Value, t.Type, t.ID],
        chunk_limit=chunk_limit,
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


def extract_named_entities(
    texts: list[str] | None = None,
    batch_size: int = 10,
    src_dir: str | None = None,
    class_filepath: str | None = None,
    class_name: str | None = None,
    model: str | None = None,
    chunk_limit: int = 512,
) -> dict:
    """Extract named entities from a collection of texts.

    Annotates each text with named entities (people, dates, locations,
    organizations, etc.) using bulk-ner. Designed for large volumes of text:
    it chunks long inputs so the language model context limit is never exceeded.

    Args:
        texts: The texts to annotate. May be ``None`` when the caller intends
            to supply the texts via an artifact reference — the agent-level
            `inflate_artifact_inputs` before_tool_callback fills this in
            before the function runs.
        batch_size: How many texts to process per batch.
        src_dir: Directory containing the NER provider script.
        class_filepath: Provider script filename (relative to src_dir).
        class_name: Provider class to instantiate from the script.
        model: Model identifier passed to the provider.
        chunk_limit: Maximum chunk size (in tokens) fed to the model.

    Returns:
        A dict with:
        - status: "success" or "error".
        - documents: list of per-text results, each containing the original
          `text`, the raw bulk-ner `annotation` (mixed spans + entities), and a
          flattened `entities` list of {value, type, id}.
        - error: present only when status is "error".
    """
    if texts is None:
        return {
            "status": "error",
            "error": (
                "extract_named_entities: provide `texts` inline or supply a "
                "`texts_artifact` filename so the before_tool_callback can "
                "inflate it."
            ),
        }
    if not texts:
        return {"status": "success", "documents": []}

    try:
        annotator = _get_annotator(
            src_dir, class_filepath, class_name, model, chunk_limit
        )
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
