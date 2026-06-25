"""Integration script for the bulk-ner tool (src/tools/ner.py).

Submits texts through `extract_named_entities` and prints the resulting
annotated output. Uses the local spaCy NER provider
(test/providers/spacy_383.py), so no API token is required.

Run:
    python test/test_ner.py
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

from tools.ner import extract_named_entities  # noqa: E402

PROVIDERS_DIR = os.path.join(ROOT, "test", "providers")

TEXTS = [
    "It was in July, 1805, and the speaker was the well-known Anna Pávlovna.",
    "Napoleon marched his army towards Moscow in the autumn of 1812.",
]


def test_extract_named_entities():
    """Submit texts and return the annotated output."""
    # Use the local spaCy provider (test/providers/spacy_383.py).
    result = extract_named_entities(
        TEXTS,
        src_dir=PROVIDERS_DIR,
        class_filepath="spacy_383.py",
        class_name="SpacyNER",
        model="en_core_web_sm",
    )

    assert result["status"] == "success", result
    assert len(result["documents"]) == len(TEXTS)

    for doc in result["documents"]:
        print("TEXT:      ", doc["text"])
        print("ANNOTATION:", doc["annotation"])
        print("ENTITIES:  ", doc["entities"])
        print("-" * 70)

    return result


if __name__ == "__main__":
    test_extract_named_entities()
