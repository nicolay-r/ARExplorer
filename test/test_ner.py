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

# Point bulk-ner at the local spaCy provider BEFORE importing the tool, since
# ner.py reads these into module-level constants at import time.
os.environ.setdefault("BULK_NER_SRC_DIR", os.path.join(ROOT, "test", "providers"))
os.environ.setdefault("BULK_NER_CLASS_FILEPATH", "spacy_383.py")
os.environ.setdefault("BULK_NER_CLASS_NAME", "SpacyNER")
os.environ.setdefault("BULK_NER_MODEL", "en_core_web_sm")

from tools.ner import extract_named_entities  # noqa: E402

TEXTS = [
    "It was in July, 1805, and the speaker was the well-known Anna Pávlovna.",
    "Napoleon marched his army towards Moscow in the autumn of 1812.",
]


def test_extract_named_entities():
    """Submit texts and return the annotated output."""
    result = extract_named_entities(TEXTS)

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
