"""Integration script for the bulk-chain relation tool (src/tools/relations.py).

Submits entity pairs through `classify_relations` and prints the resulting
attitude (positive / negative / neutral) for each pair. Uses the Replicate LLM
provider (test/providers/replicate_104.py).

Requires an API token exported before running:
    export BULK_CHAIN_API_TOKEN=<your-replicate-token>

Run:
    python test/test_relations.py
"""

import os
import sys
from dotenv import load_dotenv

load_dotenv()

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

# Point bulk-chain at the local Replicate provider BEFORE importing the tool,
# since relations.py reads these into module-level constants at import time.
os.environ.setdefault(
    "BULK_CHAIN_PROVIDER",
    os.path.join(ROOT, "test", "providers", "replicate_104.py"),
)
os.environ.setdefault("BULK_CHAIN_MODEL", "meta/meta-llama-3-70b-instruct")
os.environ.setdefault("BULK_CHAIN_API_TOKEN", os.getenv("REPLICATE_API_TOKEN"))

from tools.relations import classify_relations  # noqa: E402

PAIRS = [
    {
        "text": "Anna warmly praised Pierre for his courage and kindness.",
        "source": "Anna",
        "target": "Pierre",
    },
    {
        "text": "Napoleon despised the cowardice of General Mack.",
        "source": "Napoleon",
        "target": "General Mack",
    },
]


def test_classify_relations():
    """Submit entity pairs and return the classified relations."""
    result = classify_relations(PAIRS)

    assert result["status"] == "success", result
    assert len(result["relations"]) == len(PAIRS)

    for rel in result["relations"]:
        print("SOURCE:   ", rel["source"])
        print("TARGET:   ", rel["target"])
        print("LABEL:    ", rel["label"])
        print("REASONING:", rel["reasoning"])
        print("-" * 70)

    return result


if __name__ == "__main__":
    test_classify_relations()
