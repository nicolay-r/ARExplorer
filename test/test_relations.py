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

from tools.relations import classify_relations  # noqa: E402

PROVIDER_FILEPATH = os.path.join(ROOT, "test", "providers", "replicate_104.py")

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
    # Use the local Replicate provider (test/providers/replicate_104.py).
    result = classify_relations(
        PAIRS,
        provider_filepath=PROVIDER_FILEPATH,
        model_name="meta/meta-llama-3-70b-instruct",
        api_token=os.getenv("REPLICATE_API_TOKEN", ""),
    )

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
