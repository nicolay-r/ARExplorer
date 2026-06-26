---
name: bulk-chain
description: >-
  Apply Chain-of-Thought (CoT) prompt schemas over massive textual collections
  with any LLM using the bulk-chain framework. Use when the user wants to run
  multi-step LLM prompting/reasoning across many input texts, mentions
  bulk-chain, iter_content, CoT prompt schemas, batch/streaming/async LLM
  inference, or wants to wire up a third-party LLM provider (e.g. Replicate,
  Llama) over an iterator of input dictionaries.
---

# Bulk Chain

A no-strings-attached framework for applying Chain-of-Thought (CoT) prompt
`schema` over massive textual collections using custom third-party LLM
providers.

## Installation

From PyPI:

```bash
pip install --no-deps bulk-chain
```

Or the latest version from source:

```bash
pip install git+https://github.com/nicolay-r/bulk-chain@master
```

## Chain-of-Thought Schema

A CoT schema is declared as a list of instruction dicts. Each item has:

- `prompt`: the input prompt template (variables referenced with `{}`).
- `out`: the output variable name produced by that step.

Each `out` becomes a variable usable in later prompts, chaining the reasoning.

```python
[
    {"prompt": "extract topic: {text}", "out": "topic"},
    {"prompt": "extract subject: {text}", "out": "subject"},
]
```

## Usage (Python API)

Prepare three things:

1. A CoT [schema](#chain-of-thought-schema).
2. An LLM model from a third-party provider, loaded via `dynamic_init`.
3. Data: an iterator of dictionaries (each providing the schema's input variables).

Then call `iter_content` and iterate over the resulting batches:

```python
from bulk_chain.core.utils import dynamic_init
from bulk_chain.api import iter_content

content_it = iter_content(
    # 1. Your schema.
    schema=[
        {"prompt": "extract topic: {text}", "out": "topic"},
        {"prompt": "extract subject: {text}", "out": "subject"},
    ],
    # 2. Your third-party model implementation.
    llm=dynamic_init(class_filepath="replicate_104.py")(
        api_token="<API-KEY>",
        model_name="meta/meta-llama-3-70b-instruct"),
    # 3. Toggle streaming if needed.
    stream=False,
    # 4. Toggle Async API mode usage.
    async_mode=True,
    async_policy='prompt',
    # 5. Batch size.
    batch_size=10,
    # 6. Your iterator of dictionaries.
    input_dicts_it=[
        {"text": "Rocks are hard"},
        {"text": "Water is wet"},
        {"text": "Fire is hot"},
    ],
)

for batch in content_it:
    for entry in batch:
        print(entry)
```

Each output entry is the original input dict augmented with the schema's `out`
variables (here `topic` and `subject`):

```jsonl
{'text': 'Rocks are hard', 'topic': 'The topic is: Geology/Rocks', 'subject': 'The subject is: "Rocks"'}
{'text': 'Water is wet', 'topic': 'The topic is: Properties of Water', 'subject': 'The subject is: Water'}
{'text': 'Fire is hot', 'topic': 'The topic is: Temperature/Properties of Fire', 'subject': 'The subject is: "Fire"'}
```

## `iter_content` Parameters

- `schema`: list of CoT instruction dicts (`prompt` + `out`).
- `llm`: third-party model instance created via `dynamic_init`.
- `stream`: yield result chunks as they are generated.
- `async_mode`: enable the async API.
- `async_policy`: async scheduling granularity (e.g. `'prompt'`).
- `batch_size`: number of inputs processed per batch.
- `input_dicts_it`: iterator of input dictionaries (supports infinite streams).

## Single-Prompt and Batch API

Methods that accept a single `prompt`:

| Method                     | Mode  | Description                                                       |
|----------------------------|-------|-----------------------------------------------------------------|
| `ask(prompt)`              | Sync  | Infers the model with a single prompt.                          |
| `ask_stream(prompt)`       | Sync  | Returns a generator that yields chunks of the inferred result.  |
| `ask_async(prompt)`        | Async | Asynchronously infers the model with a single prompt.           |
| `ask_stream_async(prompt)` | Async | Asynchronously returns a generator of result chunks.            |

Methods that accept a `batch`:

| Method                   | Mode  | Description                                            |
|--------------------------|-------|-------------------------------------------------------|
| `ask_batch(batch)`       | Sync  | Infers the model over a batch.                        |
| `ask_async_batch(batch)` | Async | Asynchronously infers the model over a batch.         |

## References

- Third-party LLM providers: https://github.com/nicolay-r/nlp-thirdgate?tab=readme-ov-file#llm
- API Wiki: https://github.com/nicolay-r/bulk-chain/wiki
