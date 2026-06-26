---
name: bulk-ner
description: >-
  Run Named Entity Recognition (NER) over batches of texts using the bulk-ner
  library. Use when the user wants to extract named entities (people, dates,
  locations, organizations, etc.) from one or many texts, mentions bulk-ner,
  NERAnnotator, batch/bulk NER annotation, or asks how to wire up a NER model
  (e.g. DeepPavlov) over a collection of documents.
---

# Bulk NER

Annotate batches of texts with named entities using the `bulk-ner` library.

## Quick Start

Prepare the input as a list of dicts, each containing a `text` field:

```python
texts = [
    {"text": "It was in July, 1805, and the speaker was the well-known Anna Pávlovna"},
    # other texts ...
]
```

Initialize a NER model, wrap it in a `NERAnnotator`, and iterate over results:

```python
from bulk_ner.api import NERAnnotator
from bulk_ner.src.service_dynamic import dynamic_init

ner_model = dynamic_init(src_dir="models",
                         class_filepath="dp_130.py",
                         class_name="DeepPavlovNER")(model="ner_ontonotes_bert")

annotator = NERAnnotator(ner_model=ner_model,
                         entity_func=lambda t: [t.Value, t.Type, t.ID],
                         chunk_limit=128)

data_it = annotator.iter_annotated_data(data_dict_it=texts, prompt="{text}", batch_size=10)

for data in data_it:
    # Handle your NER data here ...
    print(data["result"])
    # Output:
    # ['It was in', ['July , 1805', 'DATE', 0], ', and the speaker was the well - known', ['Anna Pávlovna', 'PERSON', 1]]
```

## Key Parameters

- `ner_model`: NER backend created via `dynamic_init` (loads a model class from `src_dir`).
- `entity_func`: maps each recognized entity to the desired fields; the example emits `[Value, Type, ID]`.
- `chunk_limit`: maximum chunk size (in tokens) fed to the model.
- `prompt`: template applied to each input dict; `"{text}"` formats the `text` field.
- `batch_size`: number of texts processed per batch.

## Output Format

Results are returned as a `list` for accessibility. Within each list:

- Plain strings are non-entity text spans.
- Nested lists are **recognized named entities**, containing:
  - Value of the named entity
  - Type of the named entity
  - Index (ID) of the entity

Example:

![ner-formatting-small](https://github.com/user-attachments/assets/87a48788-0192-4f96-ad83-64eb3a308306)
