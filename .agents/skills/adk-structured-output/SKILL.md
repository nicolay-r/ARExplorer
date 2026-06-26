---
name: adk-structured-output
description: >-
  Produce validated, structured agent responses in Google ADK 2.0 by pairing a
  Pydantic BaseModel with an LlmAgent's output_schema, including the ADK 2.0
  capability to combine output_schema WITH tools via the auto-injected
  set_model_response tool. Use when the user wants typed/JSON agent output,
  mentions output_schema, set_model_response, Pydantic schemas for ADK agents,
  or wants an agent that calls tools and still returns a structured result.
---

# ADK 2.0 Structured Output (Pydantic)

Return typed, schema-validated responses from an ADK `Agent` by defining a
Pydantic `BaseModel` and passing it as `output_schema`.

> Requires ADK 2.0. Earlier ADK disabled tool calling whenever `output_schema`
> was set; ADK 2.0 lifts that restriction (see "Schema + tools" below).

## 1. Define the schema with Pydantic

Use `Field(description=...)` on every field — the descriptions are sent to the
model and materially improve output quality.

```python
from pydantic import BaseModel, Field
from typing import List

class Product(BaseModel):
    name: str = Field(description="The product's name")
    price: float = Field(description="The product's price in USD")
    size: str = Field(description="The product's size")
    image_url: str = Field(description="URL to the product's image")

class ProductList(BaseModel):
    products: List[Product] = Field(description="List of products")
```

## 2. Attach it to the agent

Set `output_schema` to the top-level model. The agent's final response is then
a JSON object conforming to that schema.

```python
from google.adk.agents import Agent

root_agent = Agent(
    name="product_info_agent",
    model="gemini-flash-lite-latest",
    instruction="""
You are a helpful assistant that provides information about products.

When asked about products, you should:
1. Use the get_products tool to retrieve product information
2. Compile the information into a structured response using the ProductList format

Always use the set_model_response tool to provide your final answer in the required structured format.
    """.strip(),
    output_schema=ProductList,
    tools=[get_products],
)
```

## 3. Schema + tools together (ADK 2.0)

When **both** `output_schema` and `tools` are set, ADK:

1. Does **not** put the schema on the model config directly.
2. Auto-injects a `set_model_response(result)` tool whose `result` matches your
   `output_schema`.
3. Lets the model call your regular tools to gather data, then call
   `set_model_response(...)` to emit the final structured answer.
4. Extracts that call's content as the model's response and **validates** it
   against the schema.

This is why the instruction must explicitly tell the model:
*"Always use the set_model_response tool to provide your final answer in the
required structured format."* Without that nudge the model may answer as plain
text and skip the structured step.

### Native support (no tool workaround needed)

Some models accept `output_schema` and `tools` in one request natively (e.g.
Gemini 2+/3.0 on Vertex AI, and LiteLLM providers with native `response_format`
support). There ADK skips the `set_model_response` injection, but keeping the
instruction line is harmless and portable.

## 4. Store output in state (optional)

Add `output_key` to also persist the validated result into session state for
downstream agents/nodes. State holds a plain dict; reconstruct with the model:

```python
root_agent = Agent(
    name="product_info_agent",
    model="gemini-flash-lite-latest",
    instruction="...",
    output_schema=ProductList,
    tools=[get_products],
    output_key="product_list",   # ctx.state["product_list"] -> dict
)

# downstream:
products = ProductList(**ctx.state["product_list"])
```

## Checklist

- [ ] Top-level response modeled as one `BaseModel` (wrap lists in a container
      model like `ProductList`, not a bare `List[...]`).
- [ ] Every field has `Field(description=...)`.
- [ ] `output_schema=<Model>` set on the `Agent`.
- [ ] If `tools` are also set: instruction tells the model to finish via
      `set_model_response`.
- [ ] (Optional) `output_key` set when downstream agents need the result.
