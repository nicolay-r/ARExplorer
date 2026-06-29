---
name: adk-event
description: >-
  Explains ADK Event objects — the atomic units of conversation history that
  record each user message, agent response, tool call, and tool result. Use when
  the user asks about ADK events, event stream, runner.run() output,
  EventActions, state_delta, artifact_delta, author/content/actions fields,
  session history, or how the Runner logs conversational turns.
---

# ADK Event

An **Event** is the fundamental unit of ADK conversation history — one chat
bubble per step (user message, agent text, function call, tool result). The
`Session.events` list *is* the conversation log.

## Key fields

| Field | Purpose |
|-------|---------|
| `author` | `"user"`, agent name (e.g. `"MathAgent"`), `"tool"`, or `"code_executor"` |
| `content` | `google.genai.types.Content` with `parts` (text, function_call, function_response, code_execution_result) |
| `actions` | `EventActions` side effects: `state_delta`, `artifact_delta`, `transfer_to_agent`, `escalate` |
| `id` | Unique 8-char identifier (auto-generated if omitted) |
| `timestamp` | Creation time |
| `invocation_id` | Links event to the processing turn (`InvocationContext`) |

`Event` inherits from `LlmResponse` (`content`, `error_code`, etc.) and adds
history-tracking fields. Source: `google.adk.events.event`.

## Observing events

Iterate the stream from `runner.run()` or `runner.run_async()`:

```python
from google.genai import types

event_stream = runner.run(
    user_id=user_id,
    session_id=session_id,
    new_message=types.Content(role="user", parts=[types.Part(text="What is 5 times 8?")]),
)

for event in event_stream:
    print(event.id, event.author, event.timestamp)
    if event.content and event.content.parts:
        part = event.content.parts[0]
        if part.text:
            print("text:", part.text)
        elif part.function_call:
            print("call:", part.function_call.name, part.function_call.args)
        elif part.function_response:
            print("response:", part.function_response.name, part.function_response.response)
    if event.actions and event.actions.model_dump(exclude_defaults=True):
        print("actions:", event.actions.model_dump(exclude_defaults=True))
```

Typical tool-use sequence:

1. Agent event — `FunctionCall` in `content`
2. Tool event — `author="tool"`, `FunctionResponse` in `content`
3. Agent event — final text in `content`

State changes (e.g. `ctx.state['last_result'] = 40`) appear in
`event.actions.state_delta`.

## How events are created

You rarely construct `Event` manually — the framework creates them:

1. **User input** — Runner wraps `new_message` as `Event(author="user")`, appends via `SessionService`.
2. **Agent LLM output** — LLM Flow converts `LlmResponse` → `Event(author=agent_name)`.
3. **Tool result** — Runner formats return value as `FunctionResponse`, yields `Event(author="tool")` back to agent and caller.
4. **Persistence** — Runner calls `session_service.append_event()` for non-partial events.

## EventActions

Side effects attached to an event (`google.adk.events.event_actions`):

```python
class EventActions(BaseModel):
    state_delta: dict[str, object] = Field(default_factory=dict)
    artifact_delta: dict[str, int] = Field(default_factory=dict)
    transfer_to_agent: Optional[str] = None
    escalate: Optional[bool] = None
```

Tools update state via `tool_context.state` and artifacts via
`tool_context.save_artifact`; those changes land in `EventActions`.

## Useful helpers on Event

- `get_function_calls()` — extract `FunctionCall` parts from `content`
- `get_function_responses()` — extract `FunctionResponse` parts
- `is_final_response()` — whether this event is the turn's final answer

## Related skills

- `/google-agents-cli-adk-code` — broader ADK API (Agent, Runner, Session, tools)
- `/adk-structured-output` — typed final responses via `output_schema`

## Full reference

For the complete tutorial (sequence diagram, class walkthrough, example output),
see [references/event.md](references/event.md).
