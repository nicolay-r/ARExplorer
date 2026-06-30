---
name: adk-skills
description: >-
  Explains the Google ADK Skills feature — self-contained, on-demand packages of
  instructions and resources that an ADK agent loads via SkillToolset to optimize
  the context window. Use when the user asks about ADK Skills, SkillToolset,
  load_skill_from_dir, the SKILL.md / references / assets / scripts layout, inline
  vs filesystem skill sources, the Skill/Frontmatter/Resources models, or how to
  give an ADK agent loadable capabilities.
---

# ADK Skills

An ADK **Skill** is a self-contained unit of functionality (instructions +
resources + tools) that an agent loads *on demand*. Skills are loaded
incrementally so only the instructions actually needed enter the context window.

> Experimental feature. Available in ADK Python `v1.25.0+`, TypeScript `v0.6.1+`,
> Go `v1.2.0+`. Based on the Agent Skill specification.

## Three loading levels

Skills load progressively to minimize context cost:

| Level | Content | Loaded when |
|-------|---------|-------------|
| **L1** | Frontmatter (`name`, `description`) | Always — lets the agent discover the skill |
| **L2** | `SKILL.md` instructions body | When the agent decides to use the skill |
| **L3** | Resources (`references/`, `assets/`, `scripts/`) | Only when an instruction calls for them |

## Wire skills into an agent

Make Skills available through `SkillToolset`, passed in the agent's `tools`.
`additional_tools` are regular `FunctionTool`s the skill instructions may invoke.

```python
import pathlib

from google.adk import Agent
from google.adk.skills import load_skill_from_dir
from google.adk.tools import skill_toolset

weather_skill = load_skill_from_dir(
    pathlib.Path(__file__).parent / "skills" / "weather_skill"
)

my_skill_toolset = skill_toolset.SkillToolset(
    skills=[weather_skill],
    additional_tools=[get_weather_tool],
)

root_agent = Agent(
    model="gemini-flash-latest",
    name="skill_user_agent",
    description="An agent that can use specialized skills.",
    instruction="You are a helpful assistant that can leverage skills to perform tasks.",
    tools=[my_skill_toolset],
)
```

## Skill sources

### Filesystem (recommended)

Load each skill directory with `load_skill_from_dir`, then pass them to the
toolset:

```python
import pathlib

from google.adk.skills import load_skill_from_dir
from google.adk.tools import skill_toolset

greeting_skill = load_skill_from_dir(
    pathlib.Path(__file__).parent / "skills" / "greeting-skill"
)
weather_skill = load_skill_from_dir(
    pathlib.Path(__file__).parent / "skills" / "weather-skill"
)

my_skill_toolset = skill_toolset.SkillToolset(skills=[weather_skill, greeting_skill])
```

### Inline (in code)

Define a `Skill` directly. Resources live in the `Resources` model instead of on
disk; instructions reference them by name:

```python
from google.adk.skills import models

greeting_skill = models.Skill(
    frontmatter=models.Frontmatter(
        name="greeting-skill",
        description="A friendly greeting skill that can say hello to a specific person.",
    ),
    instructions=(
        "Step 1: Read the 'references/hello_world.txt' file to understand how"
        " to greet the user. Step 2: Return a greeting based on the reference."
    ),
    resources=models.Resources(
        references={
            "hello_world.txt": "Hello! So glad to have you here!",
            "example.md": "This is an example reference.",
        },
    ),
)
```

## Directory structure

Each skill directory must follow the Agent Skill specification. Only `SKILL.md`
is required; `references/`, `assets/`, and `scripts/` are optional L3 resources.

```
my_agent/
    agent.py
    .env
    skills/
        example-skill/        # one Skill per directory
            SKILL.md          # main instructions (required)
            references/       # extra .md: extended instructions, workflows
            assets/           # templates, images, schemas, data
            scripts/          # executable utility scripts (.py / .js / .ts)
```

- `references/`: additional Markdown with extended instructions or guidance.
- `assets/`: schemas, API docs, templates, examples.
- `scripts/`: executable scripts supported by the agent runtime.

## Other languages

`SkillToolset` and `loadSkillFromDir` exist in TypeScript (`@google/adk`) with
the same shape. In Go use `skilltoolset.New` with a `skill.Source` — e.g.
`skill.NewFileSystemSource(os.DirFS("./skills"))`. Go has no built-in inline
source; implement the `skill.Source` interface yourself for in-memory skills.

## Related skills

- `/google-agents-cli-adk-code` — broader ADK API (Agent, Runner, Session, tools)
- `/adk-event` — the Event objects produced while an agent runs
- `/adk-structured-output` — typed final responses via `output_schema`
