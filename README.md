# ARExplorer

![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)

An ADK 2.0 agent for extracting **Attitudes and Relations** from documents,
exposing three tools: named-entity recognition (bulk-ner), relation/attitude
classification (bulk-chain), and graph set operations (union / intersection).

## Launching

### ARExplorer UI (chat + d3.js graph)

<img src="docs/ui-demo.png" alt="ARExplorer UI" width="640">

Two-panel web UI — left panel chats with the agent, main panel renders the
returned attitude graph with d3.js (force / radial layouts). The agent replies
with a structured `AgentResponse` (`src/schema.py`); the chat shows `message`
and the graph drives the visualization.

```bash
uvicorn src.server:app --port 8000
```

Then open http://127.0.0.1:8000/.

### ADK dev web UI

For low-level agent debugging:

```bash
adk web --port 2000 ./src/
```
## Deployment

Using docker-compose:

```bash
cd .recepie/arexplorer-demo
docker compose up --build
```

Then open http://127.0.0.1:2000/ (Compose maps host port `2000` → container
`8000`). Stop and remove the container with:

```bash
docker compose down
```

