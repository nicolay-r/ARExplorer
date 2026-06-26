# Known Issues

## bulk-chain: nested event loop when run inside an async server

**Symptom**

```
RuntimeError: Cannot run the event loop while another loop is running
```

Raised from `bulk_chain/core/service_asyncio.py` →
`event_loop.run_until_complete(...)` while serving a request.

**Cause**

`bulk-chain` with `async_mode=True` drives its batches synchronously via
`loop.run_until_complete()`. Our `classify_relations` tool is invoked by ADK
directly on the FastAPI/uvicorn **event-loop thread**, so a loop is already
running and `run_until_complete` refuses to start another on that thread.

**Workaround (in `src/tools/relations.py`)**

Run bulk-chain's `iter_content` creation *and* consumption inside a dedicated
worker thread that owns a fresh event loop (`_run_in_isolated_loop`). The thread
provides a clean, loop-free context.

**Caveats / follow-ups**

- The server's event loop is blocked while a relation batch runs (acceptable for
  single-user/dev use; revisit for concurrent requests, e.g. offload the whole
  `runner.run_async` per request).
- Ideally fixed upstream in `bulk-chain` by detecting a running loop and using
  `run_in_executor` / a thread, instead of always calling `run_until_complete`.
