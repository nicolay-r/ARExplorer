"""Tool — form candidate (source, target, context) pairs from NER documents.

Bridges `extract_named_entities` and `classify_relations`: it consumes the
per-document entity lists emitted by NER and turns them into the
`{text, source, target}` triples that the attitude classifier expects.

Three design choices keep the pair list compact and grounded:

  - **Distance filter (`window_size`)**: only pairs whose two occurrences are
    within `window_size` words of each other (gap measured between the two
    spans, not including the entities themselves) are emitted. Distant
    co-mentions rarely express a relation about each other and would only
    inflate downstream LLM cost.
  - **Local context (`context_pad`)**: each pair's `text` is a small slice of
    the source document covering the pair plus `context_pad` extra words on
    each side — NOT the full source text. This keeps the classifier's input
    short and focused on the actual context of the relation.
  - **Hard cap (`max_pairs`)**: at most `max_pairs` pairs (default 50) are
    returned overall. When more candidates pass the window filter, those
    with the SMALLEST gap are kept first — they carry the strongest signal
    for a relation. This bounds the cost of the downstream
    `classify_relations` call regardless of corpus size.

Pure Python; no third-party resources are needed.

Input shape (mirrors `extract_named_entities` output):

    documents = [
        {"text": "...", "entities": [{"value": "...", "type": "...", ...}, ...]},
        ...
    ]

Output:

    {"status": "success", "pairs": [{"text": <local-context>,
                                     "source": ..., "target": ...}, ...]}
"""


def _normalize_documents(documents: list) -> list[dict]:
    """Accept the LLM-supplied `documents` list with light tolerance.

    Allows callers (or `inflate_artifact_inputs` rehydrating a NER artifact)
    to pass either:
      - a list of `{text, entities}` dicts, or
      - a single dict matching that shape (wrapped into a 1-element list).
    """
    if isinstance(documents, dict):
        return [documents]
    return list(documents)


def _find_word_positions(words: list[str], value: str) -> list[tuple[int, int]]:
    """Return every (start, end) inclusive word-index span where `value` matches.

    Multi-word entity values are matched as a contiguous token sequence on
    whitespace-tokenized input. An empty value yields no positions.
    """
    val_words = value.split()
    n = len(val_words)
    if not n:
        return []
    return [
        (i, i + n - 1)
        for i in range(len(words) - n + 1)
        if words[i : i + n] == val_words
    ]


def _gap(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    """Word gap strictly between two inclusive spans (0 if they touch/overlap).

    For ``a = [0, 0]`` and ``b = [2, 2]`` the gap is ``1`` ("one word in
    between"). For overlapping spans the formula goes negative and we clamp
    to 0.
    """
    return max(0, max(a_start, b_start) - min(a_end, b_end) - 1)


def form_entity_pairs(
    documents: list[dict] | None = None,
    entity_types: list[str] | None = None,
    directed: bool = True,
    window_size: int = 5,
    context_pad: int = 5,
    skip_self_pairs: bool = True,
    max_pairs: int | None = 50,
) -> dict:
    """Form candidate entity pairs from NER-annotated documents.

    For each document, every entity *occurrence* (one per appearance of the
    surface form, not deduplicated) is indexed by its word position. Pairs
    are emitted between two occurrences in the same document whose gap is at
    most ``window_size`` words. Each pair's ``text`` is the local context
    spanning the pair plus ``context_pad`` words on either side — not the
    full document. The final list is truncated to ``max_pairs`` entries,
    keeping the closest (smallest-gap) pairs first.

    Args:
        documents: Per-document NER results — a list of dicts each containing
            at least `text` and `entities`. `entities` is a list of
            `{value, type, ...}` dicts (the shape produced by
            `extract_named_entities`). May be ``None`` when the caller intends
            to supply it via an artifact reference — the agent-level
            `inflate_artifact_inputs` before_tool_callback fills this in.
        entity_types: Optional whitelist of entity types (e.g.
            ``["PERSON", "ORG"]``). When set, only entities whose `type`
            matches are included. ``None`` keeps all types.
        directed: If True, emit ordered pairs (A->B and B->A as separate
            entries). If False, emit one entry per unordered pair, with the
            earlier-appearing entity as ``source``.
        window_size: Maximum allowed word gap between the two occurrences in
            a pair. The gap is the number of words STRICTLY BETWEEN the
            spans, so adjacent entities have gap 0.
        context_pad: Extra words of context to include on each side of the
            pair when extracting the local ``text``. Defaults to 5.
        skip_self_pairs: If True, drop pairs where ``source == target`` by
            surface form. Defaults to True since attitudes towards oneself
            are not meaningful here.
        max_pairs: Hard upper bound on the number of pairs returned (and
            therefore on the relations the downstream classifier will need
            to score). Defaults to 50. When more candidates pass the window
            filter, those with the SMALLEST gap are kept first; ties break
            on the order candidates were generated (document then occurrence
            order — stable). Pass ``None`` to disable truncation.

    Returns:
        A dict with:
        - status: "success" or "error".
        - pairs: list of {text, source, target} dicts where `text` is the
          local context window (not the full document).
        - error: present only when status is "error".
    """
    if documents is None:
        return {
            "status": "error",
            "error": (
                "form_entity_pairs: provide `documents` inline or supply a "
                "`documents_artifact` filename so the before_tool_callback "
                "can inflate it."
            ),
        }

    try:
        docs = _normalize_documents(documents)
    except TypeError:
        return {
            "status": "error",
            "error": "form_entity_pairs: `documents` must be a list of dicts.",
        }

    if window_size < 0:
        return {
            "status": "error",
            "error": "form_entity_pairs: `window_size` must be non-negative.",
        }
    if context_pad < 0:
        return {
            "status": "error",
            "error": "form_entity_pairs: `context_pad` must be non-negative.",
        }
    if max_pairs is not None and max_pairs < 0:
        return {
            "status": "error",
            "error": "form_entity_pairs: `max_pairs` must be non-negative or None.",
        }

    type_filter = set(entity_types) if entity_types else None
    # (gap, pair_dict) so we can keep the closest pairs after global truncation.
    candidates: list[tuple[int, dict]] = []

    for idx, doc in enumerate(docs):
        if not isinstance(doc, dict):
            return {
                "status": "error",
                "error": (
                    f"form_entity_pairs: document at index {idx} is not a dict."
                ),
            }
        text = doc.get("text")
        if not isinstance(text, str):
            return {
                "status": "error",
                "error": (
                    f"form_entity_pairs: document at index {idx} is missing a "
                    "`text` string."
                ),
            }
        words = text.split()
        if len(words) < 2:
            continue

        # Index every entity OCCURRENCE in the document, sorted by reading
        # order. We deliberately do not dedup by `value` here: two mentions
        # of the same entity in different positions yield two distinct
        # evidences for the relation classifier.
        occurrences: list[dict] = []
        for ent in doc.get("entities") or []:
            if not isinstance(ent, dict):
                continue
            value = ent.get("value")
            if not isinstance(value, str) or not value:
                continue
            etype = ent.get("type")
            if type_filter is not None and etype not in type_filter:
                continue
            for ws, we in _find_word_positions(words, value):
                occurrences.append(
                    {
                        "value": value,
                        "type": etype,
                        "word_start": ws,
                        "word_end": we,
                    }
                )
        occurrences.sort(key=lambda o: (o["word_start"], o["word_end"]))

        for i, src in enumerate(occurrences):
            for j, tgt in enumerate(occurrences):
                if i == j:
                    continue
                if not directed and src["word_start"] > tgt["word_start"]:
                    continue
                if skip_self_pairs and src["value"] == tgt["value"]:
                    continue

                gap = _gap(
                    src["word_start"], src["word_end"],
                    tgt["word_start"], tgt["word_end"],
                )
                if gap > window_size:
                    continue

                lo = min(src["word_start"], tgt["word_start"])
                hi = max(src["word_end"], tgt["word_end"])
                ctx_lo = max(0, lo - context_pad)
                ctx_hi = min(len(words), hi + context_pad + 1)
                context = " ".join(words[ctx_lo:ctx_hi])

                candidates.append(
                    (
                        gap,
                        {
                            "text": context,
                            "source": src["value"],
                            "target": tgt["value"],
                        },
                    )
                )

    # Stable sort by gap so ties preserve the document/occurrence order in
    # which they were generated. Truncate to the global max_pairs limit.
    candidates.sort(key=lambda item: item[0])
    if max_pairs is not None:
        candidates = candidates[:max_pairs]
    pairs = [pair for _, pair in candidates]

    return {"status": "success", "pairs": pairs}
