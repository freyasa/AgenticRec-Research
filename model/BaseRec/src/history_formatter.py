"""Create history-only user context without exposing held-out events.

The caller must pass training-prefix item ids (history[:-1]). All formats
preserve chronology and avoid item identifiers or target-derived labels.
"""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence

from .models.base_rec import positive_item_evidence, sanitize_item_description

HISTORY_MODES = ("history_titles", "history_full", "history_structured")


def format_history(history: Sequence[int], metadata: Mapping[int, Mapping],
                   mode: str) -> str:
    """Serialize a user's known positive interactions for the frozen ranker.

    history_titles: title/name only
    history_full: title/name and description
    history_structured: same allowlisted evidence the BaseRec updater sees

    These three modes intentionally have different information sources.
    """
    if mode not in HISTORY_MODES:
        raise ValueError(f"Unknown history mode: {mode}")
    rows = []
    for position, item_id in enumerate(history):
        item = metadata[item_id]
        if mode == "history_structured":
            facts = positive_item_evidence(item)
        else:
            title = sanitize_item_description(item.get("title") or item.get("name"))
            facts = {"title": title} if isinstance(title, str) and title else {}
            if mode == "history_full":
                description = sanitize_item_description(item.get("description"))
                if isinstance(description, str) and description:
                    facts["description"] = description
        rows.append({"position": position, "observed_positive_item": facts})
    return json.dumps(rows, ensure_ascii=False, separators=(",", ":"))
