"""
context_builder.py — Assembles the LLM user prompt from chunks + history for Health AI v3.

The context builder is intentionally stateless — it takes inputs and returns
a formatted string. No side effects.

Context structure sent to the LLM:
    [CONVERSATION HISTORY]    ← last N turns (if any)
    {history pairs}

    [PATIENT DATA]            ← retrieved chunks (lab results, prescription text)
    {chunk 1}
    ---
    {chunk 2}
    ...

    [QUESTION]
    {user query}
"""

from typing import List

from health_ai.core.logger import get_logger
from health_ai.core.character import MAX_HISTORY_TURNS

log = get_logger(__name__)


def build_context(
    query: str,
    chunks: List[str],
    history: List[str] | None = None,
) -> str:
    """
    Build the user prompt to send to the LLM.

    Args:
        query:   The user's current question.
        chunks:  List of retrieved chunk texts (pre-filtered by the phone or server).
                 Can be empty for general queries.
        history: Flat list of alternating [user_msg, ai_msg, user_msg, ai_msg, ...].
                 Most recent last. We take the last MAX_HISTORY_TURNS*2 entries.

    Returns:
        A single formatted string ready to be used as the LLM's user prompt.
    """
    sections: List[str] = []

    # ── Conversation history ──────────────────────────────────────────────────
    if history:
        # Take last MAX_HISTORY_TURNS complete turns (user + AI = 2 entries)
        recent = history[-(MAX_HISTORY_TURNS * 2):]
        history_lines = []
        for i, msg in enumerate(recent):
            role = "User" if i % 2 == 0 else "Dr. Aria"
            history_lines.append(f"{role}: {msg.strip()}")
        sections.append("[CONVERSATION HISTORY]\n" + "\n".join(history_lines))

    # ── Patient data (retrieved chunks) ──────────────────────────────────────
    if chunks:
        # Deduplicate — phones may send the same chunk twice if scores are identical
        seen = set()
        unique_chunks = []
        for c in chunks:
            c_stripped = c.strip()
            if c_stripped and c_stripped not in seen:
                seen.add(c_stripped)
                unique_chunks.append(c_stripped)

        if unique_chunks:
            patient_data = "\n---\n".join(unique_chunks)
            sections.append(f"[PATIENT DATA]\n{patient_data}")
            log.debug(f"Context includes {len(unique_chunks)} unique chunks.")

    # ── Current question ──────────────────────────────────────────────────────
    sections.append(f"[QUESTION]\n{query.strip()}")

    return "\n\n".join(sections)
