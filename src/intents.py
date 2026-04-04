"""Intent classification from user utterances (rule-based)."""

from __future__ import annotations

import re
from enum import Enum


class Intent(str, Enum):
    BOOK = "book"
    RESCHEDULE = "reschedule"
    CANCEL = "cancel"
    GENERAL = "general"
    OFF_TOPIC = "off_topic"


_BOOK = re.compile(
    r"\b(book|schedule|make|set up|need)\s+(an?\s+)?appointment\b|"
    r"\bappointment\b.*\b(book|schedule)\b|"
    r"\bnew\s+appointment\b",
    re.I,
)
_RESCHEDULE = re.compile(
    r"\b(reschedule|move|change|push)\b.*\bappointment\b|"
    r"\bappointment\b.*\b(reschedule|move|change)\b|"
    r"\bcan\s+i\s+change\s+my\s+appointment\b",
    re.I,
)


def classify_intent(text: str) -> Intent:
    t = text.strip()
    if not t:
        return Intent.GENERAL
    # cancel appointment → same workflow as reschedule (fallback when llm unavailable)
    if re.search(r"\bcancel\b", t, re.I) and re.search(
        r"\b(appointment|visit|booking)\b", t, re.I
    ):
        return Intent.CANCEL
    if _RESCHEDULE.search(t):
        return Intent.RESCHEDULE
    if _BOOK.search(t):
        return Intent.BOOK
    if re.search(r"\breschedule\b", t, re.I):
        return Intent.RESCHEDULE
    if re.search(r"\b(book|appointment|schedule)\b", t, re.I):
        return Intent.BOOK
    return Intent.GENERAL


def extract_name(text: str) -> str | None:
    """Pull a name after 'name is', 'I'm', etc."""
    patterns = [
        r"(?:my name is|i am|i'm|this is|call me)\s+([A-Za-z][A-Za-z'\-\s]{1,40})",
        r"(?:name\s*:?\s*)([A-Za-z][A-Za-z'\-\s]{1,40})",
    ]
    for p in patterns:
        m = re.search(p, text, re.I)
        if m:
            name = m.group(1).strip()
            name = re.split(r"[,.\n]", name)[0].strip()
            if len(name) >= 2:
                return name.title()
    # two capitalized words as fallback
    m = re.search(r"\b([A-Z][a-z]+)\s+([A-Z][a-z]+)\b", text)
    if m:
        return f"{m.group(1)} {m.group(2)}"
    return None


def is_unclear(text: str) -> bool:
    t = text.strip().lower()
    if len(t) < 2:
        return True
    vague = ["uh", "um", "hmm", "what", "hello", "hi"]
    if t in vague or (len(t) < 4 and t not in {"yes", "no", "ok"}):
        return True
    return False
