"""Detect user intent to end the session (works better with noisy STT)."""

from __future__ import annotations

import re


def wants_quit(text: str) -> bool:
    """
    True if the user is trying to hang up / end the conversation.
    Tuned for short phrases and common speech-to-text distortions.
    """
    raw = text.strip()
    if not raw:
        return False

    t = raw.lower()
    t = re.sub(r"^(uh|um|er|hmm)[,.\s]+", "", t, flags=re.I).strip()
    t = re.sub(r"[.!?]+$", "", t).strip()

    exact = frozenset(
        {
            "quit",
            "exit",
            "q",
            "goodbye",
            "good bye",
            "bye",
            "bye bye",
            "bye-bye",
            "that's all",
            "thats all",
            "that's it",
            "thats it",
            "end call",
            "hang up",
            "hangup",
            "i'm done",
            "im done",
            "we're done",
            "were done",
            "ciao",
        }
    )
    if t in exact:
        return True

    words = t.split()
    if len(words) <= 5:
        if t in ("bi", "by", "buy buy"):
            return True
        if words == ["bye"] or (len(words) == 1 and words[0] in ("bye", "goodbye", "quit", "exit")):
            return True

    if re.search(
        r"\b(goodbye|bye-?bye|farewell|hang\s*up|end\s+(the\s+)?call|stop\s+calling|"
        r"that'?s\s+all(\s+for\s+now)?|i'?m\s+(done|good|finished)|we'?re\s+done|"
        r"no\s+more|nothing\s+else|all\s+set|all\s+done)\b",
        t,
    ):
        return True

    if t.startswith(("quit ", "exit ", "goodbye ", "bye ", "end call")) and len(t) <= 72:
        return True

    return False
