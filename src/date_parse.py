"""Lightweight date/time extraction from natural language (demo-friendly)."""

from __future__ import annotations

import re
from datetime import date, timedelta

_WEEKDAYS = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}


def _next_weekday(d: date, weekday: int) -> date:
    """Next occurrence of weekday on or after d."""
    days_ahead = weekday - d.weekday()
    if days_ahead < 0:
        days_ahead += 7
    return d + timedelta(days=days_ahead)


def parse_date_phrase(text: str, *, today: date | None = None) -> str | None:
    """
    Return a human-readable date string like '2026-03-30 (Monday)' or None.
    Handles: next Monday, this Friday, tomorrow, MM/DD/YYYY, March 30.
    """
    t = text.lower().strip()
    today = today or date.today()

    if "tomorrow" in t:
        d = today + timedelta(days=1)
        return _fmt(d)

    m = re.search(r"\bnext\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", t)
    if m:
        wd = _WEEKDAYS[m.group(1)]
        d = _next_weekday(today + timedelta(days=7), wd)
        return _fmt(d)

    m = re.search(
        r"\b(this|coming)\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", t
    )
    if m:
        wd = _WEEKDAYS[m.group(2)]
        d = _next_weekday(today, wd)
        if d == today:
            d = d + timedelta(days=7)
        return _fmt(d)

    m = re.search(r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", t)
    if m:
        wd = _WEEKDAYS[m.group(1)]
        d = _next_weekday(today, wd)
        return _fmt(d)

    m = re.search(r"\b(\d{1,2})/(\d{1,2})/(\d{2,4})\b", t)
    if m:
        mo, da, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if y < 100:
            y += 2000
        try:
            d = date(y, mo, da)
            return _fmt(d)
        except ValueError:
            pass

    m = re.search(
        r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{1,2})(?:st|nd|rd|th)?(?:,?\s*(\d{4}))?\b",
        t,
    )
    if m:
        months = {
            "january": 1,
            "february": 2,
            "march": 3,
            "april": 4,
            "may": 5,
            "june": 6,
            "july": 7,
            "august": 8,
            "september": 9,
            "october": 10,
            "november": 11,
            "december": 12,
        }
        mo = months[m.group(1)]
        da = int(m.group(2))
        y = int(m.group(3)) if m.group(3) else today.year
        try:
            d = date(y, mo, da)
            if d < today:
                d = date(y + 1, mo, da)
            return _fmt(d)
        except ValueError:
            pass

    return None


def parse_time_phrase(text: str) -> str | None:
    """Return e.g. '10:00 AM' or None."""
    t = text.lower()
    m = re.search(r"\b(\d{1,2}):(\d{2})\s*(am|pm)?\b", t)
    if m:
        h, mi = int(m.group(1)), int(m.group(2))
        ap = m.group(3)
        if ap == "pm" and h != 12:
            h += 12
        if ap == "am" and h == 12:
            h = 0
        return f"{h % 24:02d}:{mi:02d}"

    m = re.search(r"\b(\d{1,2})\s*(am|pm)\b", t)
    if m:
        h = int(m.group(1))
        if m.group(2) == "pm" and h != 12:
            h += 12
        if m.group(2) == "am" and h == 12:
            h = 0
        return f"{h % 24:02d}:00"

    if "morning" in t:
        return "09:00"
    if "afternoon" in t:
        return "14:00"
    if "evening" in t:
        return "17:00"
    return None


def _fmt(d: date) -> str:
    return f"{d.isoformat()} ({d.strftime('%A')})"


def looks_like_correction(text: str) -> bool:
    t = text.lower()
    cues = (
        "actually",
        "instead",
        "i meant",
        "change that",
        "make it",
        "not that",
        "sorry",
        "correction",
    )
    return any(c in t for c in cues)
