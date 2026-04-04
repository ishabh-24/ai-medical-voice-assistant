"""Machine-readable schedule aligned with `CLINIC.hours` in clinic_profile."""

from __future__ import annotations

import re
from datetime import date

# Weekday: Monday=0 .. Sunday=6 (datetime.weekday())
# Mon–Fri 8:00–17:00, Sat 9:00–13:00, Sun closed (matches ClinicProfile.hours string)


def weekday_from_booking_date(date_str: str) -> int | None:
    """Parse weekday from `YYYY-MM-DD (...)` or ISO prefix in display string."""
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", date_str)
    if not m:
        return None
    y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
    try:
        return date(y, mo, d).weekday()
    except ValueError:
        return None


def _hm_to_minutes(h: int, m: int) -> int:
    return h * 60 + m


def parse_hhmm(hhmm: str) -> tuple[int, int] | None:
    s = hhmm.strip()
    m = re.fullmatch(r"(\d{1,2}):(\d{2})", s)
    if not m:
        return None
    h, mi = int(m.group(1)), int(m.group(2))
    if h > 23 or mi > 59:
        return None
    return h, mi


def is_appointment_within_hours(weekday: int, time_hhmm: str) -> bool:
    """
    True if time falls within clinic appointment window for that weekday.
    """
    parsed = parse_hhmm(time_hhmm)
    if not parsed:
        return True
    h, mi = parsed
    mins = _hm_to_minutes(h, mi)

    if weekday == 6:  # Sunday
        return False

    if weekday == 5:  # Saturday
        return _hm_to_minutes(9, 0) <= mins <= _hm_to_minutes(13, 0)

    # Mon–Fri
    return _hm_to_minutes(8, 0) <= mins <= _hm_to_minutes(17, 0)


def booking_hours_violation_message(date_str: str, time_hhmm: str) -> str | None:
    """
    If the slot is outside hours, return a user-facing message; else None.
    """
    wd = weekday_from_booking_date(date_str)
    if wd is None:
        return None
    if is_appointment_within_hours(wd, time_hhmm):
        return None
    if wd == 6:
        return (
            "We're closed on Sundays. Please pick another day. "
            "Our hours are Monday–Friday 8 AM to 5 PM, and Saturday 9 AM to 1 PM."
        )
    return (
        "That time is outside our office hours for that day. "
        "We're open Monday–Friday 8 AM to 5 PM, and Saturday 9 AM to 1 PM. "
        "What other time works for you?"
    )
