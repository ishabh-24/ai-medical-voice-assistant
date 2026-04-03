"""Simulated clinic data used for answering general questions.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ClinicProfile:
    name: str
    address: str
    phone: str
    hours: str
    website: str | None = None


CLINIC = ClinicProfile(
    name="SENA Health Family Practice",
    address="1234 Wellness Ave, Suite 200, San Francisco, CA 94107",
    phone="(415) 555-0134",
    hours="Mon–Fri 8:00 AM–5:00 PM; Sat 9:00 AM–1:00 PM; Closed Sundays",
    website="https://example.com",
)

