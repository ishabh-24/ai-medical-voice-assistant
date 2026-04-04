"""OpenAI-powered intent + slot extraction and policy-driven replies.

If OPENAI_API_KEY is not set, fall back to rule-based parsing.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Literal

from .clinic_profile import ClinicProfile

try:
    from dotenv import load_dotenv  

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

# Chat / NLU / general answers (override with OPENAI_MODEL in .env)
DEFAULT_OPENAI_MODEL = "gpt-5.3"

IntentLabel = Literal["book", "reschedule", "cancel", "general", "off_topic"]


@dataclass
class NLUResult:
    intent: IntentLabel | None = None
    patient_name: str | None = None
    date: str | None = None
    time: str | None = None
    old_date: str | None = None
    new_date: str | None = None
    is_correction: bool = False
    confidence: float | None = None
    notes: str | None = None


def llm_available() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))


def _client():
    try:
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError(f"OpenAI SDK not installed: {e}") from e

    return OpenAI()


def extract_nlu(user_text: str, *, context: dict[str, Any] | None = None) -> NLUResult | None:
    """Return NLUResult via OpenAI, or None if unavailable/error (resorts to fallback)."""
    if not llm_available():
        return None

    model = os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)
    today = date.today().isoformat()
    ctx = context or {}

    system = (
        "You are an AI phone assistant for a medical practice.\n"
        "Classify intent and extract scheduling details.\n"
        "Return ONLY valid JSON (no markdown). If a field is unknown, use null.\n"
        "Normalize dates to 'YYYY-MM-DD' when possible (resolve relative to today's date).\n"
        "Normalize time to 24-hour 'HH:MM' when possible.\n"
        "Set is_correction=true if the user is changing previously provided info.\n"
        "Intent definitions:\n"
        "- book: schedule a new appointment.\n"
        "- reschedule: change/move an existing appointment to another date/time.\n"
        "- cancel: cancel/remove an existing appointment (do NOT use off_topic for this).\n"
        "- general: questions answerable from clinic profile (hours, address, phone, parking, basic visit info).\n"
        "- off_topic: anything else (medical advice, billing disputes, unrelated topics) — not scheduling/hours/profile.\n"
        "Be conservative: if unsure, leave fields null and lower confidence.\n"
    )

    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "intent": {
                "type": ["string", "null"],
                "enum": ["book", "reschedule", "cancel", "general", "off_topic", None],
            },
            "patient_name": {"type": ["string", "null"]},
            "date": {"type": ["string", "null"], "description": "YYYY-MM-DD"},
            "time": {"type": ["string", "null"], "description": "HH:MM 24-hour"},
            "old_date": {"type": ["string", "null"], "description": "YYYY-MM-DD"},
            "new_date": {"type": ["string", "null"], "description": "YYYY-MM-DD"},
            "is_correction": {"type": "boolean"},
            "confidence": {"type": ["number", "null"], "minimum": 0, "maximum": 1},
            "notes": {"type": ["string", "null"]},
        },
        "required": [
            "intent",
            "patient_name",
            "date",
            "time",
            "old_date",
            "new_date",
            "is_correction",
            "confidence",
            "notes",
        ],
    }

    user = {
        "today": today,
        "user_text": user_text,
        "context": ctx,
        "instructions": (
            "Extract only what the user stated or clearly implied. "
            "For cancel/reschedule, fill old_date/new_date when possible. "
            "For cancel-only, old_date may be enough without new_date."
        ),
    }

    try:
        resp = _client().chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(user)},
                {
                    "role": "user",
                    "content": "Return JSON matching this schema exactly: " + json.dumps(schema),
                },
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
        )
        content = resp.choices[0].message.content or "{}"
        raw = json.loads(content)
        return NLUResult(
            intent=raw.get("intent"),
            patient_name=raw.get("patient_name"),
            date=raw.get("date"),
            time=raw.get("time"),
            old_date=raw.get("old_date"),
            new_date=raw.get("new_date"),
            is_correction=bool(raw.get("is_correction", False)),
            confidence=raw.get("confidence"),
            notes=raw.get("notes"),
        )
    except Exception:
        return None


def _clinic_json(clinic: ClinicProfile) -> dict[str, Any]:
    return {
        "clinic_name": clinic.name,
        "address": clinic.address,
        "phone": clinic.phone,
        "hours": clinic.hours,
        "website": clinic.website,
    }


def generate_greeting(clinic: ClinicProfile) -> str | None:
    """One short sentence listing what you can help with; API-first."""
    if not llm_available():
        return None
    model = os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)
    system = (
        "You are a medical office phone assistant.\n"
        "Write EXACTLY ONE sentence, max 18 words.\n"
        "Say hi, then that you can help with: booking appointments, cancelling appointments, "
        "and questions about office hours.\n"
        "No second sentence. No phone number. No front desk. No clinic name unless one word fits.\n"
        "Do not invent doctors or services."
    )
    try:
        resp = _client().chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": json.dumps({"clinic": _clinic_json(clinic)}),
                },
            ],
            temperature=0.4,
        )
        return (resp.choices[0].message.content or "").strip() or None
    except Exception:
        return None


def answer_general_question(user_text: str, *, clinic: ClinicProfile) -> str | None:
    """Clinic-profile questions only; no default 'call front desk' for routine info."""
    if not llm_available():
        return None

    model = os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)
    system = (
        "You are an AI phone assistant for a medical practice.\n"
        "Use ONLY the provided clinic profile for address/hours/phone.\n"
        "Answer briefly and helpfully.\n"
        "If the user asks for urgent/emergency care, tell them to call 911 or go to the nearest ER.\n"
        "Do not invent clinicians, prices, or insurance details.\n"
        "If the question cannot be answered from the profile, say you don't have that detail and they can "
        "call the front desk number from the profile.\n"
    )
    try:
        resp = _client().chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": json.dumps({"clinic": _clinic_json(clinic), "question": user_text}),
                },
            ],
            temperature=0.3,
        )
        return (resp.choices[0].message.content or "").strip() or None
    except Exception:
        return None


def answer_off_topic(user_text: str, *, clinic: ClinicProfile) -> str | None:
    """Direct unrelated requests to front desk (API-first)."""
    if not llm_available():
        return None
    model = os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)
    system = (
        "The caller's request is NOT about booking, cancelling/rescheduling appointments, or basic office hours/location.\n"
        "Politely say you cannot help with that topic here and ask them to contact the front desk.\n"
        "Include the clinic phone number from the profile.\n"
        "Keep it to 1–2 sentences."
    )
    try:
        resp = _client().chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": json.dumps({"clinic": _clinic_json(clinic), "user_text": user_text}),
                },
            ],
            temperature=0.3,
        )
        return (resp.choices[0].message.content or "").strip() or None
    except Exception:
        return None
