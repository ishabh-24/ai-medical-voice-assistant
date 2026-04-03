"""Multi-step dialog for booking and rescheduling.
Modes:
- OpenAI-powered NLU (intent + slot extraction) when OPENAI_API_KEY is set
- Rule-based fallback (regex/date parsing) when not set
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum

from .date_parse import looks_like_correction, parse_date_phrase, parse_time_phrase
from .intents import Intent, classify_intent, extract_name, is_unclear
from .clinic_profile import CLINIC
from .llm_nlu import (
    answer_general_question,
    answer_off_topic,
    extract_nlu,
)


class Phase(str, Enum):
    ROUTE = "route"
    BOOK = "book"
    RESCHEDULE = "reschedule"
    CANCEL = "cancel"
    GENERAL = "general"


class CancelStep(str, Enum):
    OFFER = "offer"
    NAME = "name"
    DATE = "date"
    CONFIRM = "confirm"


class BookStep(str, Enum):
    NAME = "name"
    DATE = "date"
    TIME = "time"
    CONFIRM = "confirm"
    DONE = "done"


class RescheduleStep(str, Enum):
    NAME = "name"
    OLD_DATE = "old_date"
    NEW_DATE = "new_date"
    CONFIRM = "confirm"
    DONE = "done"


@dataclass
class DialogState:
    phase: Phase = Phase.ROUTE
    book_step: BookStep = BookStep.NAME
    reschedule_step: RescheduleStep = RescheduleStep.NAME
    patient_name: str | None = None
    date: str | None = None
    time: str | None = None
    old_date: str | None = None
    new_date: str | None = None
    pending_intent: Intent | None = None
    cancel_step: CancelStep = CancelStep.OFFER


@dataclass
class TurnResult:
    reply: str
    done: bool = False
    action_log: dict | None = field(default_factory=dict)


def reset_slots(state: DialogState) -> None:
    """Clear collected slots after a completed or cancelled flow."""
    state.patient_name = None
    state.date = None
    state.time = None
    state.old_date = None
    state.new_date = None
    state.book_step = BookStep.NAME
    state.reschedule_step = RescheduleStep.NAME
    state.pending_intent = None
    state.cancel_step = CancelStep.OFFER


def _normalize_short_reply(text: str) -> str:
    """STT often mishears 'no' as 'know'."""
    t = text.strip().lower().rstrip(".,!?")
    if t == "know":
        return "no"
    return t


_YES = frozenset(
    {"yes", "yeah", "yep", "yup", "ok", "okay", "sure", "please", "confirm", "correct", "affirmative"}
)
_NO = frozenset({"no", "nope", "nah", "negative", "cancel"})
_NO_STRICT = frozenset({"no", "nope", "nah", "negative"})
_NOT_NAMES = frozenset(
    {"cancel", "reschedule", "yes", "no", "nope", "nah", "know", "hi", "hello", "ok", "okay", "sure"}
)


def _is_yes(text: str) -> bool:
    t = _normalize_short_reply(text)
    if not t:
        return False
    w = t.split()[0]
    return w in _YES


def _is_no(text: str) -> bool:
    t = _normalize_short_reply(text)
    if not t:
        return False
    w = t.split()[0]
    return w in _NO


def _try_book_confirm_shortcut(state: DialogState, text: str, action_log: dict) -> TurnResult | None:
    if state.phase != Phase.BOOK or state.book_step != BookStep.CONFIRM:
        return None
    if not (_is_yes(text) or _is_no(text)):
        return None
    if _is_yes(text):
        return _book_finalize(state, action_log)
    state.phase = Phase.ROUTE
    reset_slots(state)
    return TurnResult(
        reply="Okay, I won't book that. What else can I help with?",
        action_log=action_log,
        done=False,
    )


def _try_reschedule_confirm_shortcut(state: DialogState, text: str, action_log: dict) -> TurnResult | None:
    if state.phase != Phase.RESCHEDULE or state.reschedule_step != RescheduleStep.CONFIRM:
        return None
    if not (_is_yes(text) or _is_no(text)):
        return None
    if _is_yes(text):
        record = {
            "action": "reschedule_appointment",
            "patient_name": state.patient_name,
            "old_date": state.old_date,
            "new_date": state.new_date,
        }
        action_log["simulated_result"] = record
        pn, o, nw = state.patient_name, state.old_date, state.new_date
        state.phase = Phase.ROUTE
        reset_slots(state)
        return TurnResult(
            reply=(
                f"Done. {pn}'s appointment has been moved from {o} to {nw}. "
                f"Anything else I can help with?"
            ),
            done=True,
            action_log=action_log,
        )
    state.phase = Phase.ROUTE
    reset_slots(state)
    return TurnResult(reply="Okay, no changes made. How else can I help?", action_log=action_log)


def _try_cancel_only_confirm_shortcut(state: DialogState, text: str, action_log: dict) -> TurnResult | None:
    """Do not treat the word 'cancel' as denial here — it may confirm cancellation."""
    if state.phase != Phase.CANCEL or state.cancel_step != CancelStep.CONFIRM:
        return None
    t = _normalize_short_reply(text)
    w = t.split()[0] if t else ""
    if w not in _YES and w not in _NO_STRICT:
        return None
    if w in _YES:
        record = {
            "action": "cancel_appointment",
            "patient_name": state.patient_name,
            "date": state.old_date,
        }
        action_log["simulated_result"] = record
        pn, od = state.patient_name, state.old_date
        state.phase = Phase.ROUTE
        reset_slots(state)
        return TurnResult(
            reply=f"Done — I've cancelled the appointment for {pn} on {od}. Anything else?",
            done=True,
            action_log=action_log,
        )
    state.phase = Phase.ROUTE
    reset_slots(state)
    return TurnResult(reply="Okay, I didn't cancel anything. How else can I help?", action_log=action_log)


def _parse_cancel_offer_choice(text: str, *, first_turn: bool) -> str:
    """Return 'reschedule' | 'cancel_only' | 'unknown'."""
    t = _normalize_short_reply(text)
    t = t.strip().rstrip(".,!?")

    if not first_turn:
        if _is_no(text) or t in ("no", "nope", "nah"):
            return "cancel_only"
        if t == "reschedule" or t.startswith("reschedule "):
            return "reschedule"
        if t == "cancel" or t.startswith("cancel "):
            return "cancel_only"

    if re.search(r"\b(reschedule|rescheduling)\b", text, re.I):
        return "reschedule"
    if re.search(
        r"\b(just\s+cancel|cancel\s+entirely|cancel\s+only|remove\s+(my\s+)?appointment|drop\s+(the\s+)?appointment)\b",
        text,
        re.I,
    ):
        return "cancel_only"
    if first_turn:
        return "unknown"
    if re.search(r"\bcancel\b", text, re.I):
        return "cancel_only"
    return "unknown"


def _try_cancel_offer_turn(
    state: DialogState, text: str, action_log: dict, *, first_turn: bool
) -> TurnResult | None:
    """Resolve reschedule vs cancel-only before NLU (so 'no' is never misclassified)."""
    if state.phase != Phase.CANCEL or state.cancel_step != CancelStep.OFFER:
        return None
    choice = _parse_cancel_offer_choice(text, first_turn=first_turn)
    if choice == "unknown":
        n = extract_name(text)
        if n and str(n).lower().strip() not in _NOT_NAMES:
            state.cancel_step = CancelStep.DATE
            state.patient_name = n.strip().title() if " " in n else n
            return TurnResult(
                reply=(
                    f"What date is the appointment you want to cancel for {state.patient_name}?"
                ),
                action_log=action_log,
            )
        return TurnResult(
            reply="Say reschedule to move your visit, or cancel to cancel it entirely.",
            action_log=action_log,
        )
    if choice == "reschedule":
        state.phase = Phase.RESCHEDULE
        state.reschedule_step = RescheduleStep.NAME
        state.cancel_step = CancelStep.OFFER
        return _reschedule_flow(state, text, action_log, start=True)
    state.cancel_step = CancelStep.NAME
    return _cancel_flow(state, text, action_log, start=False)


def _resolve_intent(text: str, nlu) -> Intent:
    """Prefer OpenAI NLU intent; else rule-based fallback."""
    if nlu and nlu.intent in (
        "book",
        "reschedule",
        "cancel",
        "general",
        "off_topic",
    ):
        try:
            return Intent(nlu.intent)
        except ValueError:
            pass
    return classify_intent(text)


def _off_topic_reply(user_text: str) -> str:
    t = answer_off_topic(user_text, clinic=CLINIC)
    if t:
        return t
    return (
        "I can't help with that here. For questions outside scheduling or office hours, "
        f"please call our front desk at {CLINIC.phone}."
    )


def _general_reply(user_text: str) -> str:
    if "emergency" in user_text.lower():
        return "If this is a medical emergency, please hang up and dial 911 or go to the nearest ER."

    llm = answer_general_question(user_text, clinic=CLINIC)
    if llm:
        return llm

    t = user_text.lower()
    if any(k in t for k in ("hours", "open", "close", "closing", "weekend", "saturday", "sunday")):
        return f"Our hours are: {CLINIC.hours}."
    if any(k in t for k in ("address", "location", "where are you", "where is", "directions")):
        return f"Our address is {CLINIC.address}."
    if any(k in t for k in ("phone", "number", "call you", "contact")):
        return f"You can reach the office at {CLINIC.phone}."

    return (
        "I can help with booking, cancelling, or rescheduling appointments, or questions about hours and location. "
        f"For other topics, call {CLINIC.phone}."
    )


def process_turn(state: DialogState, user_text: str) -> TurnResult:
    text = user_text.strip()
    action_log: dict = {"raw_input": text}

    if is_unclear(text) and state.phase == Phase.ROUTE:
        return TurnResult(
            reply="I didn't quite catch that. Would you like to book, cancel or reschedule an appointment, "
            "or ask about hours?",
            action_log=action_log,
        )

    # yes/no on confirm steps before NLU so "no" is never misheard or reclassified
    r = _try_book_confirm_shortcut(state, text, action_log)
    if r:
        return r
    r = _try_reschedule_confirm_shortcut(state, text, action_log)
    if r:
        return r
    r = _try_cancel_only_confirm_shortcut(state, text, action_log)
    if r:
        return r

    # cancel: reschedule vs cancel-only (before NLU)
    if state.phase == Phase.CANCEL and state.cancel_step == CancelStep.OFFER:
        r = _try_cancel_offer_turn(state, text, action_log, first_turn=False)
        if r:
            return r

    # llm nlu for intent/slots
    nlu = extract_nlu(
        text,
        context={
            "phase": state.phase,
            "book_step": state.book_step,
            "reschedule_step": state.reschedule_step,
            "cancel_step": state.cancel_step,
            "patient_name": state.patient_name,
            "date": state.date,
            "time": state.time,
            "old_date": state.old_date,
            "new_date": state.new_date,
        },
    )
    if nlu:
        action_log["nlu"] = {
            "intent": nlu.intent,
            "patient_name": nlu.patient_name,
            "date": nlu.date,
            "time": nlu.time,
            "old_date": nlu.old_date,
            "new_date": nlu.new_date,
            "is_correction": nlu.is_correction,
            "confidence": nlu.confidence,
        }

        if nlu.patient_name:
            state.patient_name = nlu.patient_name
        if nlu.date:
            state.date = nlu.date
        if nlu.time:
            state.time = nlu.time
        if nlu.old_date:
            state.old_date = nlu.old_date
        if nlu.new_date:
            state.new_date = nlu.new_date

    if state.phase == Phase.ROUTE:
        intent = _resolve_intent(text, nlu)
        if intent == Intent.OFF_TOPIC:
            return TurnResult(reply=_off_topic_reply(text), action_log=action_log)
        if intent == Intent.CANCEL:
            state.phase = Phase.CANCEL
            state.cancel_step = CancelStep.OFFER
            state.pending_intent = None
            return _cancel_flow(state, text, action_log, start=True)
        if intent == Intent.BOOK:
            state.phase = Phase.BOOK
            state.book_step = BookStep.NAME
            state.pending_intent = None
            return _book_flow(state, text, action_log, start=True)
        if intent == Intent.RESCHEDULE:
            state.phase = Phase.RESCHEDULE
            state.reschedule_step = RescheduleStep.NAME
            state.pending_intent = None
            return _reschedule_flow(state, text, action_log, start=True)
        return TurnResult(reply=_general_reply(text), action_log=action_log)

    if state.phase == Phase.BOOK:
        return _book_flow(state, text, action_log, start=False)
    if state.phase == Phase.RESCHEDULE:
        return _reschedule_flow(state, text, action_log, start=False)
    if state.phase == Phase.CANCEL:
        return _cancel_flow(state, text, action_log, start=False)

    return TurnResult(reply=_general_reply(text), action_log=action_log)


def _cancel_flow(
    state: DialogState,
    text: str,
    action_log: dict,
    *,
    start: bool,
) -> TurnResult:
    """Offer reschedule vs cancel-only; then collect name/date and confirm cancellation."""
    if start:
        choice = _parse_cancel_offer_choice(text, first_turn=True)
        if choice == "reschedule":
            state.phase = Phase.RESCHEDULE
            state.reschedule_step = RescheduleStep.NAME
            state.cancel_step = CancelStep.OFFER
            return _reschedule_flow(state, text, action_log, start=True)
        if choice == "cancel_only":
            state.cancel_step = CancelStep.NAME
            n = extract_name(text)
            if n and str(n).lower().strip() in _NOT_NAMES:
                n = None
            if n:
                state.patient_name = n
            if state.patient_name:
                state.cancel_step = CancelStep.DATE
                return TurnResult(
                    reply=(
                        f"What date is the appointment you want to cancel for {state.patient_name}?"
                    ),
                    action_log=action_log,
                )
            return TurnResult(
                reply="What is the patient's full name for the appointment you want to cancel?",
                action_log=action_log,
            )
        state.cancel_step = CancelStep.OFFER
        return TurnResult(
            reply="Would you like to reschedule, or cancel the appointment entirely?",
            action_log=action_log,
        )

    if state.cancel_step == CancelStep.OFFER:
        return TurnResult(
            reply="Would you like to reschedule, or cancel the appointment entirely?",
            action_log=action_log,
        )

    if state.cancel_step == CancelStep.NAME:
        w0 = _normalize_short_reply(text).split()[0] if text.strip() else ""
        if w0 in _NOT_NAMES:
            return TurnResult(
                reply="What is the patient's full name for the appointment you want to cancel?",
                action_log=action_log,
            )
        n = extract_name(text)
        if not n and len(text.strip()) > 2:
            cand = text.strip()
            if cand.lower().split()[0] not in _NOT_NAMES:
                n = cand
        if not n:
            return TurnResult(
                reply="What is the patient's full name for the appointment you want to cancel?",
                action_log=action_log,
            )
        state.patient_name = n.strip().title() if " " in str(n) else n
        state.cancel_step = CancelStep.DATE
        return TurnResult(
            reply=f"What date is the appointment you want to cancel for {state.patient_name}?",
            action_log=action_log,
        )

    if state.cancel_step == CancelStep.DATE:
        od = state.old_date or parse_date_phrase(text)
        if not od:
            return TurnResult(
                reply="What date is that appointment? For example, next Tuesday or 04/15/2026.",
                action_log=action_log,
            )
        state.old_date = od
        state.cancel_step = CancelStep.CONFIRM
        return TurnResult(
            reply=(
                f"Please confirm: cancel the appointment for {state.patient_name} on {state.old_date}. "
                f"Say yes to confirm or no to stop."
            ),
            action_log=action_log,
        )

    if state.cancel_step == CancelStep.CONFIRM:
        return TurnResult(
            reply="Please say yes to confirm cancellation, or no to stop.",
            action_log=action_log,
        )

    return TurnResult(reply="How else can I help?", action_log=action_log)


def _book_flow(
    state: DialogState,
    text: str,
    action_log: dict,
    *,
    start: bool,
) -> TurnResult:
    if start:
        # try to pre-fill from first utterance (llm nlu already updated slots)
        if not state.patient_name:
            n = extract_name(text)
            if n:
                state.patient_name = n
        if not state.date:
            d = parse_date_phrase(text)
            if d:
                state.date = d
        if not state.time:
            tm = parse_time_phrase(text)
            if tm:
                state.time = tm
        if state.patient_name and state.date and state.time:
            state.book_step = BookStep.CONFIRM
            return _book_confirm(state, action_log)
        if state.patient_name and state.date:
            state.book_step = BookStep.TIME
            return TurnResult(
                reply=f"Thanks, {state.patient_name}. What time on {state.date} works best?",
                action_log=action_log,
            )
        if state.patient_name:
            state.book_step = BookStep.DATE
            return TurnResult(
                reply=f"Thanks, {state.patient_name}. What date would you like for your appointment?",
                action_log=action_log,
            )
        state.book_step = BookStep.NAME
        return TurnResult(
            reply="I can help you book an appointment. What is the patient's full name?",
            action_log=action_log,
        )

    # corrections before step handlers
    if looks_like_correction(text):
        d = parse_date_phrase(text)
        tm = parse_time_phrase(text)
        n = extract_name(text)
        if state.book_step in (BookStep.DATE, BookStep.TIME, BookStep.CONFIRM):
            if d:
                state.date = d
                action_log["correction"] = "date_updated"
            if tm:
                state.time = tm
                action_log["correction"] = action_log.get("correction", "time_updated")
            if n:
                state.patient_name = n
        if state.book_step == BookStep.NAME and n:
            state.patient_name = n
        if state.book_step == BookStep.TIME:
            if d and not tm:
                return TurnResult(
                    reply=f"Updated to {state.date}. What time would you prefer?",
                    action_log=action_log,
                )
            if d and tm:
                state.time = tm
                state.book_step = BookStep.CONFIRM
                return _book_confirm(state, action_log)
            if not d and tm:
                state.time = tm
                state.book_step = BookStep.CONFIRM
                return _book_confirm(state, action_log)

    if state.book_step == BookStep.NAME:
        n = extract_name(text) or (text.strip() if len(text.strip()) > 2 else None)
        if not n:
            return TurnResult(
                reply="I didn't get a clear name. Could you say it again, for example: My name is Jane Doe?",
                action_log=action_log,
            )
        state.patient_name = n.strip().title() if " " in n else n
        state.book_step = BookStep.DATE
        return TurnResult(
            reply=f"Thank you, {state.patient_name}. What date works for you?",
            action_log=action_log,
        )

    if state.book_step == BookStep.DATE:
        d = state.date or parse_date_phrase(text)
        if not d:
            return TurnResult(
                reply="I couldn't parse that date. Try something like next Monday, March 30, or 04/15/2026.",
                action_log=action_log,
            )
        state.date = d
        state.book_step = BookStep.TIME
        return TurnResult(
            reply=f"Got it — {d}. What time would you prefer?",
            action_log=action_log,
        )

    if state.book_step == BookStep.TIME:
        tm = state.time or parse_time_phrase(text)
        if not tm:
            return TurnResult(
                reply="What time should I book? For example, 10:30 AM or afternoon.",
                action_log=action_log,
            )
        state.time = tm
        state.book_step = BookStep.CONFIRM
        return _book_confirm(state, action_log)

    if state.book_step == BookStep.CONFIRM:
        if _is_yes(text):
            return _book_finalize(state, action_log)
        if _is_no(text):
            state.phase = Phase.ROUTE
            reset_slots(state)
            return TurnResult(
                reply="Okay, I won't book that. What else can I help with?",
                action_log=action_log,
                done=False,
            )
        # treat as correction during confirm
        if looks_like_correction(text) or parse_date_phrase(text) or parse_time_phrase(text):
            d = parse_date_phrase(text)
            tm = parse_time_phrase(text)
            if d:
                state.date = d
            if tm:
                state.time = tm
            return _book_confirm(state, action_log)
        return TurnResult(
            reply="Please say yes to confirm the appointment, or no to cancel.",
            action_log=action_log,
        )

    return TurnResult(reply="Booking is complete. Say book or reschedule to start again.", done=True)


def _book_confirm(state: DialogState, action_log: dict) -> TurnResult:
    return TurnResult(
        reply=(
            f"Please confirm: appointment for {state.patient_name} on {state.date} at {state.time}. "
            f"Say yes to confirm or no to cancel."
        ),
        action_log=action_log,
    )


def _book_finalize(state: DialogState, action_log: dict) -> TurnResult:
    record = {
        "action": "book_appointment",
        "patient_name": state.patient_name,
        "date": state.date,
        "time": state.time,
    }
    action_log["simulated_result"] = record
    pn, d, tm = state.patient_name, state.date, state.time
    state.phase = Phase.ROUTE
    reset_slots(state)
    return TurnResult(
        reply=(
            f"Your appointment is booked for {pn} on {d} at {tm}. "
            f"We'll send a reminder. Thank you for calling."
        ),
        done=True,
        action_log=action_log,
    )


def _reschedule_flow(
    state: DialogState,
    text: str,
    action_log: dict,
    *,
    start: bool,
) -> TurnResult:
    if start:
        # llm nlu may have already updated slots 
        n = state.patient_name or extract_name(text)
        od: str | None = None
        nd: str | None = None
        m_from_to = re.search(r"\bfrom\s+(.+?)\s+to\s+(.+)$", text, re.I)
        if m_from_to:
            od = parse_date_phrase(m_from_to.group(1))
            nd = parse_date_phrase(m_from_to.group(2))
        else:
            od = state.old_date or parse_date_phrase(text)
            if " to " in text.lower():
                parts = re_split_reschedule(text)
                if len(parts) >= 2:
                    nd = state.new_date or parse_date_phrase(parts[-1])
        if n:
            state.patient_name = n
        if od:
            state.old_date = od
        if nd:
            state.new_date = nd
        if state.patient_name and state.old_date and state.new_date:
            state.reschedule_step = RescheduleStep.CONFIRM
            return _reschedule_confirm(state, action_log)
        if state.patient_name and state.old_date:
            state.reschedule_step = RescheduleStep.NEW_DATE
            return TurnResult(
                reply=f"What new date would you like instead of {state.old_date}?",
                action_log=action_log,
            )
        if state.patient_name:
            state.reschedule_step = RescheduleStep.OLD_DATE
            return TurnResult(
                reply="What is the date of the appointment you want to move?",
                action_log=action_log,
            )
        state.reschedule_step = RescheduleStep.NAME
        return TurnResult(
            reply="I can reschedule your visit. What is the patient's name?",
            action_log=action_log,
        )

    if looks_like_correction(text):
        od = parse_date_phrase(text)
        if od and state.reschedule_step in (
            RescheduleStep.OLD_DATE,
            RescheduleStep.NEW_DATE,
            RescheduleStep.CONFIRM,
        ):
            if state.reschedule_step == RescheduleStep.OLD_DATE:
                state.old_date = od
            elif state.reschedule_step == RescheduleStep.NEW_DATE:
                state.new_date = od
            else:
                state.new_date = od
            action_log["correction"] = "date_updated"

    if state.reschedule_step == RescheduleStep.NAME:
        n = extract_name(text) or (text.strip() if len(text.strip()) > 2 else None)
        if not n:
            return TurnResult(
                reply="Please tell me the name on the appointment.",
                action_log=action_log,
            )
        state.patient_name = n.strip().title() if " " in str(n) else n
        state.reschedule_step = RescheduleStep.OLD_DATE
        return TurnResult(
            reply=f"Thanks, {state.patient_name}. Which appointment date should we move?",
            action_log=action_log,
        )

    if state.reschedule_step == RescheduleStep.OLD_DATE:
        od = state.old_date or parse_date_phrase(text)
        if not od:
            return TurnResult(
                reply="I need the current appointment date, e.g. next Tuesday or 04/01/2026.",
                action_log=action_log,
            )
        state.old_date = od
        state.reschedule_step = RescheduleStep.NEW_DATE
        return TurnResult(
            reply=f"Understood — we'll move the visit from {od}. What is the new date?",
            action_log=action_log,
        )

    if state.reschedule_step == RescheduleStep.NEW_DATE:
        nd = state.new_date or parse_date_phrase(text)
        if not nd:
            return TurnResult(
                reply="What date would you like instead? For example, next Monday.",
                action_log=action_log,
            )
        state.new_date = nd
        state.reschedule_step = RescheduleStep.CONFIRM
        return _reschedule_confirm(state, action_log)

    if state.reschedule_step == RescheduleStep.CONFIRM:
        if _is_yes(text):
            record = {
                "action": "reschedule_appointment",
                "patient_name": state.patient_name,
                "old_date": state.old_date,
                "new_date": state.new_date,
            }
            action_log["simulated_result"] = record
            pn, o, nw = state.patient_name, state.old_date, state.new_date
            state.phase = Phase.ROUTE
            reset_slots(state)
            return TurnResult(
                reply=(
                    f"Done. {pn}'s appointment has been moved from {o} "
                    f"to {nw}. Anything else I can help with?"
                ),
                done=True,
                action_log=action_log,
            )
        if _is_no(text):
            state.phase = Phase.ROUTE
            reset_slots(state)
            return TurnResult(reply="Okay, no changes made. How else can I help?", action_log=action_log)
        if looks_like_correction(text) or parse_date_phrase(text):
            nd = parse_date_phrase(text)
            if nd:
                state.new_date = nd
            return _reschedule_confirm(state, action_log)
        return TurnResult(
            reply="Say yes to confirm the new date or no to cancel.",
            action_log=action_log,
        )

    return TurnResult(reply="Reschedule complete.", done=True)


def _reschedule_confirm(state: DialogState, action_log: dict) -> TurnResult:
    return TurnResult(
        reply=(
            f"Confirm reschedule for {state.patient_name}: from {state.old_date} to {state.new_date}. "
            f"Say yes to confirm."
        ),
        action_log=action_log,
    )


def re_split_reschedule(text: str) -> list[str]:
    """Very small helper for 'from X to Y' style phrases."""
    m = re.search(r"\bfrom\s+(.+?)\s+to\s+(.+)$", text, re.I)
    if m:
        return [m.group(1).strip(), m.group(2).strip()]
    return text.split(" to ")
