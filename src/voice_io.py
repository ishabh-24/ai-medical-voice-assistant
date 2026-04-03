"""speech input (optional) and text-to-speech output."""

from __future__ import annotations

from typing import Callable


def speak(text: str, *, enabled: bool = True) -> None:
    if not enabled or not text.strip():
        return
    try:
        import pyttsx3

        engine = pyttsx3.init()
        engine.setProperty("rate", 175)
        engine.say(text)
        engine.runAndWait()
    except OSError as e:
        print(f"[Voice output unavailable: {e}]")
    except Exception as e:
        print(f"[Voice output error: {e}]")


def listen(
    *,
    timeout: float = 6.0,
    phrase_time_limit: float = 12.0,
    ambient_duration: float = 0.4,
    on_timeout: Callable[[], None] | None = None,
) -> str | None:
    """
    capture one utterance from the default microphone.
    returns None on silence/timeout or if speech recognition is unavailable.
    """
    try:
        import speech_recognition as sr
    except ImportError:
        print("speech_recognition is not installed; use text mode.")
        return None

    r = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            r.adjust_for_ambient_noise(source, duration=ambient_duration)
            audio = r.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
    except sr.WaitTimeoutError:
        if on_timeout:
            on_timeout()
        return None
    except OSError as e:
        print(f"Microphone not available: {e}")
        return None

    try:
        return r.recognize_google(audio)
    except sr.UnknownValueError:
        return ""
    except sr.RequestError as e:
        print(f"Speech recognition service error: {e}")
        return None


def voice_available() -> bool:
    try:
        import speech_recognition as sr  
        import pyttsx3  

        return True
    except ImportError:
        return False
