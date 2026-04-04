"""Voice: Whisper (speech-to-text), OpenAI TTS (speech output). Fallback: print-only if unavailable."""

from __future__ import annotations

import os
import platform
import subprocess
import tempfile
from io import BytesIO
from pathlib import Path
from typing import Callable


def _openai_client():
    from openai import OpenAI

    return OpenAI()


def _play_audio_file(path: Path) -> None:
    system = platform.system()
    try:
        if system == "Darwin":
            subprocess.run(["afplay", str(path)], check=False, capture_output=True)
        elif system == "Windows":
            os.startfile(str(path))  # type: ignore[attr-defined]
        else:
            subprocess.run(["xdg-open", str(path)], check=False, capture_output=True)
    except OSError as e:
        print(f"[Could not play audio: {e}]")


def speak(text: str, *, enabled: bool = True) -> None:
    if not enabled or not text.strip():
        return
    if not os.getenv("OPENAI_API_KEY"):
        _speak_pyttsx3_fallback(text)
        return
    try:
        client = _openai_client()
        model = os.getenv("OPENAI_TTS_MODEL", "tts-1")
        voice = os.getenv("OPENAI_TTS_VOICE", "alloy")
        response = client.audio.speech.create(
            model=model,
            voice=voice,
            input=text[:4096],
        )
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            response.write_to_file(str(tmp_path))
            _play_audio_file(tmp_path)
        finally:
            try:
                tmp_path.unlink()
            except OSError:
                pass
    except Exception as e:
        print(f"[OpenAI TTS error: {e}]")
        _speak_pyttsx3_fallback(text)


def _speak_pyttsx3_fallback(text: str) -> None:
    try:
        import pyttsx3

        engine = pyttsx3.init()
        engine.setProperty("rate", 175)
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print(f"[Voice output fallback error: {e}]")


def listen(
    *,
    timeout: float = 7.0,
    phrase_time_limit: float = 12.0,
    ambient_duration: float = 0.4,
    on_timeout: Callable[[], None] | None = None,
) -> str | None:
    """
    Capture microphone audio, transcribe with OpenAI Whisper (whisper-1).
    Returns None on silence/timeout or missing deps/API key.
    """
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is required for Whisper voice input. Use --text or set your key.")
        return None
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

    wav_bytes = audio.get_wav_data()
    try:
        client = _openai_client()
        stt_model = os.getenv("OPENAI_STT_MODEL", "whisper-1")
        transcript = client.audio.transcriptions.create(
            model=stt_model,
            file=("audio.wav", BytesIO(wav_bytes)),
        )
        text = (transcript.text or "").strip()
        return text if text else ""
    except Exception as e:
        print(f"Whisper transcription error: {e}")
        return None


def voice_available() -> bool:
    """True if OpenAI key present and packages can load (mic may still fail at runtime)."""
    if not os.getenv("OPENAI_API_KEY"):
        return False
    try:
        import speech_recognition as sr  # noqa: F401
        from openai import OpenAI  # noqa: F401

        return True
    except ImportError:
        return False
