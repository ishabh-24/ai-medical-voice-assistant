"""Medical practice voice assistant (voice + text)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.clinic_profile import CLINIC
from src.conversation_logger import log_turn, setup_logging
from src.dialog import DialogState, process_turn
from src.llm_nlu import generate_greeting
from src.session_exit import wants_quit
from src.voice_io import listen, speak, voice_available


def main() -> None:
    parser = argparse.ArgumentParser(description="Medical practice voice assistant (demo).")
    parser.add_argument(
        "--voice",
        action="store_true",
        help="Use microphone for input and TTS for output (requires PyAudio + network for OpenAI Whisper + TTS).",
    )
    parser.add_argument(
        "--no-tts",
        action="store_true",
        help="Disable text-to-speech even in voice mode.",
    )
    parser.add_argument(
        "--text",
        action="store_true",
        help="Text-only REPL (no microphone).",
    )
    args = parser.parse_args()

    text_only = args.text or not args.voice
    use_tts = args.voice and not args.no_tts

    logger = setup_logging()
    state = DialogState()

    greeting = generate_greeting(CLINIC) or (
        "Hi — I can help with booking, cancelling an appointment, or questions about our hours. "
        f"For anything else, call {CLINIC.phone}."
    )
    print(f"Assistant: {greeting}")
    log_turn(logger, role="assistant", text=greeting)
    if use_tts:
        speak(greeting, enabled=True)

    if not text_only and not voice_available():
        print(
            "Note: voice mode needs OPENAI_API_KEY, openai package, speechrecognition, and PyAudio.\n"
            "Falling back to text input."
        )
        text_only = True

    while True:
        user_text: str | None = None
        if text_only:
            try:
                line = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye.")
                break
            if wants_quit(line):
                print("Goodbye.")
                log_turn(logger, role="user", text=line, extra={"exit": True})
                break
            user_text = line
        else:
            print("Listening... (speak now)")
            user_text = listen(
                timeout=7.0,
                on_timeout=lambda: print("(No speech detected — try again.)"),
            )
            if user_text is None:
                continue
            if user_text == "":
                msg = "I didn't understand that. Could you repeat it?"
                print(f"Assistant: {msg}")
                log_turn(logger, role="assistant", text=msg, extra={"reason": "unknown_audio"})
                speak(msg, enabled=use_tts)
                continue
            if wants_quit(user_text):
                farewell = "Goodbye."
                print(f"You (heard): {user_text}")
                print(f"Assistant: {farewell}")
                log_turn(logger, role="user", text=user_text, extra={"exit": True})
                log_turn(logger, role="assistant", text=farewell, extra={"exit": True})
                speak(farewell, enabled=use_tts)
                break
            print(f"You (heard): {user_text}")

        log_turn(logger, role="user", text=user_text)
        result = process_turn(state, user_text)
        print(f"Assistant: {result.reply}")
        log_turn(
            logger,
            role="assistant",
            text=result.reply,
            extra=result.action_log,
        )
        speak(result.reply, enabled=use_tts)

        if result.action_log.get("simulated_result"):
            action = result.action_log["simulated_result"]
            print("\n--- Simulated action ---")
            print(action)
            print("------------------------\n")
            logger.info(f"ACTION_RESULT: {action}")

if __name__ == "__main__":
    main()
