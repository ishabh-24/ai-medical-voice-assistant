## SENA Health – AI Voice Assistant (Medical Practice Demo)

Prototype voice assistant that simulates handling patient phone calls for a medical practice.
It supports booking, rescheduling, and general questions.

### Features 
- **Voice interaction**: microphone input (optional) + text-to-speech output; **text fallback** always available
- **Basic intents**: book appointment, reschedule appointment, general question (fallback)
- **Multi-step flow**: asks for missing info (name/date/time), confirms before completing
- **Corrections**: handles changes like “actually, next Monday”
- **Simulated actions**: prints/logs a final `book_appointment` or `reschedule_appointment` result
- **Logging**: writes conversation turns to `logs/assistant.log`
- **Non-hardcoded NLU**: uses **OpenAI API** for intent + slot extraction when configured, with rule-based fallback

### Project structure
- `src/main.py`: CLI entrypoint (text / voice modes)
- `src/dialog.py`: conversation state machine (booking + rescheduling)
- `src/llm_nlu.py`: OpenAI-powered intent + slot extraction + general Q&A (optional)
- `src/voice_io.py`: speech input (optional) and text-to-speech output
- `src/conversation_logger.py`: file logging

### Prerequisites
- Python **3.10+**

### Setup

```bash
cd /Users/ishabhatia/Desktop/sena-technical
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run (text mode)

```bash
python src/main.py --text
```

Type `quit`, `exit`, `goodbye`, etc. to end (or press `Ctrl+C`).

### Run (voice mode)

1. Install microphone dependencies (macOS example):

```bash
brew install portaudio
pip install pyaudio
```

2. Run:

```bash
python src/main.py --voice
```

Notes:
- Speech recognition uses Google STT via `speech_recognition`, so network access is required.
- Use `--no-tts` to disable text-to-speech in voice mode:

```bash
python src/main.py --voice --no-tts
```

To exit by voice, say **“goodbye”**, **“that’s all”**, **“hang up”**, etc. `Ctrl+C` always works.

### Enable OpenAI

The assistant uses OpenAI when `OPENAI_API_KEY` is set (via `.env` or your shell).

**Recommended:** copy the example env file and add your key:

```bash
cp .env.example .env
# Edit `.env` and set OPENAI_API_KEY=sk-...
```

Variables are loaded automatically from `.env` in the project root (see `src/llm_nlu.py`).

Alternatively, export in the terminal:

```bash
export OPENAI_API_KEY="YOUR_KEY"
export OPENAI_MODEL="gpt-4o-mini"   # optional
```

Then run normally (text or voice). If the key is not set, it falls back to rule-based parsing.

**Default model (cost):** `gpt-4o-mini` is used as the default (`OPENAI_MODEL` in `src/llm_nlu.py`). It is a small, cost-effective model that works for JSON extraction and short replies, but a better model would allow for quicker replies.

### Logging
- Conversation and action results are written to `logs/assistant.log`.

### Technical approach (brief)
- **Conversation flow**: a small state machine in `src/dialog.py` tracks phase/step and asks for missing fields.
- **NLU**:
  - Primary: OpenAI (`src/llm_nlu.py`) classifies **book / reschedule / cancel / general / off_topic** and extracts slots (name/date/time).
  - **Cancel**: asks once whether the caller wants **reschedule** or **cancel entirely**; reschedule enters the reschedule flow; cancel-only collects details and simulates **`cancel_appointment`**.
  - **Yes/no** on confirm steps is handled **before** NLU so short answers like “no” are not misclassified.
  - **Off-topic** requests get a short reply directing them to the **front desk phone** from the clinic profile.
  - **Greeting**: one short line via OpenAI (fallback if no API key).
  - **Fallback**: lightweight regex/date parsing when OpenAI is not configured.
- **Clinic grounding**: a small simulated clinic profile in `src/clinic_profile.py` (address/hours/phone) is used to answer general questions (LLM + offline fallback).
- **Safety/robustness**: handles silence/unclear audio, confirmation steps before “finalizing” actions, and logs every turn.

