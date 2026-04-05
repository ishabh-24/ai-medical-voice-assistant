## SENA Health – AI Voice Assistant (Medical Practice Demo)

Prototype voice assistant that simulates handling patient phone calls for a medical practice.
It supports booking, rescheduling, and general questions.

### Features 
- **Voice interaction**: mic → **Whisper** STT, dialog + NLU → **OpenAI chat** (`OPENAI_MODEL`, default `gpt-5.3`), reply → **OpenAI TTS** (`tts-1`); **text mode** always available
- **Intents** (NLU): **book**, **reschedule**, **cancel**, **general**, **off_topic**; routing also uses lightweight rules when the API is unavailable
- **Multi-step flow**: asks for missing info (name/date/time), confirms before completing
- **Corrections**: handles changes like “actually, next Monday”
- **Simulated actions**: prints/logs a final `book_appointment`, `reschedule_appointment`, or `cancel_appointment` result
- **Logging**: writes conversation turns to `logs/assistant.log`
- **NLU + rules**: OpenAI JSON extraction when `OPENAI_API_KEY` is set; **regex / `date_parse`** always applied on top of model output for reliable date/time (and **clinic-hours checks** in `clinic_hours.py` for booking times)

### Project structure
- `src/main.py`: CLI entrypoint (text / voice modes)
- `src/dialog.py`: conversation state machine (booking, reschedule, cancel)
- `src/llm_nlu.py`: OpenAI intent + slot extraction + greeting + general/off-topic replies (optional without API key)
- `src/date_parse.py`: deterministic date/time parsing (used with / without the LLM)
- `src/clinic_hours.py`: office-hours validation for proposed appointment times
- `src/session_exit.py`: voice/text phrases to end the session
- `src/clinic_profile.py`: clinic facts for grounded answers
- `src/voice_io.py`: Whisper mic capture + OpenAI TTS (with offline fallbacks if needed)
- `src/conversation_logger.py`: file logging

### Prerequisites
- Python **3.10+**

### Getting started

1. **Clone the repository** and enter the project directory:

```bash
git clone https://github.com/ishabh-24/ai-medical-voice-assistant.git
cd ai-medical-voice-assistant
```

2. **Create a virtual environment** and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

3. **Configure the environment** (recommended if you use OpenAI for NLU, STT, or TTS). Copy the example file and add your API key:

```bash
cp .env.example .env
# Edit `.env` and set OPENAI_API_KEY=sk-...
```

See [Enable OpenAI](#enable-openai) below for optional variables (`OPENAI_MODEL`, voice mode, etc.). The app still runs in a reduced rule-based mode without a key.

### Run (text mode)

```bash
python src/main.py --text
```

Type `quit`, `exit`, `goodbye`, etc. to end (or press `Ctrl+C`).

### Run (voice mode)

1. Install microphone capture (macOS example):

```bash
brew install portaudio
pip install pyaudio
```

2. Set `OPENAI_API_KEY` in `.env`. Voice mode uses:
   - **Whisper** (`OPENAI_STT_MODEL`, default `whisper-1`) to transcribe the mic
   - **GPT-5.3** (`OPENAI_MODEL`, default `gpt-5.3`) for assistant logic and replies
   - **TTS** (`OPENAI_TTS_MODEL`, default `tts-1`) to speak replies; override `OPENAI_TTS_VOICE` if needed

3. Run:

```bash
python src/main.py --voice
```

Notes:
- Playback on macOS uses `afplay` for MP3 from OpenAI TTS.
- Use `--no-tts` to print replies only (still uses Whisper for input):

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
export OPENAI_MODEL="gpt-5.3"   # must match an ID your key can use
export OPENAI_STT_MODEL="whisper-1"
export OPENAI_TTS_MODEL="tts-1"
```

Then run normally (text or voice). If the key is not set, it falls back to rule-based parsing.

**Models:** Defaults are **`gpt-5.3`** for chat/NLU, **`whisper-1`** for mic transcription, and **`tts-1`** for spoken replies (see `src/llm_nlu.py` and `src/voice_io.py`).

### Logging
- Conversation and action results are written to `logs/assistant.log`.

### Technical approach 
- **Conversation flow**: a small state machine in `src/dialog.py` tracks phase/step and asks for missing fields.
- **NLU**:
  - Primary: OpenAI (`src/llm_nlu.py`) classifies **book / reschedule / cancel / general / off_topic** and extracts slots (name/date/time, plus old/new dates for reschedule when relevant).
  - Slot fields from the model are **scoped by phase** (e.g. reschedule `old_date`/`new_date` are not merged during booking) and **merged with deterministic parsing** from the same user text so one utterance can fill date/time reliably.
  - **Cancel**: asks once whether the caller wants **reschedule** or **cancel entirely**; reschedule enters the reschedule flow; cancel-only collects details and simulates **`cancel_appointment`**.
  - **Yes/no** on confirm steps is handled **before** NLU so short answers like “no” are not misclassified.
  - **Off-topic** requests get a short reply directing them to the **front desk phone** from the clinic profile.
  - **Greeting**: one short line via OpenAI (static fallback if no API key).
  - **Without OpenAI**: rule-based intent from `src/intents.py` and regex/date parsing only; no LLM slots.
- **Clinic grounding**: `src/clinic_profile.py` (address/hours/phone) for general questions; **`clinic_hours.py`** enforces bookable times against those hours.
- **Safety/robustness**: handles silence/unclear audio, confirmation steps before “finalizing” actions, and logs every turn.

