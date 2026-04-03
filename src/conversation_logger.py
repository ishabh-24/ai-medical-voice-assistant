"""Conversation logging"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def setup_logging(log_dir: Path | None = None) -> logging.Logger:
    log_dir = log_dir or Path(__file__).resolve().parent.parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "assistant.log"

    logger = logging.getLogger("medical_assistant")
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        )
        logger.addHandler(fh)
    return logger


def log_turn(
    logger: logging.Logger,
    *,
    role: str,
    text: str,
    extra: dict[str, Any] | None = None,
) -> None:
    payload = {"role": role, "text": text, "ts": datetime.now(timezone.utc).isoformat()}
    if extra:
        payload["extra"] = extra
    logger.info(json.dumps(payload, ensure_ascii=False))
