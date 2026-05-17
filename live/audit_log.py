"""JSONL audit logger for paper trading (DR v3.0.23).

Writes structured events to daily JSONL files. One file per UTC day per
session. Each line is a self-contained JSON object with mandatory `ts`
(ISO 8601 UTC) and `event` (string) fields plus arbitrary payload.

Event types (informal vocabulary, free to extend):
  - session_start, session_end
  - tick_batch_fetched
  - bars_built
  - features_built
  - l0_prediction
  - trade_decision (with skip_reason if not traded)
  - trade_entry
  - trade_exit
  - daily_summary
  - halt (manual or auto)
  - error

Usage:
    log = AuditLog("/path/to/dir", session_id="btc_thr015_v1_20260517")
    log.write("session_start", config={...})
    log.write("trade_entry", trade_id=42, direction=1, entry_price=70123.5)

Files rotate at UTC midnight automatically. Always appends, never overwrites.
"""
from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LOG = logging.getLogger("live.audit_log")


class AuditLog:
    def __init__(self, log_dir: str | Path, session_id: str):
        self.log_dir = Path(log_dir)
        self.session_id = session_id
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._current_day: str | None = None
        self._current_handle = None

    def _path_for_day(self, day_str: str) -> Path:
        return self.log_dir / f"{self.session_id}_{day_str}.jsonl"

    def _ensure_handle(self) -> None:
        """Open the file for today's UTC date, rotating if day changed."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today != self._current_day:
            # Day rolled over (or first write)
            if self._current_handle is not None:
                try:
                    self._current_handle.close()
                except Exception:
                    pass
            self._current_day = today
            path = self._path_for_day(today)
            self._current_handle = open(path, "a", buffering=1)  # line-buffered

    def write(self, event: str, **payload: Any) -> None:
        """Append a JSONL line. Always includes ts (UTC) and session_id."""
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "session_id": self.session_id,
            "event": event,
            **payload,
        }
        line = json.dumps(record, default=str)
        with self._lock:
            self._ensure_handle()
            self._current_handle.write(line + "\n")

    def close(self) -> None:
        with self._lock:
            if self._current_handle is not None:
                try:
                    self._current_handle.close()
                except Exception:
                    pass
                self._current_handle = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
