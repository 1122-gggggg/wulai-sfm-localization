"""Latest-wins background writer for the operator's /tmp debug status files.

The live-localization and detector status files are rewritten at ~5 Hz. A plain
``Path.write_text`` on the Tk thread stalls a 33 ms tick whenever the disk or
the filesystem hiccups, so the bytes are written by one daemon thread instead.
Each path holds at most one pending text (a newer text replaces an unwritten
older one), an unchanged text is never rewritten, and the thread is a daemon:
there is nothing to join at shutdown.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any


class StatusFileWriter:
    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._pending: dict[Path, str] = {}
        self._written: dict[Path, str] = {}
        self._failures: dict[Path, Exception] = {}
        self._writing = False
        self._thread: threading.Thread | None = None

    def submit(self, path: Path, text: str) -> list[tuple[Path, Exception]]:
        """Queue ``text`` for ``path``; return write failures seen since the last call."""
        with self._cond:
            failures = list(self._failures.items())
            self._failures.clear()
            if self._pending.get(path, self._written.get(path)) == text:
                return failures
            self._pending[path] = text
            if self._thread is None:
                self._thread = threading.Thread(
                    target=self._run, name="status-file-writer", daemon=True
                )
                self._thread.start()
            self._cond.notify_all()
        return failures

    def flush(self, timeout: float = 5.0) -> bool:
        """Wait until every queued text has been written; True when idle."""
        with self._cond:
            return self._cond.wait_for(
                lambda: not self._pending and not self._writing, timeout
            )

    def _run(self) -> None:
        while True:
            with self._cond:
                self._cond.wait_for(lambda: bool(self._pending))
                path, text = self._pending.popitem()
                self._writing = True
            failure: Exception | None = None
            try:
                path.write_text(text, encoding="utf-8")
            except Exception as exc:  # Tier3: reported to the Tk thread, never raised here
                failure = exc
            with self._cond:
                if failure is None:
                    self._written[path] = text
                else:
                    self._written.pop(path, None)
                    self._failures[path] = failure
                self._writing = False
                self._cond.notify_all()


STATUS_FILE_WRITER = StatusFileWriter()


def publish_status_file(app: Any, path: Path, text: str) -> None:
    """Queue a status file write; report earlier write failures on this (Tk) thread."""
    for failed_path, exc in STATUS_FILE_WRITER.submit(path, text):
        app._record_diagnostic_failure(failed_path, exc)
