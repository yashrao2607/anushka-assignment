"""Threaded frame reader for live sources.

PRD 9.5: "Decoding and inference run on **separate threads** with a bounded
queue (`maxsize=4`, drop-oldest on overflow). A single-threaded read-infer loop
is bottlenecked by decode and typically wastes 30-40% of achievable FPS. For
live feeds, dropping stale frames is correct behaviour -- a real-time system
must prefer the newest frame over a complete-but-lagging one."

The drop policy differs by source, and conflating the two is a real bug:

* **Live (webcam / RTSP):** drop the oldest frame on overflow. The camera keeps
  producing frames whether or not inference keeps up, so a queue that blocks
  builds unbounded latency until the display is showing a scene from ten
  seconds ago.
* **File:** block instead. A recorded clip has no real-time constraint, and
  silently dropping frames would change the tracking metrics -- the evaluation
  would score a different sequence than the one on disk.

NFR-8 also lives here: an RTSP stream that drops is reconnected automatically
rather than ending the run.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Iterator

import cv2
import numpy as np

from ..utils.logging import get_logger

log = get_logger("pipeline.reader")

RECONNECT_DELAY_S = 1.0
MAX_RECONNECT_ATTEMPTS = 5


@dataclass
class ReaderStats:
    frames_read: int = 0
    frames_dropped: int = 0
    reconnects: int = 0
    source_fps: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


class ThreadedFrameReader:
    """Decodes on a background thread into a bounded queue."""

    def __init__(self, source: str | int, queue_size: int = 4, is_live: bool | None = None):
        self.source = source
        self.is_live = self._infer_live(source) if is_live is None else is_live
        self.queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self.stats = ReaderStats()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._cap: cv2.VideoCapture | None = None

    @staticmethod
    def _infer_live(source: str | int) -> bool:
        if isinstance(source, int):
            return True
        s = str(source).lower()
        return s.isdigit() or s.startswith(("rtsp://", "http://", "https://", "udp://"))

    def _open(self) -> bool:
        src = int(self.source) if str(self.source).isdigit() else self.source
        self._cap = cv2.VideoCapture(src)
        if not self._cap.isOpened():
            return False
        self.stats.source_fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        return True

    def start(self) -> "ThreadedFrameReader":
        if not self._open():
            raise IOError(f"cannot open source: {self.source}")
        self._thread = threading.Thread(target=self._loop, daemon=True, name="frame-reader")
        self._thread.start()
        return self

    def _loop(self) -> None:
        attempts = 0
        while not self._stop.is_set():
            ok, frame = self._cap.read() if self._cap else (False, None)
            if not ok:
                if not self.is_live:
                    break                       # end of file: a normal ending
                attempts += 1
                if attempts > MAX_RECONNECT_ATTEMPTS:
                    log.error(f"stream {self.source} did not recover after "
                              f"{MAX_RECONNECT_ATTEMPTS} attempts",
                              extra={"event": "stream_dead"})
                    break
                log.warning(f"stream dropped; reconnecting ({attempts})",
                            extra={"event": "stream_reconnect", "attempt": attempts})
                self.stats.reconnects += 1
                if self._cap:
                    self._cap.release()
                time.sleep(RECONNECT_DELAY_S)
                self._open()
                continue

            attempts = 0
            self.stats.frames_read += 1
            if self.is_live:
                # Drop-oldest: prefer the newest frame over a complete backlog.
                while True:
                    try:
                        self.queue.put_nowait(frame)
                        break
                    except queue.Full:
                        try:
                            self.queue.get_nowait()
                            self.stats.frames_dropped += 1
                        except queue.Empty:
                            pass
            else:
                self.queue.put(frame)           # block: never alter a recording

        self.queue.put(None)                    # sentinel

    def frames(self) -> Iterator[np.ndarray]:
        while True:
            frame = self.queue.get()
            if frame is None:
                break
            yield frame

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "ThreadedFrameReader":
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()
