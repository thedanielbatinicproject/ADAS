"""Thread-safe context analysis service.

Runs context analysis periodically in a background thread, caching the
latest ContextState for the main pipeline loop to read without blocking.

Usage
-----
    service = ContextService(config=DEFAULT_CONFIG, context_interval=5)
    service.start()

    # In main loop (any thread):
    service.push_frame(frame, frame_idx, timestamp_s, fps)
    ctx_state = service.get_state()   # always non-blocking

    service.stop()

The service only analyses frames whose index is a multiple of
context_interval.  Between analysis calls the main loop reads the last
cached ContextState via get_state(), which never blocks.
"""

from __future__ import annotations

import threading
from typing import Any, Optional

from .defaults import ContextConfig, DEFAULT_CONFIG
from .router import route
from .types import ContextState


class ContextService:
    """Thread-safe, background context analysis service.

    Parameters
    ----------
    config : ContextConfig, optional
        Configuration forwarded to route(). Defaults to DEFAULT_CONFIG.
    context_interval : int
        Analyse context every N frames pushed via push_frame().
        push_frame calls where (frame_idx % context_interval != 0) are
        silently ignored.
    """

    def __init__(
        self,
        config: Optional[ContextConfig] = None,
        context_interval: int = 5,
    ) -> None:
        self._config = config or DEFAULT_CONFIG
        self._context_interval = max(1, context_interval)

        # Protects _state
        self._state_lock = threading.Lock()
        self._state: Optional[ContextState] = None

        # Pending frame slot: at most one frame waits for the worker.
        # We keep only the latest; older unprocessed frames are dropped.
        self._pending_lock = threading.Lock()
        self._pending: Optional[tuple] = None   # (frame, frame_idx, ts, fps)
        self._pending_event = threading.Event()

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------ API

    def start(self) -> None:
        """Start the background analysis thread (idempotent)."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._worker,
            daemon=True,
            name="ContextService",
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the background thread and block until it exits."""
        self._stop_event.set()
        self._pending_event.set()   # unblock worker if it is waiting
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None

    def push_frame(
        self,
        frame: Any,
        frame_idx: int,
        timestamp_s: float = 0.0,
        fps: Optional[float] = None,
    ) -> None:
        """Offer a frame to the background analyser.

        Frames whose index is not a multiple of context_interval are
        dropped immediately.  Otherwise the frame is queued (overwriting
        any previously queued but not yet processed frame).  This call
        never blocks.
        """
        if frame_idx % self._context_interval != 0:
            return
        with self._pending_lock:
            self._pending = (frame, frame_idx, timestamp_s, fps)
        self._pending_event.set()

    def get_state(self) -> Optional[ContextState]:
        """Return the latest cached ContextState (non-blocking).

        Returns None until the first analysis has completed.
        """
        with self._state_lock:
            return self._state

    def set_context_interval(self, interval: int) -> None:
        """Dynamically update context_interval."""
        self._context_interval = max(1, interval)

    def get_context_interval(self) -> int:
        return self._context_interval

    def update_config(self, config: ContextConfig) -> None:
        """Replace the ContextConfig used for future route() calls."""
        self._config = config

    # ------------------------------------------------------------ internals

    def _worker(self) -> None:
        prev_state: Optional[ContextState] = None
        while not self._stop_event.is_set():
            triggered = self._pending_event.wait(timeout=0.5)
            self._pending_event.clear()
            if self._stop_event.is_set():
                break
            if not triggered:
                continue

            with self._pending_lock:
                item = self._pending
                self._pending = None

            if item is None:
                continue

            frame, frame_idx, timestamp_s, fps = item
            try:
                new_state = route(
                    frame,
                    timestamp_s=float(timestamp_s),
                    fps=fps,
                    prev_state=prev_state,
                    config=self._config,
                )
                prev_state = new_state
                with self._state_lock:
                    self._state = new_state
            except Exception:
                # Never crash the background thread; silently keep last state.
                pass
