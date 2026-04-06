"""Audio feedback module for ADAS warnings.

Generates audible beep warnings without blocking the main loop.
Each beep is executed in a background thread.

Backends (tried in order):
  1. simpleaudio  - if installed, plays a generated sine-wave tone
  2. os.system    - uses the system 'beep' command (Linux) or 'printf' bell

Install simpleaudio for reliable audio: pip install simpleaudio
"""

from __future__ import annotations

import logging
import threading
import time

_log = logging.getLogger(__name__)
_audio_warned = False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def play_warning_beep(duration_s: float = 0.15, pause_s: float = 0.25, repeats: int = 2) -> None:
    """Play a warning beep sequence in a background thread.

    Parameters
    ----------
    duration_s : float
        Duration of each beep tone in seconds.
    pause_s : float
        Silence between beeps.
    repeats : int
        Number of beeps in the sequence.
    """
    thread = threading.Thread(
        target=_beep_sequence,
        args=(660, duration_s, pause_s, repeats),
        daemon=True,
    )
    thread.start()


def play_brake_beep(duration_s: float = 0.10, pause_s: float = 0.08, repeats: int = 4) -> None:
    """Play a rapid brake-warning beep sequence in a background thread.

    Parameters
    ----------
    duration_s : float
        Duration of each beep tone in seconds.
    pause_s : float
        Silence between beeps.
    repeats : int
        Number of beeps in the sequence.
    """
    thread = threading.Thread(
        target=_beep_sequence,
        args=(880, duration_s, pause_s, repeats),
        daemon=True,
    )
    thread.start()


# ---------------------------------------------------------------------------
# Internal implementation
# ---------------------------------------------------------------------------

def _beep_sequence(freq_hz: int, duration_s: float, pause_s: float, repeats: int) -> None:
    """Generate and play a beep sequence (runs in daemon thread)."""
    for i in range(repeats):
        _single_beep(freq_hz, duration_s)
        if i < repeats - 1:
            time.sleep(pause_s)


def _single_beep(freq_hz: int, duration_s: float) -> None:
    """Play one beep. Tries simpleaudio, PulseAudio, then system fallback."""
    global _audio_warned
    if _try_simpleaudio(freq_hz, duration_s):
        return
    if _try_pulseaudio(freq_hz, duration_s):
        return
    if not _audio_warned:
        _audio_warned = True
        _log.warning(
            "Audio unavailable. For sound in Docker, start PulseAudio on "
            "the host:  scripts/start_pulseaudio.bat  (see README)."
        )
    _try_system_beep()


def _try_simpleaudio(freq_hz: int, duration_s: float) -> bool:
    """Try to play a sine-wave beep with simpleaudio. Returns True on success."""
    try:
        import simpleaudio as sa
        import numpy as np

        sample_rate = 44100
        t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)
        wave = (np.sin(2 * np.pi * freq_hz * t) * 32767 * 0.6).astype(np.int16)
        play_obj = sa.play_buffer(wave, 1, 2, sample_rate)
        play_obj.wait_done()
        return True
    except Exception:
        return False


def _try_pulseaudio(freq_hz: int, duration_s: float) -> bool:
    """Try to play a sine-wave beep via PulseAudio (paplay).

    Works inside Docker when the host PulseAudio socket is mounted.
    Returns True on success.
    """
    try:
        import subprocess
        import numpy as np

        sample_rate = 44100
        t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)
        wave = (np.sin(2 * np.pi * freq_hz * t) * 32767 * 0.6).astype(np.int16)

        proc = subprocess.Popen(
            ["paplay", "--raw", "--format=s16le", "--channels=1",
             f"--rate={sample_rate}"],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        proc.stdin.write(wave.tobytes())  # type: ignore[union-attr]
        proc.stdin.close()  # type: ignore[union-attr]
        proc.wait(timeout=duration_s + 2.0)
        return proc.returncode == 0
    except Exception:
        return False


def _try_system_beep() -> None:
    """Fallback: try OS-level beep (Linux/Mac terminal bell)."""
    import os
    import sys
    try:
        # Terminal bell
        sys.stdout.write("\a")
        sys.stdout.flush()
    except Exception:
        pass
    try:
        # Linux 'beep' command (requires the 'beep' package)
        os.system("beep -l 100 2>/dev/null")
    except Exception:
        pass
