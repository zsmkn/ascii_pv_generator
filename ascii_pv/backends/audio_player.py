"""Audio playback whose position is the film's clock.

The song is decoded once (ffmpeg -> float32 stereo) and played from a sounddevice callback. The clock is the
sample position of what is coming out of the speaker right now: the callback records which sample starts each
block and when that block reaches the DAC, and `time()` interpolates from there with the stream clock. So
pausing, seeking and slow frames never make picture and sound drift apart.

SilentClock has the same interface without a sound device (for --mute, tests, and machines without output).
"""

from __future__ import annotations

import threading
import time as _time

import numpy as np

from ascii_pv.analysis.ffmpeg import decode


class SilentClock:
    def __init__(self, duration: float):
        self.duration = duration
        self._pos = 0.0
        self._t0: float | None = None
        self.volume = 1.0

    @property
    def playing(self) -> bool:
        return self._t0 is not None

    def time(self) -> float:
        if self._t0 is None:
            return self._pos
        return min(self.duration, self._pos + _time.perf_counter() - self._t0)

    def play(self) -> None:
        if self._t0 is None:
            self._t0 = _time.perf_counter()

    def pause(self) -> None:
        self._pos = self.time()
        self._t0 = None

    def seek(self, t: float) -> None:
        was = self.playing
        self._pos = max(0.0, min(self.duration, t))
        self._t0 = _time.perf_counter() if was else None

    def set_volume(self, v: float) -> None:
        self.volume = max(0.0, min(1.0, v))

    def close(self) -> None:
        pass


class AudioPlayer:
    def __init__(self, path: str, sr: int | None = None, blocksize: int = 1024):
        import sounddevice as sd

        self._sd = sd
        dev = sd.query_devices(kind="output")
        self.sr = int(sr or dev["default_samplerate"])
        self.samples = np.ascontiguousarray(decode(path, self.sr, mono=False))
        self.duration = len(self.samples) / self.sr
        self.volume = 0.8
        self._lock = threading.Lock()
        self._pos = 0                      # next sample to hand to the device
        self._playing = False
        self._anchor: tuple[int, float] | None = None   # (sample at block start, stream time it is heard)
        self.stream = sd.OutputStream(samplerate=self.sr, channels=2, dtype="float32", blocksize=blocksize,
                                      callback=self._callback, latency="low")
        self.stream.start()

    def _callback(self, out, frames, time_info, status) -> None:
        with self._lock:
            if not self._playing:
                out.fill(0)
                self._anchor = None
                return
            a = self._pos
            b = min(len(self.samples), a + frames)
            n = b - a
            out[:n] = self.samples[a:b] * self.volume
            out[n:] = 0
            self._anchor = (a, time_info.outputBufferDacTime)
            self._pos = b
            if b >= len(self.samples):
                self._playing = False

    @property
    def playing(self) -> bool:
        return self._playing

    def time(self) -> float:
        with self._lock:
            if self._playing and self._anchor is not None:
                a, dac = self._anchor
                t = a / self.sr + (self.stream.time - dac)
                return max(0.0, min(self.duration, t))
            return self._pos / self.sr

    def play(self) -> None:
        with self._lock:
            if self._pos >= len(self.samples):
                self._pos = 0
            self._playing = True

    def pause(self) -> None:
        t = self.time()
        with self._lock:
            self._playing = False
            self._pos = int(t * self.sr)
            self._anchor = None

    def seek(self, t: float) -> None:
        with self._lock:
            self._pos = int(max(0.0, min(self.duration, t)) * self.sr)
            self._anchor = None

    def set_volume(self, v: float) -> None:
        self.volume = max(0.0, min(1.0, v))

    def close(self) -> None:
        try:
            self.stream.stop()
            self.stream.close()
        except Exception:
            pass


def open_clock(path: str | None, duration: float, mute: bool = False):
    """AudioPlayer when possible, SilentClock otherwise (with the reason)."""
    if mute or not path:
        return SilentClock(duration), "muted"
    try:
        return AudioPlayer(path), ""
    except Exception as e:  # no device, no ffmpeg, ...
        return SilentClock(duration), f"audio unavailable ({e}); playing silently"
