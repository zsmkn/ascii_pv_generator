"""Vocal separation with demucs (htdemucs), cached per song:

  cache/vocals_rms.npy   vocal level at 48 Hz (lyric sync, vocal-reactive visuals)
  cache/vocals_16k.npy   the vocal track itself, mono 16 kHz float16 (lyric text alignment, analysis/align.py)
  cache/stems_48.npz     the other stems at 48 Hz (M10): drums / bass / other RMS, and the drum stem split into
                         a low band (kick) and a high band (snare, hats) for drum hits
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ascii_pv.analysis.audio import HOP, RATE, SR
from ascii_pv.analysis.ffmpeg import decode

MODEL = "htdemucs"
WAV_SR = 16000


KICK_BAND = (30, 150)
SNARE_BAND = (1500, 10000)


def _rms48(x: np.ndarray) -> np.ndarray:
    n = len(x) // HOP
    return np.sqrt(np.mean(x[: n * HOP].reshape(n, HOP) ** 2, axis=1)).astype(np.float32)


def _band48(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """RMS inside [lo, hi) Hz, one value per 1/48 s (same framing as analysis/audio.py)."""
    import librosa

    from ascii_pv.analysis.audio import N_FFT

    S = np.abs(librosa.stft(x, n_fft=N_FFT, hop_length=HOP, center=True)) ** 2
    f = librosa.fft_frequencies(sr=SR, n_fft=N_FFT)
    return np.sqrt(S[(f >= lo) & (f < hi)].mean(axis=0))[: len(x) // HOP].astype(np.float32)


def _separate(audio: str | Path, cache: Path, device: str, log) -> None:
    """Run demucs once and write the missing caches (existing ones are kept, so analyses and the alignment
    inputs derived from them stay stable)."""
    import librosa
    import torch
    from demucs.apply import apply_model
    from demucs.pretrained import get_model

    model = get_model(MODEL)
    model.to(device).eval()
    sr = model.samplerate
    log(f"  demucs {MODEL} on {device} ...")
    wav = torch.from_numpy(decode(audio, sr, mono=False).T.copy())        # (2, n)
    ref = wav.mean(0)
    mean, std = ref.mean(), ref.std() + 1e-8
    with torch.no_grad():
        src = apply_model(model, ((wav - mean) / std)[None], device=device, split=True, overlap=0.25,
                          progress=False)[0]
    stems = {name: (src[k] * std + mean).mean(0).cpu().numpy().astype(np.float32)
             for k, name in enumerate(model.sources)}                    # drums, bass, other, vocals
    vocals = stems["vocals"]
    del src
    if device == "cuda":
        torch.cuda.empty_cache()
    cache.mkdir(parents=True, exist_ok=True)
    rms_out = cache / "vocals_rms.npy"
    if not rms_out.exists():
        np.save(rms_out, _rms48(librosa.resample(vocals, orig_sr=sr, target_sr=SR)))
    if not (cache / "vocals_16k.npy").exists():
        np.save(cache / "vocals_16k.npy", librosa.resample(vocals, orig_sr=sr, target_sr=WAV_SR).astype(np.float16))
    if not (cache / "stems_48.npz").exists():
        at = {k: librosa.resample(stems[k], orig_sr=sr, target_sr=SR) for k in ("drums", "bass", "other")}
        np.savez(cache / "stems_48.npz", drums=_rms48(at["drums"]), bass=_rms48(at["bass"]),
                 other=_rms48(at["other"]), kick=_band48(at["drums"], *KICK_BAND),
                 snare=_band48(at["drums"], *SNARE_BAND))


def vocal_rms(audio: str | Path, cache: Path, device: str = "cuda", log=print) -> np.ndarray:
    """Raw RMS of the separated vocals, one value per 1/48 s."""
    out = cache / "vocals_rms.npy"
    if not out.exists():
        _separate(audio, cache, device, log)
    return np.load(out)


def vocal_wave(audio: str | Path, cache: Path, device: str = "cuda", log=print) -> np.ndarray:
    """The separated vocal track, mono float32 at WAV_SR (older caches only have the RMS: separate again)."""
    out = cache / "vocals_16k.npy"
    if not out.exists():
        _separate(audio, cache, device, log)
    return np.load(out).astype(np.float32)


def stem_raw(audio: str | Path, cache: Path, device: str = "cuda", log=print) -> dict[str, np.ndarray]:
    """Raw 48 Hz curves of the non-vocal stems (older caches: separate again once)."""
    out = cache / "stems_48.npz"
    if not out.exists():
        _separate(audio, cache, device, log)
    with np.load(out) as z:
        return {k: z[k] for k in z.files}


def stem_features(raw: dict[str, np.ndarray], n: int) -> dict:
    """-> analysis.json "stems": 0..1 curves of length n (the feature table's length).
    drums / bass / other: level; kick / snare: hits on the drum stem (an envelope that jumps and decays)."""
    from ascii_pv.analysis.audio import follow, norm, onset_of, r3

    def fit(v):
        v = np.asarray(v, float)[:n]
        return np.pad(v, (0, n - len(v)))

    out = {"rate": RATE}
    for k in ("drums", "bass", "other"):
        out[k] = r3(follow(norm(fit(raw[k])), 0.6, 0.15))
    out["kick"] = r3(follow(onset_of(fit(raw["kick"])), 0.9, 0.2))
    out["snare"] = r3(follow(onset_of(fit(raw["snare"])), 0.9, 0.3))
    return out


def vocal_features(rms: np.ndarray) -> tuple[dict, list[float]]:
    """-> ({"rate", "rms" (0..1)}, vocal onset times) for analysis.json."""
    from ascii_pv.analysis.audio import follow, norm
    from ascii_pv.analysis.sync import onset_envelope

    level = follow(norm(rms.astype(float)), 0.6, 0.2)
    env = onset_envelope(rms, RATE)
    peaks = np.nonzero((env[1:-1] > 1.5) & (env[1:-1] >= env[:-2]) & (env[1:-1] > env[2:]))[0] + 1
    return {"rate": RATE, "rms": [round(float(x), 3) for x in level]}, [round(p / RATE, 3) for p in peaks]
