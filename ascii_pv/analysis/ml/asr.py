"""Models for plain-text lyric alignment (analysis/align.py), run on the separated vocals (stems.vocal_wave).

  ctc_logp       torchaudio MMS_FA (wav2vec2, ~1.2 GB): per-frame character log-probs   -> cache/ctc_logp.npy
  whisper_words  Whisper large-v3-turbo (~1.6 GB): transcription with word timestamps -> cache/whisper_words.json
                 (other languages: whisper_words.<lang>.json; Chinese comes back one character per "word")

Both depend only on the audio, so they are cached per song and reused when the lyric text changes.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ascii_pv.analysis.ml.stems import WAV_SR

WHISPER = "openai/whisper-large-v3-turbo"
CHUNK, OVERLAP = 30.0, 2.0          # wav2vec2 attention is quadratic: run it on 30 s pieces with context


def ctc_vocab() -> dict[str, int]:
    import torchaudio

    return dict(torchaudio.pipelines.MMS_FA.get_dict(star=None))


def ctc_logp(wave: np.ndarray, cache: Path, device: str = "cuda", log=print) -> tuple[np.ndarray, float]:
    """-> (log-probs (frames, vocab), frames per second)."""
    out = cache / "ctc_logp.npy"
    if out.exists():
        lp = np.load(out).astype(np.float32)
        return lp, len(lp) / (len(wave) / WAV_SR)
    import torch
    import torchaudio

    log(f"  wav2vec2 aligner (MMS_FA) on {device} ...")
    model = torchaudio.pipelines.MMS_FA.get_model(with_star=False).to(device).eval()
    wav = torch.from_numpy(wave.astype(np.float32))
    step, ov = int(CHUNK * WAV_SR), int(OVERLAP * WAV_SR)
    parts = []
    with torch.inference_mode():
        for s in range(0, len(wav), step):
            a, z = max(0, s - ov), min(len(wav), s + step + ov)
            e, _ = model(wav[a:z][None].to(device))
            e = e[0].float().cpu()
            per = e.shape[0] / (z - a)
            parts.append(e[int(round((s - a) * per)): int(round((min(s + step, len(wav)) - a) * per))])
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    lp = torch.cat(parts).numpy()
    np.save(out, lp.astype(np.float16))
    return lp, len(lp) / (len(wave) / WAV_SR)


def whisper_words(wave: np.ndarray, cache: Path, device: str = "cuda", log=print,
                  language: str = "en") -> list[tuple[float, float, str]]:
    """-> [(start, end, word)] in `language` ("en", "zh", ...)."""
    out = cache / ("whisper_words.json" if language == "en" else f"whisper_words.{language}.json")
    if out.exists():
        return [tuple(w) for w in json.loads(out.read_text(encoding="utf-8"))]
    import torch
    from transformers import pipeline

    log(f"  whisper ({WHISPER}, {language}) on {device} ...")
    asr = pipeline("automatic-speech-recognition", model=WHISPER, device=device,
                   dtype=torch.float16 if device == "cuda" else torch.float32)
    # Sequential long-form decoding with temperature fallback: chunked decoding can loop on a held note
    # ("Aaaaaa..." for half a minute); a too-compressible window is decoded again at a higher temperature.
    # batch_size 1: batched fp16 attention hits a cuBLAS error on this torch / RTX 50xx combination.
    r = asr({"raw": wave.astype(np.float32), "sampling_rate": WAV_SR}, return_timestamps="word", batch_size=1,
            generate_kwargs={"language": language, "task": "transcribe", "condition_on_prev_tokens": False,
                             "compression_ratio_threshold": 1.35, "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
                             "logprob_threshold": -1.0, "no_speech_threshold": 0.6})
    del asr
    if device == "cuda":
        torch.cuda.empty_cache()
    words = []
    for c in r["chunks"]:
        s, e = c["timestamp"]
        if s is None:
            continue
        e = s + 0.3 if e is None else e
        words.append((round(float(s), 3), round(float(e), 3), c["text"].strip()))
    out.write_text(json.dumps(words), encoding="utf-8")
    return words
