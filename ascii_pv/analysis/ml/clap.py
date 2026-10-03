"""Zero-shot genre / mood tags with CLAP (LAION, via transformers).

Five 10 s windows across the song are embedded and compared with text prompts; probabilities are averaged over
the windows (softmax per label set). The result nudges the rule-based style: extra template tags and a valence
correction.
"""

from __future__ import annotations

import numpy as np

MODEL = "laion/clap-htsat-unfused"
CLAP_SR = 48000
GENRES = ["electronic dance music", "hip hop", "rock", "metal", "pop", "r&b", "jazz", "classical orchestral",
          "acoustic folk", "lo-fi", "ambient", "ballad"]
MOODS = ["happy", "sad", "dark", "energetic", "calm", "romantic", "aggressive", "dreamy"]

# CLAP label -> tags the templates understand
TAGS = {"electronic dance music": ["energetic", "busy", "percussive"], "hip hop": ["percussive"],
        "rock": ["energetic"], "metal": ["energetic", "dark"], "ambient": ["calm", "sparse"],
        "ballad": ["calm"], "lo-fi": ["calm", "sparse"], "classical orchestral": ["calm"],
        "happy": ["bright"], "sad": ["dark"], "dark": ["dark"], "energetic": ["energetic"], "calm": ["calm"],
        "romantic": ["bright"], "aggressive": ["energetic", "dark"], "dreamy": ["calm"]}


def _windows(y: np.ndarray, sr: int, n: int = 5, secs: float = 10.0) -> list[np.ndarray]:
    L = int(secs * sr)
    if len(y) <= L:
        return [y]
    starts = np.linspace(len(y) * 0.1, len(y) * 0.9 - L, n).astype(int)
    return [y[s:s + L] for s in starts]


def tags(y: np.ndarray, sr: int, device: str = "cuda") -> dict:
    import librosa
    import torch
    from transformers import ClapModel, ClapProcessor

    y48 = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=CLAP_SR)
    clips = _windows(y48, CLAP_SR)
    model = ClapModel.from_pretrained(MODEL).to(device).eval()
    proc = ClapProcessor.from_pretrained(MODEL)
    result = {}
    for name, labels, tmpl in (("genre", GENRES, "This is a {} song."), ("mood", MOODS, "This song sounds {}.")):
        texts = [tmpl.format(x) for x in labels]
        try:
            inputs = proc(text=texts, audio=clips, sampling_rate=CLAP_SR, return_tensors="pt", padding=True)
        except TypeError:   # older transformers
            inputs = proc(text=texts, audios=clips, sampling_rate=CLAP_SR, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = model(**{k: v.to(device) for k, v in inputs.items()}).logits_per_audio   # (clips, labels)
        p = logits.softmax(dim=-1).mean(0).cpu().numpy()
        order = np.argsort(-p)
        result[name] = [[labels[i], round(float(p[i]), 3)] for i in order]
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return result


def merge_into_style(style: dict, clap: dict) -> dict:
    """Add CLAP tags (label prob >= 0.25) and pull valence 30% toward the CLAP mood reading."""
    style = dict(style)
    style["clap"] = clap
    extra = []
    for name in ("genre", "mood"):
        for label, p in clap.get(name, []):
            if p >= 0.25:
                extra += TAGS.get(label, [])
    style["tags"] = list(dict.fromkeys(style.get("tags", []) + extra))
    mood = dict((lab, p) for lab, p in clap.get("mood", []))
    v_clap = 0.5 + 0.5 * (mood.get("happy", 0) + 0.5 * mood.get("romantic", 0) + 0.3 * mood.get("energetic", 0)
                          - mood.get("sad", 0) - 0.7 * mood.get("dark", 0) - 0.3 * mood.get("aggressive", 0))
    style["valence"] = round(0.7 * style.get("valence", 0.5) + 0.3 * float(np.clip(v_clap, 0, 1)), 3)
    return style
