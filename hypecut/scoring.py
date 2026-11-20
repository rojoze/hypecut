"""Combine signals into a hype curve and pick the best non-overlapping moments."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Weights:
    loudness: float = 0.6
    spike: float = 1.0
    chat: float = 1.4
    scene: float = 0.5


@dataclass
class Moment:
    start: float
    end: float
    peak: float
    score: float
    signals: dict = field(default_factory=dict)
    chat_tokens: list = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.end - self.start

    def reason(self) -> str:
        parts = sorted(self.signals.items(), key=lambda kv: kv[1], reverse=True)
        labels = {"loudness": "loud", "spike": "audio spike", "chat": "chat spam", "scene": "visual chaos"}
        tags = [labels[k] for k, v in parts if v >= 1.0][:3] or [labels[parts[0][0]]] if parts else []
        s = " + ".join(tags)
        if self.chat_tokens and self.signals.get("chat", 0) >= 1.0:
            s += " (" + ", ".join(f"{t} x{n}" for t, n in self.chat_tokens[:3]) + ")"
        return s


def robust_z(x: np.ndarray) -> np.ndarray:
    """Median/MAD z-score — one loud outlier doesn't flatten everything else."""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x
    med = np.median(x)
    mad = np.median(np.abs(x - med)) * 1.4826
    if mad < 1e-6:
        mad = x.std() or 1.0
    return (x - med) / mad


def smooth(x: np.ndarray, width: int) -> np.ndarray:
    if width <= 1 or x.size == 0:
        return x.astype(np.float64)
    k = np.ones(width) / width
    return np.convolve(x, k, mode="same")


def hype_curve(
    loudness: np.ndarray,
    spike: np.ndarray,
    chat: np.ndarray | None,
    scene: np.ndarray | None,
    weights: Weights = Weights(),
    smooth_s: int = 5,
):
    """Return (combined curve, dict of per-signal positive z-curves)."""
    n = len(loudness)
    comps = {
        "loudness": np.clip(robust_z(loudness), 0, 6),
        "spike": np.clip(robust_z(spike), 0, 6),
    }
    if chat is not None and np.any(chat):
        comps["chat"] = np.clip(robust_z(smooth(chat[:n], 3)), 0, 8)
    if scene is not None and np.any(scene):
        comps["scene"] = np.clip(robust_z(smooth(scene[:n], 3)), 0, 4)

    total_w = sum(getattr(weights, k) for k in comps)
    combined = sum(getattr(weights, k) * v for k, v in comps.items()) / max(total_w, 1e-9)
    combined = smooth(combined, smooth_s)
    comps = {k: smooth(v, smooth_s) for k, v in comps.items()}
    return combined, comps


def pick_moments(
    curve: np.ndarray,
    comps: dict[str, np.ndarray],
    duration: float,
    top: int = 5,
    min_len: float = 15.0,
    max_len: float = 45.0,
    lead_in: float = 10.0,
    tail: float = 5.0,
    min_score: float = 1.5,
    rel_score: float = 0.25,
    gap: float = 10.0,
    chat_tokens: list[Counter] | None = None,
) -> list[Moment]:
    """Greedy peak picking with non-maximum suppression.

    For each peak we grow the window while the curve stays above half the peak,
    then add a lead-in (so viewers see the build-up) and a tail (the reaction),
    and clamp to [min_len, max_len]."""
    n = len(curve)
    if n == 0:
        return []
    taken = np.zeros(n, dtype=bool)
    order = np.argsort(curve)[::-1]
    moments: list[Moment] = []

    floor = max(min_score, rel_score * float(curve.max()))
    for p in order:
        if len(moments) >= top:
            break
        peak_val = float(curve[p])
        if peak_val < floor:
            break
        if taken[p]:
            continue

        thr = peak_val * 0.5
        lo = p
        while lo > 0 and curve[lo - 1] >= thr and (p - lo) < max_len:
            lo -= 1
        hi = p
        while hi < n - 1 and curve[hi + 1] >= thr and (hi - p) < max_len:
            hi += 1

        start = lo - lead_in
        end = hi + 1 + tail
        length = end - start
        if length < min_len:
            pad = (min_len - length)
            start -= pad * 0.6  # bias padding toward build-up
            end += pad * 0.4
        if end - start > max_len:
            # keep the peak, cut evenly around it but favour the lead-in
            start = max(start, p - max_len * 0.65)
            end = start + max_len
        start = max(0.0, start)
        end = min(duration, end)
        if end - start < min_len:  # near the edges
            if start == 0.0:
                end = min(duration, start + min_len)
            else:
                start = max(0.0, end - min_len)

        a = max(0, int(start - gap))
        b = min(n, int(np.ceil(end + gap)))
        if taken[int(start):int(np.ceil(end))].any():
            taken[p] = True
            continue
        taken[a:b] = True

        s_i, e_i = int(start), max(int(start) + 1, int(end))
        signals = {k: round(float(v[s_i:e_i].max()), 2) for k, v in comps.items()}
        toks: list = []
        if chat_tokens:
            c = Counter()
            for i in range(s_i, min(e_i, len(chat_tokens))):
                c.update(chat_tokens[i])
            toks = c.most_common(5)
        moments.append(Moment(
            start=round(float(start), 2), end=round(float(end), 2), peak=float(p),
            score=round(peak_val, 3), signals=signals, chat_tokens=toks,
        ))
    return moments
