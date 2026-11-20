"""Per-second hype signals: audio loudness, chat activity and scene chaos."""

from __future__ import annotations

import csv
import json
import re
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np

from .media import require_binary

# --------------------------------------------------------------------------- audio

AUDIO_SR = 8000  # plenty for loudness, keeps memory tiny on 6h VODs


def audio_loudness(path: str | Path, duration: float) -> np.ndarray:
    """Loudness in dBFS for every second, streamed so long VODs don't eat RAM."""
    require_binary("ffmpeg")
    n_sec = max(1, int(np.ceil(duration)))
    db = np.full(n_sec, -90.0, dtype=np.float32)
    proc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", str(path), "-vn", "-ac", "1",
         "-ar", str(AUDIO_SR), "-f", "s16le", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    assert proc.stdout is not None
    chunk_bytes = AUDIO_SR * 2  # one second of int16
    sec = 0
    try:
        while sec < n_sec:
            raw = proc.stdout.read(chunk_bytes)
            if not raw:
                break
            x = np.frombuffer(raw[: len(raw) // 2 * 2], dtype=np.int16).astype(np.float32) / 32768.0
            if x.size:
                rms = float(np.sqrt(np.mean(x * x)) + 1e-9)
                db[sec] = 20.0 * np.log10(rms)
            sec += 1
    finally:
        proc.stdout.close()
        proc.wait()
    return db


def audio_spikes(db: np.ndarray, window: int = 30) -> np.ndarray:
    """How much louder each second is than the recent baseline (sudden screams,
    explosions, crashes) — robust to streamers who are just loud all the time."""
    out = np.zeros_like(db)
    for i in range(len(db)):
        lo = max(0, i - window)
        base = np.median(db[lo:i]) if i > lo else db[i]
        out[i] = db[i] - base
    return out


# --------------------------------------------------------------------------- chat

HYPE_TOKENS = {
    "kekw", "lul", "lulw", "omegalul", "lmao", "lmfao", "lol", "pog", "poggers",
    "pogchamp", "pogu", "clip", "clipit", "wtf", "holy", "nah", "noway", "w",
    "monkas", "pepelaugh", "icant", "😂", "💀", "🔥", "😭", "🤣", "ez", "gg",
}
_word_re = re.compile(r"[\w']+|[^\w\s]", re.UNICODE)


def _normalise_msg(text: str) -> list[str]:
    t = text.lower().replace("no way", "noway").replace("clip it", "clipit").replace("i cant", "icant")
    return _word_re.findall(t)


def load_chat(path: str | Path) -> list[tuple[float, str]]:
    """Load chat replay. Supports:
    - TwitchDownloader JSON  ({"comments": [{"content_offset_seconds", "message": {"body"}}]})
    - chat-downloader JSON   ([{"time_in_seconds", "message"}])
    - CSV / TXT              (seconds,message  — one per line)
    """
    p = Path(path)
    text = p.read_text(encoding="utf-8", errors="replace")
    rows: list[tuple[float, str]] = []
    if p.suffix.lower() == ".json":
        data = json.loads(text)
        if isinstance(data, dict) and "comments" in data:
            for c in data["comments"]:
                t = c.get("content_offset_seconds")
                body = (c.get("message") or {}).get("body", "") if isinstance(c.get("message"), dict) else str(c.get("message", ""))
                if t is not None:
                    rows.append((float(t), body))
        elif isinstance(data, list):
            for c in data:
                t = c.get("time_in_seconds", c.get("time", c.get("t")))
                if t is not None:
                    rows.append((float(t), str(c.get("message", c.get("text", "")))))
        else:
            raise ValueError(f"Unrecognised chat JSON format in {p}")
    else:
        for row in csv.reader(text.splitlines()):
            if not row:
                continue
            try:
                rows.append((float(row[0]), ",".join(row[1:]).strip()))
            except ValueError:
                continue  # header or junk line
    return rows


def chat_activity(messages: list[tuple[float, str]], n_sec: int, delay: float = 5.0):
    """Per-second weighted message rate. Chat reacts a few seconds *after* the
    moment (stream delay + typing), so timestamps are shifted back by `delay`.
    Returns (activity array, per-second Counter of hype tokens)."""
    act = np.zeros(n_sec, dtype=np.float32)
    tokens: list[Counter] = [Counter() for _ in range(n_sec)]
    for t, body in messages:
        i = int(t - delay)
        if i < 0 or i >= n_sec:
            continue
        words = _normalise_msg(body)
        hype = [w for w in words if w in HYPE_TOKENS]
        caps = sum(ch.isupper() for ch in body) > max(4, 0.6 * len(body.strip() or "x"))
        weight = 1.0 + 0.5 * min(len(hype), 3) + (0.5 if caps else 0.0)
        act[i] += weight
        tokens[i].update(hype)
    return act, tokens


# --------------------------------------------------------------------------- scenes


def scene_changes(path: str | Path, n_sec: int, threshold: float = 0.35) -> np.ndarray:
    """Count of hard visual changes per second (explosions, deaths, cuts, flashes).
    Video is downscaled + decimated to 10 fps so this stays fast on long VODs."""
    require_binary("ffmpeg")
    vf = f"fps=10,scale=160:-2,select='gt(scene\\,{threshold})',metadata=print:file=-"
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-an", "-vf", vf, "-f", "null", "-"],
        capture_output=True, text=True,
    )
    counts = np.zeros(n_sec, dtype=np.float32)
    for m in re.finditer(r"pts_time:([\d.]+)", proc.stdout):
        i = int(float(m.group(1)))
        if 0 <= i < n_sec:
            counts[i] += 1
    return counts
