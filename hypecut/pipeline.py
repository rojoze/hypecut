"""End-to-end: VOD in -> ranked moments -> vertical captioned clips + manifest."""

from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from . import captions, render, scoring, signals
from .media import HypeCutError, download, fetch_chat, is_url, probe


@dataclass
class Options:
    top: int = 5
    min_len: float = 15.0
    max_len: float = 45.0
    lead_in: float = 10.0
    tail: float = 5.0
    min_score: float = 1.5
    chat: str | None = None
    fetch_chat: bool = False
    chat_delay: float = 5.0
    use_scenes: bool = True
    layout: str = "blur"
    facecam: tuple[int, int, int, int] | None = None
    zoom: float = 1.0
    captions: bool = True
    whisper_model: str = "small"
    language: str | None = None
    font: str = "Montserrat Black"
    fps: int = 30
    weights: scoring.Weights = field(default_factory=scoring.Weights)


def log(msg: str) -> None:
    print(f"[hypecut] {msg}", flush=True)


def analyze(src: str, out_dir: Path, opts: Options):
    """Score the VOD. Returns (local video path, info, moments, curve)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    chat_path = opts.chat
    if is_url(src):
        log(f"downloading {src}")
        if opts.fetch_chat and not chat_path:
            got = fetch_chat(src, out_dir / "chat.json")
            if got:
                chat_path = str(got)
                log("chat replay downloaded")
            else:
                log("chat-downloader not installed, skipping chat (pip install 'hypecut[chat]')")
        video = download(src, out_dir / "_source")
    else:
        video = Path(src)
        if not video.exists():
            raise HypeCutError(f"input not found: {src}")

    info = probe(video)
    dur = info["duration"]
    if dur <= 0:
        raise HypeCutError("could not read video duration")
    n = int(np.ceil(dur))
    log(f"source: {video.name}  {dur/60:.1f} min  {info['width']}x{info['height']}")

    t0 = time.time()
    if info["has_audio"]:
        loud = signals.audio_loudness(video, dur)
        spike = signals.audio_spikes(loud)
    else:
        loud = np.zeros(n, dtype=np.float32)
        spike = np.zeros(n, dtype=np.float32)
    log(f"audio analysed ({time.time()-t0:.1f}s)")

    chat_act, chat_tok = None, None
    if chat_path:
        msgs = signals.load_chat(chat_path)
        chat_act, chat_tok = signals.chat_activity(msgs, n, delay=opts.chat_delay)
        log(f"chat: {len(msgs)} messages")

    scene = None
    if opts.use_scenes:
        t0 = time.time()
        scene = signals.scene_changes(video, n)
        log(f"scenes analysed ({time.time()-t0:.1f}s)")

    curve, comps = scoring.hype_curve(loud[:n], spike[:n], chat_act, scene, opts.weights)
    moments = scoring.pick_moments(
        curve, comps, dur, top=opts.top, min_len=opts.min_len, max_len=opts.max_len,
        lead_in=opts.lead_in, tail=opts.tail, min_score=opts.min_score, chat_tokens=chat_tok,
    )

    with open(out_dir / "hype_curve.csv", "w", newline="") as f:
        w = csv.writer(f)
        keys = list(comps)
        w.writerow(["second", "hype", *keys])
        for i in range(n):
            w.writerow([i, round(float(curve[i]), 3), *(round(float(comps[k][i]), 3) for k in keys)])
    return video, info, moments, curve


def run(src: str, out_dir: str | Path, opts: Options, render_clips: bool = True) -> dict:
    out = Path(out_dir)
    video, info, moments, _ = analyze(src, out, opts)
    if not moments:
        log("no moments above the threshold — try --min-score 0.8 or add a chat log")

    results = []
    for rank, m in enumerate(moments, 1):
        stamp = f"{int(m.start//3600):02d}h{int(m.start%3600//60):02d}m{int(m.start%60):02d}s"
        name = f"clip_{rank:02d}_{stamp}"
        entry = {
            "rank": rank, "start": m.start, "end": m.end, "duration": round(m.duration, 2),
            "peak_at": m.peak, "score": m.score, "signals": m.signals,
            "chat_tokens": m.chat_tokens, "reason": m.reason(),
        }
        log(f"#{rank} {stamp}  {m.duration:.0f}s  score {m.score:.2f}  — {m.reason()}")

        if render_clips:
            ass = None
            if opts.captions and info["has_audio"]:
                wav = out / f"{name}.wav"
                render.extract_audio(video, m.start, m.duration, wav)
                try:
                    t0 = time.time()
                    words = captions.transcribe(wav, opts.whisper_model, opts.language)
                    log(f"  transcribed {len(words)} words ({time.time() - t0:.1f}s)")
                    if words:
                        ass = captions.build_ass(words, out / f"{name}.ass", font=opts.font)
                        entry["transcript"] = " ".join(w.text for w in words)
                except RuntimeError as e:
                    log(f"captions skipped: {e}")
                    opts.captions = False
                finally:
                    wav.unlink(missing_ok=True)
            mp4 = out / f"{name}.mp4"
            log(f"  rendering {mp4.name} ({opts.layout} layout)...")
            t0 = time.time()
            render.render_clip(
                video, m.start, m.duration, mp4, layout=opts.layout, facecam=opts.facecam,
                zoom=opts.zoom, ass_path=ass, has_audio=info["has_audio"], fps=opts.fps,
            )
            log(f"  rendered ({time.time() - t0:.1f}s)")
            render.thumbnail(mp4, out / f"{name}.jpg", at=min(m.peak - m.start, m.duration - 0.5))
            entry["file"] = mp4.name
            entry["thumbnail"] = f"{name}.jpg"
        results.append(entry)

    manifest = {
        "source": str(src), "duration": info["duration"],
        "options": {k: v for k, v in asdict(opts).items() if k != "weights"},
        "weights": asdict(opts.weights), "clips": results,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log(f"done → {out.resolve()}")
    return manifest
