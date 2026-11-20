"""Thin wrappers around ffmpeg/ffprobe and optional downloaders."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path


class HypeCutError(RuntimeError):
    pass


def require_binary(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise HypeCutError(f"'{name}' not found on PATH. Install ffmpeg (https://ffmpeg.org) first.")
    return path


def is_url(s: str) -> bool:
    return s.startswith(("http://", "https://"))


def probe(path: str | Path) -> dict:
    """Return duration (s), width, height, fps of the first video stream."""
    require_binary("ffprobe")
    out = subprocess.run(
        [
            "ffprobe", "-v", "error", "-print_format", "json",
            "-show_format", "-show_streams", str(path),
        ],
        capture_output=True, text=True,
    )
    if out.returncode != 0:
        raise HypeCutError(f"ffprobe failed on {path}: {out.stderr.strip()}")
    data = json.loads(out.stdout)
    video = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), None)
    has_audio = any(s.get("codec_type") == "audio" for s in data.get("streams", []))
    duration = float(data.get("format", {}).get("duration") or 0)
    info = {"duration": duration, "has_audio": has_audio, "width": 0, "height": 0, "fps": 30.0}
    if video:
        info["width"] = int(video.get("width", 0))
        info["height"] = int(video.get("height", 0))
        num, _, den = (video.get("avg_frame_rate") or "30/1").partition("/")
        try:
            info["fps"] = float(num) / float(den or 1) if float(den or 1) else 30.0
        except ValueError:
            pass
    return info


def download(url: str, work_dir: Path) -> Path:
    """Download a VOD with yt-dlp (Twitch, YouTube, Kick, ...)."""
    try:
        import yt_dlp  # type: ignore
    except ImportError as e:
        raise HypeCutError("URL input needs yt-dlp: pip install 'hypecut[download]'") from e

    work_dir.mkdir(parents=True, exist_ok=True)
    opts = {
        "format": "bv*[height<=1080]+ba/b[height<=1080]/b",
        "outtmpl": str(work_dir / "%(id)s.%(ext)s"),
        "merge_output_format": "mp4",
        "quiet": True,
        "noprogress": True,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        path = Path(ydl.prepare_filename(info))
    if not path.exists():
        mp4 = path.with_suffix(".mp4")
        if mp4.exists():
            path = mp4
    return path


def fetch_chat(url: str, out_path: Path) -> Path | None:
    """Grab the VOD chat replay with chat-downloader, if installed."""
    try:
        from chat_downloader import ChatDownloader  # type: ignore
    except ImportError:
        return None
    rows = []
    for msg in ChatDownloader().get_chat(url):
        t = msg.get("time_in_seconds")
        if t is None:
            continue
        rows.append({"time_in_seconds": t, "message": msg.get("message", "")})
    out_path.write_text(json.dumps(rows), encoding="utf-8")
    return out_path
