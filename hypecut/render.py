"""Cut + reframe to 9:16 + burn captions + loudness-normalise, in one ffmpeg pass."""

from __future__ import annotations

import subprocess
from pathlib import Path

from .media import HypeCutError, require_binary

W, H = 1080, 1920


def layout_filter(layout: str, facecam: tuple[int, int, int, int] | None = None, zoom: float = 1.0) -> str:
    """Return a filter_complex graph that reads [0:v] and writes [v] at 1080x1920."""
    if layout == "blur":
        fg_w = int(round(W * zoom / 2) * 2)
        crop = f",crop={W}:ih" if fg_w > W else ""
        return (
            f"[0:v]split=2[bg][fg];"
            f"[bg]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},"
            f"boxblur=25:2,eq=brightness=-0.10[bgb];"
            f"[fg]scale={fg_w}:-2{crop}[fgs];"
            f"[bgb][fgs]overlay=(W-w)/2:(H-h)/2,setsar=1[v]"
        )
    if layout == "crop":
        return f"[0:v]scale=-2:{H},crop={W}:{H},setsar=1[v]"
    if layout == "facecam":
        if not facecam:
            raise HypeCutError("facecam layout needs --facecam x:y:w:h (in source pixels)")
        x, y, w, h = facecam
        cam_h, game_h = 640, H - 640
        return (
            f"[0:v]split=2[a][b];"
            f"[a]crop={w}:{h}:{x}:{y},scale={W}:{cam_h}:force_original_aspect_ratio=increase,crop={W}:{cam_h}[cam];"
            f"[b]scale=-2:{game_h},crop={W}:{game_h}[game];"
            f"[cam][game]vstack=inputs=2,setsar=1[v]"
        )
    raise HypeCutError(f"Unknown layout '{layout}' (use blur, crop or facecam)")


def extract_audio(src: str | Path, start: float, duration: float, out_wav: Path) -> Path:
    require_binary("ffmpeg")
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-ss", f"{start:.3f}", "-i", str(src), "-t", f"{duration:.3f}",
         "-vn", "-ac", "1", "-ar", "16000", str(out_wav)],
        check=True,
    )
    return out_wav


def render_clip(
    src: str | Path,
    start: float,
    duration: float,
    out_path: Path,
    layout: str = "blur",
    facecam: tuple[int, int, int, int] | None = None,
    zoom: float = 1.0,
    ass_path: Path | None = None,
    has_audio: bool = True,
    crf: int = 20,
    preset: str = "veryfast",
    fps: int = 30,
) -> Path:
    require_binary("ffmpeg")
    graph = layout_filter(layout, facecam, zoom)
    if ass_path is not None:
        # run from the output dir and use a bare filename: no path escaping pain
        graph = graph[: -len("[v]")] + f",subtitles={ass_path.name}[v]"
    cmd = [
        "ffmpeg", "-v", "error", "-y",
        "-ss", f"{start:.3f}", "-i", str(Path(src).resolve()), "-t", f"{duration:.3f}",
        "-filter_complex", graph, "-map", "[v]",
    ]
    if has_audio:
        cmd += ["-map", "0:a:0?", "-af", "loudnorm=I=-14:TP=-1.5:LRA=11", "-c:a", "aac", "-b:a", "160k", "-ar", "48000"]
    cmd += [
        "-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p",
        "-r", str(fps), "-movflags", "+faststart", out_path.name,
    ]
    proc = subprocess.run(cmd, cwd=out_path.parent, capture_output=True, text=True)
    if proc.returncode != 0:
        raise HypeCutError(f"ffmpeg render failed:\n{proc.stderr[-2000:]}")
    return out_path


def thumbnail(clip: Path, out_jpg: Path, at: float = 1.0) -> Path:
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-ss", f"{at:.2f}", "-i", str(clip), "-frames:v", "1",
         "-q:v", "3", str(out_jpg)],
        check=True,
    )
    return out_jpg
