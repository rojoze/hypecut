"""Word-level transcription + TikTok-style karaoke captions (ASS)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Word:
    start: float
    end: float
    text: str


_model_cache: dict = {}


def transcribe(audio_path: str | Path, model_size: str = "small", language: str | None = None) -> list[Word]:
    """Transcribe with faster-whisper (runs locally, CPU is fine for short clips)."""
    try:
        from faster_whisper import WhisperModel  # type: ignore
    except ImportError as e:
        raise RuntimeError("Captions need faster-whisper: pip install 'hypecut[captions]'") from e
    if model_size not in _model_cache:
        _model_cache[model_size] = WhisperModel(model_size, device="auto", compute_type="int8")
    model = _model_cache[model_size]
    segments, _ = model.transcribe(str(audio_path), word_timestamps=True, language=language, vad_filter=True)
    words: list[Word] = []
    for seg in segments:
        for w in seg.words or []:
            t = w.word.strip()
            if t:
                words.append(Word(float(w.start), float(w.end), t))
    return words


def chunk_words(words: list[Word], max_words: int = 3, max_gap: float = 0.6, max_dur: float = 1.6) -> list[list[Word]]:
    """Group words into short punchy lines (2-3 words, the TikTok look)."""
    chunks: list[list[Word]] = []
    cur: list[Word] = []
    for w in words:
        if cur and (
            len(cur) >= max_words
            or w.start - cur[-1].end > max_gap
            or w.end - cur[0].start > max_dur
            or cur[-1].text[-1:] in ".!?"
        ):
            chunks.append(cur)
            cur = []
        cur.append(w)
    if cur:
        chunks.append(cur)
    return chunks


def _ts(t: float) -> str:
    t = max(0.0, t)
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = t % 60
    return f"{h}:{m:02d}:{s:05.2f}"


def _esc(text: str) -> str:
    return text.replace("\\", "/").replace("{", "(").replace("}", ")").replace("\n", " ")


ASS_HEADER = """[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Hype,{font},{size},&H0000E5FF,&H00FFFFFF,&H00000000,&H96000000,-1,0,0,0,100,100,0,0,1,7,3,2,60,60,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def build_ass(
    words: list[Word],
    out_path: str | Path,
    font: str = "Montserrat Black",
    size: int = 92,
    margin_v: int = 560,
    uppercase: bool = True,
) -> Path:
    """Write an ASS file where each word lights up yellow as it's spoken."""
    lines = [ASS_HEADER.format(font=font, size=size, margin_v=margin_v)]
    for chunk in chunk_words(words):
        start = chunk[0].start
        end = max(chunk[-1].end, start + 0.35)
        parts = []
        cursor = start
        for w in chunk:
            # karaoke timing in centiseconds: gap before word is folded into it
            k = max(1, int(round((w.end - cursor) * 100)))
            cursor = w.end
            txt = _esc(w.text.upper() if uppercase else w.text)
            parts.append(f"{{\\k{k}}}{txt}")
        pop = "{\\fscx85\\fscy85\\t(0,90,\\fscx100\\fscy100)}"  # little pop-in
        lines.append(f"Dialogue: 0,{_ts(start)},{_ts(end)},Hype,,0,0,0,,{pop}{' '.join(parts)}\n")
    p = Path(out_path)
    p.write_text("".join(lines), encoding="utf-8")
    return p
