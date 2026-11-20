# HypeCut ✂️🔥

**Drop in a 4-hour stream VOD and get back the 5 best moments as vertical, captioned clips ready for TikTok, Reels and Shorts.**

HypeCut scores every second of a VOD for hype, picks the best moments, and renders each one as a 1080×1920 clip with word-by-word captions. Built for GTA VI streams first, but it works on any gameplay VOD.

```
hypecut run https://www.twitch.tv/videos/123456789 --fetch-chat -n 5
```

```
[hypecut] #1 01h42m07s  26s  score 6.93  — chat spam + loud + audio spike (omegalul x95, clipit x85, kekw x76)
[hypecut] #2 00h18m49s  21s  score 2.56  — loud + audio spike
...
```

## How it finds moments

| Signal | What it catches | Why |
|---|---|---|
| **Audio loudness** | Screaming, explosions, music drops | Per-second dBFS, robust z-scored |
| **Audio spikes** | *Sudden* jumps vs the last 30s | Catches the moment even if the streamer is loud all the time |
| **Chat activity** | Message flood + hype tokens (KEKW, OMEGALUL, CLIP IT, 💀, NO WAY…) | Shifted back by `--chat-delay` because chat reacts *after* the action |
| **Scene chaos** | Hard visual cuts, flashes, deaths, crashes | ffmpeg scene detection at 10 fps / 160p, so it stays fast |

The signals are normalised with median/MAD, weighted, and smoothed into one **hype curve** (saved as `hype_curve.csv` so you can chart it). Peaks are picked greedily with non-max suppression. Each clip gets a **lead-in** (the build-up), the peak, and a **tail** (the reaction), clamped to `--min-len` and `--max-len`.

## Output

```
clips/
  clip_01_01h42m07s.mp4   # 1080x1920, 30fps, H.264, loudness-normalised to -14 LUFS
  clip_01_01h42m07s.jpg   # thumbnail at the peak frame
  clip_01_01h42m07s.ass   # caption file (editable)
  manifest.json           # timestamps, scores, signal breakdown, transcript
  hype_curve.csv          # per-second scores
```

### Layouts
- `--layout blur` (default): the full gameplay frame on top of a blurred fill. Add `--zoom 1.3` to punch in.
- `--layout crop`: a centre crop, full bleed.
- `--layout facecam --facecam 1500:40:380:260`: the streamer's cam on top and gameplay below (the classic streamer clip). The box is `x:y:w:h` in source pixels.

### Captions
Local Whisper (`faster-whisper`) with word timestamps. Lines are 2–3 words, uppercase, with a yellow karaoke highlight and a pop-in. Change the font with `--font`, the model with `--whisper-model` (tiny → large-v3), and force a language with `--language sv`.

## Install

Requires Python 3.10+ and [ffmpeg](https://ffmpeg.org/download.html) on your PATH.

```bash
git clone https://github.com/rojoze/hypecut && cd hypecut
pip install -e ".[all]"
```

`[all]` pulls in yt-dlp, chat-downloader and faster-whisper.

Minimal install (local files only, no captions): `pip install -e .`

## Usage

```bash
# Just find the moments, no rendering (fast) — good for checking a VOD first
hypecut analyze stream.mp4 --chat chat.json

# Full run on a local file with a chat log
hypecut run stream.mp4 --chat chat.json -n 8 --max-len 40

# Twitch VOD URL, auto-download video + chat
hypecut run https://www.twitch.tv/videos/123456789 --fetch-chat

# Streamer layout with facecam on top
hypecut run stream.mp4 --layout facecam --facecam 1500:40:380:260

# No chat available? Lean harder on audio
hypecut run stream.mp4 --w-spike 1.5 --w-loud 0.8
```

Chat formats: TwitchDownloader JSON, chat-downloader JSON, or a CSV of `seconds,message`.

Speed: audio analysis takes about 1–2 s per hour of VOD. Scene analysis is the slow part (about 2 min per hour), so skip it with `--no-scenes`. Rendering runs at about 3–4 s per clip on a laptop CPU, plus Whisper time.

## Roadmap
- [ ] Kill-feed / wanted-stars OCR for GTA-specific events (police chases, wasted screens)
- [ ] LLM hook titles + hashtags per clip from the transcript
- [ ] Auto facecam detection
- [ ] Web UI: paste URL → review clips → export
- [ ] Batch mode for clipping campaigns (watch a channel, clip every new VOD)
- [ ] Direct upload to TikTok / YouTube Shorts

## Legal
Only clip content you have rights to or that the creator allows (e.g. official clipping programs). Game footage belongs to its publisher. This project isn't affiliated with Rockstar Games or Take-Two.

MIT License.
