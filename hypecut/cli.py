"""Command line interface."""

from __future__ import annotations

import argparse
import sys

from .media import HypeCutError
from .pipeline import Options, run
from .scoring import Weights


def _facecam(s: str) -> tuple[int, int, int, int]:
    try:
        x, y, w, h = (int(v) for v in s.replace(",", ":").split(":"))
        return x, y, w, h
    except ValueError:
        raise argparse.ArgumentTypeError("facecam must be x:y:w:h, e.g. 1500:40:380:260")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="hypecut",
        description="Find the hype moments in a long VOD and auto-cut vertical, captioned clips.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("input", help="video file or VOD URL (Twitch / YouTube / Kick via yt-dlp)")
        sp.add_argument("-o", "--out", default="clips", help="output folder (default: clips)")
        sp.add_argument("-n", "--top", type=int, default=5, help="how many clips (default 5)")
        sp.add_argument("--min-len", type=float, default=15)
        sp.add_argument("--max-len", type=float, default=45)
        sp.add_argument("--lead-in", type=float, default=10, help="seconds of build-up before the peak")
        sp.add_argument("--tail", type=float, default=5, help="seconds of reaction after the peak")
        sp.add_argument("--min-score", type=float, default=1.5, help="ignore peaks below this hype score")
        sp.add_argument("--chat", help="chat replay file (TwitchDownloader JSON, chat-downloader JSON or CSV)")
        sp.add_argument("--fetch-chat", action="store_true", help="download chat replay for URL input")
        sp.add_argument("--chat-delay", type=float, default=5, help="seconds chat lags the action (default 5)")
        sp.add_argument("--no-scenes", action="store_true", help="skip visual scene analysis (faster)")
        sp.add_argument("--w-loud", type=float, default=0.6)
        sp.add_argument("--w-spike", type=float, default=1.0)
        sp.add_argument("--w-chat", type=float, default=1.4)
        sp.add_argument("--w-scene", type=float, default=0.5)

    a = sub.add_parser("analyze", help="score the VOD and list moments, no rendering")
    common(a)

    r = sub.add_parser("run", help="analyze + render vertical clips")
    common(r)
    r.add_argument("--layout", choices=["blur", "crop", "facecam"], default="blur",
                   help="blur = full frame over blurred bg, crop = centre crop, facecam = cam on top + gameplay")
    r.add_argument("--facecam", type=_facecam, help="facecam box in source pixels x:y:w:h")
    r.add_argument("--zoom", type=float, default=1.0, help="zoom gameplay in blur layout (e.g. 1.3)")
    r.add_argument("--no-captions", action="store_true")
    r.add_argument("--whisper-model", default="small", help="tiny / base / small / medium / large-v3")
    r.add_argument("--whisper-device", choices=["cpu", "cuda", "auto"], default="cpu",
                   help="cpu is safest; cuda needs NVIDIA CUDA 12 + cuDNN 9 installed")
    r.add_argument("--language", help="force caption language, e.g. en, sv")
    r.add_argument("--font", default="Montserrat Black")
    r.add_argument("--fps", type=int, default=30)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    opts = Options(
        top=args.top, min_len=args.min_len, max_len=args.max_len, lead_in=args.lead_in, tail=args.tail,
        min_score=args.min_score, chat=args.chat, fetch_chat=args.fetch_chat, chat_delay=args.chat_delay,
        use_scenes=not args.no_scenes,
        weights=Weights(loudness=args.w_loud, spike=args.w_spike, chat=args.w_chat, scene=args.w_scene),
    )
    if args.cmd == "run":
        opts.layout, opts.facecam, opts.zoom = args.layout, args.facecam, args.zoom
        opts.captions = not args.no_captions
        opts.whisper_model, opts.language, opts.font, opts.fps = args.whisper_model, args.language, args.font, args.fps
        opts.whisper_device = args.whisper_device
    try:
        run(args.input, args.out, opts, render_clips=args.cmd == "run")
    except HypeCutError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
