#!/usr/bin/env python3
"""
Run faster-whisper on each clip referenced in a Label Studio JSON and write
its text into a chosen field. Lets you add an extra "Whisper-as-French" or
"Whisper-as-auto" column alongside BadRex's Malagasy primary text — useful
for code-switched audio where one or two words per sentence are French/English.

Run separately from any transformers/torch process: faster-whisper bundles
its own CUDA libs and conflicts with torch's bundled cuDNN if co-loaded.

Usage:
  .venv/Scripts/python.exe update_text_whisper.py <json> \\
      --language fr --field text_whisper_fr --device cuda --require-gpu
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from segment_audio import (  # type: ignore
    _add_nvidia_dll_dirs, ffmpeg_bin, pick_device, _LIVE_MODELS,
)


def transcribe_clip(model, clip_path: Path, language: str | None) -> tuple[str, float | None]:
    """Decode a single short clip in one Whisper pass."""
    segments, _info = model.transcribe(
        str(clip_path),
        language=language,
        task="transcribe",
        vad_filter=False,           # the clip is already a tight segment
        beam_size=1,
        condition_on_previous_text=False,
        word_timestamps=False,
    )
    texts = []
    logprobs = []
    for s in segments:
        texts.append((s.text or "").strip())
        if getattr(s, "avg_logprob", None) is not None:
            logprobs.append(s.avg_logprob)
    text = " ".join(t for t in texts if t).strip()
    conf = None
    if logprobs:
        # convert avg log-prob to a [0,1] proxy
        avg = sum(logprobs) / len(logprobs)
        conf = float(min(max(0.0, 1.0 + avg), 1.0))
    return text, conf


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Add a Whisper transcript column (any language) to a Label Studio JSON.")
    ap.add_argument("json", help="Label Studio JSON to update in place.")
    ap.add_argument("--language", default="auto",
                    help="Whisper language code: 'fr', 'en', 'mg', or 'auto' for per-clip detection.")
    ap.add_argument("--model", default="large-v3",
                    help="faster-whisper model size: tiny|base|small|medium|large-v3.")
    ap.add_argument("--device", default="cuda", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--require-gpu", action="store_true", default=True,
                    help="Default ON: abort if CUDA isn't usable.")
    ap.add_argument("--no-require-gpu", action="store_false", dest="require_gpu")
    ap.add_argument("--field", required=True,
                    help="JSON field to write the new text into (e.g. 'text_whisper_fr').")
    ap.add_argument("--backup", default="",
                    help="Suffix appended to a backup of the JSON. Empty to skip "
                         "(default — backup once was already made by update_text_from_asr.py).")
    args = ap.parse_args()

    _add_nvidia_dll_dirs()  # CUDA DLLs visible before ctranslate2 imports

    json_path = Path(args.json).resolve()
    if not json_path.exists():
        sys.exit(f"JSON not found: {json_path}")
    base = json_path.parent

    if args.backup:
        backup = json_path.with_suffix(args.backup)
        if not backup.exists():
            shutil.copyfile(json_path, backup)
            print(f"[backup] {backup.name}")

    tasks = json.loads(json_path.read_text(encoding="utf-8"))
    print(f"[in] {len(tasks)} tasks from {json_path.name}", flush=True)

    device, compute_type = pick_device(args.device)
    if args.require_gpu and device != "cuda":
        sys.exit(f"--require-gpu set but no usable CUDA device was found (device={device}).")

    from faster_whisper import WhisperModel
    print(f"[whisper] loading model={args.model} device={device} compute_type={compute_type} "
          f"language={args.language}", flush=True)
    model = WhisperModel(args.model, device=device, compute_type=compute_type)
    _LIVE_MODELS.append(model)

    language = None if args.language.lower() == "auto" else args.language
    conf_field = f"{args.field}_confidence"

    n_done = 0
    for t in tqdm(tasks, desc="whisper", unit="seg"):
        d = t.setdefault("data", {})
        audio_field = d.get("audio", "")
        name = audio_field.rsplit("/", 1)[-1].rsplit("=", 1)[-1]
        clip = base / name
        if not clip.exists():
            print(f"[warn] missing clip: {clip}", flush=True)
            continue
        try:
            text, conf = transcribe_clip(model, clip, language)
        except Exception as e:
            print(f"[err] {clip.name}: {e}", flush=True)
            continue
        d[args.field] = text
        if conf is not None:
            d[conf_field] = round(conf, 3)
        n_done += 1

    json_path.write_text(json.dumps(tasks, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] wrote field '{args.field}' on {n_done}/{len(tasks)} tasks -> {json_path.name}")

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
