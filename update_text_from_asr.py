#!/usr/bin/env python3
"""
Replace the `text` field of every task in a Label Studio JSON with the output
of a HuggingFace ASR model. Runs against the per-segment WAV clips already
produced by segment_audio.py.

This exists as a separate script (rather than inline in segment_audio.py) so
that ctranslate2 (used by faster-whisper) and torch don't share a process —
they bundle conflicting cuDNN/cuBLAS versions and crash when co-loaded.

Usage:
  .venv/Scripts/python.exe update_text_from_asr.py \\
      ./dataset/<project>/<file>.label_studio.json \\
      --model BadRex/w2v-bert-2.0-malagasy-asr \\
      --device cuda --language mg
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_asr import ASR, load_audio_f32  # type: ignore


def main() -> None:
    ap = argparse.ArgumentParser(description="Rewrite Label Studio JSON text using a HF ASR model.")
    ap.add_argument("json", help="Label Studio JSON to update in place.")
    ap.add_argument("--model", "-m", required=True, help="HuggingFace ASR model id.")
    ap.add_argument("--device", default="cuda", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--language", default=None, help="Language code (used by seq2seq models).")
    ap.add_argument("--backup", default=".whisper.bak.json",
                    help="Suffix appended to a backup of the original JSON. Empty to skip.")
    ap.add_argument("--no-require-gpu", action="store_true",
                    help="Allow CPU fallback (off by default — fail loudly if no GPU).")
    ap.add_argument("--field", default="text",
                    help="JSON field to write the new text into (default 'text'). "
                         "Use a different name (e.g. 'text_badrex') to keep the existing "
                         "'text' value as a second alternative for human reviewers.")
    ap.add_argument("--keep-original-as", default="",
                    help="If set, save the previous value of --field into this new field "
                         "before overwriting (e.g. 'text_whisper'). Useful for code-switched "
                         "audio: Label Studio reviewers see both transcripts side-by-side.")
    args = ap.parse_args()

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

    import test_asr as _t
    _t._REQUIRE_GPU = not args.no_require_gpu
    asr = ASR(args.model, device=args.device, language=args.language)
    if _t._REQUIRE_GPU and asr.device != "cuda":
        sys.exit(f"GPU required but ASR ended up on {asr.device}.")

    n_done = 0
    conf_field = "confidence" if args.field == "text" else f"{args.field}_confidence"
    for t in tqdm(tasks, desc="asr", unit="seg"):
        audio_field = t.get("audio", "")
        name = audio_field.rsplit("/", 1)[-1].rsplit("=", 1)[-1]
        clip = base / name
        if not clip.exists():
            print(f"[warn] missing clip: {clip}", flush=True)
            continue
        try:
            audio = load_audio_f32(clip, asr.sample_rate)
            text, conf = asr.transcribe(audio)
        except Exception as e:
            print(f"[err] {clip.name}: {e}", flush=True)
            continue
        # Preserve the prior value if requested (for code-switched audio:
        # keep Whisper's French/English-friendly text alongside BadRex's Malagasy).
        if args.keep_original_as and args.keep_original_as not in t:
            t[args.keep_original_as] = t.get(args.field, "")
            old_conf = t.get("confidence")
            if old_conf is not None:
                t[f"{args.keep_original_as}_confidence"] = old_conf
        t[args.field] = text
        t[conf_field] = round(conf, 3)
        n_done += 1

    json_path.write_text(json.dumps(tasks, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] rewrote {n_done}/{len(tasks)} tasks -> {json_path.name}")

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
