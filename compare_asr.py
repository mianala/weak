#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "transformers>=4.45.0",
#   "torch>=2.3.0",
#   "soundfile>=0.12.1",
#   "numpy>=1.26.0",
#   "tqdm>=4.66.0",
#   "imageio-ffmpeg>=0.5.1",
#   "jiwer>=3.0.0",
# ]
#
# [[tool.uv.index]]
# name = "pytorch-cu121"
# url = "https://download.pytorch.org/whl/cu121"
# explicit = true
#
# [tool.uv.sources]
# torch = { index = "pytorch-cu121" }
# torchvision = { index = "pytorch-cu121" }
# torchaudio = { index = "pytorch-cu121" }
# ///
"""
Side-by-side ASR comparison.

Loads each model in turn, transcribes the same set of clips, and produces:
  * a CSV with one column per model + reference
  * per-model WER (vs the reference text from the JSON, if any)
  * a JSON dump of all hypotheses

Usage:

  uv run compare_asr.py ./dataset/<project>/<file>.label_studio.json \\
      --model openai/whisper-large-v3 \\
      --model BadRex/<malagasy-w2vbert-repo> \\
      --model BadRex/<finetuned-whisper-repo> \\
      --language mg \\
      --device cuda \\
      --output ./compare/run1

`--model` may be repeated. Anything HuggingFace hosts works (Whisper, Wav2Vec2,
Wav2Vec2-BERT, HuBERT, etc.); architecture is auto-detected.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

from tqdm import tqdm

# Reuse the engine from test_asr.py
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_asr import ASR, discover_clips, load_audio_f32, safe_wer


def short_name(model_id: str) -> str:
    """Compact label for tables: keep last path component."""
    return model_id.rsplit("/", 1)[-1]


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare multiple HF ASR models on the same clips.")
    ap.add_argument("input", help="Audio file, directory of *.wav, or Label Studio JSON.")
    ap.add_argument("--model", "-m", action="append", required=True,
                    help="HuggingFace model id. Repeat for multiple models.")
    ap.add_argument("--device", default="cuda", choices=["auto", "cpu", "cuda"],
                    help="Default 'cuda'. Use 'auto' to allow silent CPU fallback.")
    ap.add_argument("--require-gpu", action="store_true", default=True,
                    help="Default ON: abort if CUDA isn't usable. Pass --no-require-gpu to allow CPU.")
    ap.add_argument("--no-require-gpu", action="store_false", dest="require_gpu")
    ap.add_argument("--language", default=None,
                    help="Language code for seq2seq models (e.g. 'mg'). Ignored by CTC models.")
    ap.add_argument("--limit", type=int, default=0,
                    help="Process at most N clips (0 = all).")
    ap.add_argument("--output", "-o", default="./compare",
                    help="Output directory for CSV + JSON reports.")
    args = ap.parse_args()

    input_path = Path(args.input).resolve()
    items, _ = discover_clips(input_path)
    if args.limit > 0:
        items = items[: args.limit]
    if not items:
        sys.exit("No audio items found.")

    out_dir = Path(args.output).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[compare] {len(items)} clips × {len(args.model)} models", flush=True)

    # Each model is loaded sequentially so VRAM is freed between runs.
    # Audio is decoded once per (model, clip) — cheap relative to inference.
    per_model: dict[str, list[dict]] = {}
    sample_rates: dict[str, int] = {}
    import test_asr as _t
    _t._REQUIRE_GPU = args.require_gpu
    for model_id in args.model:
        asr = ASR(model_id, device=args.device, language=args.language)
        if args.require_gpu and asr.device != "cuda":
            sys.exit(f"--require-gpu set but ASR for {model_id} ended up on {asr.device}.")
        sample_rates[model_id] = asr.sample_rate
        rows = []
        for it in tqdm(items, desc=short_name(model_id), unit="clip"):
            audio_path = it["audio"]
            if not audio_path.exists():
                rows.append({"hypothesis": "", "confidence": None, "error": "missing"})
                continue
            try:
                audio = load_audio_f32(audio_path, asr.sample_rate)
                hyp, conf = asr.transcribe(audio)
                rows.append({"hypothesis": hyp, "confidence": round(conf, 3)})
            except Exception as e:
                rows.append({"hypothesis": "", "confidence": None, "error": str(e)})
        per_model[model_id] = rows
        # Drop the model before loading the next one; CPython GC + CUDA caching
        # allocator generally reclaims VRAM, but we can't force-fix the segfault
        # path so we keep this script single-model-at-a-time.
        del asr
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    # WER per model (only over clips with a reference)
    refs = [it.get("ref") for it in items]
    wer_per_model: dict[str, float | None] = {}
    for model_id, rows in per_model.items():
        paired_refs = []
        paired_hyps = []
        for r, row in zip(refs, rows):
            if r is not None and row.get("hypothesis") is not None:
                paired_refs.append(r)
                paired_hyps.append(row["hypothesis"])
        wer_per_model[model_id] = safe_wer(paired_refs, paired_hyps) if paired_refs else None

    # ---------- write CSV ----------
    csv_path = out_dir / "comparison.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        header = ["audio", "start", "end", "reference"]
        for m in args.model:
            header += [f"{short_name(m)}__hyp", f"{short_name(m)}__conf"]
        w.writerow(header)
        for i, it in enumerate(items):
            row = [
                Path(it["audio"]).name,
                it.get("start", ""),
                it.get("end", ""),
                it.get("ref") or "",
            ]
            for m in args.model:
                rec = per_model[m][i]
                row += [rec.get("hypothesis", ""), rec.get("confidence", "")]
            w.writerow(row)

    # ---------- write JSON ----------
    json_path = out_dir / "comparison.json"
    payload = {
        "input": str(input_path),
        "language": args.language,
        "device": args.device,
        "models": args.model,
        "wer": {short_name(m): wer_per_model[m] for m in args.model},
        "items": [
            {
                "audio": Path(it["audio"]).name,
                "start": it.get("start"),
                "end": it.get("end"),
                "reference": it.get("ref"),
                "predictions": {
                    short_name(m): per_model[m][i]
                    for m in args.model
                },
            }
            for i, it in enumerate(items)
        ],
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---------- summary ----------
    print("\n[wer] vs JSON references (lower is better):")
    for m in args.model:
        wer = wer_per_model[m]
        label = short_name(m)
        if wer is None:
            print(f"  {label:40s}  (no references available)")
        else:
            print(f"  {label:40s}  {wer * 100:6.2f}%")
    print(f"\n[out] CSV:  {csv_path}")
    print(f"[out] JSON: {json_path}")

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
