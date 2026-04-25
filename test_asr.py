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
Standalone tester for a HuggingFace CTC ASR model (e.g. BadRex/* Malagasy
fine-tunes, w2v-bert-2.0-malagasy-asr, etc.).

Three input modes:

  1. Single audio file (wav/mp3/...)        — transcribe whole file in one pass.
  2. Directory of clips (*.wav)             — transcribe each clip.
  3. Label Studio JSON from segment_audio.py — transcribe each clip referenced
                                               in the JSON, optionally compute
                                               WER vs the existing weak text.

Examples:

  uv run test_asr.py ./dataset/<project>/<file>.label_studio.json \\
       --model BadRex/w2v-bert-2.0-malagasy-cv-fleurs \\
       --device cuda --output preds.json

  uv run test_asr.py ./some_clip.wav --model BadRex/<repo>

  uv run test_asr.py ./dataset/<project> --model BadRex/<repo> --limit 20

The model id is fully parameterizable — nothing is hardcoded — so swapping in
any HF CTC ASR (Wav2Vec2, Wav2Vec2-BERT, HuBERT, ...) is just a flag change.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
from tqdm import tqdm


# ---------- ffmpeg (bundled) ----------

_FFMPEG: str | None = None


def ffmpeg_bin() -> str:
    global _FFMPEG
    if _FFMPEG:
        return _FFMPEG
    try:
        import imageio_ffmpeg
        _FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
        return _FFMPEG
    except Exception:
        pass
    found = shutil.which("ffmpeg")
    if not found:
        sys.exit("No ffmpeg found. Install ffmpeg or `pip install imageio-ffmpeg`.")
    _FFMPEG = found
    return _FFMPEG


def load_audio_f32(path: Path, target_sr: int) -> np.ndarray:
    """Decode any audio file to mono float32 in [-1, 1] at target_sr via ffmpeg."""
    cmd = [
        ffmpeg_bin(), "-v", "error",
        "-i", str(path),
        "-f", "s16le", "-ac", "1", "-ar", str(target_sr),
        "-",
    ]
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if r.returncode != 0:
        sys.stderr.write(r.stderr.decode("utf-8", errors="replace"))
        raise RuntimeError(f"ffmpeg failed on {path}")
    return np.frombuffer(r.stdout, dtype=np.int16).astype(np.float32) / 32768.0


# ---------- ASR ----------

_REQUIRE_GPU = False  # set by main() when --require-gpu is passed


def _resolve_device(device: str) -> str:
    import torch
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        if _REQUIRE_GPU:
            sys.exit("CUDA requested but torch.cuda.is_available()=False. "
                     "On Windows, the default `torch` wheel is CPU-only — install the "
                     "CUDA wheel via `uv run` (the script header pins one) or "
                     "`pip install torch --index-url https://download.pytorch.org/whl/cu121`.")
        print("[asr] CUDA requested but unavailable; using CPU.", flush=True)
        return "cpu"
    return device


def _model_kind(model_id: str) -> str:
    """Inspect the model's config to dispatch on architecture.
    Returns 'ctc' for Wav2Vec2/HuBERT/Wav2Vec2-BERT-style, 'seq2seq' for Whisper-style."""
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(model_id)
    archs = list(cfg.architectures or [])
    name = (archs[0] if archs else cfg.__class__.__name__).lower()
    if "ctc" in name or "wav2vec2" in name or "hubert" in name:
        return "ctc"
    if "whisper" in name or "speechseq2seq" in name or "speech2text" in name:
        return "seq2seq"
    # Fall back: try CTC head, then seq2seq.
    return "ctc"


class ASR:
    """Generic ASR wrapper that dispatches between CTC and seq2seq (Whisper)
    HuggingFace models. Constructor takes any HF id; nothing is hardcoded."""

    def __init__(self, model_id: str, device: str = "auto", language: str | None = None):
        import torch
        from transformers import AutoProcessor

        self.torch = torch
        self.model_id = model_id
        self.device = _resolve_device(device)
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.kind = _model_kind(model_id)
        self.language = language

        print(f"[asr] loading model={model_id} kind={self.kind} "
              f"device={self.device} dtype={self.dtype}", flush=True)
        self.processor = AutoProcessor.from_pretrained(model_id)

        if self.kind == "ctc":
            from transformers import AutoModelForCTC
            self.model = (
                AutoModelForCTC.from_pretrained(model_id, torch_dtype=self.dtype)
                .to(self.device).eval()
            )
        else:
            from transformers import AutoModelForSpeechSeq2Seq
            self.model = (
                AutoModelForSpeechSeq2Seq.from_pretrained(model_id, torch_dtype=self.dtype)
                .to(self.device).eval()
            )

        fe = getattr(self.processor, "feature_extractor", self.processor)
        self.sample_rate = int(getattr(fe, "sampling_rate", 16000))

    def _to_dtype(self, inputs):
        torch = self.torch
        if self.dtype == torch.float16:
            for k in ("input_features", "input_values"):
                if k in inputs:
                    inputs[k] = inputs[k].to(torch.float16)
        return inputs

    def transcribe(self, audio_f32: np.ndarray) -> tuple[str, float]:
        """Returns (text, confidence in [0,1])."""
        torch = self.torch
        inputs = self.processor(
            audio_f32, sampling_rate=self.sample_rate, return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        inputs = self._to_dtype(inputs)

        if self.kind == "ctc":
            with torch.inference_mode():
                logits = self.model(**inputs).logits  # (1, T, V)
            pred_ids = logits.argmax(dim=-1)
            text = self.processor.batch_decode(pred_ids)[0].strip()
            probs = torch.softmax(logits.float(), dim=-1)
            conf = float(probs.max(dim=-1).values.mean().item())
            return text, conf

        # seq2seq (Whisper-family)
        gen_kwargs = {"return_dict_in_generate": True, "output_scores": True}
        if self.language and hasattr(self.processor, "get_decoder_prompt_ids"):
            try:
                forced = self.processor.get_decoder_prompt_ids(
                    language=self.language, task="transcribe"
                )
                gen_kwargs["forced_decoder_ids"] = forced
            except Exception:
                pass
        with torch.inference_mode():
            out = self.model.generate(**inputs, **gen_kwargs)
        seq = out.sequences
        text = self.processor.batch_decode(seq, skip_special_tokens=True)[0].strip()
        # Confidence proxy: mean of max-softmax over generated tokens.
        if getattr(out, "scores", None):
            scores = torch.stack(out.scores, dim=1)  # (1, T_gen, V)
            probs = torch.softmax(scores.float(), dim=-1)
            conf = float(probs.max(dim=-1).values.mean().item())
        else:
            conf = float("nan")
        return text, conf


# Back-compat alias for older callers
CTCASR = ASR


# ---------- input discovery ----------

def discover_clips(input_path: Path) -> tuple[list[dict], Path | None]:
    """
    Returns (items, json_path).

    Each item: {"audio": Path, "ref": str | None, "start": float | None, "end": float | None}

    - If input is a Label Studio JSON: load tasks, resolve audio paths relative to the JSON.
    - If input is a directory: glob *.wav.
    - If input is a single audio file: one item, no reference.
    """
    if input_path.is_file() and input_path.suffix.lower() == ".json":
        data = json.loads(input_path.read_text(encoding="utf-8"))
        base = input_path.parent
        items = []
        for t in data:
            audio_field = t.get("audio", "")
            # Strip any Label Studio URL prefix (e.g. "/data/local-files/?d=...")
            name = audio_field.rsplit("/", 1)[-1].rsplit("=", 1)[-1]
            items.append({
                "audio": base / name,
                "ref": t.get("text"),
                "start": t.get("start"),
                "end": t.get("end"),
            })
        return items, input_path

    if input_path.is_dir():
        wavs = sorted(input_path.glob("*.wav"))
        return [{"audio": p, "ref": None, "start": None, "end": None} for p in wavs], None

    if input_path.is_file():
        return [{"audio": input_path, "ref": None, "start": None, "end": None}], None

    sys.exit(f"Input not found: {input_path}")


# ---------- WER ----------

def safe_wer(refs: list[str], hyps: list[str]) -> float | None:
    refs = [r for r in refs if r is not None]
    if not refs or len(refs) != len(hyps):
        return None
    try:
        import jiwer
        return float(jiwer.wer(refs, hyps))
    except Exception as e:
        print(f"[wer] couldn't compute: {e}")
        return None


# ---------- main ----------

def main() -> None:
    ap = argparse.ArgumentParser(description="Test a HF CTC ASR model on audio clips.")
    ap.add_argument("input", help="Audio file, directory of *.wav, or Label Studio JSON.")
    ap.add_argument("--model", "-m", required=True,
                    help="HuggingFace model id (e.g. 'BadRex/w2v-bert-2.0-malagasy-cv-fleurs').")
    ap.add_argument("--device", default="cuda", choices=["auto", "cpu", "cuda"],
                    help="Default 'cuda'. Use 'auto' to allow silent CPU fallback.")
    ap.add_argument("--require-gpu", action="store_true", default=True,
                    help="Default ON: abort if CUDA isn't usable. Pass --no-require-gpu to allow CPU.")
    ap.add_argument("--no-require-gpu", action="store_false", dest="require_gpu")
    ap.add_argument("--language", default=None,
                    help="Optional language code for seq2seq models like Whisper "
                         "(e.g. 'mg'). CTC models ignore this.")
    ap.add_argument("--limit", type=int, default=0,
                    help="Process at most N clips (0 = all).")
    ap.add_argument("--output", "-o", default="",
                    help="Write per-clip predictions as JSON to this path.")
    ap.add_argument("--print-each", action="store_true",
                    help="Print every (ref, hyp) pair to stdout.")
    args = ap.parse_args()

    input_path = Path(args.input).resolve()
    items, json_path = discover_clips(input_path)
    if args.limit > 0:
        items = items[: args.limit]
    if not items:
        sys.exit("No audio items found.")
    print(f"[in] {len(items)} clip(s) from {input_path}", flush=True)

    global _REQUIRE_GPU
    _REQUIRE_GPU = args.require_gpu
    asr = ASR(args.model, device=args.device, language=args.language)
    if args.require_gpu and asr.device != "cuda":
        sys.exit(f"--require-gpu set but ASR ended up on {asr.device}.")

    refs: list[str | None] = []
    hyps: list[str] = []
    out_records: list[dict] = []

    for it in tqdm(items, desc="asr", unit="clip"):
        audio_path = it["audio"]
        if not audio_path.exists():
            print(f"[warn] missing: {audio_path}")
            continue
        try:
            audio = load_audio_f32(audio_path, asr.sample_rate)
            hyp, conf = asr.transcribe(audio)
        except Exception as e:
            print(f"[err] {audio_path.name}: {e}")
            continue
        refs.append(it["ref"])
        hyps.append(hyp)
        rec = {
            "audio": audio_path.name,
            "hypothesis": hyp,
            "confidence": round(conf, 3),
        }
        if it["ref"] is not None:
            rec["reference"] = it["ref"]
        if it["start"] is not None:
            rec["start"] = it["start"]
            rec["end"] = it["end"]
        out_records.append(rec)
        if args.print_each:
            tag = " " if it["ref"] is None else "R"
            print(f"[{tag}] {audio_path.name}")
            if it["ref"] is not None:
                print(f"      ref: {it['ref']}")
            print(f"      hyp: {hyp}")

    # WER if references exist
    paired_refs = [r for r in refs if r is not None]
    paired_hyps = [hyps[i] for i, r in enumerate(refs) if r is not None]
    wer = safe_wer(paired_refs, paired_hyps) if paired_refs else None
    if wer is not None:
        print(f"\n[wer] vs references in JSON: {wer * 100:.2f}%  "
              f"({len(paired_refs)} clips with reference text)")

    if args.output:
        out_path = Path(args.output).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": args.model,
            "input": str(input_path),
            "wer": wer,
            "records": out_records,
        }
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[out] wrote {len(out_records)} records -> {out_path}")

    # Match the segment_audio.py convention: skip interpreter shutdown so
    # ctranslate2/torch GPU destructors can't take us down with them.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
