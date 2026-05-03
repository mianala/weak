#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "transformers>=4.45.0,<5.0",
#   "torch>=2.3.0",
#   "datasets>=2.20.0,<3.0",
#   "evaluate>=0.4.0",
#   "jiwer>=3.0.0",
#   "accelerate>=0.34.0,<1.10",
#   "soundfile>=0.12.1",
#   "librosa>=0.10.0",
#   "imageio-ffmpeg>=0.5.1",
#   "numpy>=1.26.0,<2.0",
# ]
#
# [[tool.uv.index]]
# name = "pytorch-cu121"
# url = "https://download.pytorch.org/whl/cu121"
# explicit = true
#
# [tool.uv.sources]
# torch = { index = "pytorch-cu121" }
# torchaudio = { index = "pytorch-cu121" }
# ///
"""
Full fine-tune of Whisper for Malagasy, following the HuggingFace Audio Course
Chapter 5 recipe: https://huggingface.co/learn/audio-course/chapter5/fine-tuning

Differences from train_whisper_lora.py:
  * No LoRA / no 8-bit. Updates all weights — pick a smaller base if VRAM is tight.
  * Designed to start from public Malagasy data (Common Voice / FLEURS) and/or
    reviewed Label Studio JSON exported by segment_audio.py.
  * Optionally bootstraps labels from BadRex/w2v-bert-2.0-malagasy-asr for
    unreviewed clips (--pseudo-label). Use sparingly: pseudo-labels cap WER at
    the teacher's WER. The course recommends real reference text when possible.

Examples:
  # Fine-tune whisper-small on Common Voice MG only
  uv run train_whisper.py \\
      --base openai/whisper-small \\
      --language mg \\
      --hf-dataset mozilla-foundation/common_voice_17_0:mg \\
      --output ./fine-tunes/whisper-small-mg

  # Mix Common Voice + reviewed Label Studio data
  uv run train_whisper.py \\
      --base openai/whisper-small \\
      --language mg \\
      --hf-dataset mozilla-foundation/common_voice_17_0:mg \\
      --train-json ./dataset/podcast/podcast.label_studio.json \\
      --train-clip-root ./dataset/podcast \\
      --output ./fine-tunes/whisper-small-mg

  # Pseudo-label unreviewed clips with BadRex w2v-bert and add them in
  uv run train_whisper.py \\
      --base openai/whisper-small \\
      --language mg \\
      --train-json ./dataset/podcast/podcast.label_studio.json \\
      --train-clip-root ./dataset/podcast \\
      --pseudo-label badrex/w2v-bert-2.0-malagasy-asr \\
      --output ./fine-tunes/whisper-small-mg
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# ---------- dataset assembly ----------

def load_label_studio_json(
    json_path: Path,
    clip_root: Path,
    pseudo_label_pipe=None,
) -> list[dict]:
    """Each Label Studio task -> {audio_path, sentence}.

    Rows with empty `data.text` are skipped unless `pseudo_label_pipe` is given,
    in which case the audio is transcribed by the teacher model.
    """
    tasks = json.loads(json_path.read_text(encoding="utf-8"))
    rows: list[dict] = []
    for t in tasks:
        d = t.get("data") or {}
        text = (d.get("text") or "").strip()
        audio_field = d.get("audio", "")
        name = audio_field.rsplit("/", 1)[-1].rsplit("=", 1)[-1]
        path = clip_root / name
        if not path.exists():
            print(f"[warn] missing clip: {path}", file=sys.stderr)
            continue
        if not text:
            if pseudo_label_pipe is None:
                continue
            try:
                text = pseudo_label_pipe(str(path))["text"].strip()
            except Exception as e:  # noqa: BLE001
                print(f"[warn] pseudo-label failed for {path}: {e}", file=sys.stderr)
                continue
            if not text:
                continue
        rows.append({"audio_path": str(path), "sentence": text})
    return rows


def build_dataset(args, processor):
    """Return (train_ds, eval_ds) HuggingFace Datasets ready for the trainer."""
    from datasets import Audio, Dataset, DatasetDict, concatenate_datasets, load_dataset

    pseudo_pipe = None
    if args.pseudo_label:
        from transformers import pipeline
        import torch
        device = 0 if torch.cuda.is_available() else -1
        print(f"[pseudo] loading teacher {args.pseudo_label}", flush=True)
        pseudo_pipe = pipeline(
            "automatic-speech-recognition",
            model=args.pseudo_label,
            device=device,
            chunk_length_s=30,
        )

    local_rows: list[dict] = []
    for json_path in args.train_json or []:
        json_path = Path(json_path).resolve()
        clip_root = Path(args.train_clip_root or json_path.parent).resolve()
        local_rows.extend(load_label_studio_json(json_path, clip_root, pseudo_pipe))
    del pseudo_pipe  # release VRAM before training starts

    parts = []
    if local_rows:
        ds_local = Dataset.from_list(local_rows).cast_column(
            "audio_path", Audio(sampling_rate=16000)
        ).rename_columns({"audio_path": "audio"})
        parts.append(ds_local)

    for spec in args.hf_dataset or []:
        # "<repo>" or "<repo>:<config>"  (HF hub)
        # OR a local directory containing data/ subdir with parquet files
        # (e.g. one produced by `hf download --local-dir`).
        from os.path import isdir
        as_path = Path(spec).expanduser()
        if isdir(as_path) and (as_path / "data").is_dir():
            print(f"[data] loading local parquet from {as_path}", flush=True)
            files = sorted((as_path / "data").glob("train-*.parquet")) \
                or sorted((as_path / "data").glob("*.parquet"))
            if args.max_train_samples and args.max_parquet_shards:
                files = files[: args.max_parquet_shards]
                print(f"[data] limiting to first {len(files)} parquet shard(s)", flush=True)
            tr = load_dataset("parquet", data_files=[str(p) for p in files], split="train")
        else:
            repo, _, cfg = spec.partition(":")
            cfg = cfg or None
            label = f"{repo}:{cfg}" if cfg else repo
            print(f"[data] loading {label} (streaming={args.stream})", flush=True)
            ld_kwargs = dict(trust_remote_code=True, streaming=args.stream)
            try:
                tr = load_dataset(repo, cfg, split="train", **ld_kwargs)
            except Exception:
                tr = load_dataset(repo, cfg, split="train+validation", **ld_kwargs)
        if args.stream:
            # Streaming IterableDatasets can't concat with mapped local Datasets;
            # materialize to a regular Dataset, optionally capped.
            from itertools import islice
            cap = args.max_hf_samples or 5000
            tr = Dataset.from_list(list(islice(tr, cap)))
        cols = list(tr.column_names)
        col_text = next(
            (c for c in ("sentence", "transcription", "raw_transcription", "text") if c in cols),
            None,
        )
        if col_text is None:
            sys.exit(f"No text column on {spec}; columns={cols}")
        col_audio = "audio" if "audio" in cols else next((c for c in cols if "audio" in c.lower()), "audio")
        if col_audio != "audio":
            tr = tr.rename_column(col_audio, "audio")
        tr = tr.cast_column("audio", Audio(sampling_rate=16000))
        tr = tr.remove_columns([c for c in tr.column_names if c not in ("audio", col_text)])
        if col_text != "sentence":
            tr = tr.rename_column(col_text, "sentence")
        parts.append(tr)

    if not parts:
        sys.exit("No data: pass --train-json and/or --hf-dataset.")

    full = concatenate_datasets(parts) if len(parts) > 1 else parts[0]
    full = full.shuffle(seed=args.seed)
    if args.max_train_samples:
        full = full.select(range(min(args.max_train_samples, len(full))))

    eval_n = max(1, int(len(full) * args.eval_frac))
    eval_ds = full.select(range(eval_n))
    train_ds = full.select(range(eval_n, len(full)))

    def prepare(batch):
        audio = batch["audio"]
        feats = processor.feature_extractor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        ).input_features[0]
        labels = processor.tokenizer(batch["sentence"]).input_ids
        return {"input_features": feats, "labels": labels}

    train_ds = train_ds.map(prepare, remove_columns=train_ds.column_names, num_proc=1)
    eval_ds = eval_ds.map(prepare, remove_columns=eval_ds.column_names, num_proc=1)
    return train_ds, eval_ds


# ---------- collator (verbatim from the HF audio course) ----------

@dataclass
class WhisperCollator:
    processor: Any

    def __call__(self, features: list[dict]) -> dict:
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


# ---------- main ----------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Full fine-tune Whisper on Malagasy (HF audio course recipe)."
    )
    ap.add_argument("--base", default="openai/whisper-small",
                    help="Base Whisper checkpoint. tiny/base/small fit on consumer GPUs.")
    ap.add_argument("--language", required=True,
                    help="Whisper language code (e.g. 'mg').")
    ap.add_argument("--task", default="transcribe", choices=["transcribe", "translate"])

    ap.add_argument("--train-json", action="append",
                    help="Label Studio JSON with reviewed transcripts. Repeatable.")
    ap.add_argument("--train-clip-root", default="",
                    help="Clip directory referenced by the JSON. Defaults to JSON's parent.")
    ap.add_argument("--hf-dataset", action="append",
                    help="HF dataset 'repo' or 'repo:config'. Repeatable. "
                         "E.g. 'badrex/malagasy-speech-full', "
                         "'mozilla-foundation/common_voice_17_0:mg', "
                         "'google/fleurs:mg_mg'.")
    ap.add_argument("--stream", action="store_true",
                    help="Stream HF datasets instead of downloading. "
                         "Useful for huge datasets like badrex/malagasy-speech-full (124 GB).")
    ap.add_argument("--max-hf-samples", type=int, default=0,
                    help="When --stream, cap rows pulled per HF dataset (default 5000).")
    ap.add_argument("--max-parquet-shards", type=int, default=0,
                    help="When loading a local parquet directory, only read this "
                         "many shard files. Useful for smoke tests; ignored when 0.")
    ap.add_argument("--pseudo-label",
                    help="HF model id used to label unreviewed clips "
                         "(e.g. 'badrex/w2v-bert-2.0-malagasy-asr'). "
                         "Off by default — prefer human-corrected text.")

    ap.add_argument("--output", "-o", required=True, help="Output directory.")

    # training knobs (defaults match the audio-course tutorial)
    ap.add_argument("--max-steps", type=int, default=4000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--warmup-steps", type=int, default=500)
    ap.add_argument("--eval-steps", type=int, default=1000)
    ap.add_argument("--save-steps", type=int, default=1000)
    ap.add_argument("--eval-frac", type=float, default=0.05)
    ap.add_argument("--max-train-samples", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--push-to-hub", action="store_true")
    ap.add_argument("--hub-model-id", default=None)
    args = ap.parse_args()

    if not args.train_json and not args.hf_dataset:
        sys.exit("Provide at least one of --train-json or --hf-dataset.")

    # Import datasets/pyarrow BEFORE torch — on Windows, importing pyarrow after
    # torch's CUDA DLLs is loaded triggers an access-violation segfault.
    import datasets  # noqa: F401
    import pyarrow   # noqa: F401

    import torch
    from transformers import (
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        WhisperForConditionalGeneration,
        WhisperProcessor,
    )

    out_dir = Path(args.output).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[base] {args.base} (language={args.language}, task={args.task})", flush=True)
    processor = WhisperProcessor.from_pretrained(
        args.base, language=args.language, task=args.task
    )
    model = WhisperForConditionalGeneration.from_pretrained(args.base)

    # The course pins language/task on the generation config so eval decodes Malagasy.
    model.generation_config.language = args.language
    model.generation_config.task = args.task
    model.generation_config.forced_decoder_ids = None
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.config.use_cache = False  # incompatible with gradient checkpointing

    print("[data] building train/eval datasets", flush=True)
    train_ds, eval_ds = build_dataset(args, processor)
    print(f"[data] train={len(train_ds)} eval={len(eval_ds)}", flush=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        fp16=torch.cuda.is_available(),
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        logging_steps=25,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        predict_with_generate=True,
        generation_max_length=225,
        report_to=[],
        remove_unused_columns=False,
        label_names=["labels"],
        seed=args.seed,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
    )

    import evaluate
    metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        return {"wer": 100 * metric.compute(predictions=pred_str, references=label_str)}

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=WhisperCollator(processor),
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )

    trainer.train()
    trainer.save_model(str(out_dir))
    processor.save_pretrained(str(out_dir))
    if args.push_to_hub:
        trainer.push_to_hub()
    print(f"[done] model saved to {out_dir}")
    print(f"       Use it via: --model {out_dir}")

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
