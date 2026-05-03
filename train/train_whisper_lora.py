#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "transformers>=4.45.0",
#   "torch>=2.3.0",
#   "datasets>=2.20.0",
#   "evaluate>=0.4.0",
#   "jiwer>=3.0.0",
#   "accelerate>=0.34.0",
#   "peft>=0.13.0",
#   "bitsandbytes>=0.43.0; platform_system != 'Darwin'",
#   "soundfile>=0.12.1",
#   "imageio-ffmpeg>=0.5.1",
#   "numpy>=1.26.0",
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
Fine-tune Whisper for a low-resource language (e.g. Malagasy) with LoRA + 8-bit
quantization so it fits on a single 12 GB GPU.

Inputs:
  * One or more Label Studio JSON files exported from segment_audio.py /
    Label Studio (after humans corrected the weak transcripts).
  * Optionally, additional HuggingFace datasets (e.g. mozilla-foundation/common_voice_17_0
    config 'mg', google/fleurs config 'mg_mg').

Output:
  * A local directory containing the LoRA adapter + tokenizer + processor.
    Pass that directory as --model to compare_asr.py / test_asr.py.

Why LoRA + 8-bit?
  * Whisper-small full-FT in fp16 needs ~4-6 GB just for activations on 30s
    inputs; large-v3 won't fit on a 3060 at all.
  * LoRA adds 1-3% params, trains in hours, and the resulting adapter is
    ~50 MB instead of 1.5 GB.

Run (example):
  uv run train_whisper_lora.py \\
      --base openai/whisper-small \\
      --language mg \\
      --train-json ./dataset/<project>/<file>.label_studio.json \\
      --train-clip-root ./dataset/<project> \\
      --extra-dataset mozilla-foundation/common_voice_17_0:mg \\
      --output ./fine-tunes/whisper-small-mg-lora

This script is intentionally NOT auto-run — it expects corrected reference
text. Without that, "fine-tuning" just memorizes Whisper's own mistakes.
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

def load_label_studio_json(json_path: Path, clip_root: Path) -> list[dict]:
    """Each Label Studio task -> {audio_path, sentence}.
    Discards rows with empty text — that's where the human hasn't reviewed yet."""
    tasks = json.loads(json_path.read_text(encoding="utf-8"))
    rows: list[dict] = []
    for t in tasks:
        d = t.get("data") or {}
        text = (d.get("text") or "").strip()
        if not text:
            continue
        audio_field = d.get("audio", "")
        name = audio_field.rsplit("/", 1)[-1].rsplit("=", 1)[-1]
        path = clip_root / name
        if not path.exists():
            print(f"[warn] missing clip referenced in JSON: {path}", file=sys.stderr)
            continue
        rows.append({"audio_path": str(path), "sentence": text})
    return rows


def build_dataset(args, processor):
    """Return (train_ds, eval_ds) HuggingFace Datasets, ready to feed the trainer."""
    from datasets import Dataset, Audio, concatenate_datasets, load_dataset

    rows: list[dict] = []
    for json_path in args.train_json or []:
        json_path = Path(json_path).resolve()
        clip_root = Path(args.train_clip_root or json_path.parent).resolve()
        rows.extend(load_label_studio_json(json_path, clip_root))

    parts = []
    if rows:
        ds_local = Dataset.from_list(rows).cast_column("audio_path", Audio(sampling_rate=16000))
        ds_local = ds_local.rename_columns({"audio_path": "audio"})
        parts.append(ds_local)

    for spec in args.extra_dataset or []:
        # Format: "<repo>:<config>" e.g. "mozilla-foundation/common_voice_17_0:mg"
        # Or a local path to a HF dataset directory (no ':cfg' needed).
        local_path = Path(spec)
        if local_path.exists() and local_path.is_dir():
            print(f"[data] loading local dataset {local_path}", flush=True)
            extra = load_dataset(str(local_path), split="train", trust_remote_code=True)
        else:
            repo, _, cfg = spec.partition(":")
            if not cfg:
                sys.exit(f"--extra-dataset must be 'repo:config' or a local path, got '{spec}'.")
            print(f"[data] loading {repo} ({cfg})", flush=True)
            extra = load_dataset(repo, cfg, split="train", trust_remote_code=True)
        # Normalize column names. Common Voice uses 'sentence', FLEURS / BadRex use 'transcription'.
        col_text = "sentence" if "sentence" in extra.column_names else (
            "transcription" if "transcription" in extra.column_names else None
        )
        if col_text is None:
            sys.exit(f"Don't know which text column to use for {spec}; columns={extra.column_names}")
        extra = extra.cast_column("audio", Audio(sampling_rate=16000))
        extra = extra.remove_columns([c for c in extra.column_names if c not in ("audio", col_text)])
        if col_text != "sentence":
            extra = extra.rename_column(col_text, "sentence")
        parts.append(extra)

    if not parts:
        sys.exit("No data: provide --train-json with reviewed clips and/or --extra-dataset.")

    full = concatenate_datasets(parts) if len(parts) > 1 else parts[0]
    full = full.shuffle(seed=args.seed)
    if args.max_train_samples:
        full = full.select(range(min(args.max_train_samples, len(full))))

    # 95/5 split for eval. With tiny data, override via --eval-frac.
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


# ---------- collator ----------

@dataclass
class WhisperCollator:
    processor: Any

    def __call__(self, features: list[dict]) -> dict:
        import torch
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


# ---------- main ----------

def main() -> None:
    ap = argparse.ArgumentParser(description="LoRA fine-tune Whisper for a low-resource language.")
    ap.add_argument("--base", default="openai/whisper-small",
                    help="Base Whisper checkpoint (small/medium recommended for 12 GB GPUs).")
    ap.add_argument("--language", required=True,
                    help="Whisper language code (e.g. 'mg' for Malagasy).")
    ap.add_argument("--train-json", action="append", default=[],
                    help="Label Studio JSON file(s). Repeat for multiple. Optional if --extra-dataset is given.")
    ap.add_argument("--train-clip-root", default="",
                    help="Directory containing the clips referenced by the JSON. "
                         "Defaults to the JSON's parent.")
    ap.add_argument("--extra-dataset", action="append",
                    help="HF dataset to mix in, format 'repo:config'. "
                         "E.g. 'mozilla-foundation/common_voice_17_0:mg'. Repeatable.")
    ap.add_argument("--output", "-o", required=True, help="Output directory for the adapter.")

    # training knobs
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup-steps", type=int, default=50)
    ap.add_argument("--eval-frac", type=float, default=0.05)
    ap.add_argument("--max-train-samples", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    quant_group = ap.add_mutually_exclusive_group()
    quant_group.add_argument("--4bit", dest="use_4bit", action="store_true",
                             help="Load base model in 4-bit (QLoRA). Required for large-v3 on 15 GB GPUs.")
    quant_group.add_argument("--no-8bit", action="store_true",
                             help="Disable quantization entirely (needs more VRAM).")
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    args = ap.parse_args()

    import faulthandler
    faulthandler.enable()
    import torch
    from transformers import (
        WhisperProcessor, WhisperForConditionalGeneration,
        Seq2SeqTrainer, Seq2SeqTrainingArguments,
        BitsAndBytesConfig,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    out_dir = Path(args.output).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[base] {args.base} (language={args.language})", flush=True)
    processor = WhisperProcessor.from_pretrained(args.base, language=args.language, task="transcribe")

    load_kwargs: dict = {}
    if torch.cuda.is_available() and not args.no_8bit:
        if args.use_4bit:
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        else:
            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        load_kwargs["device_map"] = "auto"
    else:
        load_kwargs["torch_dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32

    model = WhisperForConditionalGeneration.from_pretrained(args.base, **load_kwargs)
    # Pin language/task in the generation config so eval generates Malagasy.
    model.generation_config.language = args.language
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    if torch.cuda.is_available() and not args.no_8bit:
        model = prepare_model_for_kbit_training(model)

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        # No task_type — Whisper's input_features signature collides with
        # SEQ_2_SEQ_LM's auto-injection of input_ids.
    )
    print("[peft] wrapping with LoRA", flush=True)
    model = get_peft_model(model, lora_cfg)
    print("[peft] wrapped", flush=True)
    model.print_trainable_parameters()
    sys.stdout.flush()

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
        num_train_epochs=args.epochs,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        fp16=torch.cuda.is_available(),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=25,
        report_to=[],
        remove_unused_columns=False,  # PEFT needs original columns kept
        label_names=["labels"],
        predict_with_generate=True,
        generation_max_length=225,
        seed=args.seed,
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
        processing_class=processor.feature_extractor,  # for padding only
    )
    model.config.use_cache = False  # required for gradient checkpointing

    trainer.train()
    trainer.save_model(str(out_dir))
    processor.save_pretrained(str(out_dir))
    print(f"[done] adapter saved to {out_dir}")
    print(f"       Use it via: --model {out_dir}")

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
