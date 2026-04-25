#!/usr/bin/env python3
"""
LLM cleanup pass: takes the existing ASR transcripts in a Label Studio JSON
(BadRex Malagasy + Whisper French/English) and asks a small multilingual LLM
to produce a cleaned, code-switching-aware version. Writes it into a new field.

The LLM never sees the audio — only the two text hypotheses. So it's effectively
doing post-hoc disambiguation: when BadRex's Malagasy makes sense, keep it;
when Whisper has a recognizable French/English word in the same span, prefer
that. When both are gibberish, the LLM is told to leave it empty.

Default model: Qwen/Qwen2.5-3B-Instruct (3B params, ~2GB at 4-bit, no gating).
Override with --model.

Run separately from the ASR scripts; loads its own LLM.

  .venv/Scripts/python.exe update_text_llm.py <json> \\
      --model Qwen/Qwen2.5-3B-Instruct \\
      --device cuda --field text_llm
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

from tqdm import tqdm


PROMPT_TEMPLATE = """You are cleaning a weak speech transcription.

Audio language: Malagasy, with occasional French or English words mixed into sentences (the speaker is from Madagascar).

Two automatic speech recognition systems gave different transcripts of the same short audio clip:

- Malagasy-specialized model (usually correct on Malagasy words, butchers French/English):
  "{badrex}"

- General multilingual Whisper (correct on French/English words, weak on Malagasy):
  "{whisper}"

Your task: output the single most likely correct transcription of what the speaker actually said.

Rules:
- Use Malagasy words from the first transcript when they look like real Malagasy.
- Use French or English words from the second transcript when they appear to be borrowed/code-switched terms (community manager, exactement, business, etc.).
- Do not invent content that isn't supported by either transcript.
- If both transcripts are clearly nonsense or empty, output exactly: <UNCLEAR>
- Output ONLY the cleaned transcript on a single line. No quotes, no explanation, no prefix.
"""


def build_prompt(badrex: str, whisper: str) -> str:
    return PROMPT_TEMPLATE.format(badrex=(badrex or "").strip(), whisper=(whisper or "").strip())


def main() -> None:
    ap = argparse.ArgumentParser(description="LLM cleanup of weak ASR transcripts.")
    ap.add_argument("json", help="Label Studio JSON to update.")
    ap.add_argument("--model", "-m", default="Qwen/Qwen2.5-3B-Instruct",
                    help="HuggingFace LLM id (default: Qwen2.5-3B-Instruct).")
    ap.add_argument("--device", default="cuda", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--require-gpu", action="store_true", default=True)
    ap.add_argument("--no-require-gpu", action="store_false", dest="require_gpu")
    ap.add_argument("--field", default="text_llm",
                    help="JSON field to write the cleaned text into.")
    ap.add_argument("--source-malagasy-field", default="text",
                    help="Field that holds the Malagasy ASR transcript (default 'text').")
    ap.add_argument("--source-whisper-field", default="text_whisper",
                    help="Field that holds the Whisper transcript (default 'text_whisper').")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--no-4bit", action="store_true",
                    help="Disable 4-bit quantization. Needs more VRAM.")
    ap.add_argument("--backup", default="",
                    help="Suffix for a JSON backup. Empty to skip.")
    ap.add_argument("--limit", type=int, default=0,
                    help="Process at most N tasks (0 = all).")
    ap.add_argument("--checkpoint-every", type=int, default=20,
                    help="Write JSON back to disk every N tasks so progress survives crashes (default 20).")
    ap.add_argument("--resume", action="store_true", default=True,
                    help="Skip tasks that already have a non-empty value in --field (default ON).")
    ap.add_argument("--no-resume", action="store_false", dest="resume",
                    help="Re-run on all tasks even if they already have output.")
    args = ap.parse_args()

    json_path = Path(args.json).resolve()
    if not json_path.exists():
        sys.exit(f"JSON not found: {json_path}")

    if args.backup:
        backup = json_path.with_suffix(args.backup)
        if not backup.exists():
            shutil.copyfile(json_path, backup)
            print(f"[backup] {backup.name}")

    tasks = json.loads(json_path.read_text(encoding="utf-8"))
    if args.limit > 0:
        process = tasks[: args.limit]
    else:
        process = tasks
    print(f"[in] {len(process)}/{len(tasks)} tasks from {json_path.name}", flush=True)

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    if args.require_gpu and not torch.cuda.is_available():
        sys.exit("CUDA required but unavailable.")
    device = "cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu"

    load_kwargs: dict = {}
    if not args.no_4bit and device == "cuda":
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
        load_kwargs["device_map"] = "auto"
    else:
        load_kwargs["torch_dtype"] = torch.float16 if device == "cuda" else torch.float32

    print(f"[llm] loading {args.model} device={device} 4bit={not args.no_4bit}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    if device == "cpu":
        model = model.to(device)
    model.eval()

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Resume: count how many already have output and skip them mid-loop.
    n_already = sum(1 for t in process if args.resume and (t.get(args.field) or "").strip())
    if n_already:
        print(f"[resume] {n_already}/{len(process)} tasks already have '{args.field}'; skipping them.",
              flush=True)

    def flush_to_disk():
        json_path.write_text(json.dumps(tasks, ensure_ascii=False, indent=2), encoding="utf-8")

    n_done = 0
    n_unclear = 0
    n_since_flush = 0
    for t in tqdm(process, desc="llm", unit="seg"):
        if args.resume and (t.get(args.field) or "").strip():
            continue
        badrex = t.get(args.source_malagasy_field, "") or ""
        whisper = t.get(args.source_whisper_field, "") or ""
        if not badrex and not whisper:
            t[args.field] = ""
            continue

        prompt = build_prompt(badrex, whisper)
        messages = [{"role": "user", "content": prompt}]
        try:
            input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            input_text = prompt + "\n"
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.pad_token_id,
            )
        gen = out[0, inputs["input_ids"].shape[1]:]
        text = tokenizer.decode(gen, skip_special_tokens=True).strip()
        # Common artifacts: model echoes "Cleaned transcript:" etc. — strip.
        for prefix in ("Cleaned transcript:", "Output:", "Transcript:", "Result:"):
            if text.lower().startswith(prefix.lower()):
                text = text[len(prefix):].strip()
        # Single line only
        text = text.splitlines()[0].strip() if text else ""
        # Strip surrounding quotes if model added them
        if len(text) >= 2 and text[0] in "\"'" and text[-1] in "\"'":
            text = text[1:-1].strip()
        if text == "<UNCLEAR>":
            n_unclear += 1
            text = ""
        t[args.field] = text
        n_done += 1
        n_since_flush += 1
        if n_since_flush >= args.checkpoint_every:
            flush_to_disk()
            n_since_flush = 0

    flush_to_disk()
    print(f"[done] cleaned {n_done} tasks, {n_unclear} marked unclear -> {json_path.name}")
    print(f"       field: {args.field}")

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
