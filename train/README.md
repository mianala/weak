# Whisper fine-tuning for Malagasy

Two scripts, both following the recipe from the
[HuggingFace Audio Course — Chapter 5](https://huggingface.co/learn/audio-course/chapter5/fine-tuning):

| Script | What it does | When to use |
| --- | --- | --- |
| `train_whisper.py` | Full fine-tune (all weights). | You have ≥ 16 GB VRAM or you're using a small base (tiny/base/small). |
| `train_whisper_lora.py` | LoRA + 8-bit quantized base. ~50 MB adapter. | Tight VRAM (12 GB GPU) or you want to keep the base model frozen. |

Both expect 16 kHz audio with reference text. Sources:

- Reviewed Label Studio JSON exported by `../segment_audio.py` (humans corrected the weak transcripts).
- Public HF datasets — `badrex/malagasy-speech-full` (150 h, 124 GB), `mozilla-foundation/common_voice_17_0:mg`, `google/fleurs:mg_mg`.
- Optional pseudo-labels from [`badrex/w2v-bert-2.0-malagasy-asr`](https://huggingface.co/badrex/w2v-bert-2.0-malagasy-asr) for unreviewed clips (`--pseudo-label`, full-FT script only).

> Pseudo-labeling caps WER at the teacher's WER. Prefer reviewed text whenever you have it.

## Downloading the BadRex Malagasy dataset

[`badrex/malagasy-speech-full`](https://huggingface.co/datasets/badrex/malagasy-speech-full)
— 150 hours of transcribed Malagasy, ~31.6k clips across train/validation/test,
total **124 GB** in parquet. The script can either download to disk or stream:

**Option A — stream (no full download).** Pulls only what training consumes:

```bash
uv run train/train_whisper.py \
  --base openai/whisper-small --language mg \
  --hf-dataset badrex/malagasy-speech-full --stream --max-hf-samples 5000 \
  --output ./fine-tunes/whisper-small-mg
```

**Option B — download once with the HF CLI**, then train normally.

Install the CLI if needed (the modern command is `hf`; `huggingface-cli` is deprecated):

```bash
uv tool install huggingface_hub
```

Or run it ad-hoc with `uvx --from huggingface_hub hf ...` (no global install).

PowerShell (Windows) — point `HF_HOME` at a drive with space first:

```powershell
$env:HF_HOME = "E:\hf-cache"
hf download badrex/malagasy-speech-full `
  --repo-type dataset `
  --local-dir E:\hf-data\malagasy-speech-full
```

Bash / macOS / Linux:

```bash
export HF_HOME=/mnt/data/hf-cache
hf download badrex/malagasy-speech-full \
  --repo-type dataset \
  --local-dir /mnt/data/hf-data/malagasy-speech-full
```

Then train against the cached dataset (no `--stream` needed):

```bash
uv run train/train_whisper.py \
  --base openai/whisper-small --language mg \
  --hf-dataset badrex/malagasy-speech-full \
  --output ./fine-tunes/whisper-small-mg
```

## Quick run — full fine-tune

Train `whisper-small` on Common Voice MG:

```bash
uv run train/train_whisper.py \
  --base openai/whisper-small \
  --language mg \
  --hf-dataset mozilla-foundation/common_voice_17_0:mg \
  --output ./fine-tunes/whisper-small-mg
```

Mix Common Voice + your reviewed Label Studio data:

```bash
uv run train/train_whisper.py \
  --base openai/whisper-small \
  --language mg \
  --hf-dataset mozilla-foundation/common_voice_17_0:mg \
  --train-json dataset/storytown_bolo/storytown_bolo.label_studio.json \
  --train-clip-root dataset/storytown_bolo \
  --output ./fine-tunes/whisper-small-mg
```

Bootstrap unreviewed clips with the BadRex w2v-bert teacher:

```bash
uv run train/train_whisper.py \
  --base openai/whisper-small \
  --language mg \
  --train-json dataset/storytown_bolo/storytown_bolo.label_studio.json \
  --train-clip-root dataset/storytown_bolo \
  --pseudo-label badrex/w2v-bert-2.0-malagasy-asr \
  --output ./fine-tunes/whisper-small-mg
```

Smoke test (verify install, < 1 min, no real training):

```bash
uv run train/train_whisper.py \
  --base openai/whisper-tiny \
  --language mg \
  --train-json dataset/storytown_bolo/storytown_bolo.label_studio.json \
  --train-clip-root dataset/storytown_bolo \
  --max-steps 10 --max-train-samples 20 --batch-size 2 \
  --warmup-steps 2 --eval-steps 5 --save-steps 10 --eval-frac 0.2 \
  --output ./fine-tunes/smoketest
```

## Quick run — LoRA (12 GB GPU, whisper-small/medium)

```bash
uv run train/train_whisper_lora.py \
  --base openai/whisper-small \
  --language mg \
  --train-json dataset/storytown_bolo/storytown_bolo.label_studio.json \
  --train-clip-root dataset/storytown_bolo \
  --extra-dataset mozilla-foundation/common_voice_17_0:mg \
  --output ./fine-tunes/whisper-small-mg-lora
```

## Fine-tuning whisper-large-v3 for free (Google Colab / Kaggle T4)

The free tier on **Google Colab** and **Kaggle** both give a T4 GPU (15 GB VRAM).
`whisper-large-v3` (~1.5B params) needs **4-bit QLoRA** (`--4bit`) to fit there.

> Note: `badrex/w2v-bert-2.0-malagasy-asr` is the **model**. The training dataset
> is [`badrex/malagasy-speech-full`](https://huggingface.co/datasets/badrex/malagasy-speech-full)
> (150 h, ~31 k clips).

### Step 1 — pick a free GPU platform

| Platform | Free GPU | Hours/week | Notes |
|---|---|---|---|
| [Kaggle](https://kaggle.com) | T4 × 1 (15 GB) | 30 h | Best free option; connect to internet to pull HF datasets |
| [Google Colab](https://colab.research.google.com) | T4 (15 GB) | variable | Session-limited; upgrade to Colab Pro ($10/mo) for A100 |
| [HF Spaces ZeroGPU](https://huggingface.co/spaces) | A10G (24 GB) | free via Spaces | Must wrap in a Gradio/Streamlit app |

### Step 2 — install and run (Colab / Kaggle cell)

```python
# Cell 1 — install
!pip install -q transformers datasets accelerate peft bitsandbytes \
             evaluate jiwer soundfile imageio-ffmpeg

# Cell 2 — clone repo (or upload train_whisper_lora.py directly)
!git clone https://github.com/YOUR_ORG/weak .

# Cell 3 — log in to HF (needed to stream badrex/malagasy-speech-full)
from huggingface_hub import login
login()   # paste your HF read token

# Cell 4 — train (streams dataset, no 124 GB download needed)
!python train/train_whisper_lora.py \
  --base openai/whisper-large-v3 \
  --language mg \
  --4bit \
  --extra-dataset badrex/malagasy-speech-full \
  --max-train-samples 5000 \
  --batch-size 2 \
  --grad-accum 8 \
  --lr 1e-4 \
  --warmup-steps 50 \
  --epochs 3 \
  --output ./fine-tunes/whisper-large-v3-mg-lora
```

**Memory breakdown on T4 15 GB:**
- Model weights in 4-bit: ~900 MB
- LoRA trainable params (q_proj + v_proj, r=32): ~10 M params × fp16 ≈ 20 MB
- Optimizer states (AdamW on LoRA only): ~80 MB
- Activations at batch 2, 30 s audio: ~2–4 GB
- Total: ~6–7 GB → fits comfortably with headroom

If you OOM, add `--lora-r 16` (halves adapter size) or keep `--batch-size 1`.

### Step 3 — save the adapter back to HF Hub

```python
# Cell 5
!python train/train_whisper_lora.py \
  ... \
  --push-to-hub \
  --hub-model-id YOUR_HF_USERNAME/whisper-large-v3-malagasy-lora
```

Or from Python after training:

```python
from peft import PeftModel
from transformers import WhisperForConditionalGeneration
base  = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3")
model = PeftModel.from_pretrained(base, "./fine-tunes/whisper-large-v3-mg-lora")
model.push_to_hub("YOUR_HF_USERNAME/whisper-large-v3-malagasy-lora")
```

### Why not whisper-large full fine-tune?

Full FT of large-v3 in fp16 needs ~40 GB VRAM. That requires an A100 80 GB or
multi-GPU setup — not available for free. QLoRA is the standard approach for
large models on consumer/free GPUs and typically recovers >95% of full-FT WER.

## Use the result

Pass the output directory as `--model` to the comparison/test scripts in the repo root:

```bash
uv run compare_asr.py input.mp3 --model ./fine-tunes/whisper-small-mg
uv run test_asr.py    clip.wav  --model ./fine-tunes/whisper-small-mg
```

## CLI reference (full FT)

```
--base ID            Base Whisper checkpoint (default openai/whisper-small).
--language CODE      Whisper language code, e.g. 'mg' (required).
--task {transcribe,translate}
--train-json PATH    Label Studio JSON. Repeatable.
--train-clip-root D  Clips directory. Defaults to JSON's parent.
--hf-dataset SPEC    'repo:config'. Repeatable. e.g. mozilla-foundation/common_voice_17_0:mg
--pseudo-label ID    Teacher model for unreviewed clips, e.g. badrex/w2v-bert-2.0-malagasy-asr
--output / -o DIR    Output directory (required).

--max-steps N        Default 4000 (HF audio-course default).
--batch-size N       Default 16.
--grad-accum N       Default 1.
--lr FLOAT           Default 1e-5.
--warmup-steps N     Default 500.
--eval-steps N       Default 1000.
--save-steps N       Default 1000.
--eval-frac F        Held-out fraction (default 0.05).
--max-train-samples  Cap rows for quick experiments.
--push-to-hub        Push the trained model to HuggingFace.
--hub-model-id ID
```

## Notes

- The first run downloads the base model (~ 500 MB for `small`) and any HF datasets — Common Voice MG is small (< 1 GB), FLEURS too.
- `gradient_checkpointing=True` and `fp16` are on by default; that's how `whisper-small` fits at batch size 16 on a 12–16 GB GPU.
- The trainer evaluates with `predict_with_generate=True` so reported WER is on actual greedy decodes, not teacher-forced loss.
- `model.generation_config.language` and `task` are pinned so eval decodes Malagasy — without this, Whisper drifts back to English mid-training.
