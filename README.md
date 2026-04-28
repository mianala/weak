# Weak Transcription + Segmentation

Local pipeline that turns long audio (podcasts, radio, interviews) into
**draft transcripts + 10–30 s clips** ready for human correction in
[Label Studio](https://labelstud.io/). Built for **low-resource languages**
like **Malagasy** where Whisper output is noisy but still useful as a
starting point.

## What it does

1. Loads audio from a local path or URL (mp3, wav, m4a, …).
2. Normalizes to 16 kHz mono WAV via `ffmpeg`.
3. Runs **faster-whisper** with built-in VAD to find speech regions and
   produce a weak transcript.
4. Re-splits any segment > `--max-seg` at word boundaries so clips stay
   roughly 10–30 s.
5. Writes a Label Studio-compatible JSON file and (optionally) one WAV
   per segment.

## Requirements

- Python 3.10+
- [`uv`](https://github.com/astral-sh/uv) (recommended) or pip
- `ffmpeg` on PATH
- A CUDA-capable GPU is optional — script falls back to CPU automatically.

## Run with uv (zero-setup)

The script declares its dependencies inline (PEP 723), so `uv` will create
an ephemeral environment and run it:

```bash
uv run segment_audio.py input.mp3 --output ./dataset
```

## Or install manually

```bash
uv venv
uv pip install faster-whisper pydub numpy requests tqdm
python segment_audio.py input.mp3 --output ./dataset
```

## CLI

```
python segment_audio.py <input> [options]

  input                  Path or URL to audio.
  --output / -o DIR      Output directory (default ./dataset).
  --language CODE        Whisper language code. Default: mg (Malagasy).
                         Use 'auto' to let Whisper detect.
  --model NAME           tiny | base | small | medium | large-v3 (default).
  --device {auto,cpu,cuda}
  --min-seg SEC          Drop clips shorter than this (default 2).
  --max-seg SEC          Re-split clips longer than this (default 30).
  --target-seg SEC       Target chunk length when splitting (default 20).
  --no-clips             Skip per-segment audio export.
  --audio-url-prefix S   Prefix for the JSON 'audio' field, e.g.
                         '/data/local-files/?d=clips/' for Label Studio
                         local file serving.
```

## Output

```
dataset/
├── clips/
│   ├── podcast_00000_00012300_00023700.wav
│   └── ...
└── podcast.label_studio.json
```

Each task in the JSON file is shaped for direct Label Studio import — the
weak transcript is injected as a prediction so it pre-fills the editable
`<TextArea>`, while `data.*` stays flat so `$audio` / `$text` /
`$confidence` references in your `<Audio>` and `<Header>` tags resolve:

```json
{
  "data": {
    "audio": "podcast_00000_00012300_00023700.wav",
    "start": 12.3,
    "end": 23.7,
    "text": "weak transcription here",
    "confidence": 0.71
  },
  "predictions": [
    {
      "model_version": "large-v3",
      "score": 0.71,
      "result": [
        {
          "from_name": "transcription",
          "to_name": "audio",
          "type": "textarea",
          "value": { "text": ["weak transcription here"] }
        }
      ]
    }
  ]
}
```

`from_name` must match the `name` on your `<TextArea>` (default
`transcription`) and `to_name` must match the `<Audio>` element (default
`audio`). `start` / `end` are timestamps in the **original recording**;
the clip file itself starts at 0.

## Notes for Malagasy / mixed-language audio

- The default `--language mg` forces Malagasy decoding. If recordings mix
  Malagasy and French, try `--language auto` and post-filter, or run the
  pipeline twice with different language codes.
- Transcripts are intentionally **weak** — beam size is 1 and the model
  is not conditioned on prior context, which keeps drift low when humans
  re-read it segment-by-segment in Label Studio.
