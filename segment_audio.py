#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "faster-whisper>=1.0.3",
#   "pydub>=0.25.1",
#   "numpy>=1.26.0",
#   "requests>=2.32.0",
#   "tqdm>=4.66.0",
#   "imageio-ffmpeg>=0.5.1",
#   "nvidia-cublas-cu12; platform_system != 'Darwin'",
#   "nvidia-cudnn-cu12; platform_system != 'Darwin'",
# ]
# ///
"""
Weak transcription + segmentation pipeline for long audio (e.g. Malagasy podcasts).

Pipeline:
  1. Load audio from a local path or URL (mp3/wav/m4a/...).
  2. Convert to 16 kHz mono WAV via ffmpeg (required by faster-whisper).
  3. Run faster-whisper with built-in VAD to get speech segments + draft text.
  4. Re-split any segment longer than --max-seg seconds at the closest word
     boundary so each clip is roughly 10-30 s.
  5. Export each segment as its own .wav and a Label Studio compatible JSON.

Run with uv (no manual venv needed):
    uv run segment_audio.py input.mp3 --output ./dataset

Or install deps first:
    uv pip install faster-whisper pydub numpy requests tqdm
    python segment_audio.py input.mp3 --output ./dataset

ffmpeg must be on PATH.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse

import requests
from tqdm import tqdm


# ---------- audio I/O ----------

def is_url(s: str) -> bool:
    p = urlparse(s)
    return p.scheme in ("http", "https")


def download(url: str, dst: Path) -> Path:
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dst, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 16):
                if chunk:
                    f.write(chunk)
    return dst


_FFMPEG: str | None = None


def ffmpeg_bin() -> str:
    """Resolve a working ffmpeg executable. Prefer the bundled imageio-ffmpeg
    binary because Windows shims (e.g. Scoop) fail on non-ASCII argv."""
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


def ensure_ffmpeg() -> None:
    ffmpeg_bin()


def to_wav_16k_mono(src: Path, dst: Path) -> Path:
    """Convert any input audio to 16 kHz mono PCM WAV (Whisper's expected format)."""
    cmd = [
        ffmpeg_bin(), "-y", "-i", str(src),
        "-ac", "1", "-ar", "16000",
        "-vn", "-sn",
        "-c:a", "pcm_s16le",
        str(dst),
    ]
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if r.returncode != 0:
        sys.stderr.write(r.stderr.decode("utf-8", errors="replace"))
        raise RuntimeError(f"ffmpeg failed (exit {r.returncode}) on {src}")
    return dst


def slice_wav(src_wav: Path, start: float, end: float, dst: Path) -> None:
    """Cut [start,end] from a WAV losslessly via ffmpeg."""
    cmd = [
        ffmpeg_bin(), "-y",
        "-ss", f"{start:.3f}", "-to", f"{end:.3f}",
        "-i", str(src_wav),
        "-c", "copy",
        str(dst),
    ]
    # PCM WAV with stream copy works fine for sample-accurate-ish cuts at these scales.
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


# ---------- segmentation ----------

@dataclass
class Segment:
    start: float
    end: float
    text: str
    confidence: float | None = None


def split_long_segment(words, max_len: float, target_len: float) -> list[Segment]:
    """
    Split a Whisper segment whose duration > max_len into chunks ~target_len long,
    cutting at word boundaries. `words` is faster-whisper's per-word list.
    """
    if not words:
        return []

    chunks: list[Segment] = []
    cur_words = []
    cur_start = words[0].start

    for w in words:
        cur_words.append(w)
        dur = w.end - cur_start
        if dur >= target_len:
            text = "".join(x.word for x in cur_words).strip()
            probs = [x.probability for x in cur_words if x.probability is not None]
            conf = float(sum(probs) / len(probs)) if probs else None
            chunks.append(Segment(cur_start, w.end, text, conf))
            cur_words = []
            # next chunk starts at next word's start (set on next iteration)
            cur_start = None  # type: ignore

        elif cur_start is None:
            cur_start = w.start

    if cur_words:
        text = "".join(x.word for x in cur_words).strip()
        probs = [x.probability for x in cur_words if x.probability is not None]
        conf = float(sum(probs) / len(probs)) if probs else None
        start = cur_words[0].start
        chunks.append(Segment(start, cur_words[-1].end, text, conf))

    # Merge a tiny tail (< min) into the previous chunk if needed
    return chunks


def normalize_segments(
    raw_segments: Iterable,
    min_len: float,
    max_len: float,
    target_len: float,
) -> list[Segment]:
    """Apply the 10-30 s rule: split too-long segments, drop empty ones."""
    out: list[Segment] = []
    for seg in raw_segments:
        text = (seg.text or "").strip()
        if not text:
            continue
        dur = seg.end - seg.start
        avg_logprob = getattr(seg, "avg_logprob", None)
        # crude confidence proxy
        conf = float(min(max(0.0, 1.0 + avg_logprob), 1.0)) if avg_logprob is not None else None

        if dur <= max_len:
            out.append(Segment(float(seg.start), float(seg.end), text, conf))
            continue

        words = getattr(seg, "words", None) or []
        if words:
            out.extend(split_long_segment(words, max_len, target_len))
        else:
            # No word timestamps — fall back to uniform cuts
            n = max(1, int(round(dur / target_len)))
            step = dur / n
            chunk_text = text  # we don't know where words fall, so duplicate text or blank
            for i in range(n):
                s = seg.start + i * step
                e = seg.start + (i + 1) * step if i < n - 1 else seg.end
                out.append(Segment(float(s), float(e), chunk_text if i == 0 else "", conf))
    return out


# ---------- main pipeline ----------

def _add_nvidia_dll_dirs() -> None:
    """Make the bundled nvidia-cublas/cudnn DLLs discoverable on Windows."""
    if os.name != "nt":
        return
    try:
        import importlib.util
        for pkg in ("nvidia.cublas", "nvidia.cudnn"):
            spec = importlib.util.find_spec(pkg)
            if not spec or not spec.submodule_search_locations:
                continue
            for loc in spec.submodule_search_locations:
                bin_dir = Path(loc) / "bin"
                if bin_dir.is_dir():
                    os.add_dll_directory(str(bin_dir))
    except Exception:
        pass


def transcribe(
    wav_path: Path,
    model_size: str,
    language: str | None,
    device: str,
    compute_type: str,
):
    _add_nvidia_dll_dirs()
    from faster_whisper import WhisperModel

    print(f"[whisper] loading model={model_size} device={device} compute_type={compute_type}", flush=True)
    try:
        model = WhisperModel(model_size, device=device, compute_type=compute_type)
    except Exception as e:
        if device == "cuda":
            print(f"[whisper] CUDA init failed ({e}); falling back to CPU/int8.", flush=True)
            device, compute_type = "cpu", "int8"
            model = WhisperModel(model_size, device=device, compute_type=compute_type)
        else:
            raise

    segments, info = model.transcribe(
        str(wav_path),
        language=language,            # None = auto-detect
        task="transcribe",
        vad_filter=True,              # speech-activity detection
        vad_parameters={"min_silence_duration_ms": 500},
        word_timestamps=True,         # needed for clean re-splitting
        beam_size=1,                  # weak transcription, keep it fast
        condition_on_previous_text=False,
    )
    print(f"[whisper] detected language={info.language} (p={info.language_probability:.2f}) "
          f"duration={info.duration:.1f}s", flush=True)
    return list(tqdm(segments, desc="decoding", unit="seg"))


def pick_device(arg: str) -> tuple[str, str]:
    """Return (device, compute_type). Auto = use CUDA if available, else CPU."""
    if arg == "cpu":
        return "cpu", "int8"
    if arg == "cuda":
        return "cuda", "float16"
    # auto
    try:
        import ctranslate2
        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda", "float16"
    except Exception:
        pass
    return "cpu", "int8"


def main() -> None:
    ap = argparse.ArgumentParser(description="Weak transcription + segmentation for Label Studio.")
    ap.add_argument("input", help="Path or URL to an audio file (mp3/wav/m4a/...).")
    ap.add_argument("--output", "-o", default="./dataset", help="Output directory.")
    ap.add_argument("--language", default="mg",
                    help="Language code (e.g. 'mg' for Malagasy, 'fr', 'en'). "
                         "Use 'auto' to let Whisper detect.")
    ap.add_argument("--model", default="large-v3",
                    help="faster-whisper model size: tiny|base|small|medium|large-v3|...")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--min-seg", type=float, default=2.0, help="Drop segments shorter than this (s).")
    ap.add_argument("--max-seg", type=float, default=30.0, help="Re-split segments longer than this (s).")
    ap.add_argument("--target-seg", type=float, default=20.0, help="Target chunk length when splitting (s).")
    ap.add_argument("--no-clips", action="store_true", help="Skip per-segment audio export.")
    ap.add_argument("--audio-url-prefix", default="",
                    help="Prefix to prepend to clip filenames in the JSON 'audio' field "
                         "(e.g. '/data/local-files/?d=clips/' for Label Studio local storage).")
    args = ap.parse_args()

    ensure_ffmpeg()

    out_dir = Path(args.output).resolve()
    clips_dir = out_dir / "clips"
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_clips:
        clips_dir.mkdir(exist_ok=True)

    language = None if args.language.lower() == "auto" else args.language

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)

        # 1. fetch input
        if is_url(args.input):
            ext = Path(urlparse(args.input).path).suffix or ".bin"
            src = download(args.input, tmp_dir / f"input{ext}")
            stem = Path(urlparse(args.input).path).stem or "audio"
        else:
            src = Path(args.input).resolve()
            if not src.exists():
                sys.exit(f"Input not found: {src}")
            stem = src.stem

        # 2. normalize to 16k mono wav
        # Copy to an ASCII-safe path first — Windows + non-ANSI filenames break
        # subprocess argv encoding to ffmpeg.
        safe_src = tmp_dir / f"input{src.suffix.lower() or '.bin'}"
        shutil.copyfile(src, safe_src)
        # Sanitize stem for output filenames too.
        ascii_stem = "".join(c if (c.isalnum() or c in "-_") else "_" for c in stem).strip("_") or "audio"
        stem = ascii_stem
        wav = tmp_dir / f"{stem}.16k.wav"
        print(f"[ffmpeg] {src.name} -> 16k mono wav", flush=True)
        to_wav_16k_mono(safe_src, wav)

        # 3. transcribe (with CUDA->CPU fallback if cuBLAS/cuDNN are missing
        # or the GPU run fails mid-decode)
        device, compute_type = pick_device(args.device)
        try:
            raw = transcribe(wav, args.model, language, device, compute_type)
        except RuntimeError as e:
            if device == "cuda":
                print(f"[whisper] CUDA decode failed ({e}); retrying on CPU.", flush=True)
                raw = transcribe(wav, args.model, language, "cpu", "int8")
            else:
                raise

        # 4. normalize lengths
        segs = normalize_segments(
            raw, min_len=args.min_seg, max_len=args.max_seg, target_len=args.target_seg
        )
        segs = [s for s in segs if (s.end - s.start) >= args.min_seg]
        print(f"[seg] {len(raw)} raw -> {len(segs)} normalized segments", flush=True)

        # 5. write clips + JSON
        tasks = []
        for i, s in enumerate(tqdm(segs, desc="writing", unit="clip")):
            clip_name = f"{stem}_{i:05d}_{int(s.start*1000):08d}_{int(s.end*1000):08d}.wav"
            if not args.no_clips:
                slice_wav(wav, s.start, s.end, clips_dir / clip_name)
            audio_ref = f"{args.audio_url_prefix}{clip_name}" if args.audio_url_prefix else clip_name
            task = {
                "audio": audio_ref,
                "start": round(s.start, 3),
                "end": round(s.end, 3),
                "text": s.text,
            }
            if s.confidence is not None:
                task["confidence"] = round(s.confidence, 3)
            tasks.append(task)

        json_path = out_dir / f"{stem}.label_studio.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(tasks, f, ensure_ascii=False, indent=2)

        print(f"[done] {len(tasks)} segments")
        print(f"       JSON  : {json_path}")
        if not args.no_clips:
            print(f"       clips : {clips_dir}")


if __name__ == "__main__":
    main()
