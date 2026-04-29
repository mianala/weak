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
#   "nvidia-cuda-runtime-cu12; platform_system != 'Darwin'",
#   # Optional: HuggingFace CTC ASR (e.g. w2v-bert-2.0-malagasy-asr) for the
#   # `--asr-model` hybrid path. Only loaded if that flag is set.
#   "transformers>=4.45.0",
#   "torch>=2.3.0",
#   "soundfile>=0.12.1",
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


def slice_wav(
    src_wav: Path,
    start: float,
    end: float,
    dst: Path,
    pad_head: float = 0.15,
    pad_tail: float = 0.25,
    total_duration: float | None = None,
) -> None:
    """Cut [start,end] from a WAV with small head/tail padding so words at the
    boundary aren't truncated. Re-encodes to PCM s16le for sample-accurate cuts
    (ffmpeg -c copy is only frame-accurate)."""
    s = max(0.0, start - pad_head)
    e = end + pad_tail
    if total_duration is not None:
        e = min(e, total_duration)
    # ffmpeg won't create missing parent dirs and reports the failure as a
    # generic ENOENT — make sure the destination directory exists.
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg_bin(), "-y",
        "-ss", f"{s:.3f}", "-to", f"{e:.3f}",
        "-i", str(src_wav),
        "-ac", "1", "-ar", "16000",
        "-c:a", "pcm_s16le",
        str(dst),
    ]
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if r.returncode != 0:
        sys.stderr.write(r.stderr.decode("utf-8", errors="replace"))
        raise RuntimeError(f"ffmpeg slice failed (exit {r.returncode}) at {start:.2f}-{end:.2f}")


# ---------- segmentation ----------

@dataclass
class Segment:
    start: float
    end: float
    text: str
    confidence: float | None = None
    # Filled when --asr-model is set. We keep Whisper's draft (above) AND the
    # CTC ASR's draft so reviewers in Label Studio see both versions.
    text_badrex: str | None = None
    confidence_badrex: float | None = None


def _emit_chunk(words_slice) -> Segment:
    text = "".join(x.word for x in words_slice).strip()
    probs = [x.probability for x in words_slice if x.probability is not None]
    conf = float(sum(probs) / len(probs)) if probs else None
    # Use the *midpoint of the gap* on each side so adjacent chunks meet in
    # silence rather than straddling a word boundary. The caller sets the actual
    # start/end after looking at neighbouring words.
    return Segment(float(words_slice[0].start), float(words_slice[-1].end), text, conf)


def split_long_segment(
    words,
    min_len: float,
    max_len: float,
    target_len: float,
) -> list[Segment]:
    """
    Split a Whisper segment whose duration > max_len into chunks aiming for
    ~target_len, but cutting at the *largest inter-word silence* inside the
    [min_len, max_len] window — never inside a tight word boundary.

    Boundary timestamps are placed at the *midpoint* of the chosen silence so
    neighbouring clips don't fight over the same audio.
    """
    if not words:
        return []

    n = len(words)
    chunks: list[Segment] = []
    i = 0  # index of first word in current chunk
    chunk_start = float(words[0].start)

    while i < n:
        # Find the latest word j such that words[j].end - chunk_start <= max_len.
        # Inside [min_len, max_len], pick the gap (between j and j+1) with the
        # largest silence; prefer gaps near target_len when silences tie.
        best_j = None
        best_score = -1.0  # silence size, tie-broken by closeness to target

        j = i
        while j < n:
            dur = float(words[j].end) - chunk_start
            if dur > max_len:
                break
            if dur >= min_len and j + 1 < n:
                gap = float(words[j + 1].start) - float(words[j].end)
                # closeness to target ∈ [0,1]; bigger is better
                proximity = 1.0 - min(abs(dur - target_len) / max(target_len, 1e-3), 1.0)
                # silence dominates; proximity is a tiebreaker
                score = gap + 0.05 * proximity
                if score > best_score:
                    best_score = score
                    best_j = j
            j += 1

        if best_j is None:
            # No acceptable gap found inside the window. Two cases:
            #  a) the remaining tail fits within max_len → emit it as one chunk.
            #  b) a single word/run is longer than max_len → force-cut at j-1
            #     to make progress (rare for natural speech).
            if j >= n:
                # tail fits
                chunk = _emit_chunk(words[i:n])
                # Snap start to chunk_start (which may already be a midpoint)
                chunk.start = chunk_start
                chunks.append(chunk)
                break
            else:
                # forced cut: take everything up to j-1 (or just word i if j==i)
                cut = max(j - 1, i)
                chunk = _emit_chunk(words[i:cut + 1])
                chunk.start = chunk_start
                # boundary at midpoint of gap to next word (if any), else end
                if cut + 1 < n:
                    mid = (float(words[cut].end) + float(words[cut + 1].start)) / 2.0
                    chunk.end = mid
                    chunk_start = mid
                else:
                    chunk_start = float(words[cut].end)
                chunks.append(chunk)
                i = cut + 1
                continue

        # Normal path: cut at midpoint of gap after best_j
        chunk = _emit_chunk(words[i:best_j + 1])
        chunk.start = chunk_start
        mid = (float(words[best_j].end) + float(words[best_j + 1].start)) / 2.0
        chunk.end = mid
        chunks.append(chunk)
        chunk_start = mid
        i = best_j + 1

    # Soft-merge: if the last chunk is shorter than min_len, fold it into the
    # previous one rather than dropping it.
    if len(chunks) >= 2 and (chunks[-1].end - chunks[-1].start) < min_len:
        tail = chunks.pop()
        prev = chunks[-1]
        merged_text = (prev.text + " " + tail.text).strip()
        # average confidence weighted by duration
        d1 = prev.end - prev.start
        d2 = tail.end - tail.start
        if prev.confidence is not None and tail.confidence is not None and (d1 + d2) > 0:
            conf = (prev.confidence * d1 + tail.confidence * d2) / (d1 + d2)
        else:
            conf = prev.confidence if prev.confidence is not None else tail.confidence
        chunks[-1] = Segment(prev.start, tail.end, merged_text, conf)

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
            out.extend(split_long_segment(words, min_len, max_len, target_len))
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


# ---------- silence snapping ----------

def _rms_min_time(audio, sr: int, t0: float, t1: float, frame_ms: int = 20) -> float | None:
    """Return the timestamp of the lowest-energy short frame in audio[t0:t1].
    Returns None if the window is too short to evaluate."""
    import numpy as np
    n = len(audio)
    i0 = max(0, int(t0 * sr))
    i1 = min(n, int(t1 * sr))
    frame_len = max(1, int(sr * frame_ms / 1000))
    hop = max(1, frame_len // 2)
    if i1 - i0 < frame_len * 2:
        return None
    # vectorised: sliding window via stride trick on a power-of-2 frame
    starts = np.arange(i0, i1 - frame_len + 1, hop)
    if len(starts) < 2:
        return None
    # compute RMS per frame
    rms = np.empty(len(starts), dtype=np.float32)
    for k, a in enumerate(starts):
        seg = audio[a:a + frame_len]
        rms[k] = float(np.sqrt(np.mean(seg.astype(np.float32) ** 2)))
    best = int(np.argmin(rms))
    return float((starts[best] + frame_len / 2.0) / sr)


def snap_boundaries_to_silence(
    segs: list[Segment],
    audio,
    sr: int,
    radius: float = 0.30,
    max_shift: float = 0.40,
) -> None:
    """Adjust every internal boundary between consecutive segments to land at
    the lowest-energy frame within ±radius seconds of the current cut.

    Mutates `segs` in place. The first segment's start and the last segment's
    end are left alone (they're at the audio edges)."""
    if len(segs) < 2:
        return
    moved = 0
    for i in range(len(segs) - 1):
        a = segs[i]
        b = segs[i + 1]
        cut = (a.end + b.start) / 2.0
        # Search a window around the cut, but don't cross adjacent segments' midpoints.
        lo = max(a.start + 0.05, cut - radius)
        hi = min(b.end - 0.05, cut + radius)
        if hi <= lo:
            continue
        snapped = _rms_min_time(audio, sr, lo, hi)
        if snapped is None:
            continue
        if abs(snapped - cut) > max_shift:
            continue  # silence too far from intended cut — Whisper probably knew better
        a.end = snapped
        b.start = snapped
        moved += 1
    print(f"[snap] adjusted {moved}/{len(segs) - 1} internal boundaries to nearest silence", flush=True)


# ---------- main pipeline ----------

def _add_nvidia_dll_dirs() -> None:
    """Make the bundled nvidia-* DLLs discoverable on Windows.

    ctranslate2 dynamically loads cuBLAS/cuDNN with plain LoadLibrary, which
    doesn't honor `os.add_dll_directory` (that only applies when the caller
    passes LOAD_LIBRARY_SEARCH_USER_DIRS). So we also prepend the dirs to
    PATH, which every LoadLibrary call respects."""
    if os.name != "nt":
        return
    bin_dirs: list[str] = []
    try:
        import importlib.util
        # Walk all top-level subpackages under `nvidia` and pick up any /bin dir.
        nvidia_spec = importlib.util.find_spec("nvidia")
        roots: list[Path] = []
        if nvidia_spec and nvidia_spec.submodule_search_locations:
            roots = [Path(p) for p in nvidia_spec.submodule_search_locations]
        for root in roots:
            for child in root.iterdir():
                bin_dir = child / "bin"
                if bin_dir.is_dir():
                    bin_dirs.append(str(bin_dir))
    except Exception:
        pass

    for d in bin_dirs:
        try:
            os.add_dll_directory(d)
        except (OSError, FileNotFoundError):
            pass
    if bin_dirs:
        os.environ["PATH"] = os.pathsep.join(bin_dirs) + os.pathsep + os.environ.get("PATH", "")


# Anchor for native model objects so ctranslate2's CUDA destructor doesn't run
# at function-return time — it segfaults during cleanup of large-v3/CUDA fp16
# state, taking the whole process with it (exit 127, no Python traceback).
# We pair this with os._exit() at the end of main() to skip interpreter
# shutdown entirely.
_LIVE_MODELS: list = []


def transcribe(
    wav_path: Path,
    model_size: str,
    language: str | None,
    device: str,
    compute_type: str,
    require_gpu: bool = False,
):
    from faster_whisper import WhisperModel

    print(f"[whisper] loading model={model_size} device={device} compute_type={compute_type}", flush=True)
    try:
        model = WhisperModel(model_size, device=device, compute_type=compute_type)
    except Exception as e:
        if device == "cuda" and not require_gpu:
            print(f"[whisper] CUDA init failed ({e}); falling back to CPU/int8.", flush=True)
            device, compute_type = "cpu", "int8"
            model = WhisperModel(model_size, device=device, compute_type=compute_type)
        else:
            raise
    _LIVE_MODELS.append(model)

    segments, info = model.transcribe(
        str(wav_path),
        language=language,            # None = auto-detect
        task="transcribe",
        vad_filter=True,              # speech-activity detection
        vad_parameters={"min_silence_duration_ms": 800},
        word_timestamps=True,         # needed for clean re-splitting
        beam_size=1,                  # weak transcription, keep it fast
        condition_on_previous_text=False,
    )
    print(f"[whisper] detected language={info.language} (p={info.language_probability:.2f}) "
          f"duration={info.duration:.1f}s", flush=True)

    # Materialize the generator carefully. Each segment is a frozen-ish object
    # with .start/.end/.text/.avg_logprob/.words; convert to a plain dict so
    # nothing native is held when ctranslate2's GPU buffers are torn down.
    out = []
    for seg in tqdm(segments, desc="decoding", unit="seg"):
        words = []
        for w in (getattr(seg, "words", None) or []):
            words.append(type("W", (), {
                "start": float(w.start), "end": float(w.end),
                "word": w.word, "probability": getattr(w, "probability", None),
            })())
        out.append(type("S", (), {
            "start": float(seg.start), "end": float(seg.end),
            "text": seg.text,
            "avg_logprob": getattr(seg, "avg_logprob", None),
            "words": words,
        })())
    print(f"[whisper] decoding finished: {len(out)} raw segments", flush=True)
    return out


# ---------- optional CTC re-transcriber (e.g. w2v-bert-2.0-malagasy-asr) ----------

class CTCTranscriber:
    """Loads a HuggingFace CTC ASR model (e.g. Wav2Vec2-BERT) and re-transcribes
    audio chunks. Used to replace Whisper's weak text with output from a model
    fine-tuned on the target language.

    Both the model id and the language are runtime parameters — nothing is
    hardcoded — so this works for any HF CTC ASR fine-tune."""

    def __init__(self, model_id: str, device: str, language_hint: str | None = None):
        import torch
        from transformers import AutoProcessor, AutoModelForCTC

        self.torch = torch
        self.device = "cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.language_hint = language_hint  # informational only; CTC models are usually monolingual

        print(f"[asr] loading model={model_id} device={self.device} dtype={self.dtype}", flush=True)
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForCTC.from_pretrained(model_id, torch_dtype=self.dtype).to(self.device).eval()
        # Sample rate the processor expects (almost always 16000 for w2v-bert).
        fe = getattr(self.processor, "feature_extractor", self.processor)
        self.sample_rate = int(getattr(fe, "sampling_rate", 16000))

    def transcribe_chunk(self, audio_f32: "np.ndarray") -> tuple[str, float | None]:
        """Run one forward pass on a 1-D float32 numpy array. Returns (text, confidence)."""
        torch = self.torch
        inputs = self.processor(
            audio_f32, sampling_rate=self.sample_rate, return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        # input_features for w2v-bert (Wav2Vec2BertProcessor) or input_values for plain w2v2.
        if self.dtype == torch.float16:
            for k in ("input_features", "input_values"):
                if k in inputs:
                    inputs[k] = inputs[k].to(torch.float16)

        with torch.inference_mode():
            logits = self.model(**inputs).logits  # (1, T, V)
        pred_ids = logits.argmax(dim=-1)
        text = self.processor.batch_decode(pred_ids)[0].strip()

        # Confidence proxy: mean max-softmax over predicted frames.
        probs = torch.softmax(logits.float(), dim=-1)
        conf = float(probs.max(dim=-1).values.mean().item())
        return text, conf


def load_master_audio_f32(wav_path: Path) -> tuple["np.ndarray", int]:
    """Load the 16 kHz mono master WAV into a float32 numpy array in [-1, 1]."""
    import wave
    import numpy as np
    with wave.open(str(wav_path), "rb") as wf:
        sr = wf.getframerate()
        n = wf.getnframes()
        raw = wf.readframes(n)
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return audio, sr


def retranscribe_segments(
    segs: list[Segment],
    master_audio,
    sample_rate: int,
    transcriber: CTCTranscriber,
    pad_head: float = 0.15,
    pad_tail: float = 0.25,
) -> None:
    """Add a CTC-ASR draft (text_badrex / confidence_badrex) to each segment
    alongside Whisper's draft. Mutates `segs` in place."""
    total = len(master_audio)
    for s in tqdm(segs, desc="asr", unit="seg"):
        i0 = max(0, int((s.start - pad_head) * sample_rate))
        i1 = min(total, int((s.end + pad_tail) * sample_rate))
        if i1 <= i0:
            continue
        chunk = master_audio[i0:i1]
        try:
            text, conf = transcriber.transcribe_chunk(chunk)
        except Exception as e:
            print(f"[asr] segment {s.start:.2f}-{s.end:.2f} failed: {e}", flush=True)
            continue
        s.text_badrex = text
        s.confidence_badrex = conf


def pick_device(arg: str) -> tuple[str, str]:
    """Return (device, compute_type). Auto = use CUDA if available, else CPU.

    int8_float16 is preferred on CUDA over plain float16 — it uses less VRAM
    and avoids a stability bug we hit in ctranslate2's float16 cleanup path
    when word_timestamps is enabled on long-form audio (process crashes with
    no Python traceback right after the last segment is emitted)."""
    if arg == "cpu":
        return "cpu", "int8"
    if arg == "cuda":
        return "cuda", "int8_float16"
    # auto
    try:
        import ctranslate2
        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda", "int8_float16"
    except Exception:
        pass
    return "cpu", "int8"


def main() -> None:
    # Must run BEFORE anything imports ctranslate2 (e.g. pick_device's cuda check),
    # otherwise the native lib is loaded with the original DLL search path and
    # cublas64_12.dll / cudnn64_9.dll won't be findable later.
    _add_nvidia_dll_dirs()

    ap = argparse.ArgumentParser(description="Weak transcription + segmentation for Label Studio.")
    ap.add_argument("input", help="Path or URL to an audio file (mp3/wav/m4a/...).")
    ap.add_argument("--output", "-o", default="./dataset", help="Output directory.")
    ap.add_argument("--language", default="mg",
                    help="Language code (e.g. 'mg' for Malagasy, 'fr', 'en'). "
                         "Use 'auto' to let Whisper detect.")
    ap.add_argument("--model", default="large-v3",
                    help="faster-whisper model size: tiny|base|small|medium|large-v3|...")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--require-gpu", action="store_true",
                    help="Abort with an error if the GPU isn't usable, instead of falling back to CPU.")
    ap.add_argument("--asr-model", default="BadRex/w2v-bert-2.0-malagasy-asr",
                    help="HuggingFace CTC ASR model id used to add a second draft "
                         "alongside Whisper's (Whisper still does VAD + segmentation). "
                         "Defaults to BadRex/w2v-bert-2.0-malagasy-asr. "
                         "Pass --asr-model '' to disable and keep Whisper-only output.")
    ap.add_argument("--min-seg", type=float, default=2.0, help="Drop segments shorter than this (s).")
    ap.add_argument("--max-seg", type=float, default=30.0, help="Re-split segments longer than this (s).")
    ap.add_argument("--target-seg", type=float, default=20.0, help="Target chunk length when splitting (s).")
    ap.add_argument("--no-clips", action="store_true", help="Skip per-segment audio export.")
    ap.add_argument("--pad-head", type=float, default=0.15,
                    help="Seconds of audio to keep before each clip's start (default 0.15).")
    ap.add_argument("--pad-tail", type=float, default=0.25,
                    help="Seconds of audio to keep after each clip's end (default 0.25).")
    ap.add_argument("--no-snap", action="store_true",
                    help="Disable snapping segment boundaries to the lowest-energy point "
                         "in the surrounding waveform. Boundaries from Whisper alone are "
                         "often off by 50-200ms and chop word tails.")
    ap.add_argument("--snap-radius", type=float, default=0.30,
                    help="Search ±this many seconds around each boundary for true silence "
                         "(default 0.30).")
    ap.add_argument("--snap-max-shift", type=float, default=0.40,
                    help="Don't move a boundary by more than this many seconds even if a "
                         "lower-energy point exists further away (default 0.40).")
    ap.add_argument("--audio-url-prefix", default="",
                    help="Prefix to prepend to clip filenames in the JSON 'audio' field "
                         "(e.g. '/data/local-files/?d=clips/' for Label Studio local storage).")
    args = ap.parse_args()

    ensure_ffmpeg()

    out_root = Path(args.output).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

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

        # One folder per source: clips and JSON live together inside it.
        project_dir = out_root / stem
        if project_dir.exists() and not args.no_clips:
            # Drop stale clips from prior runs — they don't match new segment indices.
            for old in project_dir.glob("*.wav"):
                old.unlink()
        project_dir.mkdir(parents=True, exist_ok=True)

        wav = tmp_dir / f"{stem}.16k.wav"
        print(f"[ffmpeg] {src.name} -> 16k mono wav", flush=True)
        to_wav_16k_mono(safe_src, wav)

        # Probe total duration so we can clamp clip padding to the file end.
        try:
            import wave
            with wave.open(str(wav), "rb") as wf:
                total_duration = wf.getnframes() / float(wf.getframerate())
        except Exception:
            total_duration = None

        # 3. transcribe. CUDA failures fall back to CPU unless --require-gpu is set.
        device, compute_type = pick_device(args.device)
        if args.require_gpu and device != "cuda":
            sys.exit(f"--require-gpu set but no usable CUDA device was found (device={device}). "
                     f"Check `nvidia-smi` and that nvidia-cublas-cu12 / nvidia-cudnn-cu12 are installed.")
        try:
            raw = transcribe(wav, args.model, language, device, compute_type,
                             require_gpu=args.require_gpu)
        except RuntimeError as e:
            if device == "cuda" and not args.require_gpu:
                print(f"[whisper] CUDA decode failed ({e}); retrying on CPU.", flush=True)
                raw = transcribe(wav, args.model, language, "cpu", "int8")
            else:
                # In require-gpu mode, surface the original CUDA error verbatim.
                raise

        # 4. normalize lengths
        print(f"[seg] decoding done; {len(raw)} raw segments. normalizing...", flush=True)
        segs = normalize_segments(
            raw, min_len=args.min_seg, max_len=args.max_seg, target_len=args.target_seg
        )
        print(f"[seg] normalize_segments returned {len(segs)} segments. filtering...", flush=True)
        segs = [s for s in segs if (s.end - s.start) >= args.min_seg]
        print(f"[seg] {len(raw)} raw -> {len(segs)} normalized segments", flush=True)

        # 4a. snap every internal boundary to the local silence in the actual
        # waveform. Whisper's reported word/segment ends are off by 50-200ms,
        # which is enough to clip the tail of words like "exactement".
        if not args.no_snap:
            master_audio, sr = load_master_audio_f32(wav)
            snap_boundaries_to_silence(
                segs, master_audio, sr,
                radius=args.snap_radius, max_shift=args.snap_max_shift,
            )

        # 4b. optional: re-transcribe each segment with a language-specific CTC
        # ASR (e.g. w2v-bert-2.0-malagasy-asr). Whisper's text is overwritten.
        if args.asr_model:
            transcriber = CTCTranscriber(
                args.asr_model,
                device=("cuda" if device == "cuda" else "cpu"),
                language_hint=language,
            )
            if 'master_audio' not in locals():
                master_audio, sr = load_master_audio_f32(wav)
            if sr != transcriber.sample_rate:
                sys.exit(f"Master WAV is {sr} Hz but ASR model expects {transcriber.sample_rate} Hz.")
            retranscribe_segments(
                segs, master_audio, sr, transcriber,
                pad_head=args.pad_head, pad_tail=args.pad_tail,
            )

        # 5. write clips + JSON
        tasks = []
        for i, s in enumerate(tqdm(segs, desc="writing", unit="clip")):
            clip_name = f"{stem}_{i:05d}_{int(s.start*1000):08d}_{int(s.end*1000):08d}.wav"
            if not args.no_clips:
                slice_wav(
                    wav, s.start, s.end, project_dir / clip_name,
                    pad_head=args.pad_head, pad_tail=args.pad_tail,
                    total_duration=total_duration,
                )
            audio_ref = f"{args.audio_url_prefix}{clip_name}" if args.audio_url_prefix else clip_name
            whisper_text = s.text or ""
            badrex_text = s.text_badrex or ""
            # Primary `text` is the BadRex draft when present (target-language
            # fine-tune wins over Whisper's noisy multilingual decode), else Whisper.
            primary_text = badrex_text if s.text_badrex is not None else whisper_text
            primary_conf = s.confidence_badrex if s.text_badrex is not None else s.confidence
            data = {
                "audio": audio_ref,
                "start": round(s.start, 3),
                "end": round(s.end, 3),
                "text": primary_text,
                "text_whisper": whisper_text,
            }
            if s.confidence is not None:
                data["confidence"] = round(primary_conf, 3) if primary_conf is not None else 0.0
                data["confidence_whisper"] = round(s.confidence, 3)
            if s.text_badrex is not None:
                data["text_badrex"] = badrex_text
                if s.confidence_badrex is not None:
                    data["confidence_badrex"] = round(s.confidence_badrex, 3)

            # One prediction per model so Label Studio shows both drafts side-by-side
            # under the same <TextArea>. The first prediction is the one Label Studio
            # pre-fills; we put the stronger model first.
            predictions = []
            if s.text_badrex is not None:
                predictions.append({
                    "model_version": args.asr_model,
                    "score": round(s.confidence_badrex, 3) if s.confidence_badrex is not None else 0.0,
                    "result": [{
                        "from_name": "transcription",
                        "to_name": "audio",
                        "type": "textarea",
                        "value": {"text": [badrex_text]},
                    }],
                })
            predictions.append({
                "model_version": f"whisper-{args.model}",
                "score": round(s.confidence, 3) if s.confidence is not None else 0.0,
                "result": [{
                    "from_name": "transcription",
                    "to_name": "audio",
                    "type": "textarea",
                    "value": {"text": [whisper_text]},
                }],
            })
            task = {"data": data, "predictions": predictions}
            tasks.append(task)

        json_path = project_dir / f"{stem}.label_studio.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(tasks, f, ensure_ascii=False, indent=2)

        print(f"[done] {len(tasks)} segments")
        print(f"       project: {project_dir}")
        print(f"       JSON   : {json_path.name}")

    # Skip interpreter shutdown to avoid the ctranslate2 CUDA destructor segfault.
    # All outputs are flushed to disk by this point; there's nothing left to clean.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
