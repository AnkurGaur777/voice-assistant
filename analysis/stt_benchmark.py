"""
Local Jarvis - Speech-to-Text Benchmark Suite (Phase 2)

This benchmark evaluates faster-whisper models (tiny, base, small) on CPU.
It records model loading time, transcription latency, real-time factor (RTF),
and transcribed output, saving all metrics to `analysis/stt_benchmark_results.csv`.
"""

import argparse
import csv
import datetime
import os
import sys
import time
import wave
from pathlib import Path
from typing import List, Optional

# Suppress HuggingFace Windows symlink warnings
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Add project root to sys.path so we can import src.stt cleanly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.stt import WhisperTranscriber, DEFAULT_AUDIO_CACHE_DIR, DEFAULT_MODEL_DIR

DEFAULT_CSV_PATH = PROJECT_ROOT / "analysis" / "stt_benchmark_results.csv"
DEFAULT_MODELS = ["tiny", "base", "small"]


def get_audio_duration(file_path: Path) -> float:
    """Calculates the duration of a .wav audio file in seconds."""
    with wave.open(str(file_path), "rb") as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        return frames / float(rate)


def run_benchmark(
    audio_path: Path,
    models: List[str] = DEFAULT_MODELS,
    device: str = "cpu",
    compute_type: str = "int8",
    cpu_threads: int = 4,
    csv_path: Path = DEFAULT_CSV_PATH,
    language: Optional[str] = "en",
) -> List[dict]:
    """
    Runs STT benchmark across the specified models on a given audio clip.
    Logs results to a CSV file and returns the result records.
    """
    if not audio_path.is_file():
        raise FileNotFoundError(f"Sample audio clip not found: {audio_path}")

    audio_duration = get_audio_duration(audio_path)
    print("\n" + "=" * 80)
    print(" FASTER-WHISPER CPU BENCHMARK (PHASE 2)")
    print("=" * 80)
    print(f"Target Audio Clip:  {audio_path.name}")
    print(f"Audio Duration:     {audio_duration:.2f} seconds")
    print(f"Device:             {device.upper()} (GPU reserved for LLM)")
    print(f"Compute Type:       {compute_type}")
    print(f"CPU Worker Threads: {cpu_threads}")
    print(f"Models to Evaluate: {', '.join(models)}")
    print(f"Results CSV:        {csv_path}")
    print("=" * 80 + "\n")

    results = []

    for model_name in models:
        print(f">>> Evaluating model: '{model_name}' ...")

        # Measure Model Load Time
        load_start = time.perf_counter()
        transcriber = WhisperTranscriber(
            model_size=model_name,
            device=device,
            compute_type=compute_type,
            cpu_threads=cpu_threads,
            download_root=DEFAULT_MODEL_DIR,
        )
        load_time = time.perf_counter() - load_start

        # Warm-up / Run Transcription Inference
        infer_start = time.perf_counter()
        transcribed_text = transcriber.transcribe(
            audio_path=audio_path,
            language=language,
        )
        infer_latency = time.perf_counter() - infer_start

        # Real-time factor (RTF): latency / audio duration (< 1.0 means faster than real-time)
        rtf = infer_latency / audio_duration if audio_duration > 0 else 0.0

        record = {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model_name": model_name,
            "device": device,
            "compute_type": compute_type,
            "cpu_threads": cpu_threads,
            "audio_file": audio_path.name,
            "audio_duration_s": round(audio_duration, 3),
            "load_time_s": round(load_time, 3),
            "latency_s": round(infer_latency, 3),
            "rtf": round(rtf, 3),
            "transcribed_text": transcribed_text,
        }
        results.append(record)

        print(f"    [+] Load Time: {load_time:.2f}s | Latency: {infer_latency:.3f}s | RTF: {rtf:.3f}x")
        print(f"    [+] Text: \"{transcribed_text}\"\n")

    # Append results to CSV
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.is_file()

    fieldnames = [
        "timestamp",
        "model_name",
        "device",
        "compute_type",
        "cpu_threads",
        "audio_file",
        "audio_duration_s",
        "load_time_s",
        "latency_s",
        "rtf",
        "transcribed_text",
    ]

    with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for rec in results:
            writer.writerow(rec)

    print(f"Successfully logged {len(results)} benchmark records to: {csv_path.resolve()}\n")
    display_summary_table(results, audio_duration)
    return results


def display_summary_table(results: List[dict], audio_duration: float) -> None:
    """Prints a clear comparison table of benchmark results."""
    print("=" * 95)
    print(f"{'Model':<8} | {'Load (s)':<9} | {'Latency (s)':<12} | {'RTF':<7} | {'Speed vs RT':<12} | {'Transcribed Text'}")
    print("-" * 95)

    for r in results:
        rtf = r["rtf"]
        speed_factor = f"{1.0 / rtf:.1f}x faster" if rtf > 0 else "N/A"
        # Truncate text if very long for table formatting
        text_disp = (r["transcribed_text"][:40] + "...") if len(r["transcribed_text"]) > 43 else r["transcribed_text"]
        print(
            f"{r['model_name']:<8} | "
            f"{r['load_time_s']:<9.2f} | "
            f"{r['latency_s']:<12.3f} | "
            f"{r['rtf']:<7.3f} | "
            f"{speed_factor:<12} | "
            f"\"{text_disp}\""
        )
    print("=" * 95)
    print(" * RTF (Real-Time Factor) = Latency / Audio Duration. Lower is faster.")
    print(" * Speed vs RT = How many times faster than actual speech playback.")
    print("=" * 95 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Local Jarvis - STT Whisper Model Benchmark (tiny, base, small)"
    )
    parser.add_argument(
        "--audio",
        "-a",
        type=str,
        default=None,
        help="Path to .wav sample clip. Defaults to largest/latest sample from audio_cache/.",
    )
    parser.add_argument(
        "--models",
        "-m",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Whisper models to benchmark (default: tiny base small).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Inference device (default: cpu).",
    )
    parser.add_argument(
        "--compute-type",
        type=str,
        default="int8",
        help="Quantization type (default: int8).",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="CPU worker threads (default: 4).",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=str(DEFAULT_CSV_PATH),
        help="Path to results CSV (default: analysis/stt_benchmark_results.csv).",
    )
    parser.add_argument(
        "--language",
        "-l",
        type=str,
        default="en",
        help="Spoken language code (default: en).",
    )
    args = parser.parse_args()

    # Determine audio sample
    target_audio_path = None
    if args.audio:
        target_audio_path = Path(args.audio).resolve()
    else:
        if DEFAULT_AUDIO_CACHE_DIR.exists():
            wav_files = sorted(
                DEFAULT_AUDIO_CACHE_DIR.glob("*.wav"),
                key=lambda p: p.stat().st_size,
                reverse=True,
            )
            if wav_files:
                target_audio_path = wav_files[0]
                print(f"Auto-selected sample clip with speech: {target_audio_path}")

    if not target_audio_path or not target_audio_path.is_file():
        print(
            "Error: No audio clip specified and none found in audio_cache/.",
            file=sys.stderr,
        )
        sys.exit(1)

    run_benchmark(
        audio_path=target_audio_path,
        models=args.models,
        device=args.device,
        compute_type=args.compute_type,
        cpu_threads=args.threads,
        csv_path=Path(args.csv),
        language=args.language,
    )


if __name__ == "__main__":
    main()
