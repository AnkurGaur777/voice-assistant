"""
Local Jarvis - Speech-to-Text (STT) Module (Phase 2)

This module provides CPU-based speech-to-text transcription using `faster-whisper`.
CPU inference (device="cpu", compute_type="int8") is used deliberately so that
the dedicated GPU remains entirely available for local LLM inference (e.g. Llama 3.1).
"""

import argparse
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Union

# Suppress Windows HuggingFace Hub symlink cache warning
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from faster_whisper import WhisperModel

# Default paths
BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_DIR = BASE_DIR / "models" / "whisper"
DEFAULT_AUDIO_CACHE_DIR = BASE_DIR / "audio_cache"

# Module-level transcriber cache for lazy singleton reuse across pipeline turns
_TRANSCRIBER_CACHE: Dict[str, "WhisperTranscriber"] = {}


def clean_transcription(text: str) -> str:
    """
    Post-processing step that strips leading short hallucinated fragments
    (under ~15 characters followed by a period, comma, or punctuation)
    that commonly appear as Whisper artifacts on short clips (e.g. 'at this,', 'of this.').

    :param text: Raw transcribed text from Whisper.
    :return: Cleaned text with leading artifacts removed.
    """
    if not text:
        return ""

    cleaned = text.strip()

    # Guard loop for up to 2 chained artifact fragments
    for _ in range(2):
        match = re.match(r"^([^.,;!?]{1,15}[.,;!?])\s+(.+)$", cleaned)
        if match:
            fragment, remainder = match.group(1), match.group(2).strip()
            if remainder:
                cleaned = remainder[0].upper() + remainder[1:] if len(remainder) > 1 else remainder.upper()
            else:
                break
        else:
            break

    return cleaned


class WhisperTranscriber:
    """
    Speech-to-Text engine powered by faster-whisper on CPU.
    """

    def __init__(
        self,
        model_size: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",
        cpu_threads: int = 4,
        download_root: Optional[Union[str, Path]] = DEFAULT_MODEL_DIR,
    ):
        """
        Initialize the Whisper model.

        :param model_size: Whisper model size ('tiny', 'base', 'small', 'medium', etc.).
        :param device: Inference device ('cpu' is required to reserve GPU for LLM).
        :param compute_type: Quantization type ('int8' recommended for fast CPU inference).
        :param cpu_threads: Number of CPU worker threads (defaults to 4).
        :param download_root: Directory where model weights are stored.
        """
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.cpu_threads = cpu_threads
        self.download_root = Path(download_root) if download_root else DEFAULT_MODEL_DIR

        self.download_root.mkdir(parents=True, exist_ok=True)

        print(
            f"Loading faster-whisper model '{self.model_size}' "
            f"(device={self.device}, compute_type={self.compute_type}, threads={self.cpu_threads})..."
        )
        load_start = time.perf_counter()
        self.model = WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type,
            cpu_threads=self.cpu_threads,
            download_root=str(self.download_root),
        )
        self.load_duration = time.perf_counter() - load_start
        print(f"Whisper model '{self.model_size}' loaded in {self.load_duration:.2f}s.")

    def transcribe(
        self,
        audio_path: Union[str, Path],
        language: Optional[str] = "en",
        beam_size: int = 5,
        vad_filter: bool = False,
    ) -> str:
        """
        Transcribes a given .wav audio file and returns the full recognized text.

        :param audio_path: Path to the .wav audio file.
        :param language: Spoken language code (e.g., 'en' for English, None for auto-detect).
        :param beam_size: Beam search size for decoding (default: 5).
        :param vad_filter: Enable faster-whisper internal VAD filtering if needed.
        :return: Transcribed text as a single stripped string with artifact cleanup.
        """
        path = Path(audio_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Audio file not found: {path}")

        if path.stat().st_size == 0:
            print(f"Warning: Audio file {path.name} is empty.")
            return ""

        # Run transcription on faster-whisper
        segments, info = self.model.transcribe(
            str(path),
            beam_size=beam_size,
            language=language,
            vad_filter=vad_filter,
            condition_on_previous_text=False,
        )

        # Collect segment texts
        transcription_parts = [segment.text for segment in segments]
        transcribed_text = " ".join(transcription_parts).strip()

        # Post-processing: strip leading short artifact fragments
        return clean_transcription(transcribed_text)


def get_transcriber(
    model_size: str = "base",
    device: str = "cpu",
    compute_type: str = "int8",
    cpu_threads: int = 4,
    download_root: Optional[Union[str, Path]] = DEFAULT_MODEL_DIR,
) -> WhisperTranscriber:
    """
    Returns a cached WhisperTranscriber instance to avoid reloading weights repeatedly.
    """
    cache_key = f"{model_size}_{device}_{compute_type}_{cpu_threads}"
    if cache_key not in _TRANSCRIBER_CACHE:
        _TRANSCRIBER_CACHE[cache_key] = WhisperTranscriber(
            model_size=model_size,
            device=device,
            compute_type=compute_type,
            cpu_threads=cpu_threads,
            download_root=download_root,
        )
    return _TRANSCRIBER_CACHE[cache_key]


def transcribe_audio(
    audio_path: Union[str, Path],
    model_size: str = "base",
    device: str = "cpu",
    compute_type: str = "int8",
    language: Optional[str] = "en",
    beam_size: int = 5,
) -> str:
    """
    Transcribes an audio clip (.wav file path) and returns the transcribed text.
    Uses a cached transcriber instance to ensure high performance in continuous usage.

    :param audio_path: Path to the .wav audio clip (e.g. from wake_word.py's audio_cache/).
    :param model_size: Whisper model size (default: "base").
    :param device: Inference device (default: "cpu").
    :param compute_type: Quantization (default: "int8").
    :param language: Spoken language (default: "en").
    :param beam_size: Beam search width (default: 5).
    :return: Transcribed text string.
    """
    transcriber = get_transcriber(
        model_size=model_size,
        device=device,
        compute_type=compute_type,
    )
    return transcriber.transcribe(
        audio_path=audio_path,
        language=language,
        beam_size=beam_size,
    )


def main():
    """CLI test harness for standalone STT transcription."""
    parser = argparse.ArgumentParser(
        description="Local Jarvis - Speech-to-Text (faster-whisper on CPU)"
    )
    parser.add_argument(
        "--audio",
        "-a",
        type=str,
        default=None,
        help="Path to .wav file. If not provided, the latest file from audio_cache/ is used.",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="base",
        choices=["tiny", "tiny.en", "base", "base.en", "small", "small.en", "medium", "medium.en"],
        help="Whisper model size (default: base).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run on (default: cpu).",
    )
    parser.add_argument(
        "--compute-type",
        type=str,
        default="int8",
        choices=["int8", "float32", "auto"],
        help="Quantization compute type (default: int8).",
    )
    parser.add_argument(
        "--language",
        "-l",
        type=str,
        default="en",
        help="Language code (default: en).",
    )
    args = parser.parse_args()

    # Determine audio clip to transcribe
    target_audio = args.audio
    if not target_audio:
        # Pick latest .wav from audio_cache
        if DEFAULT_AUDIO_CACHE_DIR.exists():
            wav_files = sorted(
                DEFAULT_AUDIO_CACHE_DIR.glob("*.wav"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if wav_files:
                target_audio = str(wav_files[0])
                print(f"No --audio specified. Using latest clip: {target_audio}")

    if not target_audio:
        print("Error: No audio file provided and audio_cache/ has no .wav files.", file=sys.stderr)
        sys.exit(1)

    print(f"\n--- Transcribing: {target_audio} ---")
    transcriber = get_transcriber(
        model_size=args.model,
        device=args.device,
        compute_type=args.compute_type,
    )

    start_time = time.perf_counter()
    transcribed_text = transcriber.transcribe(
        audio_path=target_audio,
        language=args.language,
    )
    inference_latency = time.perf_counter() - start_time

    print("\n" + "=" * 60)
    print(" TRANSCRIPTION RESULT")
    print("=" * 60)
    print(f"Model:              {args.model} (device={args.device}, compute_type={args.compute_type})")
    print(f"Model Load Time:    {transcriber.load_duration:.3f}s")
    print(f"Inference Latency:  {inference_latency:.3f}s")
    print(f"Text:               \"{transcribed_text}\"")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
