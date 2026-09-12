"""
Local Jarvis - Text-to-Speech (TTS) Standalone Test Harness (Phase 5)

Tests Piper speech synthesis, voice quality, streaming sentence-by-sentence
playback, and benchmark metrics independently from the agent pipeline.

Usage:
  # Basic test with default voice (ryan)
  python src/test_tts.py "Hello, I am Jarvis. System diagnostics are all green."

  # Compare with female voice (lessac)
  python src/test_tts.py "Hello, I am Jarvis. System diagnostics are all green." --voice lessac

  # List all available voices and download status
  python src/test_tts.py --list-voices

  # Streaming benchmark with latency metrics
  python src/test_tts.py --benchmark "Sentence one starts now. Sentence two synthesizes concurrently in the background."

  # Save to .wav file
  python src/test_tts.py "Saving to audio file." --output sample.wav
"""

import argparse
from pathlib import Path
import sys
import time

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.tts import (
    DEFAULT_VOICE,
    SUPPORTED_VOICES,
    list_available_voices,
    speak,
    synthesize_sentence,
    synthesize_to_wav,
    split_into_sentences,
    get_piper_voice,
)


def print_banner(voice: str) -> None:
    """Prints a styled startup banner."""
    print("\n" + "=" * 75)
    print(" LOCAL JARVIS - TEXT-TO-SPEECH TEST HARNESS (PHASE 5: PIPER TTS)")
    print("=" * 75)
    print(f" Engine:         Piper TTS (CPU-based, offline, ONNX runtime)")
    print(f" Selected Voice: {voice} ({SUPPORTED_VOICES.get(voice, {}).get('description', 'Custom')})")
    print(" Audio Output:   sounddevice (16-bit PCM)")
    print("=" * 75 + "\n")


def print_voices() -> None:
    """Prints all configured Piper voices and their download status."""
    voices = list_available_voices()
    print("\n" + "=" * 75)
    print(" CONFIGURED PIPER VOICES")
    print("=" * 75)
    for key, info in voices.items():
        status = " [Downloaded]" if info["downloaded"] else " [Not downloaded - auto-downloads on first use]"
        print(f"  * Voice: '{key}' ({info['gender'].capitalize()}){status}")
        print(f"    Model: {info['model_name']}")
        print(f"    Desc:  {info['description']}")
        print()
    print("=" * 75 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Local Jarvis - Piper Text-to-Speech Test Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "positional_text",
        nargs="?",
        default=None,
        help="Text to speak aloud (optional if --text is provided)",
    )
    parser.add_argument(
        "--text",
        "-t",
        type=str,
        default=None,
        help="Text string to speak aloud",
    )
    parser.add_argument(
        "--voice",
        "-v",
        type=str,
        default=DEFAULT_VOICE,
        choices=list(SUPPORTED_VOICES.keys()),
        help=f"Piper voice model to use (default: '{DEFAULT_VOICE}')",
    )
    parser.add_argument(
        "--list-voices",
        action="store_true",
        help="List available voices and whether they are downloaded locally",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Optional destination path to save synthesized audio as a .wav file",
    )
    parser.add_argument(
        "--benchmark",
        "-b",
        action="store_true",
        help="Measure and display Time-to-First-Audio (TTFA) streaming latency",
    )
    parser.add_argument(
        "--no-playback",
        action="store_true",
        help="Synthesize without playing audio through speakers",
    )

    args = parser.parse_args()

    if args.list_voices:
        print_voices()
        return

    # Determine input text
    input_text = args.text or args.positional_text
    if not input_text:
        input_text = (
            "Hello, I am Jarvis. System diagnostics are fully operational. "
            "Text to speech output is powered by Piper and running entirely offline on your CPU."
        )

    print_banner(args.voice)
    print(f"Input Text:\n  \"{input_text}\"\n")

    sentences = split_into_sentences(input_text)
    print(f"Parsed Sentences ({len(sentences)}):")
    for idx, s in enumerate(sentences, 1):
        print(f"  [{idx}] {s}")
    print()

    # Optional WAV file export
    if args.output:
        print(f"Exporting audio to: {args.output}...")
        start_export = time.perf_counter()
        out_file = synthesize_to_wav(input_text, args.output, voice=args.voice)
        export_duration = time.perf_counter() - start_export
        print(f"Audio file saved successfully: {out_file.resolve()} ({export_duration * 1000:.1f}ms)\n")

    if args.no_playback:
        print("Playback skipped (--no-playback specified).")
        return

    # Synthesize and speak
    ttfa_recorded = [0.0]

    def on_playback_start(ttfa: float):
        ttfa_recorded[0] = ttfa
        print(f">> [Audio Playback Started] Time-to-First-Audio (TTFA): {ttfa * 1000:.1f}ms")

    print("Playing speech...")
    start_time = time.perf_counter()
    speak(
        input_text,
        voice=args.voice,
        blocking=True,
        on_start_playback=on_playback_start if args.benchmark else None,
    )
    total_elapsed = time.perf_counter() - start_time
    print("Playback completed.")

    if args.benchmark:
        print("\n" + "-" * 45)
        print(" BENCHMARK RESULTS")
        print("-" * 45)
        print(f" Total Playback Duration: {total_elapsed:.2f}s")
        print(f" Time-to-First-Audio:     {ttfa_recorded[0] * 1000:.1f}ms")
        print("-" * 45 + "\n")


if __name__ == "__main__":
    main()
