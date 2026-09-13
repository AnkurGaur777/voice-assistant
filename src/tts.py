"""
Local Jarvis - Text-to-Speech (TTS) Module (Phase 5)

Provides fast, high-quality, offline, CPU-based speech synthesis using Piper
and audio playback via sounddevice.

Features:
1. Automatic on-demand voice model downloading with retry, timeout, and progress bar
   (matching the openWakeWord pattern from Phase 1).
2. Streaming-friendly sentence-by-sentence synthesis: playback begins immediately on the
   first sentence while subsequent sentences synthesize concurrently in the background.
3. Multiple voice options (e.g. 'ryan' for Jarvis male voice, 'lessac' for female voice).
4. Audio export to .wav files.
"""

from collections import deque
import os
from pathlib import Path
import queue
import re
import sys
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import requests
import sounddevice as sd
from tqdm import tqdm

from piper import PiperVoice

# --- Configuration & Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE_MODEL_DIR = PROJECT_ROOT / "models" / "piper"

SUPPORTED_VOICES: Dict[str, Dict[str, Any]] = {
    "ryan": {
        "model_name": "en_US-ryan-medium",
        "gender": "male",
        "description": "Natural, articulate American English male voice (default Jarvis persona)",
        "model_url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/medium/en_US-ryan-medium.onnx",
        "config_url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/medium/en_US-ryan-medium.onnx.json",
    },
    "lessac": {
        "model_name": "en_US-lessac-medium",
        "gender": "female",
        "description": "Clear, articulate American English female voice",
        "model_url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx",
        "config_url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json",
    },
}

DEFAULT_VOICE = "ryan"


# ==============================================================================
# 1. Model Downloader & Verifier (Phase 1 Pattern)
# ==============================================================================

def download_file_with_retry(
    url: str,
    target_path: Path,
    timeout: int = 45,
    max_retries: int = 2,
) -> None:
    """
    Downloads a file with a per-attempt timeout, retry mechanism, and progress display.
    Uses an atomic .tmp file to prevent corrupt partially-downloaded models.
    """
    if target_path.exists() and target_path.stat().st_size > 0:
        return

    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target_path.with_suffix(".tmp")
    filename = target_path.name
    total_attempts = max_retries + 1

    for attempt in range(1, total_attempts + 1):
        print(f"[TTS] Downloading {filename} (attempt {attempt}/{total_attempts})...")
        try:
            with requests.get(url, stream=True, timeout=(10, timeout)) as response:
                response.raise_for_status()
                total_size = int(response.headers.get("content-length", 0))

                with open(temp_path, "wb") as f, tqdm(
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    desc=filename,
                    ncols=80,
                ) as pbar:
                    for chunk in response.iter_content(chunk_size=32768):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))

            temp_path.replace(target_path)
            print(f"[TTS] Successfully downloaded {filename}")
            return
        except Exception as exc:
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    pass

            print(f"[TTS] Warning: Download attempt {attempt} failed for {filename}: {exc}")
            if attempt < total_attempts:
                backoff = 2 * attempt
                print(f"[TTS] Retrying in {backoff} seconds...")
                time.sleep(backoff)
            else:
                print(f"\n[TTS] ERROR: Failed to download {filename} after {total_attempts} attempts.")
                print(f"Please download manually from: {url}")
                print(f"and place it at: {target_path.resolve()}\n")
                raise RuntimeError(f"Could not download required Piper voice file: {filename}") from exc


def resolve_voice_key(voice_name: Optional[str] = None) -> str:
    """Resolves a voice key or alias (e.g. 'ryan' or 'en_US-ryan-medium') to the canonical voice key."""
    if not voice_name:
        return DEFAULT_VOICE

    cleaned = voice_name.strip().lower()
    if cleaned in SUPPORTED_VOICES:
        return cleaned

    for key, info in SUPPORTED_VOICES.items():
        if cleaned == info["model_name"].lower():
            return key

    available = ", ".join(SUPPORTED_VOICES.keys())
    raise ValueError(f"Unknown voice '{voice_name}'. Supported voices are: {available}")


def ensure_voice_model(
    voice_name: Optional[str] = None,
    model_dir: Optional[Path] = None,
) -> Tuple[Path, Path]:
    """
    Verifies that the requested Piper voice ONNX model and config JSON exist.
    Downloads them on first run if missing.

    Returns:
        (model_onnx_path, config_json_path)
    """
    voice_key = resolve_voice_key(voice_name)
    info = SUPPORTED_VOICES[voice_key]
    target_dir = model_dir if model_dir else BASE_MODEL_DIR

    model_file = target_dir / f"{info['model_name']}.onnx"
    config_file = target_dir / f"{info['model_name']}.onnx.json"

    download_file_with_retry(info["model_url"], model_file)
    download_file_with_retry(info["config_url"], config_file)

    return model_file, config_file


# ==============================================================================
# 2. In-Memory Voice Cache & Loading
# ==============================================================================

_LOADED_VOICES: Dict[str, PiperVoice] = {}
_VOICE_LOCK = threading.Lock()


def get_piper_voice(
    voice_name: Optional[str] = None,
    model_dir: Optional[Path] = None,
) -> PiperVoice:
    """
    Loads and caches a PiperVoice instance to avoid reloading the ONNX model
    on every synthesis call.
    """
    voice_key = resolve_voice_key(voice_name)
    with _VOICE_LOCK:
        if voice_key in _LOADED_VOICES:
            return _LOADED_VOICES[voice_key]

        model_path, config_path = ensure_voice_model(voice_key, model_dir=model_dir)
        print(f"[TTS] Loading Piper voice '{voice_key}' ({model_path.name})...")
        start = time.perf_counter()
        voice = PiperVoice.load(model_path, config_path=config_path)
        elapsed = time.perf_counter() - start
        print(f"[TTS] Piper voice '{voice_key}' loaded in {elapsed * 1000:.1f}ms")

        _LOADED_VOICES[voice_key] = voice
        return voice


# ==============================================================================
# 3. Text & Sentence Tokenizer
# ==============================================================================

def split_into_sentences(text: str) -> List[str]:
    """
    Splits text into natural sentences for streaming synthesis.
    Preserves punctuation marks and avoids splitting on numbers, acronyms, or decimals.
    """
    cleaned = text.strip()
    if not cleaned:
        return []

    # Clean markdown headers, bullet symbols, or excess formatting
    cleaned = re.sub(r"^[#*>-]+\s*", "", cleaned, flags=re.MULTILINE)

    # First split on linebreaks as primary boundaries
    raw_lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    sentences: List[str] = []

    for line in raw_lines:
        # Match punctuation followed by whitespace and a word or quote
        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'“‘]|$)", line)
        for part in parts:
            p = part.strip()
            if p:
                sentences.append(p)

    return sentences if sentences else [cleaned]


# ==============================================================================
# 4. Synthesis & Streaming Playback Pipeline
# ==============================================================================

def synthesize_sentence(sentence: str, voice: PiperVoice) -> Tuple[np.ndarray, int]:
    """
    Synthesizes a single sentence with Piper into an int16 numpy array.

    Returns:
        (audio_data, sample_rate)
    """
    sample_rate = voice.config.sample_rate
    chunks: List[np.ndarray] = []

    for chunk in voice.synthesize(sentence):
        if chunk and chunk.audio_int16_array is not None and len(chunk.audio_int16_array) > 0:
            chunks.append(chunk.audio_int16_array)

    if chunks:
        audio = np.concatenate(chunks)
    else:
        audio = np.zeros(0, dtype=np.int16)

    return audio, sample_rate


# Trailing silence durations (in seconds) to prevent buffer cutoff
INTER_SENTENCE_PAUSE_SECONDS = 0.15      # 150ms natural pause between sentences
END_OF_SPEECH_PADDING_SECONDS = 0.25     # 250ms flush padding so last syllables are never clipped


def sanitize_speech_text(text: str) -> str:
    """
    Cleans and filters text intended for TTS playback to ensure that:
    1. Raw JSON tool calls {"name": "...", "parameters": ...} are stripped.
    2. Tool narration boilerplate ("I'll call the `web_search` tool...", "Calling `run_python`...") is removed.
    3. Markdown code blocks, backticks, and raw tool syntax are cleaned.
    4. Malformed tool outputs are not spoken aloud as raw code or symbols.
    """
    if not text or not text.strip():
        return ""

    cleaned = text.strip()

    # Strip code fences containing json or tool syntax
    cleaned = re.sub(r"```(?:json)?\s*\{.*?\}\s*```", "", cleaned, flags=re.DOTALL)

    # Strip raw JSON function calls: {"name": ..., "parameters": ...} or {"name": ..., "arguments": ...}
    cleaned = re.sub(
        r'\{[^{}]*"name"\s*:\s*"[^"]+"\s*,\s*"(?:parameters|arguments|args)"\s*:\s*\{[^{}]*\}[^{}]*\}',
        "",
        cleaned,
        flags=re.DOTALL,
    )

    # Strip narration boilerplate like:
    # "To calculate 358% of 340, I'll call the `web_search` tool."
    # "I will call the run_python tool to calculate this."
    cleaned = re.sub(
        r"(?:To (?:calculate|search|find|check|run) [^,.]*,\s*)?I(?:'ll| will)\s+(?:call|use|run|execute)\s+the\s+[`'\"]?\w+[`'\"]?\s+tool\.?",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"(?:Calling|Running|Using)\s+(?:the\s+)?(?:tool\s+)?[`'\"]?\w+[`'\"]?\s*(?:tool)?\.?\.\.?",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )

    # Strip tool logging tags
    cleaned = re.sub(
        r"\[(Tool Invoked|Clipboard|Desktop Automation|System Clock|Sandbox|Reminders|Memory|Ollama)[^\]]*\]\s*",
        "",
        cleaned,
    )

    # Convert raw datetime tool dump if echoed verbatim into natural spoken sentence
    dt_match = re.search(
        r"(?:Current system date and time:?\s*)?- Date:\s*([^-\n\r]+?)(?:[\r\n]+[ \t]*|\s+)- Time:\s*([^\(\n\r]+)",
        cleaned,
        re.IGNORECASE,
    )
    if dt_match:
        date_part = dt_match.group(1).strip()
        time_part = dt_match.group(2).strip()
        replacement = f"It's {date_part}, {time_part}."
        cleaned = re.sub(
            r"(?:Current system date and time:?\s*)?- Date:.*?(?:Timezone:[^\.\n\r]+(?:\.|$)|$)",
            replacement,
            cleaned,
            flags=re.DOTALL | re.IGNORECASE,
        )


    # Strip remaining backticks
    cleaned = cleaned.replace("`", "")

    # Clean redundant whitespace and punctuation
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = re.sub(r"^[:\-\s]+", "", cleaned).strip()

    return cleaned


def speak(
    text: str,
    voice: str = DEFAULT_VOICE,
    blocking: bool = True,
    on_start_playback: Optional[Callable[[float], None]] = None,
) -> None:
    """
    Synthesizes and plays speech using Piper and sounddevice.

    Streaming-Friendly & Complete Audio Delivery:
    - Splits multi-sentence input and plays sentence-by-sentence so audio begins
      almost immediately without waiting for the full response to finish synthesizing.
    - Uses a continuous sounddevice.OutputStream across all sentences, preventing
      abrupt stream restarts or clicks between chunks.
    - Appends trailing silence padding to each sentence and explicitly drains the
      operating system / hardware audio buffer before closing the stream, ensuring
      the final syllables of words are never cut off.

    Args:
        text: The text string to speak.
        voice: Voice name or alias ('ryan', 'lessac'). Defaults to 'ryan' (DEFAULT_VOICE).
        blocking: If True, blocks until speech playback finishes.
        on_start_playback: Optional callback invoked with the Time-to-First-Audio (TTFA)
            in seconds when playback actually starts.
    """
    clean_text = sanitize_speech_text(text)
    if not clean_text:
        print("[TTS] Text contained only raw tool syntax or was empty after sanitization; skipping speech playback.")
        return

    def _execute_speak():
        piper_voice = get_piper_voice(voice)
        sentences = split_into_sentences(clean_text)
        if not sentences:
            return

        sample_rate = piper_voice.config.sample_rate

        # Multi-sentence streaming producer-consumer queue
        # Producer synthesizes sentences ahead into the queue while consumer feeds the audio stream
        audio_queue: queue.Queue = queue.Queue(maxsize=3)
        stop_event = threading.Event()
        producer_error: List[Exception] = []

        total_sentences = len(sentences)

        def synthesis_producer():
            try:
                for idx, sentence in enumerate(sentences):
                    if stop_event.is_set():
                        break
                    audio_data, sr = synthesize_sentence(sentence, piper_voice)
                    is_last = (idx == total_sentences - 1)

                    # Append trailing silence:
                    # - Between sentences: 150ms natural pause
                    # - End of speech: 250ms flush padding to prevent cutting off trailing syllables
                    if is_last:
                        pad_sec = END_OF_SPEECH_PADDING_SECONDS
                    else:
                        pad_sec = INTER_SENTENCE_PAUSE_SECONDS

                    if len(audio_data) > 0:
                        pad_samples = int(sr * pad_sec)
                        if pad_samples > 0:
                            silence = np.zeros(pad_samples, dtype=np.int16)
                            audio_data = np.concatenate([audio_data, silence])

                    audio_queue.put((idx, audio_data, sr))
            except Exception as e:
                producer_error.append(e)
            finally:
                audio_queue.put(None)  # Sentinel to signal end of stream

        producer_thread = threading.Thread(
            target=synthesis_producer,
            name="TTS_SynthesisProducer",
            daemon=True,
        )

        playback_start_time = time.perf_counter()
        producer_thread.start()

        stream = None
        try:
            first_sentence = True
            while True:
                item = audio_queue.get()
                if item is None:
                    break

                idx, audio_data, sr = item
                if first_sentence:
                    first_sentence = False
                    ttfa = time.perf_counter() - playback_start_time
                    if on_start_playback:
                        on_start_playback(ttfa)

                    # Initialize continuous output stream
                    stream = sd.OutputStream(
                        samplerate=sr,
                        channels=1,
                        dtype=np.int16,
                    )
                    stream.start()

                if stream is not None and len(audio_data) > 0:
                    stream.write(audio_data)

            # After all chunks are written, wait for hardware/OS audio buffer to completely drain
            if stream is not None:
                latency = stream.latency
                output_latency = latency[1] if isinstance(latency, (list, tuple)) else (latency if isinstance(latency, (int, float)) else 0.2)
                drain_delay = max(float(output_latency), 0.15)
                time.sleep(drain_delay)

        except KeyboardInterrupt:
            stop_event.set()
            if stream is not None:
                stream.abort()
            raise
        finally:
            stop_event.set()
            if stream is not None:
                try:
                    stream.stop()
                    stream.close()
                except Exception:
                    pass
            producer_thread.join(timeout=1.0)
            if producer_error:
                raise producer_error[0]

    if blocking:
        _execute_speak()
    else:
        bg_thread = threading.Thread(target=_execute_speak, name="TTS_AsyncSpeak", daemon=True)
        bg_thread.start()


def synthesize_to_wav(
    text: str,
    output_path: Union[str, Path],
    voice: str = DEFAULT_VOICE,
) -> Path:
    """
    Synthesizes the entire text and writes it to a .wav audio file.

    Args:
        text: Text to synthesize.
        output_path: Destination .wav file path.
        voice: Voice name or alias. Defaults to 'ryan' (DEFAULT_VOICE).

    Returns:
        Resolved destination Path.
    """
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    piper_voice = get_piper_voice(voice)
    sentences = split_into_sentences(text)

    all_chunks: List[np.ndarray] = []
    sample_rate = piper_voice.config.sample_rate

    for s in sentences:
        audio, sr = synthesize_sentence(s, piper_voice)
        if len(audio) > 0:
            all_chunks.append(audio)
            sample_rate = sr

    if all_chunks:
        full_audio = np.concatenate(all_chunks)
    else:
        full_audio = np.zeros(0, dtype=np.int16)

    # Write WAV file using standard scipy.io.wavfile
    from scipy.io import wavfile
    wavfile.write(str(out_path), sample_rate, full_audio)
    return out_path


def list_available_voices(model_dir: Optional[Path] = None) -> Dict[str, Dict[str, Any]]:
    """
    Returns a dictionary of supported voices with download status.
    """
    target_dir = model_dir if model_dir else BASE_MODEL_DIR
    result = {}

    for key, info in SUPPORTED_VOICES.items():
        model_path = target_dir / f"{info['model_name']}.onnx"
        config_path = target_dir / f"{info['model_name']}.onnx.json"
        is_downloaded = model_path.exists() and config_path.exists()

        result[key] = {
            "key": key,
            "model_name": info["model_name"],
            "gender": info["gender"],
            "description": info["description"],
            "downloaded": is_downloaded,
            "model_path": str(model_path) if is_downloaded else None,
        }

    return result
