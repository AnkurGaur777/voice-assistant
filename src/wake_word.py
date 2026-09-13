"""
Local Jarvis - Wake Word Detection & Audio Capture Module (Phase 1)

This module handles:
1. Verifying and downloading openWakeWord pretrained ONNX models with retry & timeout logic.
2. Real-time audio capture via sounddevice (16kHz, 16-bit mono).
3. Detecting the "Hey Jarvis" wake word using openWakeWord.
4. Recording the user prompt into a ring-buffer with silence-based VAD (~1s silence cutoff).
5. Saving the captured speech to a timestamped .wav audio clip.
"""

import argparse
import collections
import os
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Deque, List, Optional, Tuple

import numpy as np
import requests
import sounddevice as sd
from scipy.io import wavfile
from tqdm import tqdm

import openwakeword
from openwakeword.model import Model

# --- Audio Configuration ---
SAMPLE_RATE = 16000          # 16 kHz sample rate required by openWakeWord
CHUNK_SAMPLES = 1280         # 80 ms per frame (1280 samples at 16kHz)
CHANNELS = 1                 # Mono
DTYPE = np.int16             # 16-bit signed PCM
CHUNK_DURATION = CHUNK_SAMPLES / SAMPLE_RATE  # 0.080s (80 ms)

# --- VAD & Timing Defaults ---
PRE_ROLL_SECONDS = 0.5       # Keep ~500ms audio prior to wake word detection
SILENCE_DURATION = 1.0       # Seconds of silence to trigger end-of-speech
SILENCE_CHUNKS = int(SILENCE_DURATION / CHUNK_DURATION)  # ~12 chunks (0.96s - 1.04s)
INITIAL_SPEECH_TIMEOUT = 3.5 # Seconds to wait for speech after wake word
MAX_RECORDING_SECONDS = 15.0 # Max recording duration limit
DEFAULT_SILENCE_RMS = 350.0  # Base RMS energy threshold for speech vs silence

# --- Model URLs and Paths ---
BASE_MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
AUDIO_CACHE_DIR = Path(__file__).resolve().parent.parent / "audio_cache"

OFFICIAL_MODEL_URLS = {
    "melspectrogram.onnx": "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/melspectrogram.onnx",
    "embedding_model.onnx": "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/embedding_model.onnx",
    "hey_jarvis_v0.1.onnx": "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/hey_jarvis_v0.1.onnx",
}


def list_audio_devices() -> None:
    """
    Prints all available audio input devices and clearly highlights
    the current default input device index.
    """
    devices = sd.query_devices()
    default_input_idx = sd.default.device[0]

    print("\n" + "=" * 75)
    print(" AVAILABLE AUDIO INPUT DEVICES")
    print("=" * 75)

    has_input_devices = False
    for idx, dev in enumerate(devices):
        max_in = dev.get("max_input_channels", 0)
        if max_in > 0:
            has_input_devices = True
            is_default = (idx == default_input_idx)
            marker = " <== [CURRENT DEFAULT]" if is_default else ""
            try:
                host_api_name = sd.query_hostapis(dev["hostapi"])["name"]
            except Exception:
                host_api_name = f"API {dev['hostapi']}"

            print(f"  [{idx:2d}] {dev['name']} ({host_api_name}, {max_in} in){marker}")

    if not has_input_devices:
        print("  No audio input devices found on this system!")

    print("-" * 75)
    if default_input_idx is not None and 0 <= default_input_idx < len(devices):
        print(f"Current Default Input Device: Index [{default_input_idx}] -> '{devices[default_input_idx]['name']}'")
    else:
        print(f"Current Default Input Device Index: {default_input_idx}")
    print("=" * 75 + "\n")


def download_file_with_retry(
    url: str,
    target_path: Path,
    timeout: int = 30,
    max_retries: int = 2
) -> None:
    """
    Downloads a model file with a per-attempt timeout, retry mechanism, and progress display.
    If all retries fail, prints the exact URL so the user can download manually.
    """
    if target_path.exists() and target_path.stat().st_size > 0:
        return

    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target_path.with_suffix(".tmp")
    filename = target_path.name
    total_attempts = max_retries + 1

    for attempt in range(1, total_attempts + 1):
        print(f"Downloading {filename} (attempt {attempt}/{total_attempts})...")
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
                    for chunk in response.iter_content(chunk_size=16384):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))

            temp_path.replace(target_path)
            print(f"Successfully downloaded {filename}")
            return
        except Exception as exc:
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    pass

            print(f"Warning: Download attempt {attempt} failed for {filename}: {exc}")
            if attempt >= total_attempts:
                error_msg = (
                    f"\n[ERROR] Failed to download {filename} after {total_attempts} attempts.\n"
                    f"Please manually download the file from:\n"
                    f"  {url}\n"
                    f"and place it at:\n"
                    f"  {target_path.resolve()}\n"
                )
                print(error_msg, file=sys.stderr)
                raise RuntimeError(error_msg) from exc

            time.sleep(1.5 * attempt)


def ensure_models(model_dir: Path = BASE_MODEL_DIR) -> Tuple[str, str, str]:
    """
    Verifies that required ONNX models exist locally; downloads them if missing.
    Returns a tuple of (melspec_path, embedding_path, wakeword_path).
    """
    model_dir.mkdir(parents=True, exist_ok=True)

    for filename, url in OFFICIAL_MODEL_URLS.items():
        dest = model_dir / filename
        if not dest.exists() or dest.stat().st_size == 0:
            download_file_with_retry(url, dest, timeout=30, max_retries=2)

    melspec_path = str((model_dir / "melspectrogram.onnx").resolve())
    embedding_path = str((model_dir / "embedding_model.onnx").resolve())
    wakeword_path = str((model_dir / "hey_jarvis_v0.1.onnx").resolve())

    return melspec_path, embedding_path, wakeword_path


class WakeWordDetector:
    """
    Real-time wake word detector and speech capture pipeline.
    """

    def __init__(
        self,
        model_dir: Path = BASE_MODEL_DIR,
        threshold: float = 0.35,
        input_device: Optional[int] = None,
        debug: bool = False,
    ):
        self.threshold = threshold
        self.input_device = input_device
        self.model_dir = model_dir
        self.debug = debug

        # Ensure models are ready
        print("Checking openWakeWord models...")
        self.melspec_path, self.embedding_path, self.wakeword_path = ensure_models(model_dir)

        # Initialize openWakeWord Model
        print("Initializing openWakeWord ONNX inference session...")
        self.oww_model = Model(
            wakeword_models=[self.wakeword_path],
            melspec_model_path=self.melspec_path,
            embedding_model_path=self.embedding_path,
            inference_framework="onnx",
        )
        self.model_name = list(self.oww_model.models.keys())[0]

        # Audio thread-safe queue and stream
        self.audio_queue: queue.Queue = queue.Queue()
        self.stream: Optional[sd.InputStream] = None
        self._pre_roll_chunks = max(1, int(PRE_ROLL_SECONDS / CHUNK_DURATION))
        self.pre_roll_buffer: Deque[np.ndarray] = collections.deque(maxlen=self._pre_roll_chunks)

        # Ensure audio cache directory exists
        AUDIO_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        """Sounddevice stream callback running on the audio thread."""
        if status:
            pass  # Suppress non-critical buffer under/overflow messages
        self.audio_queue.put(indata.copy().flatten())

    def start_stream(self) -> None:
        """Starts the audio input stream."""
        if self.stream is None or not self.stream.active:
            # Drain queue
            while not self.audio_queue.empty():
                try:
                    self.audio_queue.get_nowait()
                except queue.Empty:
                    break

            self.stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype=DTYPE,
                blocksize=CHUNK_SAMPLES,
                device=self.input_device,
                callback=self._audio_callback,
            )
            self.stream.start()

    def stop_stream(self) -> None:
        """Stops the audio input stream."""
        if self.stream is not None:
            if self.stream.active:
                self.stream.stop()
            self.stream.close()
            self.stream = None

    def close(self) -> None:
        """Cleans up resources."""
        self.stop_stream()

    def listen_and_record(
        self,
        on_wake_word_detected: Optional[Callable[[float], None]] = None,
        stop_event: Optional[threading.Event] = None,
    ) -> Optional[str]:
        """
        Listens for 'Hey Jarvis'. Once detected, records the user's utterance
        until ~1 second of silence, and saves it to a timestamped .wav file.

        :param on_wake_word_detected: Optional callback invoked with the detection score.
        :param stop_event: Optional threading.Event to signal graceful termination.
        :return: Absolute file path to the saved .wav audio clip, or None if cancelled.
        """
        self.start_stream()
        self.oww_model.reset()
        self.pre_roll_buffer.clear()

        # Rolling background RMS energy estimation
        ambient_energies: Deque[float] = collections.deque(maxlen=40)
        speech_energy_threshold = DEFAULT_SILENCE_RMS

        print("Listening for wake word 'Hey Jarvis'...")

        # -----------------------------------------------------------------
        # Phase 1a: Wait for Wake Word
        # -----------------------------------------------------------------
        wake_word_detected = False
        detected_score = 0.0

        while not wake_word_detected:
            if stop_event is not None and stop_event.is_set():
                return None
            try:
                chunk = self.audio_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            # Update pre-roll buffer
            self.pre_roll_buffer.append(chunk)

            # Track ambient noise floor for adaptive silence threshold
            chunk_rms = float(np.sqrt(np.mean(chunk.astype(np.float32) ** 2)))
            ambient_energies.append(chunk_rms)

            # Predict wake word on current frame
            prediction = self.oww_model.predict(chunk)
            score = float(prediction.get(self.model_name, 0.0))

            # Debug logging of confidence score for every frame
            if self.debug:
                print(f"[DEBUG] Chunk RMS: {chunk_rms:6.1f} | Wake Score: {score:7.4f} (threshold: {self.threshold:.2f})")

            if score >= self.threshold:
                wake_word_detected = True
                detected_score = score

        # Wake word detected!
        print("Wake word detected")
        if on_wake_word_detected:
            on_wake_word_detected(detected_score)
            # Drain audio queue and reset pre-roll buffer so that any spoken
            # wake-word acknowledgment (e.g. "Yes?") is not recorded or transcribed
            while not self.audio_queue.empty():
                try:
                    self.audio_queue.get_nowait()
                except queue.Empty:
                    break
            self.pre_roll_buffer.clear()

        # Compute adaptive silence threshold (background RMS * 2.2, with fallback floor)
        if len(ambient_energies) > 5:
            avg_ambient = float(np.median(ambient_energies))
            speech_energy_threshold = max(DEFAULT_SILENCE_RMS, avg_ambient * 2.2)

        # -----------------------------------------------------------------
        # Phase 1b: Record User Speech Until ~1s Silence (VAD)
        # -----------------------------------------------------------------
        print(f"Recording speech (VAD threshold RMS: {speech_energy_threshold:.1f})...")
        recorded_frames: List[np.ndarray] = list(self.pre_roll_buffer)

        has_user_started_speaking = False
        silence_chunk_count = 0
        record_start_time = time.time()

        while True:
            if stop_event is not None and stop_event.is_set():
                return None
            try:
                chunk = self.audio_queue.get(timeout=0.5)
            except queue.Empty:
                break

            recorded_frames.append(chunk)
            chunk_rms = float(np.sqrt(np.mean(chunk.astype(np.float32) ** 2)))
            elapsed = time.time() - record_start_time

            if chunk_rms >= speech_energy_threshold:
                has_user_started_speaking = True
                silence_chunk_count = 0
            else:
                if has_user_started_speaking:
                    silence_chunk_count += 1
                elif elapsed > INITIAL_SPEECH_TIMEOUT:
                    # User said "Jarvis" but didn't speak a follow-up command
                    print("No follow-up speech detected after wake word.")
                    break

            # Stop condition 1: ~1 second of continuous silence after speech
            if has_user_started_speaking and silence_chunk_count >= SILENCE_CHUNKS:
                break

            # Stop condition 2: Exceeded safety max duration
            if elapsed >= MAX_RECORDING_SECONDS:
                print(f"Reached maximum recording limit ({MAX_RECORDING_SECONDS}s).")
                break

        # -----------------------------------------------------------------
        # Phase 1c: Save Captured Audio to .wav
        # -----------------------------------------------------------------
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        output_filename = f"clip_{timestamp_str}_{int(time.time() * 1000) % 1000}.wav"
        output_path = AUDIO_CACHE_DIR / output_filename

        if recorded_frames:
            combined_audio = np.concatenate(recorded_frames).astype(np.int16)
        else:
            combined_audio = np.zeros(CHUNK_SAMPLES, dtype=np.int16)

        wavfile.write(str(output_path), SAMPLE_RATE, combined_audio)
        abs_path = str(output_path.resolve())
        print(f"Saved audio clip to: {abs_path}")

        return abs_path

    def record_utterance(
        self,
        stop_event: Optional[threading.Event] = None,
        speech_timeout: float = 60.0,
        max_duration: float = MAX_RECORDING_SECONDS,
    ) -> Optional[str]:
        """
        Directly records user speech without waiting for a wake word.
        Used for continuous conversation mode.

        Waits up to `speech_timeout` seconds for user speech onset.
        If no speech is detected within `speech_timeout`, returns None (silence timeout).
        Once speech begins, records until ~1 second of continuous silence (VAD)
        and saves the utterance to a timestamped .wav file.

        :param stop_event: Optional threading.Event to signal termination.
        :param speech_timeout: Seconds of initial silence before timing out back to standby.
        :param max_duration: Hard safety limit in seconds.
        :return: Absolute path to saved .wav clip, or None if timed out or cancelled.
        """
        self.start_stream()

        # Drain any residual audio frames in the queue (e.g. from recent TTS playback)
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break

        print(f"[Conversation] Listening hands-free for next utterance (timeout: {speech_timeout:.1f}s)...")

        recorded_frames: List[np.ndarray] = []
        has_user_started_speaking = False
        silence_chunk_count = 0
        record_start_time = time.time()
        speech_energy_threshold = DEFAULT_SILENCE_RMS
        ambient_energies: Deque[float] = collections.deque(maxlen=20)

        while True:
            if stop_event is not None and stop_event.is_set():
                return None
            try:
                chunk = self.audio_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            chunk_rms = float(np.sqrt(np.mean(chunk.astype(np.float32) ** 2)))
            elapsed = time.time() - record_start_time

            if not has_user_started_speaking:
                ambient_energies.append(chunk_rms)
                if len(ambient_energies) > 5:
                    avg_ambient = float(np.median(ambient_energies))
                    speech_energy_threshold = max(DEFAULT_SILENCE_RMS, avg_ambient * 2.2)

                if chunk_rms >= speech_energy_threshold:
                    has_user_started_speaking = True
                    silence_chunk_count = 0
                    recorded_frames.append(chunk)
                elif elapsed > speech_timeout:
                    print(f"[Conversation] No speech detected within {speech_timeout:.1f}s timeout.")
                    return None
            else:
                recorded_frames.append(chunk)
                if chunk_rms >= speech_energy_threshold:
                    silence_chunk_count = 0
                else:
                    silence_chunk_count += 1
                    if silence_chunk_count >= SILENCE_CHUNKS:
                        # ~1 second continuous silence detected post-speech
                        break

            if elapsed >= max_duration:
                print(f"[Conversation] Reached maximum recording limit ({max_duration}s).")
                break

        if not recorded_frames or not has_user_started_speaking:
            return None

        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        output_filename = f"clip_{timestamp_str}_{int(time.time() * 1000) % 1000}.wav"
        output_path = AUDIO_CACHE_DIR / output_filename

        combined_audio = np.concatenate(recorded_frames).astype(np.int16)
        wavfile.write(str(output_path), SAMPLE_RATE, combined_audio)
        abs_path = str(output_path.resolve())
        print(f"Saved audio clip to: {abs_path}")
        return abs_path


def main():
    """CLI entrypoint supporting --list-devices, --debug, and custom device/threshold."""
    parser = argparse.ArgumentParser(
        description="Local Jarvis - Wake Word Detection & Audio Capture (Phase 1)"
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List all available audio input devices and current default device, then exit."
    )
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="Specific audio input device index to use (default: system default input device)."
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.35,
        help="Wake-word detection confidence threshold between 0.0 and 1.0 (default: 0.35)."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print real-time confidence scores and RMS energy for every chunk processed."
    )

    args = parser.parse_args()

    if args.list_devices:
        list_audio_devices()
        return

    print("=" * 68)
    print(" Local Jarvis — Wake Word Detection & Audio Capture (Phase 1)")
    print("=" * 68)

    # Determine input device
    default_dev_idx = args.device if args.device is not None else sd.default.device[0]
    try:
        dev_info = sd.query_devices(default_dev_idx)
        print(f"Using Audio Input Device [{default_dev_idx}]: {dev_info['name']}")
    except Exception as e:
        print(f"Warning: Could not query device [{default_dev_idx}]: {e}")

    if args.debug:
        print("[DEBUG MODE ACTIVE] Printing wake-word confidence score for every chunk.")

    detector = WakeWordDetector(
        threshold=args.threshold,
        input_device=args.device,
        debug=args.debug,
    )

    print("\n[READY] Standalone listener is active.")
    print(f"Say 'Hey Jarvis' into your microphone followed by your command (threshold: {args.threshold}).")
    print("Press Ctrl+C to exit.\n")

    try:
        while True:
            clip_path = detector.listen_and_record()
            print(f"[SUCCESS] Utterance captured at: {clip_path}\n")
            print("Listening for next 'Hey Jarvis'...\n")
    except KeyboardInterrupt:
        print("\nExiting wake word listener. Goodbye!")
    finally:
        detector.close()


if __name__ == "__main__":
    main()
