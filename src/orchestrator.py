"""
Local Jarvis - Full Voice Assistant Pipeline Orchestrator

Wires the complete autonomous hands-free loop:
1. Wake Word: Listens for "Hey Jarvis" and records follow-up speech to a .wav clip (openWakeWord).
2. STT: Transcribes speech to text via faster-whisper ("small" model on CPU).
3. Brain & Tools: Passes transcribed text to LangGraph agent with ChromaDB memory & 9 tools.
4. TTS: Synthesizes and speaks the response aloud via Piper ("ryan" voice on CPU).
5. Loop: Gracefully returns to listening for the wake word again.

System Tray Integration:
- Uses `tray.tray_app.JarvisTrayApp` to show real-time visual states:
  * LISTENING / IDLE (Cyan circle with microphone)
  * PROCESSING (Amber circle with activity dots)
  * SPEAKING (Emerald circle with speaker & sound waves)
  * ERROR (Crimson circle with exclamation mark)

Error Resilience:
- Exceptions in any single stage (STT, agent, or TTS) are caught, logged, and reset to
  listening mode without crashing the orchestrator process.

Graceful Shutdown:
- Supports Ctrl+C (SIGINT/SIGTERM) and right-click -> "Quit Jarvis" on the system tray icon.

Known Limitations:
1. Wake word detection confidence varies with mic distance/volume — may need 1-2 attempts
   in noisy conditions, adjustable via the `--threshold` flag (default: 0.35).
2. STT occasionally adds filler words on short utterances despite correct number/keyword
   transcription.
3. Very small (3B) LLM occasionally reasons through simple math in plain text instead
   of using the sandbox tool.
"""

import argparse
import collections
import os
import queue
import re
import signal
import sys
import threading
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.io import wavfile

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Safe guard for headless / pythonw.exe execution where stdout/stderr might be None
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w", encoding="utf-8")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w", encoding="utf-8")

from langchain_core.messages import BaseMessage

from src.agent.graph import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    get_agent_app,
    run_agent,
)
from src.agent.tools.reminders import start_reminder_scheduler, stop_reminder_scheduler
from src.analytics.logger import flush_logging, log_interaction, shutdown_logging
from src.stt import transcribe_audio, get_transcriber
from src.tts import DEFAULT_VOICE, get_piper_voice, speak, sanitize_speech_text
from src.ui.orb_overlay import OrbOverlay, OrbState
from src.wake_word import (
    AUDIO_CACHE_DIR,
    CHUNK_DURATION,
    CHUNK_SAMPLES,
    DEFAULT_SILENCE_RMS,
    MAX_RECORDING_SECONDS,
    SAMPLE_RATE,
    SILENCE_CHUNKS,
    SILENCE_DURATION,
    WakeWordDetector,
)
from tray.tray_app import JarvisTrayApp, JarvisTrayState


# Configurable stop phrases that signal the agent to exit continuous conversation mode
DEFAULT_STOP_PHRASES: List[str] = [
    "stop",
    "goodbye",
    "bye",
    "that's all",
    "thats all",
    "thank you jarvis",
    "thank you, jarvis",
    "thank you",
    "exit",
    "quit",
    "cancel",
    "nevermind",
]

# Configurable short dismissal phrases that silently cancel TTS playback without starting a new query
DEFAULT_DISMISSAL_PHRASES: List[str] = [
    "okay",
    "ok",
    "got it",
    "stop",
    "never mind",
    "nevermind",
    "that's enough",
    "thats enough",
    "thanks",
    "thank you",
    "cancel",
    "shh",
    "quiet",
    "enough",
    "hush",
]


def is_stop_phrase(text: str, stop_phrases: Optional[List[str]] = None) -> bool:
    """
    Checks if transcribed user text matches or contains an exit / stop phrase.
    Normalizes punctuation, whitespace, and case.
    """
    if not text or not text.strip():
        return False
    clean = re.sub(r"[^\w\s]", "", text).lower().strip()
    phrases = stop_phrases or DEFAULT_STOP_PHRASES
    for phrase in phrases:
        clean_phrase = re.sub(r"[^\w\s]", "", phrase).lower().strip()
        if not clean_phrase:
            continue
        if clean == clean_phrase:
            return True
        if clean.startswith(f"{clean_phrase} ") or clean.endswith(f" {clean_phrase}"):
            return True
    return False


def is_dismissal_phrase(text: str, dismissal_phrases: Optional[List[str]] = None) -> bool:
    """
    Checks if transcribed user interruption text is a short dismissal phrase.
    Normalizes punctuation, whitespace, and case.
    """
    if not text or not text.strip():
        return False
    clean = re.sub(r"[^\w\s]", "", text).lower().strip()
    if not clean:
        return False
    phrases = dismissal_phrases or DEFAULT_DISMISSAL_PHRASES
    tokens = clean.split()
    for phrase in phrases:
        clean_phrase = re.sub(r"[^\w\s]", "", phrase).lower().strip()
        if not clean_phrase:
            continue
        if clean == clean_phrase:
            return True
        # Allow short trailing phrases like "ok thanks", "stop jarvis", "got it jarvis" (<= 5 words)
        if (clean.startswith(f"{clean_phrase} ") or clean.endswith(f" {clean_phrase}")) and len(tokens) <= 5:
            return True
    return False


def is_jarvis_dismissal(text: str, dismissal_phrases: Optional[List[str]] = None) -> bool:
    """
    Checks if an utterance during TTS playback is a dismissal targeted at Jarvis
    (e.g., 'Jarvis stop', 'Jarvis okay', 'Hey Jarvis that's enough', 'Stop Jarvis').

    Crucially, requires the word 'jarvis' to be present to prevent accidental
    cancellations from background noise, solitary 'okay's, or coughs.
    """
    if not text or not text.strip():
        return False

    clean = re.sub(r"[^\w\s]", "", text).lower().strip()
    tokens = clean.split()
    if not tokens or "jarvis" not in tokens:
        return False

    # Remove 'jarvis' and polite/greeting prefix tokens
    non_jarvis_tokens = [w for w in tokens if w != "jarvis"]
    if not non_jarvis_tokens:
        return False

    # Filter greeting/filler words when evaluating the dismissal intent
    semantic_tokens = [w for w in non_jarvis_tokens if w not in {"hey", "hi", "hello", "yo", "please"}]
    if not semantic_tokens:
        return False

    semantic_str = " ".join(semantic_tokens)
    non_jarvis_str = " ".join(non_jarvis_tokens)

    phrases = dismissal_phrases or DEFAULT_DISMISSAL_PHRASES
    for phrase in phrases:
        clean_phrase = re.sub(r"[^\w\s]", "", phrase).lower().strip()
        if not clean_phrase:
            continue
        if semantic_str == clean_phrase or non_jarvis_str == clean_phrase:
            return True
        # For multi-word utterances, match phrase if inside semantic tokens and utterance is short (<= 5 tokens)
        if len(tokens) <= 5:
            phrase_tokens = clean_phrase.split()
            if all(pt in semantic_tokens for pt in phrase_tokens):
                query_words = {"what", "how", "why", "where", "who", "when", "can", "could", "tell", "show", "open", "search", "find", "play"}
                if not any(qw in tokens for qw in query_words):
                    return True

    return False


def is_speaker_echo(interruption_text: str, spoken_text: str) -> bool:
    """
    Checks if the transcribed interruption text is an echo of the assistant's
    own voice currently being spoken through the speakers.
    """
    if not interruption_text or not spoken_text:
        return False
    clean_interruption = re.sub(r"[^\w\s]", "", interruption_text).lower().strip()
    clean_spoken = re.sub(r"[^\w\s]", "", spoken_text).lower().strip()
    if not clean_interruption or not clean_spoken:
        return False
    if clean_interruption in clean_spoken:
        return True
    tokens_i = clean_interruption.split()
    if len(tokens_i) >= 3:
        for i in range(len(tokens_i) - 2):
            trigram = " ".join(tokens_i[i : i + 3])
            if trigram in clean_spoken:
                return True
    return False


def is_noise_or_wake_word_artifact(text: str) -> bool:
    """
    Detects if a transcribed utterance in conversation mode is ambient noise,
    a solitary wake-word artifact, or a common short acoustic hallucination.

    Returns True if the utterance should be discarded as noise rather than
    treated as a genuine conversational command.
    """
    if not text or not text.strip():
        return True

    # Check for Whisper bracketed audio captions (e.g. "[music]", "(clears throat)", "*sigh*")
    stripped = text.strip()
    if (
        (stripped.startswith("[") and stripped.endswith("]"))
        or (stripped.startswith("(") and stripped.endswith(")"))
        or (stripped.startswith("*") and stripped.endswith("*"))
    ):
        return True

    # Normalize tokens
    tokens = [w for w in re.sub(r"[^\w\s]", "", text).lower().split() if w]
    if not tokens:
        return True

    # Filter very short utterances (under 3-4 words) consisting solely of wake words or fillers
    if len(tokens) <= 3:
        clean_str = " ".join(tokens)
        wake_word_phrases = {
            "jarvis",
            "hey jarvis",
            "hi jarvis",
            "hello jarvis",
            "ok jarvis",
            "okay jarvis",
            "yo jarvis",
            "jarvis jarvis",
        }
        if clean_str in wake_word_phrases:
            return True

        # Pure acoustic filler artifacts produced by Whisper on silence / noise
        noise_fillers = {
            "you",
            "uh",
            "um",
            "ah",
            "er",
            "yeah",
            "yes",
            "oh",
            "huh",
            "hmm",
            "hm",
            "mm",
            "so",
            "the",
            "a",
            "i",
        }
        if clean_str in noise_fillers:
            return True

        # Utterances composed solely of combinations of wake-words and fillers
        if all(w in (wake_word_phrases | noise_fillers | {"hey", "hi", "ok", "okay", "hello"}) for w in tokens):
            return True

    return False


def extract_tools_from_messages(messages: Sequence[BaseMessage]) -> List[str]:
    """
    Extracts tool names invoked during a turn from LangChain messages.
    Inspects ToolMessage instances and AIMessage tool_calls while preserving call order.
    """
    tools: List[str] = []
    for msg in messages:
        if type(msg).__name__ == "ToolMessage" or (hasattr(msg, "name") and getattr(msg, "tool_call_id", None)):
            tool_name = getattr(msg, "name", None)
            if tool_name and tool_name not in tools:
                tools.append(str(tool_name))
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls and isinstance(tool_calls, list):
            for tc in tool_calls:
                if isinstance(tc, dict) and "name" in tc:
                    name = tc["name"]
                    if name and name not in tools:
                        tools.append(str(name))
    return tools


def print_banner(
    model: str,
    voice: str,
    whisper_model: str,
    enable_tray: bool,
    enable_memory: bool,
    continuous_mode: bool = True,
    conversation_timeout: float = 60.0,
    enable_orb: bool = True,
    enable_logging: bool = True,
    enable_greeting: bool = True,
    greeting_text: str = "Hello! I'm ready, how can I help you today?",
    enable_barge_in: bool = True,
    barge_in_threshold: float = 850.0,
    enable_wake_ack: bool = True,
    wake_ack_text: str = "Yes?",
) -> None:
    """Prints a styled startup banner with system configuration."""
    print("\n" + "=" * 78)
    print("      LOCAL JARVIS - AUTONOMOUS VOICE ASSISTANT PIPELINE")
    print("=" * 78)
    print(f"  Wake Word:         'Hey Jarvis' (openWakeWord ONNX)")
    print(f"  Wake Ack:          {'Active (\"' + wake_ack_text + '\")' if enable_wake_ack else 'Disabled (--no-wake-ack)'}")
    print(f"  STT Engine:        faster-whisper '{whisper_model}' (CPU INT8)")
    print(f"  LLM Brain:         {model} via local Ollama (RTX 3050 GPU)")
    print(f"  TTS Engine:        Piper '{voice}' (CPU offline playback)")
    print(f"  Agent Tools:       10 Tools (DateTime, Reminders, Search, Desktop, Clipboard, Sandbox)")
    print(f"  Vector Memory:     {'ChromaDB (all-MiniLM-L6-v2, CPU)' if enable_memory else 'Disabled'}")
    print(f"  Continuous Mode:   {'Active (' + str(conversation_timeout) + 's silence timeout safety net)' if continuous_mode else 'Disabled (Wake-word only)'}")
    print(f"  Startup Greeting:  {'Active (\"' + greeting_text + '\")' if enable_greeting else 'Disabled (--no-greeting)'}")
    print(f"  Speech Barge-In:   {'Active (threshold: ' + str(barge_in_threshold) + ' RMS)' if enable_barge_in else 'Disabled (--no-barge-in)'}")
    print(f"  Interaction Log:   {'Active (analysis/interactions.db)' if enable_logging else 'Disabled (--no-logging)'}")
    print(f"  System Tray Icon:  {'Active (pystray thread)' if enable_tray else 'Disabled (--no-tray)'}")
    print(f"  Floating Orb UI:   {'Active (Tkinter Canvas)' if enable_orb else 'Disabled (--no-orb)'}")
    print("  Controls:")
    print("    - Speak 'Hey Jarvis' followed by your command/question hands-free.")
    print("    - Speak follow-up questions hands-free without repeating 'Hey Jarvis'.")
    print("    - Say 'stop', 'goodbye', or 'that's all' to exit active conversation.")
    print("    - Interrupt Jarvis while speaking ('barge-in') to stop or ask a new question.")
    print("    - Right-click tray icon or orb -> 'Quit Jarvis' OR press Ctrl+C to exit.")
    print("=" * 78 + "\n")


class VoiceAssistantOrchestrator:
    """
    Main controller for the autonomous voice assistant pipeline.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = 150,
        whisper_model: str = "small",
        voice: str = DEFAULT_VOICE,
        wake_threshold: float = 0.35,
        audio_device: Optional[int] = None,
        enable_tray: bool = True,
        enable_orb: bool = True,
        enable_memory: bool = True,
        enable_logging: bool = True,
        is_test: bool = False,
        db_path: Optional[Union[str, Path]] = None,
        continuous_mode: bool = True,
        conversation_timeout: float = 60.0,
        stop_phrases: Optional[List[str]] = None,
        enable_greeting: bool = True,
        greeting_text: str = "Hello! I'm ready, how can I help you today?",
        enable_barge_in: bool = True,
        barge_in_threshold: float = 850.0,
        dismissal_phrases: Optional[List[str]] = None,
        enable_wake_ack: bool = True,
        wake_ack_text: str = "Yes?",
        debug: bool = False,
    ):
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.whisper_model = whisper_model
        self.voice = voice
        self.wake_threshold = wake_threshold
        self.audio_device = audio_device
        self.enable_tray = enable_tray
        self.enable_orb = enable_orb
        self.enable_memory = enable_memory
        self.enable_logging = enable_logging
        self.is_test = is_test or (os.environ.get("JARVIS_TEST_MODE") == "1")
        self.db_path = db_path
        self.continuous_mode = continuous_mode
        self.conversation_timeout = conversation_timeout
        self.stop_phrases = stop_phrases or list(DEFAULT_STOP_PHRASES)
        self.enable_greeting = enable_greeting
        self.greeting_text = greeting_text
        self.enable_barge_in = enable_barge_in
        self.barge_in_threshold = barge_in_threshold
        self.dismissal_phrases = dismissal_phrases or list(DEFAULT_DISMISSAL_PHRASES)
        self.enable_wake_ack = enable_wake_ack
        self.wake_ack_text = wake_ack_text
        self.pending_user_text: Optional[str] = None
        self.debug = debug

        self.shutdown_event = threading.Event()
        self.conversation_history: List[BaseMessage] = []

        # Components
        self.detector: Optional[WakeWordDetector] = None
        self.agent_app = None
        self.tray_app: Optional[JarvisTrayApp] = None
        self.orb_overlay: Optional[OrbOverlay] = None
        self.scheduler = None

    def _log_turn(
        self,
        user_text: str,
        assistant_response: str,
        tools_called: Optional[Sequence[str]] = None,
        stt_seconds: float = 0.0,
        brain_seconds: float = 0.0,
        tts_seconds: float = 0.0,
        total_seconds: Optional[float] = None,
        success: bool = True,
        trigger_mode: str = "wake_word",
    ) -> None:
        """Internal helper to dispatch interaction logging with orchestrator configuration."""
        if not self.enable_logging:
            return
        log_interaction(
            user_text=user_text,
            assistant_response=assistant_response,
            tools_called=tools_called,
            stt_seconds=stt_seconds,
            brain_seconds=brain_seconds,
            tts_seconds=tts_seconds,
            total_seconds=total_seconds,
            success=success,
            trigger_mode=trigger_mode,
            db_path=self.db_path,
            is_test=self.is_test,
        )

    def initialize(self) -> None:
        """Prepares and verifies all pipeline models and sub-systems before entering loop."""
        print("[Orchestrator] Initializing Local Jarvis pipeline components...")

        # 1. Start background reminder scheduler
        print("[Orchestrator] Starting background reminder scheduler...")
        self.scheduler = start_reminder_scheduler(interval=10.0)

        # 2. Initialize wake word detector
        print("[Orchestrator] Initializing WakeWordDetector (threshold: {:.2f})...".format(self.wake_threshold))
        self.detector = WakeWordDetector(
            threshold=self.wake_threshold,
            input_device=self.audio_device,
            debug=self.debug,
        )

        # 3. Pre-load / cache Whisper STT model
        print(f"[Orchestrator] Loading Whisper STT model ('{self.whisper_model}')...")
        get_transcriber(model_size=self.whisper_model)

        # 4. Compile LangGraph agent app
        print(f"[Orchestrator] Compiling LangGraph agent app for model '{self.model}'...")
        self.agent_app = get_agent_app(
            model=self.model,
            base_url=self.base_url,
            temperature=self.temperature,
            num_predict=self.max_tokens,
            enable_memory=self.enable_memory,
        )

        # 5. Verify Piper TTS voice model
        print(f"[Orchestrator] Verifying Piper TTS voice ('{self.voice}')...")
        get_piper_voice(self.voice)

        # 6. Initialize System Tray App if enabled
        if self.enable_tray:
            print("[Orchestrator] Launching System Tray icon...")

            def toggle_orb():
                if self.orb_overlay is not None:
                    self.orb_overlay.toggle_visibility()

            self.tray_app = JarvisTrayApp(
                on_quit=self.stop,
                on_toggle_orb=toggle_orb if self.enable_orb else None,
                initial_state=JarvisTrayState.LISTENING,
            )
            self.tray_app.start()

        # 7. Initialize Floating Orb Overlay UI if enabled
        if self.enable_orb:
            print("[Orchestrator] Launching Floating Orb Overlay UI...")
            self.orb_overlay = OrbOverlay(
                on_quit=self.stop,
                initial_state=JarvisTrayState.LISTENING.value,
            )
            self.orb_overlay.start()

        print("[Orchestrator] All components initialized successfully.\n")

    def _set_ui_state(self, state: JarvisTrayState | str, custom_msg: Optional[str] = None) -> None:
        """Helper to update both tray icon and floating orb overlay in sync."""
        if self.tray_app is not None:
            self.tray_app.set_state(state, custom_message=custom_msg)
        if self.orb_overlay is not None:
            state_val = state.value if isinstance(state, JarvisTrayState) else str(state)
            self.orb_overlay.set_state(state_val)

    def _set_tray_state(self, state: JarvisTrayState | str, custom_msg: Optional[str] = None) -> None:
        """Alias for _set_ui_state to preserve backward compatibility."""
        self._set_ui_state(state, custom_msg)

    def _speak_startup_greeting(self) -> None:
        """Speaks the initial startup greeting via TTS before entering listening loop."""
        if not self.enable_greeting or not self.greeting_text or self.shutdown_event.is_set():
            return
        print(f"[Pipeline] Startup greeting: \"{self.greeting_text}\"")
        self._set_ui_state(JarvisTrayState.SPEAKING, self.greeting_text)
        try:
            interrupted, next_query, speak_elapsed = self._speak_with_barge_in(
                text=self.greeting_text,
                trigger_mode="startup_greeting",
            )
            if interrupted and next_query:
                print(f"[Pipeline] User interrupted startup greeting with command: \"{next_query}\"")
                self.pending_user_text = next_query
        except Exception as e:
            print(f"[Pipeline Warning] Startup greeting TTS error: {e}")
        finally:
            self._set_ui_state(JarvisTrayState.LISTENING, "Listening for 'Hey Jarvis'...")

    def _handle_wake_word_detected(self, score: float) -> None:
        """
        Callback invoked when the wake word 'Hey Jarvis' is detected.
        Optionally speaks a brief audible acknowledgment ('Yes?') before speech recording begins.
        """
        print(f"\n[Pipeline] Wake word detected! (confidence: {score:.3f})")
        if self.enable_wake_ack and self.wake_ack_text and not self.shutdown_event.is_set():
            print(f"[Pipeline] Acknowledging wake word with: \"{self.wake_ack_text}\"")
            self._set_ui_state(JarvisTrayState.SPEAKING, f"Speaking: \"{self.wake_ack_text}\"")
            try:
                speak(self.wake_ack_text, voice=self.voice, blocking=True)
            except Exception as e:
                print(f"[Pipeline Error] Failed to speak wake acknowledgment: {e}")
            finally:
                self._set_ui_state(JarvisTrayState.LISTENING, "Listening for command...")

    def _speak_with_barge_in(
        self,
        text: str,
        trigger_mode: str = "wake_word",
    ) -> Tuple[bool, Optional[str], float]:
        """
        Speaks the given text via Piper TTS while monitoring microphone input in the
        background for user speech interruption ("barge-in").

        Redesign:
        Instead of aborting playback immediately on microphone energy, playback
        continues while speech is captured in the background.
        Once the user pauses (~0.8s silence), Whisper transcribes the speech:
        1. If acoustic noise, artifact, or empty: ignored, playback continues seamlessly.
        2. If speaker echo of assistant voice: ignored, playback continues seamlessly.
        3. If bare dismissal without 'Jarvis' (e.g. solitary 'stop', 'okay'): ignored to
           prevent false interruptions from ambient chatter.
        4. If confirmed Jarvis dismissal ('Jarvis stop', 'Jarvis okay', 'Stop Jarvis'):
           aborts playback immediately, silently ends current response, returns (True, None, tts_duration).
        5. If valid new command or question: aborts playback immediately, returns
           (True, new_command, tts_duration) to chain directly into the next turn.

        Args:
            text: Text to speak.
            trigger_mode: Calling context ('wake_word', 'conversation', or 'startup_greeting').

        Returns:
            Tuple of (interrupted: bool, next_user_text: Optional[str], tts_duration: float)
        """
        if not text or not text.strip():
            return False, None, 0.0

        if not self.enable_barge_in or self.detector is None:
            speak_start = time.perf_counter()
            speak(
                text=text,
                voice=self.voice,
                blocking=True,
                on_start_playback=lambda ttfa: print(f"[TTS] Playback started in {ttfa:.2f}s (TTFA)"),
            )
            tts_duration = time.perf_counter() - speak_start
            return False, None, tts_duration

        interrupt_event = threading.Event()
        speak_start = time.perf_counter()
        producer_error: List[Exception] = []

        def tts_worker():
            try:
                speak(
                    text=text,
                    voice=self.voice,
                    blocking=True,
                    interrupt_event=interrupt_event,
                    on_start_playback=lambda ttfa: print(f"[TTS] Playback started in {ttfa:.2f}s (TTFA)"),
                )
            except Exception as exc:
                producer_error.append(exc)

        tts_thread = threading.Thread(
            target=tts_worker,
            name="TTS_BargeInWorker",
            daemon=True,
        )

        try:
            self.detector.start_stream()
        except Exception as stream_err:
            if self.debug:
                print(f"[DEBUG Barge-in] Failed to start input stream: {stream_err}")

        # Drain residual audio in queue before playback begins
        if hasattr(self.detector, "audio_queue") and hasattr(self.detector.audio_queue, "empty"):
            while not self.detector.audio_queue.empty():
                try:
                    self.detector.audio_queue.get_nowait()
                except Exception:
                    break

        tts_thread.start()

        pre_roll_chunks = max(2, int(0.35 / CHUNK_DURATION))
        pre_roll: collections.deque = collections.deque(maxlen=pre_roll_chunks)
        consecutive_speech_chunks = 0
        playback_begin = time.time()
        grace_period = 0.25

        collecting_speech = False
        recorded_frames: List[np.ndarray] = []
        silence_chunks = 0
        record_start = 0.0
        silence_limit_chunks = int(0.8 / CHUNK_DURATION)

        while tts_thread.is_alive() or collecting_speech:
            if self.shutdown_event.is_set():
                interrupt_event.set()
                break

            try:
                chunk = self.detector.audio_queue.get(timeout=0.04)
            except Exception:
                continue

            if not isinstance(chunk, np.ndarray):
                continue

            chunk_rms = float(np.sqrt(np.mean(chunk.astype(np.float32) ** 2)))

            if not collecting_speech:
                pre_roll.append(chunk)

                if time.time() - playback_begin < grace_period:
                    continue

                if self.debug:
                    print(f"[DEBUG Barge-in] Mic RMS: {chunk_rms:6.1f} | Threshold: {self.barge_in_threshold:.1f}")

                if chunk_rms >= self.barge_in_threshold:
                    consecutive_speech_chunks += 1
                    if consecutive_speech_chunks >= 2:
                        print(
                            f"\n[Barge-in] Speech sound detected during playback "
                            f"(RMS: {chunk_rms:.1f} >= {self.barge_in_threshold:.1f}). Verifying speech..."
                        )
                        collecting_speech = True
                        recorded_frames = list(pre_roll)
                        silence_chunks = 0
                        record_start = time.time()
                else:
                    consecutive_speech_chunks = max(0, consecutive_speech_chunks - 1)
            else:
                # Active speech collection while TTS continues playing
                recorded_frames.append(chunk)

                if chunk_rms < DEFAULT_SILENCE_RMS * 1.5:
                    silence_chunks += 1
                else:
                    silence_chunks = 0

                speech_elapsed = time.time() - record_start
                speech_finished = (silence_chunks >= silence_limit_chunks) or (speech_elapsed > MAX_RECORDING_SECONDS)

                if speech_finished:
                    # User finished speaking. Transcribe to verify intent before aborting TTS
                    AUDIO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
                    timestamp_str = time.strftime("%Y%m%d_%H%M%S")
                    clip_path = AUDIO_CACHE_DIR / f"interruption_{timestamp_str}_{int(time.time() * 1000) % 1000}.wav"

                    if recorded_frames:
                        combined_audio = np.concatenate(recorded_frames).astype(np.int16)
                    else:
                        combined_audio = np.zeros(CHUNK_SAMPLES, dtype=np.int16)

                    wavfile.write(str(clip_path), SAMPLE_RATE, combined_audio)

                    interruption_text = ""
                    try:
                        stt_start = time.perf_counter()
                        interruption_text = transcribe_audio(
                            audio_path=str(clip_path.resolve()),
                            model_size=self.whisper_model,
                        )
                        stt_elapsed = time.perf_counter() - stt_start
                        print(f"[Barge-in STT] Transcribed in {stt_elapsed:.2f}s: \"{interruption_text}\"")
                    except Exception as stt_err:
                        print(f"[Barge-in Error] Transcription failed: {stt_err}")
                        interruption_text = ""

                    # Case A: Empty speech or acoustic noise/artifact
                    if not interruption_text or not interruption_text.strip() or is_noise_or_wake_word_artifact(interruption_text):
                        print(f"[Barge-in] Sound was noise/artifact (\"{interruption_text.strip()}\"). Continuing playback.")
                        collecting_speech = False
                        recorded_frames = []
                        consecutive_speech_chunks = 0
                        silence_chunks = 0
                        pre_roll.clear()
                        continue

                    # Case B: Speaker echo of Jarvis's own TTS output
                    if is_speaker_echo(interruption_text, text):
                        print(f"[Barge-in] Detected speaker echo of assistant voice (\"{interruption_text.strip()}\"). Continuing playback.")
                        collecting_speech = False
                        recorded_frames = []
                        consecutive_speech_chunks = 0
                        silence_chunks = 0
                        pre_roll.clear()
                        continue

                    # Case C: Confirmed Jarvis dismissal ("Jarvis stop", "Jarvis okay", etc.)
                    if is_jarvis_dismissal(interruption_text, self.dismissal_phrases):
                        print(f"[Barge-in] Confirmed Jarvis dismissal: \"{interruption_text}\". Aborting TTS playback.")
                        interrupt_event.set()
                        tts_thread.join(timeout=1.0)
                        tts_duration = time.perf_counter() - speak_start
                        return True, None, tts_duration

                    # Case D: Bare dismissal without "Jarvis" (e.g. solitary "stop", "okay")
                    if is_dismissal_phrase(interruption_text, self.dismissal_phrases):
                        print(
                            f"[Barge-in] Dismissal word \"{interruption_text}\" lacked required \"Jarvis\" prefix. "
                            "Ignoring to prevent accidental interruption."
                        )
                        collecting_speech = False
                        recorded_frames = []
                        consecutive_speech_chunks = 0
                        silence_chunks = 0
                        pre_roll.clear()
                        continue

                    # Case E: Valid new command or question!
                    print(f"[Barge-in] Valid new command received during playback: \"{interruption_text}\". Aborting TTS and chaining.")
                    interrupt_event.set()
                    tts_thread.join(timeout=1.0)
                    tts_duration = time.perf_counter() - speak_start
                    return True, interruption_text, tts_duration

        tts_thread.join(timeout=1.0)
        tts_duration = time.perf_counter() - speak_start

        if producer_error:
            raise producer_error[0]

        return False, None, tts_duration

    def run_turn(self, pending_user_text: Optional[str] = None) -> bool:
        """
        Executes a single end-to-end voice pipeline turn:
        Listening -> Heard wake word -> Transcribing -> Thinking -> Speaking -> Listening again.

        Supports pending_user_text or self.pending_user_text from barge-in interruptions,
        bypassing the wake-word detection stage when the user already spoke their next command.

        Returns True if the loop should continue, False if shutdown was requested.
        """
        if self.shutdown_event.is_set():
            return False

        user_text = pending_user_text or self.pending_user_text
        self.pending_user_text = None
        stt_elapsed = 0.0

        if not user_text:
            # ---------------------------------------------------------------------
            # STAGE 1: Wake Word Listening
            # ---------------------------------------------------------------------
            print("\n" + "-" * 60)
            print("[Pipeline] Stage 1: LISTENING for 'Hey Jarvis' (hands-free)...")
            self._set_tray_state(JarvisTrayState.LISTENING, "Listening for 'Hey Jarvis'...")

            try:
                audio_clip_path = self.detector.listen_and_record(
                    on_wake_word_detected=self._handle_wake_word_detected,
                    stop_event=self.shutdown_event,
                )
            except Exception as e:
                if self.shutdown_event.is_set():
                    return False
                print(f"[Pipeline Error] Wake word detection error: {e}")
                self._set_tray_state(JarvisTrayState.ERROR, "Wake Word Error")
                time.sleep(1.0)
                return True

            if self.shutdown_event.is_set() or not audio_clip_path:
                return False

            # ---------------------------------------------------------------------
            # STAGE 2: Transcribing Speech (STT)
            # ---------------------------------------------------------------------
            print("[Pipeline] Stage 2: TRANSCRIBING speech with Whisper...")
            self._set_tray_state(JarvisTrayState.PROCESSING, "Transcribing user speech...")

            try:
                stt_start = time.perf_counter()
                user_text = transcribe_audio(
                    audio_path=audio_clip_path,
                    model_size=self.whisper_model,
                )
                stt_elapsed = time.perf_counter() - stt_start
                print(f"[STT] Transcribed in {stt_elapsed:.2f}s: \"{user_text}\"")
            except Exception as e:
                stt_elapsed = time.perf_counter() - stt_start if "stt_start" in locals() else 0.0
                print(f"[Pipeline Error] STT transcription failed: {e}")
                self._set_tray_state(JarvisTrayState.ERROR, f"STT Error: {e}")
                self._log_turn(
                    user_text="",
                    assistant_response="",
                    tools_called=[],
                    stt_seconds=stt_elapsed,
                    brain_seconds=0.0,
                    tts_seconds=0.0,
                    total_seconds=stt_elapsed,
                    success=False,
                    trigger_mode="wake_word",
                )
                time.sleep(1.0)
                return True

            # Check for empty or silent utterance
            if not user_text or not user_text.strip():
                print("[Pipeline] No speech recognized in audio clip. Returning to listening mode.")
                return True
        else:
            print(f"\n[Pipeline] Direct turn received from speech interruption: \"{user_text}\"")

        print(f"\nUser > {user_text}")

        # ---------------------------------------------------------------------
        # STAGE 3: Agent Brain & Tool Execution (LangGraph)
        # ---------------------------------------------------------------------
        print("[Pipeline] Stage 3: THINKING (LangGraph agent + memory + tools)...")
        self._set_tray_state(JarvisTrayState.PROCESSING, f"Thinking about: {user_text[:25]}...")

        response_text = ""
        agent_elapsed = 0.0
        turn_tools: List[str] = []
        history_len_before = len(self.conversation_history)
        try:
            agent_start = time.perf_counter()
            response_text, self.conversation_history = run_agent(
                user_input=user_text,
                history=self.conversation_history,
                app=self.agent_app,
                model=self.model,
                base_url=self.base_url,
                temperature=self.temperature,
                num_predict=self.max_tokens,
                enable_memory=self.enable_memory,
            )
            agent_elapsed = time.perf_counter() - agent_start
            print(f"[Brain] Response generated in {agent_elapsed:.2f}s")
            turn_tools = extract_tools_from_messages(self.conversation_history[history_len_before:])
        except Exception as e:
            agent_elapsed = time.perf_counter() - agent_start if "agent_start" in locals() else 0.0
            print(f"[Pipeline Error] Agent execution error: {e}")
            self._set_tray_state(JarvisTrayState.ERROR, f"Agent Error: {e}")
            self._log_turn(
                user_text=user_text,
                assistant_response="",
                tools_called=[],
                stt_seconds=stt_elapsed,
                brain_seconds=agent_elapsed,
                tts_seconds=0.0,
                total_seconds=stt_elapsed + agent_elapsed,
                success=False,
                trigger_mode="wake_word",
            )
            time.sleep(1.0)
            return True

        # Sanitize response text so raw tool calls or narration are never displayed or spoken
        response_text = sanitize_speech_text(response_text)
        print(f"\nJarvis > {response_text}\n")

        # ---------------------------------------------------------------------
        # STAGE 4: Speaking Response Aloud (Piper TTS with Barge-In)
        # ---------------------------------------------------------------------
        speak_elapsed = 0.0
        interrupted = False
        next_user_query: Optional[str] = None
        if response_text and response_text.strip():
            print(f"[Pipeline] Stage 4: SPEAKING response aloud (voice: '{self.voice}')...")
            self._set_tray_state(JarvisTrayState.SPEAKING, "Speaking response aloud...")

            try:
                interrupted, next_user_query, speak_elapsed = self._speak_with_barge_in(
                    text=response_text,
                    trigger_mode="wake_word",
                )
                if interrupted:
                    print(f"[TTS] Playback interrupted after {speak_elapsed:.2f}s.")
                else:
                    print(f"[TTS] Finished speaking in {speak_elapsed:.2f}s.")
            except Exception as e:
                speak_elapsed = time.perf_counter() - speak_start if "speak_start" in locals() else 0.0
                print(f"[Pipeline Error] TTS playback error: {e}")
                self._set_tray_state(JarvisTrayState.ERROR, f"TTS Error: {e}")
                self._log_turn(
                    user_text=user_text,
                    assistant_response=response_text,
                    tools_called=turn_tools,
                    stt_seconds=stt_elapsed,
                    brain_seconds=agent_elapsed,
                    tts_seconds=speak_elapsed,
                    total_seconds=stt_elapsed + agent_elapsed + speak_elapsed,
                    success=False,
                    trigger_mode="wake_word",
                )
                time.sleep(1.0)
                return True

        # Log completed wake-word turn
        self._log_turn(
            user_text=user_text,
            assistant_response=response_text,
            tools_called=turn_tools,
            stt_seconds=stt_elapsed,
            brain_seconds=agent_elapsed,
            tts_seconds=speak_elapsed,
            total_seconds=stt_elapsed + agent_elapsed + speak_elapsed,
            success=True,
            trigger_mode="wake_word",
        )

        # Handle speech interruption outcome
        if interrupted:
            if next_user_query:
                # User spoke a new question/command during playback!
                # Chain directly to the next turn without asking for wake-word or listening again.
                self.pending_user_text = next_user_query
                return True
            else:
                # User spoke a dismissal phrase ("stop", "got it", "okay", etc.)
                # Silently end current response and return to standby listening.
                self._set_tray_state(JarvisTrayState.LISTENING, "Listening for 'Hey Jarvis'...")
                return True

        # ---------------------------------------------------------------------
        # STAGE 5: Hands-Free Continuous Conversation Loop (Multi-Turn)
        # ---------------------------------------------------------------------
        if self.continuous_mode and not self.shutdown_event.is_set():
            print("\n" + "=" * 60)
            print("[Pipeline] Stage 5: ENTERING ACTIVE CONVERSATION MODE")
            print("  - Speak follow-up hands-free without repeating 'Hey Jarvis'.")
            print("  - Say 'stop', 'goodbye', or 'that's all' to exit conversation.")
            print(f"  - Times out quietly after {self.conversation_timeout:.1f}s of silence (safety net).")
            print("=" * 60)

            conv_pending_text: Optional[str] = None
            while not self.shutdown_event.is_set():
                if conv_pending_text is not None:
                    followup_text = conv_pending_text
                    conv_pending_text = None
                    followup_stt_elapsed = 0.0
                else:
                    self._set_tray_state(
                        JarvisTrayState.CONVERSATION,
                        "Active Conversation (Listening hands-free)...",
                    )
                    try:
                        followup_clip_path = self.detector.record_utterance(
                            stop_event=self.shutdown_event,
                            speech_timeout=self.conversation_timeout,
                        )
                    except Exception as e:
                        print(f"[Conversation Error] Error capturing audio: {e}")
                        break

                    if self.shutdown_event.is_set():
                        return False

                    # Check for silence timeout (no speech detected within window)
                    if not followup_clip_path:
                        print(
                            f"[Conversation] Silence timeout ({self.conversation_timeout:.1f}s). "
                            "Quietly reverting to wake-word standby (conversation history retained)."
                        )
                        break

                    # STT Transcribe
                    self._set_tray_state(JarvisTrayState.PROCESSING, "Transcribing user speech...")
                    followup_text = ""
                    followup_stt_elapsed = 0.0
                    try:
                        followup_start = time.perf_counter()
                        followup_text = transcribe_audio(
                            audio_path=followup_clip_path,
                            model_size=self.whisper_model,
                        )
                        followup_stt_elapsed = time.perf_counter() - followup_start
                        print(f"[STT] Transcribed in {followup_stt_elapsed:.2f}s: \"{followup_text}\"")
                    except Exception as e:
                        followup_stt_elapsed = time.perf_counter() - followup_start if "followup_start" in locals() else 0.0
                        print(f"[Conversation Error] STT failed: {e}")
                        self._set_tray_state(JarvisTrayState.ERROR, f"STT Error: {e}")
                        self._log_turn(
                            user_text="",
                            assistant_response="",
                            tools_called=[],
                            stt_seconds=followup_stt_elapsed,
                            brain_seconds=0.0,
                            tts_seconds=0.0,
                            total_seconds=followup_stt_elapsed,
                            success=False,
                            trigger_mode="conversation",
                        )
                        time.sleep(1.0)
                        continue

                    if not followup_text or not followup_text.strip():
                        print("[Conversation] Empty utterance. Continuing active listening...")
                        continue

                print(f"\nUser (Conversation) > {followup_text}")

                # Check if user spoke a stop phrase
                if is_stop_phrase(followup_text, self.stop_phrases):
                    print(f"[Conversation] Stop phrase detected ('{followup_text}'). Exiting conversation mode.")
                    self._set_tray_state(JarvisTrayState.SPEAKING, "Goodbye!")
                    try:
                        speak("Goodbye!", voice=self.voice, blocking=True)
                    except Exception as tts_err:
                        if self.debug:
                            print(f"[DEBUG] Stop phrase TTS error: {tts_err}")
                    self._log_turn(
                        user_text=followup_text,
                        assistant_response="Goodbye!",
                        tools_called=[],
                        stt_seconds=followup_stt_elapsed,
                        brain_seconds=0.0,
                        tts_seconds=0.0,
                        total_seconds=followup_stt_elapsed,
                        success=True,
                        trigger_mode="conversation",
                    )
                    break

                # Filter ambient noise or solitary wake-word artifacts in conversation mode
                if is_noise_or_wake_word_artifact(followup_text):
                    print(
                        f"[Conversation] Filtered ambient noise / wake-word artifact ('{followup_text}'). "
                        "Continuing hands-free listening..."
                    )
                    continue

                # Brain & LangGraph Agent
                self._set_tray_state(JarvisTrayState.PROCESSING, f"Thinking about: {followup_text[:25]}...")
                followup_resp = ""
                agent_elapsed = 0.0
                conv_turn_tools: List[str] = []
                history_len_before = len(self.conversation_history)
                try:
                    agent_start = time.perf_counter()
                    followup_resp, self.conversation_history = run_agent(
                        user_input=followup_text,
                        history=self.conversation_history,
                        app=self.agent_app,
                        model=self.model,
                        base_url=self.base_url,
                        temperature=self.temperature,
                        num_predict=self.max_tokens,
                        enable_memory=self.enable_memory,
                    )
                    agent_elapsed = time.perf_counter() - agent_start
                    print(f"[Brain] Response generated in {agent_elapsed:.2f}s")
                    conv_turn_tools = extract_tools_from_messages(self.conversation_history[history_len_before:])
                except Exception as e:
                    agent_elapsed = time.perf_counter() - agent_start if "agent_start" in locals() else 0.0
                    print(f"[Conversation Error] Agent execution error: {e}")
                    self._set_tray_state(JarvisTrayState.ERROR, f"Agent Error: {e}")
                    self._log_turn(
                        user_text=followup_text,
                        assistant_response="",
                        tools_called=[],
                        stt_seconds=followup_stt_elapsed,
                        brain_seconds=agent_elapsed,
                        tts_seconds=0.0,
                        total_seconds=followup_stt_elapsed + agent_elapsed,
                        success=False,
                        trigger_mode="conversation",
                    )
                    time.sleep(1.0)
                    continue

                followup_resp = sanitize_speech_text(followup_resp)
                print(f"\nJarvis (Conversation) > {followup_resp}\n")

                # Speaking response aloud (Piper TTS with barge-in)
                speak_elapsed = 0.0
                conv_interrupted = False
                next_conv_query: Optional[str] = None
                if followup_resp and followup_resp.strip():
                    self._set_tray_state(JarvisTrayState.SPEAKING, "Speaking response aloud...")
                    try:
                        conv_interrupted, next_conv_query, speak_elapsed = self._speak_with_barge_in(
                            text=followup_resp,
                            trigger_mode="conversation",
                        )
                        if conv_interrupted:
                            print(f"[TTS] Conversation playback interrupted after {speak_elapsed:.2f}s.")
                        else:
                            print(f"[TTS] Finished speaking in {speak_elapsed:.2f}s.")
                    except Exception as e:
                        speak_elapsed = time.perf_counter() - speak_start if "speak_start" in locals() else 0.0
                        print(f"[Conversation Error] TTS playback error: {e}")
                        self._set_tray_state(JarvisTrayState.ERROR, f"TTS Error: {e}")
                        self._log_turn(
                            user_text=followup_text,
                            assistant_response=followup_resp,
                            tools_called=conv_turn_tools,
                            stt_seconds=followup_stt_elapsed,
                            brain_seconds=agent_elapsed,
                            tts_seconds=speak_elapsed,
                            total_seconds=followup_stt_elapsed + agent_elapsed + speak_elapsed,
                            success=False,
                            trigger_mode="conversation",
                        )
                        time.sleep(1.0)
                        continue

                # Log completed conversation-mode turn
                self._log_turn(
                    user_text=followup_text,
                    assistant_response=followup_resp,
                    tools_called=conv_turn_tools,
                    stt_seconds=followup_stt_elapsed,
                    brain_seconds=agent_elapsed,
                    tts_seconds=speak_elapsed,
                    total_seconds=followup_stt_elapsed + agent_elapsed + speak_elapsed,
                    success=True,
                    trigger_mode="conversation",
                )

                if conv_interrupted:
                    if next_conv_query:
                        if is_stop_phrase(next_conv_query, self.stop_phrases):
                            print(f"[Conversation] Stop phrase detected during interruption ('{next_conv_query}'). Exiting conversation mode.")
                            self._set_tray_state(JarvisTrayState.SPEAKING, "Goodbye!")
                            try:
                                speak("Goodbye!", voice=self.voice, blocking=True)
                            except Exception:
                                pass
                            break
                        # Directly chain to next conversation turn without re-recording
                        conv_pending_text = next_conv_query
                        continue
                    else:
                        # Dismissal phrase: silently end response and continue hands-free listening
                        print("[Conversation] Interruption dismissal detected. Continuing hands-free listening...")
                        continue

        print("[Pipeline] Reverting to wake word listening.")
        return True

    def run(self) -> None:
        """Main continuous execution loop."""
        self.initialize()

        # Startup greeting
        self._speak_startup_greeting()

        print("[Pipeline] System ready and listening for 'Hey Jarvis'. Press Ctrl+C to stop.")
        try:
            while not self.shutdown_event.is_set():
                continue_loop = self.run_turn()
                if not continue_loop:
                    break
        except KeyboardInterrupt:
            print("\n[Orchestrator] KeyboardInterrupt received. Stopping pipeline...")
        finally:
            self.stop()

    def stop(self) -> None:
        """Gracefully halts all pipeline threads, audio streams, and background tasks."""
        if self.shutdown_event.is_set():
            return

        print("\n[Orchestrator] Shutting down Local Jarvis...")
        self.shutdown_event.set()

        # Stop wake word detector stream
        if self.detector is not None:
            try:
                self.detector.close()
            except Exception as e:
                if self.debug:
                    print(f"[DEBUG] Error closing detector: {e}")

        # Stop reminder scheduler
        try:
            stop_reminder_scheduler()
        except Exception as e:
            if self.debug:
                print(f"[DEBUG] Error stopping scheduler: {e}")

        # Stop tray icon
        if self.tray_app is not None:
            try:
                self.tray_app.stop()
            except Exception as e:
                if self.debug:
                    print(f"[DEBUG] Error stopping tray app: {e}")

        # Stop floating orb overlay
        if self.orb_overlay is not None:
            try:
                self.orb_overlay.stop()
            except Exception as e:
                if self.debug:
                    print(f"[DEBUG] Error stopping orb overlay: {e}")

        # Flush and shutdown interaction logging
        if self.enable_logging:
            try:
                flush_logging(timeout=2.0)
                shutdown_logging(timeout=2.0)
            except Exception as e:
                if self.debug:
                    print(f"[DEBUG] Error shutting down logging: {e}")

        print("[Orchestrator] Local Jarvis shutdown complete.")


def main():
    parser = argparse.ArgumentParser(
        description="Local Jarvis - Autonomous Voice Assistant Pipeline Orchestrator (Phase 7)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Ollama LLM model to use (default: '{DEFAULT_MODEL}')",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=DEFAULT_BASE_URL,
        help=f"Ollama server base URL (default: '{DEFAULT_BASE_URL}')",
    )
    parser.add_argument(
        "--whisper-model",
        type=str,
        default="small",
        help="faster-whisper model size ('tiny', 'base', 'small', default: 'small')",
    )
    parser.add_argument(
        "--voice",
        type=str,
        default=DEFAULT_VOICE,
        choices=["ryan", "lessac"],
        help=f"Piper TTS voice model (default: '{DEFAULT_VOICE}')",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.35,
        help="openWakeWord detection confidence threshold (0.0 to 1.0, default: 0.35)",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="Audio input device index (default: system default input device)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=150,
        help="Maximum tokens for agent response generation (default: 150)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help=f"Temperature for LLM reasoning (default: {DEFAULT_TEMPERATURE})",
    )
    parser.add_argument(
        "--no-continuous",
        action="store_true",
        help="Disable hands-free continuous conversation mode (wake-word required for every turn)",
    )
    parser.add_argument(
        "--conversation-timeout",
        type=float,
        default=60.0,
        help="Seconds of silence in continuous mode before reverting to wake-word listening safety net (default: 60.0)",
    )
    parser.add_argument(
        "--stop-phrases",
        type=str,
        default=None,
        help="Comma-separated custom stop phrases (default: 'stop,goodbye,bye,that\\'s all,thank you jarvis')",
    )
    parser.add_argument(
        "--no-tray",
        action="store_true",
        help="Run in headless console mode without system tray icon",
    )
    parser.add_argument(
        "--no-orb",
        action="store_true",
        help="Disable floating glowing orb UI overlay",
    )
    parser.add_argument(
        "--no-memory",
        action="store_true",
        help="Disable ChromaDB persistent vector memory",
    )
    parser.add_argument(
        "--no-logging",
        action="store_true",
        help="Disable interaction logging to SQLite database",
    )
    parser.add_argument(
        "--no-greeting",
        action="store_true",
        help="Disable spoken startup greeting on launch",
    )
    parser.add_argument(
        "--greeting-text",
        type=str,
        default="Hello! I'm ready, how can I help you today?",
        help="Custom spoken text for the startup greeting",
    )
    parser.add_argument(
        "--no-barge-in",
        action="store_true",
        help="Disable speech interruption ('barge-in') during TTS playback",
    )
    parser.add_argument(
        "--barge-in-threshold",
        type=float,
        default=850.0,
        help="Microphone RMS energy threshold for detecting speech interruption during playback (default: 850.0)",
    )
    parser.add_argument(
        "--dismissal-phrases",
        type=str,
        default=None,
        help="Comma-separated list of short dismissal phrases that cancel TTS without starting a new turn",
    )
    parser.add_argument(
        "--no-wake-ack",
        action="store_true",
        help="Disable audible acknowledgment ('Yes?') spoken immediately after wake word detection",
    )
    parser.add_argument(
        "--wake-ack-text",
        type=str,
        default="Yes?",
        help="Custom short spoken text for the wake-word acknowledgment (default: 'Yes?')",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose debug logging for wake word and barge-in detection",
    )
    parser.add_argument(
        "--list-launchable-apps",
        action="store_true",
        help="Scan and list all discoverable Start Menu, Desktop, and Windows Store applications, then exit",
    )
    parser.add_argument(
        "--list-ui-elements",
        type=str,
        default=None,
        metavar="APP_NAME",
        help="Inspect and list accessible UI elements for a running application, then exit",
    )

    args = parser.parse_args()

    if args.list_launchable_apps:
        from src.agent.tools.desktop import list_launchable_apps
        list_launchable_apps()
        sys.exit(0)

    if args.list_ui_elements:
        from src.agent.tools.desktop import list_ui_elements
        list_ui_elements(args.list_ui_elements)
        sys.exit(0)

    enable_tray = not args.no_tray
    enable_orb = not args.no_orb
    enable_memory = not args.no_memory
    enable_logging = not args.no_logging
    continuous_mode = not args.no_continuous
    enable_greeting = not args.no_greeting
    enable_barge_in = not args.no_barge_in
    enable_wake_ack = not args.no_wake_ack
    stop_phrases_list = (
        [p.strip() for p in args.stop_phrases.split(",") if p.strip()]
        if args.stop_phrases
        else None
    )
    dismissal_phrases_list = (
        [p.strip() for p in args.dismissal_phrases.split(",") if p.strip()]
        if args.dismissal_phrases
        else None
    )

    print_banner(
        model=args.model,
        voice=args.voice,
        whisper_model=args.whisper_model,
        enable_tray=enable_tray,
        enable_orb=enable_orb,
        enable_memory=enable_memory,
        enable_logging=enable_logging,
        continuous_mode=continuous_mode,
        conversation_timeout=args.conversation_timeout,
        enable_greeting=enable_greeting,
        greeting_text=args.greeting_text,
        enable_barge_in=enable_barge_in,
        barge_in_threshold=args.barge_in_threshold,
        enable_wake_ack=enable_wake_ack,
        wake_ack_text=args.wake_ack_text,
    )

    orchestrator = VoiceAssistantOrchestrator(
        model=args.model,
        base_url=args.base_url,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        whisper_model=args.whisper_model,
        voice=args.voice,
        wake_threshold=args.threshold,
        audio_device=args.device,
        enable_tray=enable_tray,
        enable_orb=enable_orb,
        enable_memory=enable_memory,
        enable_logging=enable_logging,
        continuous_mode=continuous_mode,
        conversation_timeout=args.conversation_timeout,
        stop_phrases=stop_phrases_list,
        enable_greeting=enable_greeting,
        greeting_text=args.greeting_text,
        enable_barge_in=enable_barge_in,
        barge_in_threshold=args.barge_in_threshold,
        dismissal_phrases=dismissal_phrases_list,
        enable_wake_ack=enable_wake_ack,
        wake_ack_text=args.wake_ack_text,
        debug=args.debug,
    )

    # Attach signal handler for graceful Ctrl+C
    def sigint_handler(signum, frame):
        print("\n[Orchestrator] Interrupt signal caught.")
        orchestrator.stop()

    signal.signal(signal.SIGINT, sigint_handler)
    signal.signal(signal.SIGTERM, sigint_handler)

    orchestrator.run()


if __name__ == "__main__":
    main()
