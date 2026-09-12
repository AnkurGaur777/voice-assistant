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
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import List, Optional

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.messages import BaseMessage

from src.agent.graph import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    get_agent_app,
    run_agent,
)
from src.agent.tools.reminders import start_reminder_scheduler, stop_reminder_scheduler
from src.stt import transcribe_audio, get_transcriber
from src.tts import DEFAULT_VOICE, get_piper_voice, speak, sanitize_speech_text
from src.wake_word import WakeWordDetector
from tray.tray_app import JarvisTrayApp, JarvisTrayState


def print_banner(model: str, voice: str, whisper_model: str, enable_tray: bool, enable_memory: bool) -> None:
    """Prints a styled startup banner with system configuration."""
    print("\n" + "=" * 78)
    print("      LOCAL JARVIS - AUTONOMOUS VOICE ASSISTANT PIPELINE")
    print("=" * 78)
    print(f"  Wake Word:         'Hey Jarvis' (openWakeWord ONNX)")
    print(f"  STT Engine:        faster-whisper '{whisper_model}' (CPU INT8)")
    print(f"  LLM Brain:         {model} via local Ollama (RTX 3050 GPU)")
    print(f"  TTS Engine:        Piper '{voice}' (CPU offline playback)")
    print(f"  Agent Tools:       9 Tools (DateTime, Reminders, Search, Desktop, Clipboard, Sandbox)")
    print(f"  Vector Memory:     {'ChromaDB (all-MiniLM-L6-v2, CPU)' if enable_memory else 'Disabled'}")
    print(f"  System Tray Icon:  {'Active (pystray thread)' if enable_tray else 'Disabled (--no-tray)'}")
    print("  Controls:")
    print("    - Speak 'Hey Jarvis' followed by your command/question hands-free.")
    print("    - Right-click tray icon -> 'Quit Jarvis' OR press Ctrl+C in console to exit.")
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
        enable_memory: bool = True,
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
        self.enable_memory = enable_memory
        self.debug = debug

        self.shutdown_event = threading.Event()
        self.conversation_history: List[BaseMessage] = []

        # Components
        self.detector: Optional[WakeWordDetector] = None
        self.agent_app = None
        self.tray_app: Optional[JarvisTrayApp] = None
        self.scheduler = None

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
            self.tray_app = JarvisTrayApp(
                on_quit=self.stop,
                initial_state=JarvisTrayState.LISTENING,
            )
            self.tray_app.start()

        print("[Orchestrator] All components initialized successfully.\n")

    def _set_tray_state(self, state: JarvisTrayState, custom_msg: Optional[str] = None) -> None:
        """Helper to update tray icon state if tray is enabled."""
        if self.tray_app is not None:
            self.tray_app.set_state(state, custom_message=custom_msg)

    def run_turn(self) -> bool:
        """
        Executes a single end-to-end voice pipeline turn:
        Listening -> Heard wake word -> Transcribing -> Thinking -> Speaking -> Listening again.

        Returns True if the loop should continue, False if shutdown was requested.
        """
        if self.shutdown_event.is_set():
            return False

        # ---------------------------------------------------------------------
        # STAGE 1: Wake Word Listening
        # ---------------------------------------------------------------------
        print("\n" + "-" * 60)
        print("[Pipeline] Stage 1: LISTENING for 'Hey Jarvis' (hands-free)...")
        self._set_tray_state(JarvisTrayState.LISTENING, "Listening for 'Hey Jarvis'...")

        try:
            audio_clip_path = self.detector.listen_and_record(
                on_wake_word_detected=lambda score: print(
                    f"\n[Pipeline] Wake word detected! (confidence: {score:.3f})"
                ),
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

        user_text = ""
        try:
            stt_start = time.perf_counter()
            user_text = transcribe_audio(
                audio_path=audio_clip_path,
                model_size=self.whisper_model,
            )
            stt_elapsed = time.perf_counter() - stt_start
            print(f"[STT] Transcribed in {stt_elapsed:.2f}s: \"{user_text}\"")
        except Exception as e:
            print(f"[Pipeline Error] STT transcription failed: {e}")
            self._set_tray_state(JarvisTrayState.ERROR, f"STT Error: {e}")
            time.sleep(1.0)
            return True

        # Check for empty or silent utterance
        if not user_text or not user_text.strip():
            print("[Pipeline] No speech recognized in audio clip. Returning to listening mode.")
            return True

        print(f"\nUser > {user_text}")

        # ---------------------------------------------------------------------
        # STAGE 3: Agent Brain & Tool Execution (LangGraph)
        # ---------------------------------------------------------------------
        print("[Pipeline] Stage 3: THINKING (LangGraph agent + memory + tools)...")
        self._set_tray_state(JarvisTrayState.PROCESSING, f"Thinking about: {user_text[:25]}...")

        response_text = ""
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
        except Exception as e:
            print(f"[Pipeline Error] Agent execution error: {e}")
            self._set_tray_state(JarvisTrayState.ERROR, f"Agent Error: {e}")
            time.sleep(1.0)
            return True

        # Sanitize response text so raw tool calls or narration are never displayed or spoken
        response_text = sanitize_speech_text(response_text)
        print(f"\nJarvis > {response_text}\n")

        # ---------------------------------------------------------------------
        # STAGE 4: Speaking Response Aloud (Piper TTS)
        # ---------------------------------------------------------------------
        if response_text and response_text.strip():
            print(f"[Pipeline] Stage 4: SPEAKING response aloud (voice: '{self.voice}')...")
            self._set_tray_state(JarvisTrayState.SPEAKING, "Speaking response aloud...")

            try:
                speak_start = time.perf_counter()
                speak(
                    text=response_text,
                    voice=self.voice,
                    blocking=True,
                    on_start_playback=lambda ttfa: print(f"[TTS] Playback started in {ttfa:.2f}s (TTFA)"),
                )
                speak_elapsed = time.perf_counter() - speak_start
                print(f"[TTS] Finished speaking in {speak_elapsed:.2f}s.")
            except Exception as e:
                print(f"[Pipeline Error] TTS playback error: {e}")
                self._set_tray_state(JarvisTrayState.ERROR, f"TTS Error: {e}")
                time.sleep(1.0)
                return True

        # ---------------------------------------------------------------------
        # STAGE 5: Loop back to listening
        # ---------------------------------------------------------------------
        print("[Pipeline] Stage 5: Turn completed. Returning to wake word listening.")
        return True

    def run(self) -> None:
        """Main continuous execution loop."""
        self.initialize()

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
        "--no-tray",
        action="store_true",
        help="Run in headless console mode without system tray icon",
    )
    parser.add_argument(
        "--no-memory",
        action="store_true",
        help="Disable ChromaDB persistent vector memory",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose debug logging for wake word detection",
    )

    args = parser.parse_args()

    enable_tray = not args.no_tray
    enable_memory = not args.no_memory

    print_banner(
        model=args.model,
        voice=args.voice,
        whisper_model=args.whisper_model,
        enable_tray=enable_tray,
        enable_memory=enable_memory,
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
        enable_memory=enable_memory,
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
