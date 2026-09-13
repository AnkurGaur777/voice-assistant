"""
Unit Tests for Voice Assistant Pipeline Orchestrator (src/orchestrator.py)

Validates:
1. Orchestrator initialization and component wiring with mocks.
2. Successful end-to-end turn: Wake word -> STT -> LangGraph agent -> TTS -> returns True.
3. Empty transcription handling: gracefully returns to listening without invoking agent or crashing.
4. Error resilience: single-stage failures (STT error, Agent error, TTS error) are caught and logged,
   updating tray to ERROR and returning to listening rather than terminating the pipeline.
5. Graceful shutdown: stop() cleans up streams, scheduler, and tray icon, and run_turn() exits.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.orchestrator import VoiceAssistantOrchestrator, is_stop_phrase, DEFAULT_STOP_PHRASES
from tray.tray_app import JarvisTrayState


class TestVoiceAssistantOrchestrator(unittest.TestCase):
    """Unit tests for the orchestrator."""

    def setUp(self):
        self.orchestrator = VoiceAssistantOrchestrator(
            model="llama3.2:3b",
            enable_tray=True,
            enable_memory=False,
            continuous_mode=True,
            conversation_timeout=6.0,
        )
        # Mock components to isolate logic from hardware
        self.orchestrator.detector = MagicMock()
        self.orchestrator.detector.record_utterance.return_value = None  # Default: silence timeout
        self.orchestrator.tray_app = MagicMock()
        self.orchestrator.agent_app = MagicMock()
        self.orchestrator.scheduler = MagicMock()

    def test_default_parameters(self):
        """Verifies defaults: threshold=0.35, whisper_model='small', and continuous_mode=True."""
        default_orch = VoiceAssistantOrchestrator()
        self.assertEqual(default_orch.wake_threshold, 0.35)
        self.assertEqual(default_orch.whisper_model, "small")
        self.assertTrue(default_orch.continuous_mode)
        self.assertEqual(default_orch.conversation_timeout, 6.0)

    def test_is_stop_phrase(self):
        """Verifies stop phrase matching with punctuation, whitespace, and case normalization."""
        # Exact and normalized matches
        self.assertTrue(is_stop_phrase("stop"))
        self.assertTrue(is_stop_phrase("STOP!"))
        self.assertTrue(is_stop_phrase("goodbye"))
        self.assertTrue(is_stop_phrase("Goodbye."))
        self.assertTrue(is_stop_phrase("that's all"))
        self.assertTrue(is_stop_phrase("thats all!"))
        self.assertTrue(is_stop_phrase("thank you jarvis"))
        self.assertTrue(is_stop_phrase("Thank you, Jarvis!"))
        self.assertTrue(is_stop_phrase("bye"))
        self.assertTrue(is_stop_phrase("exit"))

        # Leading / trailing prefix / suffix
        self.assertTrue(is_stop_phrase("ok goodbye"))
        self.assertTrue(is_stop_phrase("stop now"))

        # Non-stop phrases
        self.assertFalse(is_stop_phrase("what time is it"))
        self.assertFalse(is_stop_phrase("hello jarvis"))
        self.assertFalse(is_stop_phrase(""))
        self.assertFalse(is_stop_phrase("   "))

    def test_successful_turn_execution(self):
        """Verifies a full successful pipeline turn flows through all stages."""
        self.orchestrator.detector.listen_and_record.return_value = "dummy_clip.wav"

        with patch("src.orchestrator.transcribe_audio", return_value="what time is it") as mock_stt, \
             patch("src.orchestrator.run_agent", return_value=("The time is 10:00 PM", [])) as mock_agent, \
             patch("src.orchestrator.speak") as mock_speak:

            continue_loop = self.orchestrator.run_turn()

            self.assertTrue(continue_loop)
            self.orchestrator.detector.listen_and_record.assert_called_once()
            mock_stt.assert_called_once_with(audio_path="dummy_clip.wav", model_size="small")
            mock_agent.assert_called_once()
            mock_speak.assert_called_once_with(
                text="The time is 10:00 PM",
                voice="ryan",
                blocking=True,
                on_start_playback=unittest.mock.ANY,
            )
            # Verify tray transitions occurred
            tray_states = [call[0][0] for call in self.orchestrator.tray_app.set_state.call_args_list]
            self.assertIn(JarvisTrayState.LISTENING, tray_states)
            self.assertIn(JarvisTrayState.PROCESSING, tray_states)
            self.assertIn(JarvisTrayState.SPEAKING, tray_states)
            self.assertIn(JarvisTrayState.CONVERSATION, tray_states)

    def test_continuous_conversation_stop_phrase(self):
        """Verifies that speaking a stop phrase in conversation mode politely exits back to wake-word."""
        self.orchestrator.detector.listen_and_record.return_value = "query1.wav"
        self.orchestrator.detector.record_utterance.side_effect = ["stop_clip.wav"]

        with patch("src.orchestrator.transcribe_audio", side_effect=["hello", "that's all"]) as mock_stt, \
             patch("src.orchestrator.run_agent", return_value=("Hi there!", [])) as mock_agent, \
             patch("src.orchestrator.speak") as mock_speak:

            continue_loop = self.orchestrator.run_turn()

            self.assertTrue(continue_loop)
            self.assertEqual(mock_stt.call_count, 2)
            # Agent only ran once (for 'hello'), NOT for the stop phrase ('that's all')
            self.assertEqual(mock_agent.call_count, 1)
            # Speak ran twice: once for answer 'Hi there!', and once for 'Goodbye!'
            self.assertEqual(mock_speak.call_count, 2)
            speak_texts = [call[1]["text"] if "text" in call[1] else call[0][0] for call in mock_speak.call_args_list]
            self.assertIn("Goodbye!", speak_texts)

    def test_continuous_conversation_multi_turn_hands_free(self):
        """Verifies multi-turn dialogue where user asks a follow-up without saying 'Hey Jarvis'."""
        self.orchestrator.detector.listen_and_record.return_value = "turn1.wav"
        # Turn 2 asks follow-up, Turn 3 is silence timeout
        self.orchestrator.detector.record_utterance.side_effect = ["turn2.wav", None]

        with patch("src.orchestrator.transcribe_audio", side_effect=["what is 2 plus 2", "and plus 5"]) as mock_stt, \
             patch("src.orchestrator.run_agent", side_effect=[("4", []), ("9", [])]) as mock_agent, \
             patch("src.orchestrator.speak") as mock_speak:

            continue_loop = self.orchestrator.run_turn()

            self.assertTrue(continue_loop)
            self.assertEqual(mock_stt.call_count, 2)
            self.assertEqual(mock_agent.call_count, 2)
            self.assertEqual(mock_speak.call_count, 2)
            # Verify tray entered CONVERSATION state
            tray_states = [call[0][0] for call in self.orchestrator.tray_app.set_state.call_args_list]
            self.assertIn(JarvisTrayState.CONVERSATION, tray_states)

    def test_empty_transcription_handling(self):
        """Verifies that an empty/silent speech transcription returns to listening without calling agent."""
        self.orchestrator.detector.listen_and_record.return_value = "silent_clip.wav"

        with patch("src.orchestrator.transcribe_audio", return_value="   ") as mock_stt, \
             patch("src.orchestrator.run_agent") as mock_agent, \
             patch("src.orchestrator.speak") as mock_speak:

            continue_loop = self.orchestrator.run_turn()

            self.assertTrue(continue_loop)
            mock_stt.assert_called_once()
            mock_agent.assert_not_called()
            mock_speak.assert_not_called()

    def test_resilience_on_stt_failure(self):
        """Verifies an exception in STT is caught, sets ERROR state, and returns True to keep listening."""
        self.orchestrator.detector.listen_and_record.return_value = "corrupted.wav"

        with patch("src.orchestrator.transcribe_audio", side_effect=RuntimeError("Whisper decode error")), \
             patch("src.orchestrator.run_agent") as mock_agent, \
             patch("src.orchestrator.speak") as mock_speak:

            continue_loop = self.orchestrator.run_turn()

            # Orchestrator does NOT crash
            self.assertTrue(continue_loop)
            mock_agent.assert_not_called()
            mock_speak.assert_not_called()
            # Tray state set to ERROR
            tray_states = [call[0][0] for call in self.orchestrator.tray_app.set_state.call_args_list]
            self.assertIn(JarvisTrayState.ERROR, tray_states)

    def test_resilience_on_agent_failure(self):
        """Verifies an exception in the LangGraph agent is caught, sets ERROR state, and continues."""
        self.orchestrator.detector.listen_and_record.return_value = "query.wav"

        with patch("src.orchestrator.transcribe_audio", return_value="hello"), \
             patch("src.orchestrator.run_agent", side_effect=ConnectionError("Ollama unreachable")), \
             patch("src.orchestrator.speak") as mock_speak:

            continue_loop = self.orchestrator.run_turn()

            self.assertTrue(continue_loop)
            mock_speak.assert_not_called()
            tray_states = [call[0][0] for call in self.orchestrator.tray_app.set_state.call_args_list]
            self.assertIn(JarvisTrayState.ERROR, tray_states)

    def test_resilience_on_tts_failure(self):
        """Verifies an exception in TTS is caught and does not crash the loop."""
        self.orchestrator.detector.listen_and_record.return_value = "query.wav"

        with patch("src.orchestrator.transcribe_audio", return_value="hello"), \
             patch("src.orchestrator.run_agent", return_value=("Hello!", [])), \
             patch("src.orchestrator.speak", side_effect=RuntimeError("Audio output stream error")):

            continue_loop = self.orchestrator.run_turn()

            self.assertTrue(continue_loop)
            tray_states = [call[0][0] for call in self.orchestrator.tray_app.set_state.call_args_list]
            self.assertIn(JarvisTrayState.ERROR, tray_states)

    def test_graceful_shutdown(self):
        """Verifies that calling stop() signals shutdown and run_turn returns False."""
        self.orchestrator.stop()
        self.assertTrue(self.orchestrator.shutdown_event.is_set())

        # Calling run_turn when shutdown_event is set returns False immediately
        continue_loop = self.orchestrator.run_turn()
        self.assertFalse(continue_loop)

        # Verify detector, scheduler, and tray were cleanly closed
        self.orchestrator.detector.close.assert_called_once()
        self.orchestrator.tray_app.stop.assert_called_once()


if __name__ == "__main__":
    unittest.main()
