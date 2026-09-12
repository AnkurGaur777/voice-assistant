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

from src.orchestrator import VoiceAssistantOrchestrator
from tray.tray_app import JarvisTrayState


class TestVoiceAssistantOrchestrator(unittest.TestCase):
    """Unit tests for the orchestrator."""

    def setUp(self):
        self.orchestrator = VoiceAssistantOrchestrator(
            model="llama3.2:3b",
            enable_tray=True,
            enable_memory=False,
        )
        # Mock components to isolate logic from hardware
        self.orchestrator.detector = MagicMock()
        self.orchestrator.tray_app = MagicMock()
        self.orchestrator.agent_app = MagicMock()
        self.orchestrator.scheduler = MagicMock()

    def test_default_parameters(self):
        """Verifies defaults: threshold=0.35 and whisper_model='small'."""
        default_orch = VoiceAssistantOrchestrator()
        self.assertEqual(default_orch.wake_threshold, 0.35)
        self.assertEqual(default_orch.whisper_model, "small")

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
