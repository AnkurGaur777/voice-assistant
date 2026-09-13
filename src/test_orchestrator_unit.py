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

from src.orchestrator import (
    DEFAULT_STOP_PHRASES,
    VoiceAssistantOrchestrator,
    is_noise_or_wake_word_artifact,
    is_stop_phrase,
)
from src.agent.graph import is_tool_or_action_query
from tray.tray_app import JarvisTrayState


class TestVoiceAssistantOrchestrator(unittest.TestCase):
    """Unit tests for the orchestrator."""

    def setUp(self):
        self.orchestrator = VoiceAssistantOrchestrator(
            model="llama3.2:3b",
            enable_tray=True,
            enable_memory=False,
            continuous_mode=True,
            conversation_timeout=60.0,
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
        self.assertEqual(default_orch.conversation_timeout, 60.0)

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

    def test_is_noise_or_wake_word_artifact(self):
        """Verifies detection of ambient noise, solitary wake words, and acoustic filler artifacts."""
        # Solitary wake words and wake-word variants
        self.assertTrue(is_noise_or_wake_word_artifact("Jarvis."))
        self.assertTrue(is_noise_or_wake_word_artifact("jarvis"))
        self.assertTrue(is_noise_or_wake_word_artifact("Hey Jarvis."))
        self.assertTrue(is_noise_or_wake_word_artifact("hello jarvis"))
        self.assertTrue(is_noise_or_wake_word_artifact("Ok Jarvis!"))
        self.assertTrue(is_noise_or_wake_word_artifact("Jarvis Jarvis"))

        # Pure acoustic filler artifacts produced by Whisper on silence / noise
        self.assertTrue(is_noise_or_wake_word_artifact("you"))
        self.assertTrue(is_noise_or_wake_word_artifact("uh"))
        self.assertTrue(is_noise_or_wake_word_artifact("um."))
        self.assertTrue(is_noise_or_wake_word_artifact("[music]"))
        self.assertTrue(is_noise_or_wake_word_artifact("(clears throat)"))
        self.assertTrue(is_noise_or_wake_word_artifact("*sigh*"))
        self.assertTrue(is_noise_or_wake_word_artifact(""))
        self.assertTrue(is_noise_or_wake_word_artifact("   "))

        # Genuine user commands and questions should NEVER be filtered
        self.assertFalse(is_noise_or_wake_word_artifact("what is the date today"))
        self.assertFalse(is_noise_or_wake_word_artifact("what time is it"))
        self.assertFalse(is_noise_or_wake_word_artifact("calculate 5 + 5"))
        self.assertFalse(is_noise_or_wake_word_artifact("Jarvis what time is it"))
        self.assertFalse(is_noise_or_wake_word_artifact("tell me a joke"))

    def test_is_tool_or_action_query(self):
        """Verifies detection of real-time action/tool queries vs conversational memory queries."""
        # Action queries requiring tools (never retrieve stale vector memory)
        self.assertTrue(is_tool_or_action_query("what is the date today"))
        self.assertTrue(is_tool_or_action_query("what's today's date"))
        self.assertTrue(is_tool_or_action_query("what time is it"))
        self.assertTrue(is_tool_or_action_query("tell me the time"))
        self.assertTrue(is_tool_or_action_query("calculate 358% of 340"))
        self.assertTrue(is_tool_or_action_query("what is 25 * 4"))
        self.assertTrue(is_tool_or_action_query("open notepad"))
        self.assertTrue(is_tool_or_action_query("type hello and press enter"))
        self.assertTrue(is_tool_or_action_query("remind me to call mom in 10 minutes"))
        self.assertTrue(is_tool_or_action_query("search the web for python"))

        # Conversational / memory queries (allowed to retrieve vector memory)
        self.assertFalse(is_tool_or_action_query("my favorite color is blue"))
        self.assertFalse(is_tool_or_action_query("what is my name"))
        self.assertFalse(is_tool_or_action_query("who is my brother"))
        self.assertFalse(is_tool_or_action_query("how are you doing"))

    def test_continuous_conversation_filters_ambient_noise(self):
        """Verifies that an ambient noise 'Jarvis.' artifact in conversation mode is filtered and ignored."""
        self.orchestrator.detector.listen_and_record.return_value = "query1.wav"
        # Turn 2 produces 'Jarvis.' (ambient noise), Turn 3 produces silence timeout
        self.orchestrator.detector.record_utterance.side_effect = ["noise.wav", None]

        with patch("src.orchestrator.transcribe_audio", side_effect=["hello", "Jarvis."]) as mock_stt, \
             patch("src.orchestrator.run_agent", return_value=("Hello!", [])) as mock_agent, \
             patch("src.orchestrator.speak") as mock_speak:

            continue_loop = self.orchestrator.run_turn()

            self.assertTrue(continue_loop)
            self.assertEqual(mock_stt.call_count, 2)
            # Agent only ran once (for 'hello'), NOT for 'Jarvis.' noise
            self.assertEqual(mock_agent.call_count, 1)
            # Speak only ran once (for 'hello' response)
            self.assertEqual(mock_speak.call_count, 1)

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

    def test_llm_node_preserves_tool_binding_across_multi_turn_history(self):
        """Verifies that in turn 2, having ToolMessage from turn 1 does NOT strip tools from the LLM."""
        from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
        from src.agent.graph import create_llm_node

        mock_bound_llm = MagicMock()
        mock_bound_llm.invoke.return_value = AIMessage(content="Today is September 13, 2026")

        # Mock RunnableBinding with .bound attribute
        mock_binding = MagicMock()
        mock_binding.bound = MagicMock()  # unbound client
        mock_binding.invoke = mock_bound_llm.invoke

        node = create_llm_node(llm=mock_binding, enable_memory=False)

        # Multi-turn state: Turn 1 had tool call and tool message, now user asks turn 2 query
        multi_turn_messages = [
            HumanMessage(content="what time is it"),
            AIMessage(content="", tool_calls=[{"name": "get_current_datetime", "args": {}, "id": "call_1"}]),
            ToolMessage(content="10:00 AM", tool_call_id="call_1"),
            AIMessage(content="The time is 10:00 AM"),
            HumanMessage(content="what is the date today"),  # Turn 2 query!
        ]

        result = node({"messages": multi_turn_messages})
        self.assertIsNotNone(result)
        # Verify that mock_binding.invoke was called (tools BOUND), NOT mock_binding.bound.invoke!
        mock_binding.invoke.assert_called_once()
        mock_binding.bound.invoke.assert_not_called()

    def test_set_ui_state_syncs_tray_and_orb(self):
        """Verifies _set_ui_state updates both tray app and orb overlay in lockstep."""
        orchestrator = VoiceAssistantOrchestrator(enable_tray=True, enable_orb=True)
        orchestrator.tray_app = MagicMock()
        orchestrator.orb_overlay = MagicMock()

        orchestrator._set_ui_state(JarvisTrayState.PROCESSING, "Thinking...")

        orchestrator.tray_app.set_state.assert_called_once_with(
            JarvisTrayState.PROCESSING,
            custom_message="Thinking...",
        )
        orchestrator.orb_overlay.set_state.assert_called_once_with("processing")

    def test_orchestrator_orb_disabled_via_flag(self):
        """Verifies enable_orb=False does not start or update orb overlay."""
        orchestrator = VoiceAssistantOrchestrator(enable_tray=False, enable_orb=False)
        self.assertFalse(orchestrator.enable_orb)
        self.assertIsNone(orchestrator.orb_overlay)

        # Calling _set_ui_state with no tray/orb should not raise any exceptions
        orchestrator._set_ui_state(JarvisTrayState.SPEAKING, "Speaking...")


if __name__ == "__main__":
    unittest.main()

