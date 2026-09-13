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

import os
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.orchestrator import (
    DEFAULT_DISMISSAL_PHRASES,
    DEFAULT_STOP_PHRASES,
    VoiceAssistantOrchestrator,
    is_dismissal_phrase,
    is_jarvis_dismissal,
    is_noise_or_wake_word_artifact,
    is_stop_phrase,
)
from src.agent.graph import is_tool_or_action_query
from tray.tray_app import JarvisTrayState


class TestVoiceAssistantOrchestrator(unittest.TestCase):
    """Unit tests for the orchestrator."""

    @classmethod
    def setUpClass(cls):
        cls._orig_test_mode = os.environ.get("JARVIS_TEST_MODE")
        os.environ["JARVIS_TEST_MODE"] = "1"

    @classmethod
    def tearDownClass(cls):
        if cls._orig_test_mode is not None:
            os.environ["JARVIS_TEST_MODE"] = cls._orig_test_mode
        else:
            os.environ.pop("JARVIS_TEST_MODE", None)

    def setUp(self):
        self.orchestrator = VoiceAssistantOrchestrator(
            model="llama3.2:3b",
            enable_tray=True,
            enable_memory=False,
            enable_logging=False,  # Isolate automated unit tests from production interactions.db
            is_test=True,
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

    def test_is_dismissal_phrase(self):
        """Verifies short dismissal phrase matching with punctuation, whitespace, and case normalization."""
        # Exact and normalized matches
        self.assertTrue(is_dismissal_phrase("stop"))
        self.assertTrue(is_dismissal_phrase("STOP!"))
        self.assertTrue(is_dismissal_phrase("okay"))
        self.assertTrue(is_dismissal_phrase("ok"))
        self.assertTrue(is_dismissal_phrase("got it"))
        self.assertTrue(is_dismissal_phrase("Got it!"))
        self.assertTrue(is_dismissal_phrase("never mind"))
        self.assertTrue(is_dismissal_phrase("nevermind."))
        self.assertTrue(is_dismissal_phrase("that's enough"))
        self.assertTrue(is_dismissal_phrase("thats enough!"))
        self.assertTrue(is_dismissal_phrase("thanks"))
        self.assertTrue(is_dismissal_phrase("thank you"))
        self.assertTrue(is_dismissal_phrase("cancel"))
        self.assertTrue(is_dismissal_phrase("quiet"))

        # Leading/trailing conversational prefixes/suffixes
        self.assertTrue(is_dismissal_phrase("ok thanks"))
        self.assertTrue(is_dismissal_phrase("stop jarvis"))
        self.assertTrue(is_dismissal_phrase("got it jarvis"))
        self.assertTrue(is_dismissal_phrase("thanks jarvis"))

        # Non-dismissal phrases (e.g. real questions or long utterances)
        self.assertFalse(is_dismissal_phrase("what time is it"))
        self.assertFalse(is_dismissal_phrase("open notepad"))
        self.assertFalse(is_dismissal_phrase("tell me more about this"))
        self.assertFalse(is_dismissal_phrase("calculate 25 * 4"))
        self.assertFalse(is_dismissal_phrase(""))
        self.assertFalse(is_dismissal_phrase("   "))

    def test_is_jarvis_dismissal(self):
        """Verifies that is_jarvis_dismissal strictly requires 'Jarvis' and dismissal phrase."""
        # Confirmed Jarvis dismissals
        self.assertTrue(is_jarvis_dismissal("Jarvis stop"))
        self.assertTrue(is_jarvis_dismissal("Jarvis, stop!"))
        self.assertTrue(is_jarvis_dismissal("Jarvis okay"))
        self.assertTrue(is_jarvis_dismissal("Jarvis ok"))
        self.assertTrue(is_jarvis_dismissal("Hey Jarvis that's enough"))
        self.assertTrue(is_jarvis_dismissal("Stop Jarvis"))
        self.assertTrue(is_jarvis_dismissal("Jarvis got it"))
        self.assertTrue(is_jarvis_dismissal("Jarvis thanks"))
        self.assertTrue(is_jarvis_dismissal("Jarvis quiet"))
        self.assertTrue(is_jarvis_dismissal("Jarvis cancel"))
        self.assertTrue(is_jarvis_dismissal("Jarvis never mind"))

        # Bare dismissals WITHOUT 'Jarvis' MUST RETURN FALSE to prevent accidental interruption
        self.assertFalse(is_jarvis_dismissal("stop"))
        self.assertFalse(is_jarvis_dismissal("okay"))
        self.assertFalse(is_jarvis_dismissal("ok"))
        self.assertFalse(is_jarvis_dismissal("got it"))
        self.assertFalse(is_jarvis_dismissal("that's enough"))
        self.assertFalse(is_jarvis_dismissal("thanks"))
        self.assertFalse(is_jarvis_dismissal("quiet"))

        # Real questions or commands MUST RETURN FALSE
        self.assertFalse(is_jarvis_dismissal("what time is it"))
        self.assertFalse(is_jarvis_dismissal("open notepad"))
        self.assertFalse(is_jarvis_dismissal("Jarvis what time is it"))
        self.assertFalse(is_jarvis_dismissal("Jarvis open chrome"))
        self.assertFalse(is_jarvis_dismissal("hey jarvis"))
        self.assertFalse(is_jarvis_dismissal(""))
        self.assertFalse(is_jarvis_dismissal("   "))

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
                interrupt_event=unittest.mock.ANY,
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

    def test_startup_greeting_speaks_on_run(self):
        """Verifies startup greeting is spoken aloud via TTS when orchestrator.run() starts."""
        self.orchestrator.initialize = MagicMock()
        # End loop after 1 turn check
        self.orchestrator.run_turn = MagicMock(return_value=False)

        with patch("src.orchestrator.speak") as mock_speak:
            self.orchestrator.run()

            # Startup greeting must be spoken before listening begins
            self.orchestrator.initialize.assert_called_once()
            mock_speak.assert_called_once()
            spoken_text = mock_speak.call_args[1].get("text") or mock_speak.call_args[0][0]
            self.assertEqual(spoken_text, "Hello! I'm ready, how can I help you today?")

    def test_startup_greeting_disabled_via_flag(self):
        """Verifies enable_greeting=False skips speaking startup greeting."""
        orch = VoiceAssistantOrchestrator(enable_greeting=False, is_test=True, enable_logging=False)
        orch.initialize = MagicMock()
        orch.run_turn = MagicMock(return_value=False)

        with patch("src.orchestrator.speak") as mock_speak:
            orch.run()

            mock_speak.assert_not_called()

    def test_barge_in_dismissal_silently_ends_response(self):
        """Verifies speech interruption with a dismissal phrase halts TTS and returns to standby."""
        self.orchestrator.detector.listen_and_record.return_value = "query.wav"

        with patch("src.orchestrator.transcribe_audio", return_value="explain quantum computing"), \
             patch("src.orchestrator.run_agent", return_value=("Quantum computing uses qubits...", [])), \
             patch.object(self.orchestrator, "_speak_with_barge_in", return_value=(True, None, 1.5)) as mock_barge:

            continue_loop = self.orchestrator.run_turn()

            self.assertTrue(continue_loop)
            mock_barge.assert_called_once_with(
                text="Quantum computing uses qubits...",
                trigger_mode="wake_word",
            )
            # Pending user text should remain None (dismissed)
            self.assertIsNone(self.orchestrator.pending_user_text)
            # Tray state set back to LISTENING
            tray_states = [call[0][0] for call in self.orchestrator.tray_app.set_state.call_args_list]
            self.assertIn(JarvisTrayState.LISTENING, tray_states)

    def test_barge_in_new_question_chains_to_next_turn(self):
        """Verifies interruption with a new question sets pending_user_text and skips wake word next turn."""
        self.orchestrator.detector.listen_and_record.return_value = "query1.wav"

        # Turn 1: Interrupted with new question "what time is it"
        with patch("src.orchestrator.transcribe_audio", return_value="tell me a long story"), \
             patch("src.orchestrator.run_agent", return_value=("Once upon a time...", [])), \
             patch.object(self.orchestrator, "_speak_with_barge_in", return_value=(True, "what time is it", 0.8)):

            continue_loop = self.orchestrator.run_turn()
            self.assertTrue(continue_loop)
            self.assertEqual(self.orchestrator.pending_user_text, "what time is it")

        # Turn 2: Starts with pending_user_text, skipping listen_and_record
        self.orchestrator.detector.listen_and_record.reset_mock()
        with patch("src.orchestrator.transcribe_audio") as mock_stt, \
             patch("src.orchestrator.run_agent", return_value=("It is 10:00 PM", [])) as mock_agent, \
             patch.object(self.orchestrator, "_speak_with_barge_in", return_value=(False, None, 1.0)) as mock_barge2:

            continue_loop = self.orchestrator.run_turn()
            self.assertTrue(continue_loop)
            # Wake word and STT were skipped!
            self.orchestrator.detector.listen_and_record.assert_not_called()
            mock_stt.assert_not_called()
            # Agent executed directly with the interrupted command
            mock_agent.assert_called_once()
            self.assertEqual(mock_agent.call_args[1]["user_input"], "what time is it")
            self.assertIsNone(self.orchestrator.pending_user_text)

    def test_barge_in_in_continuous_conversation(self):
        """Verifies barge-in in continuous conversation mode chains immediately to the interrupted command."""
        self.orchestrator.detector.listen_and_record.return_value = "turn1.wav"
        # Continuous turn 2: normal record, turn 3: silence timeout
        self.orchestrator.detector.record_utterance.side_effect = ["turn2.wav", None]

        # In turn 2 playback, user interrupts with "open notepad"
        with patch("src.orchestrator.transcribe_audio", side_effect=["initial question", "what is 2 plus 2"]), \
             patch("src.orchestrator.run_agent", side_effect=[
                 ("initial answer", []),
                 ("4", []),
                 ("Opening notepad", []),
             ]) as mock_agent, \
             patch.object(self.orchestrator, "_speak_with_barge_in", side_effect=[
                 (False, None, 1.0),  # Turn 1 normal speak
                 (True, "open notepad", 0.4),  # Turn 2 interrupted with "open notepad"
                 (False, None, 0.8),  # Turn 3 (open notepad answer) normal speak
             ]):

            continue_loop = self.orchestrator.run_turn()
            self.assertTrue(continue_loop)
            # Agent should have been called 3 times: initial query, follow-up, and interrupted query
            self.assertEqual(mock_agent.call_count, 3)
            agent_queries = [call[1]["user_input"] for call in mock_agent.call_args_list]
            self.assertEqual(agent_queries, ["initial question", "what is 2 plus 2", "open notepad"])

    def test_wake_ack_speaks_on_detection(self):
        """Verifies _handle_wake_word_detected speaks wake-word acknowledgment when enabled."""
        with patch("src.orchestrator.speak") as mock_speak:
            self.orchestrator.enable_wake_ack = True
            self.orchestrator.wake_ack_text = "Yes?"
            self.orchestrator._handle_wake_word_detected(0.88)

            mock_speak.assert_called_once_with("Yes?", voice=self.orchestrator.voice, blocking=True)

    def test_wake_ack_disabled_via_flag(self):
        """Verifies _handle_wake_word_detected does not speak when enable_wake_ack is False."""
        with patch("src.orchestrator.speak") as mock_speak:
            self.orchestrator.enable_wake_ack = False
            self.orchestrator._handle_wake_word_detected(0.88)

            mock_speak.assert_not_called()

    def test_speak_with_barge_in_direct_detection(self):
        """Verifies _speak_with_barge_in detects energy, verifies 'Jarvis stop' dismissal, and cancels playback."""
        import queue
        from src.wake_word import CHUNK_SAMPLES, SILENCE_CHUNKS

        audio_q = queue.Queue()
        self.orchestrator.detector.audio_queue = audio_q
        self.orchestrator.barge_in_threshold = 400.0

        loud_chunk = np.full(CHUNK_SAMPLES, 1000, dtype=np.int16)
        silent_chunk = np.zeros(CHUNK_SAMPLES, dtype=np.int16)

        def mock_speak_impl(text, voice, blocking, interrupt_event, on_start_playback):
            # Feed frames while TTS playback thread is active (after grace period)
            time.sleep(0.3)
            for _ in range(4):
                audio_q.put(loud_chunk)
            for _ in range(SILENCE_CHUNKS + 2):
                audio_q.put(silent_chunk)

            # Wait for interrupt_event to be set by barge-in monitor
            interrupt_event.wait(timeout=1.5)
            return not interrupt_event.is_set()

        with patch("src.orchestrator.speak", side_effect=mock_speak_impl), \
             patch("src.orchestrator.wavfile.write"), \
             patch("src.orchestrator.transcribe_audio", return_value="Jarvis stop"):

            interrupted, next_query, duration = self.orchestrator._speak_with_barge_in(
                text="A long answer to test interruption",
                trigger_mode="wake_word",
            )

            self.assertTrue(interrupted)
            self.assertIsNone(next_query)  # "Jarvis stop" is recognized as a confirmed dismissal

    def test_speak_with_barge_in_ignores_bare_dismissal_and_noise(self):
        """Verifies bare dismissal words ('stop') and acoustic noise do NOT abort playback."""
        import queue
        from src.wake_word import CHUNK_SAMPLES, SILENCE_CHUNKS

        audio_q = queue.Queue()
        self.orchestrator.detector.audio_queue = audio_q
        self.orchestrator.barge_in_threshold = 400.0

        loud_chunk = np.full(CHUNK_SAMPLES, 1000, dtype=np.int16)
        silent_chunk = np.zeros(CHUNK_SAMPLES, dtype=np.int16)

        def mock_speak_impl(text, voice, blocking, interrupt_event, on_start_playback):
            # Feed frames while TTS playback thread is active
            time.sleep(0.3)
            for _ in range(4):
                audio_q.put(loud_chunk)
            for _ in range(SILENCE_CHUNKS + 2):
                audio_q.put(silent_chunk)
            # Sleep briefly so playback finishes without interrupt
            time.sleep(0.2)
            return True

        # Test A: Bare "stop" without "Jarvis" -> playback continues, not interrupted
        with patch("src.orchestrator.speak", side_effect=mock_speak_impl), \
             patch("src.orchestrator.wavfile.write"), \
             patch("src.orchestrator.transcribe_audio", return_value="stop"):

            interrupted, next_query, duration = self.orchestrator._speak_with_barge_in(
                text="A long answer that should not be interrupted by bare stop",
                trigger_mode="wake_word",
            )

            self.assertFalse(interrupted)
            self.assertIsNone(next_query)

        # Test B: Noise artifact "[cough]" -> playback continues, not interrupted
        with patch("src.orchestrator.speak", side_effect=mock_speak_impl), \
             patch("src.orchestrator.wavfile.write"), \
             patch("src.orchestrator.transcribe_audio", return_value="[cough]"):

            interrupted, next_query, duration = self.orchestrator._speak_with_barge_in(
                text="A long answer that should not be interrupted by cough",
                trigger_mode="wake_word",
            )

            self.assertFalse(interrupted)
            self.assertIsNone(next_query)

    def test_speak_with_barge_in_new_question_abort(self):
        """Verifies new questions without 'Jarvis' (e.g. 'what time is it') abort playback and chain."""
        import queue
        from src.wake_word import CHUNK_SAMPLES, SILENCE_CHUNKS

        audio_q = queue.Queue()
        self.orchestrator.detector.audio_queue = audio_q
        self.orchestrator.barge_in_threshold = 400.0

        loud_chunk = np.full(CHUNK_SAMPLES, 1000, dtype=np.int16)
        silent_chunk = np.zeros(CHUNK_SAMPLES, dtype=np.int16)

        def mock_speak_impl(text, voice, blocking, interrupt_event, on_start_playback):
            time.sleep(0.3)
            for _ in range(4):
                audio_q.put(loud_chunk)
            for _ in range(SILENCE_CHUNKS + 2):
                audio_q.put(silent_chunk)

            interrupt_event.wait(timeout=1.5)
            return not interrupt_event.is_set()

        with patch("src.orchestrator.speak", side_effect=mock_speak_impl), \
             patch("src.orchestrator.wavfile.write"), \
             patch("src.orchestrator.transcribe_audio", return_value="what time is it"):

            interrupted, next_query, duration = self.orchestrator._speak_with_barge_in(
                text="A long answer being spoken",
                trigger_mode="wake_word",
            )

            self.assertTrue(interrupted)
            self.assertEqual(next_query, "what time is it")


if __name__ == "__main__":
    unittest.main()



