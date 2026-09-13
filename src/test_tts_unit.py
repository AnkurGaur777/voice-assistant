"""
Local Jarvis - Text-to-Speech (TTS) Unit Tests

Verifies:
1. Sentence tokenizer logic (punctuation boundaries, decimals, linebreaks, empty strings).
2. Voice key resolution and alias mapping.
3. Voice catalog URLs and metadata.
4. Streaming synthesis pipeline with mock Piper voice and sounddevice.
"""

from pathlib import Path
import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.tts import (
    DEFAULT_VOICE,
    SUPPORTED_VOICES,
    list_available_voices,
    resolve_voice_key,
    sanitize_speech_text,
    speak,
    split_into_sentences,
    synthesize_sentence,
)


class TestSentenceTokenizer(unittest.TestCase):
    """Tests the sentence splitting logic used for streaming synthesis."""

    def test_single_sentence(self):
        """Single sentence returns a one-element list."""
        text = "Hello world, this is a test."
        sentences = split_into_sentences(text)
        self.assertEqual(sentences, ["Hello world, this is a test."])

    def test_multi_sentence_punctuation(self):
        """Splits on periods, exclamation marks, and question marks."""
        text = "First sentence! Is this sentence two? Yes, it is sentence three."
        sentences = split_into_sentences(text)
        self.assertEqual(len(sentences), 3)
        self.assertEqual(sentences[0], "First sentence!")
        self.assertEqual(sentences[1], "Is this sentence two?")
        self.assertEqual(sentences[2], "Yes, it is sentence three.")

    def test_multiline_text(self):
        """Splits on linebreaks as well as punctuation."""
        text = "Line one.\nLine two.\nLine three."
        sentences = split_into_sentences(text)
        self.assertEqual(len(sentences), 3)
        self.assertEqual(sentences[0], "Line one.")
        self.assertEqual(sentences[1], "Line two.")
        self.assertEqual(sentences[2], "Line three.")

    def test_empty_and_whitespace(self):
        """Empty or whitespace text returns an empty list."""
        self.assertEqual(split_into_sentences(""), [])
        self.assertEqual(split_into_sentences("   \n\t  "), [])


class TestVoiceCatalog(unittest.TestCase):
    """Tests voice model alias resolution and configuration metadata."""

    def test_default_voice_exists(self):
        """DEFAULT_VOICE must be defined in SUPPORTED_VOICES."""
        self.assertIn(DEFAULT_VOICE, SUPPORTED_VOICES)
        self.assertIn("ryan", SUPPORTED_VOICES)
        self.assertIn("lessac", SUPPORTED_VOICES)

    def test_resolve_voice_key_aliases(self):
        """Short aliases and full model names resolve to canonical keys."""
        self.assertEqual(resolve_voice_key("ryan"), "ryan")
        self.assertEqual(resolve_voice_key("en_US-ryan-medium"), "ryan")
        self.assertEqual(resolve_voice_key("lessac"), "lessac")
        self.assertEqual(resolve_voice_key("en_US-lessac-medium"), "lessac")
        self.assertEqual(resolve_voice_key(None), DEFAULT_VOICE)

    def test_resolve_unknown_voice_raises_error(self):
        """Unknown voice names raise ValueError with available choices."""
        with self.assertRaises(ValueError) as ctx:
            resolve_voice_key("non_existent_voice")
        self.assertIn("Supported voices", str(ctx.exception))

    def test_list_available_voices(self):
        """list_available_voices returns all configured voices."""
        voices = list_available_voices()
        self.assertIn("ryan", voices)
        self.assertIn("lessac", voices)
        self.assertEqual(voices["ryan"]["gender"], "male")
        self.assertEqual(voices["lessac"]["gender"], "female")


class TestSynthesisPipeline(unittest.TestCase):
    """Tests the synthesis and streaming playback pipeline with mocks."""

    def setUp(self):
        # Create a mock PiperVoice
        self.mock_voice = MagicMock()
        self.mock_voice.config.sample_rate = 22050

        # Mock chunk returning a small int16 numpy array
        mock_chunk = MagicMock()
        mock_chunk.audio_int16_array = np.array([100, 200, 300, 400], dtype=np.int16)
        self.mock_voice.synthesize.return_value = [mock_chunk]

    def test_synthesize_sentence(self):
        """synthesize_sentence returns concatenated audio array and sample rate."""
        audio, sr = synthesize_sentence("Testing audio.", self.mock_voice)
        self.assertEqual(sr, 22050)
        self.assertEqual(len(audio), 4)
        self.assertEqual(audio.dtype, np.int16)

    @patch("time.sleep")
    @patch("sounddevice.OutputStream")
    @patch("src.tts.get_piper_voice")
    def test_speak_single_sentence(self, mock_get_voice, mock_stream_cls, mock_sleep):
        """Single sentence speaks using sounddevice.OutputStream and writes sliced audio with trailing padding."""
        mock_get_voice.return_value = self.mock_voice
        mock_stream = MagicMock()
        mock_stream.latency = 0.18
        mock_stream_cls.return_value = mock_stream

        result = speak("One short sentence.", voice="ryan", blocking=True)

        self.assertTrue(result)
        mock_stream_cls.assert_called_once()
        mock_stream.start.assert_called_once()
        self.assertGreaterEqual(mock_stream.write.call_count, 1)
        mock_stream.stop.assert_called_once()
        mock_stream.close.assert_called_once()
        mock_sleep.assert_called()

    @patch("time.sleep")
    @patch("sounddevice.OutputStream")
    @patch("src.tts.get_piper_voice")
    def test_speak_multi_sentence_streaming(self, mock_get_voice, mock_stream_cls, mock_sleep):
        """Multi-sentence input triggers streaming synthesis and sliced stream writes for each sentence."""
        mock_get_voice.return_value = self.mock_voice
        mock_stream = MagicMock()
        mock_stream.latency = 0.18
        mock_stream_cls.return_value = mock_stream

        ttfa_list = []
        result = speak(
            "Sentence one. Sentence two. Sentence three.",
            voice="ryan",
            blocking=True,
            on_start_playback=lambda t: ttfa_list.append(t),
        )

        self.assertTrue(result)
        mock_stream_cls.assert_called_once()
        mock_stream.start.assert_called_once()
        self.assertGreater(mock_stream.write.call_count, 3)
        self.assertEqual(len(ttfa_list), 1)
        self.assertGreaterEqual(ttfa_list[0], 0.0)
        mock_stream.stop.assert_called_once()
        mock_stream.close.assert_called_once()

    @patch("time.sleep")
    @patch("sounddevice.OutputStream")
    @patch("src.tts.get_piper_voice")
    def test_speak_with_interrupt_event_aborts_playback(self, mock_get_voice, mock_stream_cls, mock_sleep):
        """Setting interrupt_event causes speak() to immediately abort stream and return False."""
        import threading
        mock_get_voice.return_value = self.mock_voice
        mock_stream = MagicMock()
        mock_stream.latency = 0.18
        mock_stream_cls.return_value = mock_stream

        interrupt_event = threading.Event()
        # Trigger interrupt after the first stream.write call
        def on_write(data):
            interrupt_event.set()

        mock_stream.write.side_effect = on_write

        result = speak(
            "Sentence one. Sentence two. Sentence three.",
            voice="ryan",
            blocking=True,
            interrupt_event=interrupt_event,
        )

        self.assertFalse(result)
        mock_stream.abort.assert_called_once()
        mock_stream.stop.assert_called_once()
        mock_stream.close.assert_called_once()

    def test_sanitize_speech_text_converts_datetime_dump(self):
        """Verifies multi-line datetime tool dump is synthesized into a concise natural sentence."""
        raw_dump = (
            "Current system date and time:\n"
            "- Date: Sunday, September 13, 2026\n"
            "- Time: 1:46 PM (13:46)\n"
            "- Day of the week: Sunday\n"
            "- Timezone: India Standard Time"
        )
        cleaned = sanitize_speech_text(raw_dump)
        self.assertEqual(cleaned, "It's Sunday, September 13, 2026, 1:46 PM.")
        self.assertNotIn("Timezone:", cleaned)
        self.assertNotIn("Day of the week:", cleaned)



def main():
    unittest.main(verbosity=2)


if __name__ == "__main__":
    main()
