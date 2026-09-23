"""Tests for the transcription module."""

import unittest


class TestTranscribeModule(unittest.TestCase):
    def test_module_imports_and_exposes_transcribe(self):
        import yt_transcriber.transcribe as transcribe_mod

        self.assertTrue(callable(transcribe_mod.transcribe))
        self.assertIn("base", transcribe_mod.MODEL_SIZES)

    def test_validates_unknown_model(self):
        from pathlib import Path

        from yt_transcriber.fetch import FetchError
        from yt_transcriber.transcribe import transcribe

        with self.assertRaises(FetchError):
            transcribe(Path("nope.wav"), model_size="does-not-exist")


if __name__ == "__main__":
    unittest.main()