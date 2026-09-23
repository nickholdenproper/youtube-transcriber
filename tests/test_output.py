"""Tests for output writing and the CLI runner."""

import tempfile
import unittest
from pathlib import Path

from typer.testing import CliRunner

from yt_transcriber.cli import app
from yt_transcriber.fetch import VideoMeta
from yt_transcriber.output import write_outputs

runner = CliRunner()


class TestOutput(unittest.TestCase):
    def test_writes_transcript_files(self):
        meta = VideoMeta(
            id="abc123", title="Test Video", channel="Tester",
            url="https://youtube.com/watch?v=abc123", duration=120.0,
        )
        segments = [{"start": 0.0, "end": 5.0, "text": "Hello"}]
        with tempfile.TemporaryDirectory() as tmp:
            paths = write_outputs(Path(tmp) / "abc123", meta, segments, "captions")
            text = (Path(tmp) / "abc123" / "transcript.md").read_text(encoding="utf-8")
            self.assertIn("Test Video", text)
            self.assertIn("[00:00]", text)
            self.assertEqual(len(paths), 2)

    def test_writes_context_files_when_report_present(self):
        meta = VideoMeta(id="x", title="T", channel="C", url="u")
        with tempfile.TemporaryDirectory() as tmp:
            paths = write_outputs(
                Path(tmp) / "x", meta, [{"start": 0, "end": 1, "text": "hi"}],
                "captions", context_report="# TL;DR\nstuff", context_chunks=[],
            )
            self.assertEqual(len(paths), 4)
            self.assertTrue((Path(tmp) / "x" / "context.md").exists())


class TestCli(unittest.TestCase):
    def test_help_exits_zero(self):
        result = runner.invoke(app, ["--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Usage", result.output)

    def test_missing_url_is_an_error(self):
        result = runner.invoke(app, [])
        self.assertEqual(result.exit_code, 2)


if __name__ == "__main__":
    unittest.main()