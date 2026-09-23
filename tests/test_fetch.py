"""Tests for the fetch module (captions parsing)."""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from yt_transcriber.fetch import FetchError, fetch_video, parse_vtt


def _info():
    return {
        "id": "abc123",
        "title": "Test Video",
        "channel": "Tester",
        "webpage_url": "https://youtube.com/watch?v=abc123",
        "duration": 120,
    }


class TestParseVtt(unittest.TestCase):
    def test_parses_simple_cues(self):
        vtt = """WEBVTT

00:00:00.000 --> 00:00:03.000
Hello world

00:00:03.000 --> 00:00:06.500
Second line here
"""
        segs = parse_vtt(vtt)
        self.assertEqual(len(segs), 2)
        self.assertEqual(segs[0]["start"], 0.0)
        self.assertEqual(segs[0]["end"], 3.0)
        self.assertEqual(segs[0]["text"], "Hello world")
        self.assertEqual(segs[1]["start"], 3.0)

    def test_strips_inline_tags(self):
        vtt = "WEBVTT\n\n00:00:01.000 --> 00:00:02.000\n<c>Hi</c> there\n"
        segs = parse_vtt(vtt)
        self.assertEqual(segs[0]["text"], "Hi there")

    def test_skips_notes_and_settings(self):
        vtt = """WEBVTT

NOTE this is a comment

00:00:01.000 --> 00:00:02.000 align:start position:0%
Real caption
"""
        segs = parse_vtt(vtt)
        self.assertEqual(len(segs), 1)
        self.assertEqual(segs[0]["text"], "Real caption")

    def test_empty_input(self):
        self.assertEqual(parse_vtt(""), [])


class TestFetchFallback(unittest.TestCase):
    def test_caption_failure_falls_back_to_audio(self):
        """A caption download error (e.g. HTTP 429) must not abort the run."""
        with tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp) / "abc123.mp3"
            audio.write_bytes(b"fake")

            with (
                mock.patch("yt_transcriber.fetch.extract_info", return_value=_info()),
                mock.patch("yt_transcriber.fetch.pick_caption", return_value=("subtitles", "en")),
                mock.patch(
                    "yt_transcriber.fetch.download_captions",
                    side_effect=FetchError("HTTP Error 429: Too Many Requests"),
                ),
                mock.patch("yt_transcriber.fetch.download_audio", return_value=audio),
            ):
                result = fetch_video("https://youtube.com/watch?v=abc123", Path(tmp))

            self.assertEqual(result.source, "audio")
            self.assertEqual(result.audio_path, audio)
            self.assertTrue(any("429" in w for w in result.warnings))


if __name__ == "__main__":
    unittest.main()