"""Tests for the fetch module (captions parsing)."""

import unittest

from yt_transcriber.fetch import parse_vtt


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


if __name__ == "__main__":
    unittest.main()