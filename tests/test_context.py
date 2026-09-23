"""Tests for the context analysis module (chunking + formatting)."""

import unittest

from yt_transcriber.context import chunk_segments, fmt_ts


def _segs(word_groups):
    segs, t = [], 0.0
    for words in word_groups:
        text = "word " * (words // 4) + "x" * (words % 4)
        segs.append({"start": t, "end": t + 5, "text": text.strip()})
        t += 5
    return segs


class TestChunking(unittest.TestCase):
    def test_single_chunk_when_small(self):
        segs = _segs([20, 20])
        chunks = chunk_segments(segs, max_chars=1000)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["start"], segs[0]["start"])
        self.assertEqual(chunks[0]["end"], segs[-1]["end"])

    def test_multiple_chunks_when_large(self):
        segs = _segs([800, 800, 800, 800, 800])
        chunks = chunk_segments(segs, max_chars=1000)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(len(chunk["text"]), 1000 + 50)

    def test_empty_segments(self):
        self.assertEqual(chunk_segments([]), [])


class TestTimestampFormatting(unittest.TestCase):
    def test_minutes(self):
        self.assertEqual(fmt_ts(125), "02:05")

    def test_hours(self):
        self.assertEqual(fmt_ts(3661), "1:01:01")

    def test_zero(self):
        self.assertEqual(fmt_ts(0), "00:00")


if __name__ == "__main__":
    unittest.main()