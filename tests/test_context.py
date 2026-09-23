"""Tests for the context analysis module (chunking + reconciliation)."""

import unittest

from yt_transcriber.context import (
    REPORT_PROMPT,
    STAMP_NOTE,
    analyze,
    chunk_segments,
    fmt_ts,
    verify_report,
)

from fake_client import FakeContextClient


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
        self.assertEqual(len(chunks[0]["segments"]), 2)

    def test_multiple_chunks_when_large(self):
        segs = _segs([800, 800, 800, 800, 800])
        chunks = chunk_segments(segs, max_chars=1000)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(len(chunk["text"]), 1000 + 50)
            self.assertEqual(chunk["segments"][0]["start"], chunk["start"])

    def test_empty_segments(self):
        self.assertEqual(chunk_segments([]), [])


class TestChunkRendering(unittest.TestCase):
    def test_chunk_text_stamped_with_real_timestamps(self):
        client = FakeContextClient()
        segs = _segs([20, 20])
        analyze(segs, client)
        chunk_prompt = client.prompts[0][0]
        self.assertIn("[00:00]", chunk_prompt)
        self.assertIn("[00:05]", chunk_prompt)
        self.assertIn("word word word word word", chunk_prompt)


class TestTimestampFormatting(unittest.TestCase):
    def test_minutes(self):
        self.assertEqual(fmt_ts(125), "02:05")

    def test_hours(self):
        self.assertEqual(fmt_ts(3661), "1:01:01")

    def test_zero(self):
        self.assertEqual(fmt_ts(0), "00:00")


class TestReportGroundRules(unittest.TestCase):
    def test_forbids_inventing_content(self):
        self.assertIn("NEVER invent topics, people, tools, websites, names", REPORT_PROMPT)
        self.assertIn("do not guess", REPORT_PROMPT)

    def test_pins_real_product_name_to_title(self):
        self.assertIn('The video is titled "{title}"', REPORT_PROMPT)
        self.assertIn("not a phonetically-misspoken", REPORT_PROMPT)

    def test_requires_distinct_chapters(self):
        self.assertIn("DISTINCT and content-driven", REPORT_PROMPT)

    def test_transcript_is_ground_truth_anchor(self):
        self.assertIn("GROUND TRUTH TRANSCRIPT", REPORT_PROMPT)
        self.assertIn("the transcript wins and the claim is DROPPED", REPORT_PROMPT)

    def test_quotes_must_be_verbatim(self):
        self.assertIn("VERBATIM in the GROUND TRUTH TRANSCRIPT", REPORT_PROMPT)
        self.assertIn("quoted in analysis", REPORT_PROMPT)

    def test_full_prompt_formats(self):
        formatted = REPORT_PROMPT.format(
            title="Fanvue Tips",
            channel="Answer ASAP",
            duration="02:41",
            transcript="- [00:00] Intro",
            analyses="- opening\n- steps",
            visual="- 00:00-00:03 - grid UI",
            voice="- 00:00 - measured pace",
        )
        self.assertIn('The video is titled "Fanvue Tips"', formatted)
        self.assertIn("The whole video spans 02:41.", formatted)
        self.assertIn("GROUND TRUTH TRANSCRIPT", formatted)

    def test_stamp_note_teaches_model_the_stamp_is_not_video(self):
        self.assertIn("NOT part of the video", STAMP_NOTE)
        self.assertIn("never describe it as video content", STAMP_NOTE)


class TestVerifyReport(unittest.TestCase):
    GOOD = (
        "# TL;DR\nsummary\n\n"
        "# What happened\n"
        "- 00:00 - Opens dashboard\n"
        "- 01:00 - Clicks continue\n"
        "- 02:00 - Done\n\n"
        "# Chapters\n"
        "- 00:00 - Opening\n"
        "- 01:00 - Setup\n"
        "- 02:00 - Finish\n\n"
        "# Key quotes\n"
        '- "Click continue" (01:00)\n\n'
    )

    @staticmethod
    def _segs():
        return [
            {"start": 0.0, "end": 5.0, "text": "Let's begin."},
            {"start": 55.0, "end": 62.0, "text": "Click continue, please."},
            {"start": 120.0, "end": 125.0, "text": "We are all done."},
        ]

    def test_healthy_report_produces_no_warnings(self):
        report = self.GOOD
        filtered, warnings = verify_report(report, self._segs(), 125.0)
        self.assertEqual(warnings, [])
        self.assertEqual(filtered, report)

    def test_fabricated_quote_is_dropped(self):
        report = self.GOOD + '- "This was never said on camera" (00:55 - quoted in analysis)\n\n'
        filtered, warnings = verify_report(report, self._segs(), 125.0)
        self.assertIn("fabricated quote", warnings[0])
        self.assertIn('"Click continue" (01:00)', filtered)
        self.assertNotIn("This was never said on camera", filtered)

    def test_compressed_clock_warns(self):
        report = (
            "# Chapters\n"
            "- 00:00 - Opening\n"
            "- 00:12 - Middle\n"
            "- 00:20 - End\n\n"
        )
        _, warnings = verify_report(report, self._segs(), 125.0)
        self.assertTrue(any("Chapters" in w and "compressed" in w for w in warnings))

    def test_overflowing_timestamp_warns(self):
        report = self.GOOD.replace("- 02:00 - Done\n", "- 02:00 - Done\n- 99:00 - far beyond the video\n")
        _, warnings = verify_report(report, self._segs(), 125.0)
        self.assertTrue(any("past the video length" in w for w in warnings))

    def test_empty_report_no_warnings(self):
        filtered, warnings = verify_report("", self._segs(), 125.0)
        self.assertEqual(warnings, [])
        self.assertEqual(filtered, "")


if __name__ == "__main__":
    unittest.main()