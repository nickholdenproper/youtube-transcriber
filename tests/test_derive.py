"""Tests for the derived-document (second LLM) stage."""

from __future__ import annotations

import unittest

from fake_client import FakeBackend, FakeTextClient

from yt_transcriber.context import ContextError
from yt_transcriber.derive import (
    DERIVE_TYPES,
    DeriveError,
    build_digest,
    derive,
)
from yt_transcriber.output import write_outputs


def _segs():
    return [
        {"start": 0.0, "end": 5.0, "text": "Welcome, today we verify your account."},
        {"start": 55.0, "end": 62.0, "text": "Click the verify button on the screen."},
        {"start": 120.0, "end": 125.0, "text": "Your account is now fully verified."},
    ]


def _timeline():
    return [
        {"ts": 1.0, "end": 4.0, "description": "A verify button appears bottom right."},
        {"ts": 58.0, "end": 61.0, "description": "The screen shows a green Continue button."},
    ]


def _voice():
    return [
        {"ts": 0.0, "end": 5.0, "description": "Calm measured pace."},
    ]


class TestBuildDigest(unittest.TestCase):
    def test_includes_ground_truth_sections(self):
        digest = build_digest(
            _segs(),
            title="Verify My Account",
            channel="Test Ch",
            report="# TL;DR\nNothing.",
            visual_timeline=_timeline(),
            voice_notes=_voice(),
            chunks=[{"start": 0.0, "end": 30.0, "analysis": "Intro present."}],
        )
        self.assertIn("# Verify My Account", digest)
        self.assertIn("- Channel: Test Ch", digest)
        self.assertIn("## Verbatim transcript (ground truth)", digest)
        self.assertIn("[00:00] Welcome, today we verify your account.", digest)
        self.assertIn("## Segment analyses", digest)
        self.assertIn("Intro present.", digest)
        self.assertIn("## Visual evidence", digest)
        self.assertIn("00:01\u201300:04 - A verify button appears bottom right.", digest)
        self.assertIn("## Voice evidence (delivery)", digest)
        self.assertIn("Calm measured pace.", digest)
        self.assertIn("## Existing context report", digest)
        self.assertIn("Nothing.", digest)

    def test_digest_uses_real_ranges_for_evidence(self):
        digest = build_digest(_segs(), visual_timeline=_timeline())
        self.assertIn("00:01\u201300:04 - A verify button appears bottom right.", digest)
        self.assertIn("00:58\u201301:01 - The screen shows a green Continue button.", digest)


class TestDerive(unittest.TestCase):
    def test_each_type_uses_digest_and_type_rules(self):
        for kind in DERIVE_TYPES:
            client = FakeTextClient("Some markdown.")
            out = derive(
                kind,
                client,
                _segs(),
                title="T",
                report="R",
                visual_timeline=_timeline(),
                voice_notes=_voice(),
                chunks=[],
            )
            self.assertEqual(out, "Some markdown.")
            prompt = client.prompts[0]
            self.assertIn("## Verbatim transcript (ground truth)", prompt)
            self.assertIn("[00:00] Welcome, today we verify your account.", prompt)
            self.assertIn("## Existing context report", prompt)
            self.assertIn("never invent", prompt.lower())
            self.assertGreater(len(prompt), 1200)

    def test_how_to_prompt_asks_for_numbered_steps_with_timestamps(self):
        client = FakeTextClient("done")
        derive("how-to", client, _segs(), report="r")
        prompt = client.prompts[0]
        self.assertIn("step-by-step how-to guide", prompt)
        self.assertIn("Never invent", prompt)

    def test_quiz_prompt_requests_flashcards(self):
        client = FakeTextClient("done")
        derive("quiz", client, _segs(), report="r")
        self.assertIn("exactly 10 flashcard pairs", client.prompts[0])

    def test_unknown_type_raises(self):
        with self.assertRaises(DeriveError):
            derive("meme", FakeTextClient("x"), _segs(), report="r")

    def test_empty_reply_raises(self):
        with self.assertRaises(DeriveError):
            derive("faq", FakeTextClient("   "), _segs(), report="r")

    def test_empty_transcript_raises(self):
        with self.assertRaises(DeriveError):
            derive("faq", FakeTextClient("x"), [], report="r")

    def test_handles_empty_visual_and_voice(self):
        client = FakeTextClient("ok")
        derive("article", client, _segs(), report="r", visual_timeline=[], voice_notes=[])
        prompt = client.prompts[0]
        self.assertIn("None - no video frames were analyzed.", prompt)
        self.assertIn("None - no voice analysis was performed.", prompt)


class TestOutputDerivedFiles(unittest.TestCase):
    def test_writes_derived_markdown(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            from dataclasses import dataclass

            @dataclass
            class _Meta:
                title: str = "Verify"
                channel: str = "Ch"
                url: str = "https://example.com/v"
                id: str = "abc"
                duration: float = 125

            written = write_outputs(
                Path(tmp),
                _Meta(),
                _segs(),
                "audio",
                context_report="# TL;DR\n.",
                derived={"how-to": "- 1. Do it `[00:00]`", "quiz": "**Q:** ?\n**A:** !"},
            )
            names = [p.name for p in written]
            self.assertIn("derived_how-to.md", names)
            self.assertIn("derived_quiz.md", names)
            how = (Path(tmp) / "derived_how-to.md").read_text(encoding="utf-8")
            self.assertIn("# Verify", how)
            self.assertIn("- 1. Do it `[00:00]`", how)


class TestCLIAndModelOptions(unittest.TestCase):
    def test_pipeline_options_accept_derive(self):
        from yt_transcriber.pipeline import PipelineOptions

        opts = PipelineOptions(derive=["how-to", "faq"], derive_model="llama3.3")
        self.assertEqual(opts.derive, ["how-to", "faq"])
        self.assertEqual(opts.derive_model, "llama3.3")

    def test_api_options_model_round_trip(self):
        from yt_transcriber.api import PipelineOptionsModel, _options_from_model

        m = PipelineOptionsModel(derive=["article"], derive_model="gemma4")
        opts = _options_from_model(m)
        self.assertEqual(opts.derive, ["article"])
        self.assertEqual(opts.derive_model, "gemma4")


class TestDeriveErrorIsContextError(unittest.TestCase):
    def test_hierarchy(self):
        self.assertTrue(issubclass(DeriveError, ContextError))

    def test_fake_backend_works_for_derive(self):
        client = FakeBackend()
        out = derive("checklist", client, _segs(), report="r")
        self.assertEqual(out, "A busy dashboard with clear on-screen text.")


if __name__ == "__main__":
    unittest.main()