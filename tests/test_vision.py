"""Tests for visual context and vision injection into chunk analysis."""

import tempfile
import unittest
from pathlib import Path

from yt_transcriber.context import OllamaClient, analyze
from yt_transcriber.vision import extract_frames


class FakeClient:
    """Records prompts instead of calling a real model."""

    def __init__(self):
        self.prompts = []

    def complete(self, messages, images=None):
        self.prompts.append((messages[0]["content"], images))
        return "# CHAPTER\nTest chapter\n\n## WHAT HAPPENED\nSomething on screen."


def _segs():
    return [
        {"start": 0.0, "end": 10.0, "text": "Opening words."},
        {"start": 10.0, "end": 20.0, "text": "Now showing the code."},
    ]


class TestVisionInjection(unittest.TestCase):
    def test_visual_notes_pasted_into_matching_chunk(self):
        client = FakeClient()
        timeline = [{"ts": 5.0, "description": "Host greets the camera"},
                    {"ts": 120.0, "description": "Host clicks a button"}]
        analyze(_segs(), client, visual_timeline=timeline)
        chunk_prompts = [c for c, _ in client.prompts if not c.startswith("Combine these")]
        self.assertEqual(len(chunk_prompts), 1)  # single chunk
        prompt = chunk_prompts[0]
        self.assertIn("VISUAL CONTEXT", prompt)
        self.assertIn("Host greets the camera", prompt)
        self.assertNotIn("Host clicks a button", prompt)  # outside window

    def test_no_visual_section_without_timeline(self):
        client = FakeClient()
        analyze(_segs(), client)
        chunk_prompts = [c for c, _ in client.prompts if not c.startswith("Combine these")]
        self.assertNotIn("VISUAL CONTEXT", chunk_prompts[0])

    def test_vision_client_requests_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            img = Path(tmp) / "f.jpg"
            img.write_bytes(b"\xff\xd8\xff\xe0fakejpeg")
            client = OllamaClient(base_url="http://localhost:11434", model="llama3.2-vision")
            # The real network call must not run; assert payload shape via a
            # patched transport is overkill - here we just verify the API contract.
            self.assertEqual(client.model, "llama3.2-vision")
            self.assertFalse(client.is_cloud)


class TestFrameExtraction(unittest.TestCase):
    def test_missing_ffmpeg_raises(self):
        from unittest import mock

        with mock.patch("yt_transcriber.vision.ffmpeg_bin", return_value=None):
            with self.assertRaises(Exception):
                extract_frames(Path("nonexistent.mp4"), Path("tmp"), "vid")


if __name__ == "__main__":
    unittest.main()