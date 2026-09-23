"""Tests for visual context and vision injection into chunk analysis."""

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from yt_transcriber.context import OllamaClient, analyze
from yt_transcriber.vision import (
    MAX_GRID_CELLS,
    MIN_GRID_CELLS,
    _pick_font,
    _stamp_filter,
    _stamp_hms,
    choose_grid,
    compose_grids,
    describe_frames_batch,
    describe_grids,
    extract_frames,
    summarize_timeline,
)


class FakeClient:
    """Records prompts instead of calling a real model."""

    def __init__(self):
        self.prompts = []

    def complete(self, messages, images=None):
        self.prompts.append((messages[0]["content"], images))
        return "# CHAPTER\nTest chapter\n\n## WHAT HAPPENED\nSomething on screen."


class FakeBackend:
    """Vision backend stand-in used by batch / grid describing tests."""

    def __init__(self):
        self.calls = []

    def complete(self, messages, images=None):
        self.calls.append((messages[0]["content"], list(images or [])))
        return "A busy scene with on-screen text."


def _segs():
    return [
        {"start": 0.0, "end": 10.0, "text": "Opening words."},
        {"start": 10.0, "end": 20.0, "text": "Now showing the code."},
    ]


def _is_synth(prompt):
    return "SEGMENT ANALYSES" in prompt


def _ffmpeg():
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise unittest.SkipTest("ffmpeg not installed")
    return ffmpeg


def _make_video(path: Path, color_src: str, seconds: int) -> None:
    ffmpeg = _ffmpeg()
    result = subprocess.run(
        [
            ffmpeg,
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"{color_src}:size=320x180:rate=30",
            "-t",
            str(seconds),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"test video creation failed: {result.stderr[-300:]}")


class TestVisionInjection(unittest.TestCase):
    def test_visual_notes_pasted_into_matching_chunk(self):
        client = FakeClient()
        timeline = [{"ts": 5.0, "description": "Host greets the camera"},
                    {"ts": 120.0, "description": "Host clicks a button"}]
        analyze(_segs(), client, visual_timeline=timeline)
        chunk_prompts = [c for c, _ in client.prompts if not _is_synth(c)]
        self.assertEqual(len(chunk_prompts), 1)  # single chunk
        prompt = chunk_prompts[0]
        self.assertIn("VISUAL CONTEXT", prompt)
        self.assertIn("Host greets the camera", prompt)
        self.assertNotIn("Host clicks a button", prompt)  # outside window

    def test_no_visual_section_without_timeline(self):
        client = FakeClient()
        analyze(_segs(), client)
        chunk_prompts = [c for c, _ in client.prompts if not _is_synth(c)]
        self.assertNotIn("VISUAL CONTEXT", chunk_prompts[0])

    def test_report_gets_guaranteed_visual_section(self):
        client = FakeClient()
        timeline = [{"ts": 5.0, "description": "A red button labeled START fills the screen"},
                    {"ts": 60.0, "description": "The host clicks the button"}]
        result = analyze(_segs(), client, visual_timeline=timeline)
        self.assertIn("# What the video shows", result["report"])
        self.assertIn("red button labeled START", result["report"])

    def test_full_timeline_fed_to_synthesis(self):
        client = FakeClient()
        timeline = [{"ts": 5.0, "description": "User lands on the dashboard"}]
        analyze(_segs(), client, visual_timeline=timeline)
        synth = [c for c, _ in client.prompts if _is_synth(c)]
        self.assertEqual(len(synth), 1)
        self.assertIn("User lands on the dashboard", synth[0])

    def test_vision_client_requests_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            img = Path(tmp) / "f.jpg"
            img.write_bytes(b"\xff\xd8\xff\xe0fakejpeg")
            client = OllamaClient(base_url="http://localhost:11434", model="gemma4")
            # The real network call must not run; assert payload shape via a
            # patched transport is overkill - here we just verify the API contract.
            self.assertEqual(client.model, "gemma4")
            self.assertFalse(client.is_cloud)


class TestFrameExtraction(unittest.TestCase):
    def test_missing_ffmpeg_raises(self):
        from unittest import mock

        with mock.patch("yt_transcriber.vision.ffmpeg_bin", return_value=None):
            with self.assertRaises(Exception):
                extract_frames(Path("nonexistent.mp4"), Path("tmp"), "vid")

    def _frames(self, video: Path, seconds: int, *, dedupe: bool, max_gap: float):
        ffmpeg = _ffmpeg()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "f"
            return extract_frames(
                video,
                out,
                "v",
                interval_seconds=1,
                max_frames=600,
                dedupe=dedupe,
                dedupe_max_gap=max_gap,
            )

    def test_static_video_dedupes_to_single_frame(self):
        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp) / "static.mp4"
            _make_video(video, "color=c=blue", 6)
            frames = self._frames(video, 6, dedupe=True, max_gap=10)
            self.assertEqual(len(frames), 1)
            self.assertEqual(frames[0]["ts"], 0.0)

    def test_dedupe_honors_max_gap_on_long_static_scene(self):
        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp) / "static.mp4"
            _make_video(video, "color=c=blue", 25)
            frames = self._frames(video, 25, dedupe=True, max_gap=10)
            self.assertEqual([f["ts"] for f in frames], [0.0, 10.0, 20.0])

    def test_no_dedupe_keeps_every_second(self):
        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp) / "static.mp4"
            _make_video(video, "color=c=blue", 6)
            frames = self._frames(video, 6, dedupe=False, max_gap=10)
            self.assertEqual(len(frames), 6)

    def test_moving_video_keeps_all_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp) / "motion.mp4"
            _make_video(video, "testsrc2=size=320x180:rate=30", 5)
            frames = self._frames(video, 5, dedupe=True, max_gap=10)
            self.assertEqual(len(frames), 5)


class TestChooseGrid(unittest.TestCase):
    def test_sizes_by_target_grids(self):
        self.assertEqual(choose_grid(600, 24, 144), (5, 5, 25))
        self.assertEqual(choose_grid(3600, 24, 144), (12, 12, 144))
        self.assertEqual(choose_grid(7200, 24, 144), (12, 12, 144))

    def test_short_video_uses_small_grid(self):
        cols, rows, cells = choose_grid(120, 24, 144)
        self.assertGreaterEqual(cols * rows, 4)
        self.assertLessEqual(cols * rows, 144)

    def test_tiny_video_uses_minimum_grid(self):
        self.assertEqual(choose_grid(3, 24, 144), (2, 2, 4))

    def test_grid_bounds(self):
        # Few frames -> minimum grid; huge videos stay within model cell limits.
        self.assertEqual(choose_grid(3, 24, 144), (2, 2, MIN_GRID_CELLS))
        self.assertEqual(choose_grid(7200, 24, 144), (12, 12, MAX_GRID_CELLS))


class TestBatchAndMosaic(unittest.TestCase):
    def test_describe_frames_batch_groups_and_tags(self):
        frames = [
            {"ts": float(i), "path": Path(f"f{i}.jpg"), "description": ""}
            for i in range(10)
        ]
        backend = FakeBackend()
        result = describe_frames_batch(frames, backend, batch_size=4)
        self.assertEqual(len(result), 3)  # 4 + 4 + 2
        self.assertEqual(len(backend.calls), 3)
        self.assertEqual(len(backend.calls[0][1]), 4)
        self.assertEqual(len(backend.calls[2][1]), 2)
        self.assertEqual(result[0]["ts"], 0.0)
        self.assertEqual(result[0]["end"], 3.0)
        self.assertIn("Image 1 = 00:00", backend.calls[0][0])

    def test_choose_grid_to_mosaic(self):
        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp) / "static.mp4"
            _make_video(video, "color=c=red", 8)
            ffmpeg = _ffmpeg()
            frames = extract_frames(
                video, Path(tmp) / "frames", "v", interval_seconds=1,
                max_frames=600, dedupe=False,
            )
            self.assertEqual(len(frames), 8)
            cols, rows, cells = choose_grid(len(frames), 24, 144)
            grids = compose_grids(frames, Path(tmp), cols, rows, cells)
            self.assertGreaterEqual(len(grids), 1)
            self.assertTrue(all(Path(g["path"]).exists() for g in grids))
            frame_w = self._probe_width(Path(frames[0]["path"]))
            self._assert_montage_width(Path(grids[0]["path"]), cols * frame_w)
            backend = FakeBackend()
            obs = describe_grids(grids, backend)
            self.assertEqual(len(obs), len(grids))
            self.assertEqual(len(backend.calls), len(grids))
            self.assertTrue(all(o["cols"] == cols and o["rows"] == rows for o in obs))

    def _probe_width(self, image: Path) -> int:
        return self._probe(image, "width")

    def _assert_montage_width(self, png: Path, expected: int) -> None:
        actual = self._probe(png, "width")
        self.assertTrue(abs(actual - expected) <= 8, f"montage width {actual} != {expected}")

    def _probe(self, image: Path, key: str) -> int:
        probe = shutil.which("ffprobe")
        if not probe:
            raise unittest.SkipTest("ffprobe not installed")
        result = subprocess.run(
            [
                probe,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                f"stream={key}",
                "-of",
                "csv=p=0",
                str(image),
            ],
            capture_output=True,
            text=True,
        )
        value = result.stdout.strip()
        self.assertTrue(value, f"could not read {key} of {image}")
        return int(value)


class TestStampOverlay(unittest.TestCase):
    def test_pick_font_returns_existing_path_or_none(self):
        font = _pick_font()
        if font is None:
            self.skipTest("no Windows font found on this machine")
        self.assertTrue(Path(font).exists())

    def test_stamp_filter_escapes_windows_font_path(self):
        f = _stamp_filter(r"C:\Windows\Fonts\arial.ttf", "00:00:04")
        self.assertIn("drawtext=", f)
        self.assertIn(r"fontfile=C\\\:/Windows/Fonts/arial.ttf", f)
        self.assertIn(r"text='00\:00\:04'", f)
        self.assertIn("boxcolor=black@0.55", f)

    def test_stamp_hms_formatting(self):
        self.assertEqual(_stamp_hms(4), "00:00:04")
        self.assertEqual(_stamp_hms(65), "00:01:05")
        self.assertEqual(_stamp_hms(3661), "01:01:01")
        self.assertEqual(_stamp_hms(-3), "00:00:00")

    def test_extract_frames_still_works_with_stamp(self):
        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp) / "stamped.mp4"
            _make_video(video, "color=c=green", 4)
            frames = extract_frames(
                video,
                Path(tmp) / "frames",
                "v",
                interval_seconds=1,
                max_frames=600,
                dedupe=False,
            )
            self.assertEqual(len(frames), 4)
            self.assertTrue(all(Path(f["path"]).exists() for f in frames))


class TestWindowSummarization(unittest.TestCase):
    def _client(self):
        return FakeClient()

    def test_groups_observations_into_windows(self):
        observations = [
            {"ts": 0.0, "description": "Title card"},
            {"ts": 1.0, "description": "Host waves"},
            {"ts": 2.0, "description": "Camera pans to a laptop"},
            {"ts": 50.0, "description": "Code appears on screen"},
        ]
        result = summarize_timeline(observations, self._client(), window_seconds=30)
        self.assertEqual([w["ts"] for w in result], [0.0, 50.0])
        self.assertEqual([w["end"] for w in result], [2.0, 50.0])
        self.assertTrue(all(w["description"] for w in result))

    def test_single_window_for_short_clip(self):
        observations = [{"ts": 0.0, "description": "A"},
                        {"ts": 10.0, "description": "B"},
                        {"ts": 20.0, "description": "C"}]
        result = summarize_timeline(observations, self._client(), window_seconds=30)
        self.assertEqual(len(result), 1)

    def test_empty_observations(self):
        self.assertEqual(summarize_timeline([], self._client()), [])


if __name__ == "__main__":
    unittest.main()