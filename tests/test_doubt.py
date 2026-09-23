"""Tests for the two-pass adaptive vision (doubt pass)."""

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from yt_transcriber.api import PipelineOptionsModel
from yt_transcriber.pipeline import PipelineOptions
from yt_transcriber.vision import (
    VisionError,
    deterministic_fill,
    extract_at_timestamps,
    merge_observations,
    request_doubt_frames,
    score_description,
    score_observations,
)

from fake_client import FakeBackend, FakeTextClient


def _ffmpeg():
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise unittest.SkipTest("ffmpeg not installed")
    return ffmpeg


def _make_video(path: Path, seconds: int = 8) -> None:
    ffmpeg = _ffmpeg()
    result = subprocess.run(
        [
            ffmpeg,
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:size=320x180:rate=30",
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


class TestScore(unittest.TestCase):
    def test_classifies_descriptions(self):
        self.assertEqual(score_description("I cannot fulfill this request"), "refused")
        self.assertEqual(score_description("I am unable to describe this image"), "refused")
        self.assertEqual(score_description("The frames are mostly black frames."), "blank")
        self.assertEqual(score_description("Same as the previous frame."), "lowinfo")
        self.assertEqual(score_description("A green button labeled Verify fills the screen."), "ok")
        self.assertEqual(score_description(""), "blank")

    def test_score_observations_adds_provenance(self):
        obs = [{"ts": 1.0, "description": "I cannot fulfill this request."}]
        scored = score_observations(obs)
        self.assertEqual(scored[0]["status"], "refused")
        self.assertEqual(scored[0]["source"], "pass1")
        self.assertEqual(scored[0]["end"], 1.0)


class TestRequestDoubtFrames(unittest.TestCase):
    def _client_and_call(self, reply, **kwargs) -> tuple[list, str]:
        client = FakeTextClient(reply)
        segments = [{"start": 0.0, "end": 5.0, "text": "Now click the button."}]
        voice = [{"ts": 2.0, "description": "Energetic."}]
        obs = [{"ts": 1.0, "description": "A screen with a button."}]
        result = request_doubt_frames(
            client,
            "Test title",
            60.0,
            segments,
            voice,
            {"pace": "fast"},
            obs,
            max_requests=10,
            **kwargs,
        )
        return result, client.prompts[0]

    def test_parses_and_de_dupes(self):
        reply = (
            "t=5.0 | high | host references a button not shown\n"
            "t=5.5 | mid | too close to previous, dropped\n"
            "t=30 | low | verify state\n"
            "t=999 | low | out of range, dropped\n"
        )
        result, _ = self._client_and_call(reply)
        self.assertEqual([r["t"] for r in result], [5.0, 30.0])
        self.assertEqual(result[0]["priority"], "hig")
        self.assertIn("button not shown", result[0]["reason"])

    def test_none_reply_returns_empty(self):
        result, _ = self._client_and_call("NONE")
        self.assertEqual(result, [])

    def test_garbage_reply_raises(self):
        client = FakeTextClient("Sure, here is my full prose analysis...")
        with self.assertRaises(VisionError):
            self._client_and_call("Sure, here is my full prose analysis...")

    def test_prompt_contains_pane_sources(self):
        _, prompt = self._client_and_call(
            "t=2 | mid | fix this\n",
        )
        self.assertIn("Test title", prompt)
        self.assertIn("Now click the button.", prompt)
        self.assertIn("Energetic.", prompt)
        self.assertIn("A screen with a button.", prompt)


class TestExtractAtTimestamps(unittest.TestCase):
    def test_captures_frames_near_requested_times(self):
        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp) / "dut.mp4"
            _make_video(video, 8)
            out = Path(tmp) / "doubt"
            frames = extract_at_timestamps(
                video,
                out,
                "vid",
                [{"t": 1.0, "reason": "x"}, {"t": 4.5, "reason": "y"}],
                duration=8.0,
                budget=6,
            )
            self.assertTrue(frames)
            self.assertLessEqual(len(frames), 6)
            self.assertTrue(all(Path(f["path"]).exists() for f in frames))
            for f in frames:
                self.assertTrue(-0.5 <= f["ts"] <= 8.0)
            tss = [f["ts"] for f in frames]
            self.assertLessEqual(min(tss), 1.1)  # ~{0,1,2} window present

    def test_no_timestamps_is_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp) / "dut.mp4"
            _make_video(video, 4)
            frames = extract_at_timestamps(
                video, Path(tmp), "vid", [], duration=4.0
            )
            self.assertEqual(frames, [])

    def test_clamps_to_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp) / "dut.mp4"
            _make_video(video, 6)
            frames = extract_at_timestamps(
                video, Path(tmp), "vid", [{"t": 0.0, "reason": "intro"}], duration=6.0
            )
            self.assertTrue(all(f["ts"] >= 0.0 for f in frames))


class TestDescribeDoubtFrames(unittest.TestCase):
    def test_ok_batches_pass_through(self):
        from yt_transcriber.vision import describe_doubt_frames

        frames = [{"ts": float(i), "path": Path(f"x{i}.jpg"), "reason": "d"} for i in range(4)]
        backend = FakeBackend()
        obs = describe_doubt_frames(frames, backend, batch_size=4)
        self.assertEqual(len(obs), 1)
        self.assertEqual(obs[0]["source"], "doubt")
        self.assertEqual(obs[0]["status"], "ok")

    def test_refused_batch_retried_per_frame(self):
        from yt_transcriber.vision import describe_doubt_frames

        frames = [{"ts": float(i), "path": Path(f"x{i}.jpg"), "reason": "d"} for i in range(4)]
        backend = FakeBackend(batch_mode="refuse-batches")
        obs = describe_doubt_frames(frames, backend, batch_size=4)
        self.assertEqual(len(obs), 4)  # retried as single frames
        self.assertTrue(all(o["source"] == "doubt" for o in obs))
        self.assertEqual(len(backend.calls), 5)  # 1 batch + 4 singles


class TestMerge(unittest.TestCase):
    def test_merge_keeps_provenance_but_drops_refused_from_usable(self):
        pass1 = [
            {"ts": 1.0, "description": "I cannot fulfill this request."},
            {"ts": 5.0, "description": "The dashboard is open."},
        ]
        doubt = [{"ts": 2.0, "description": "Now the popup is visible."}]
        provenance, usable = merge_observations(pass1, doubt)
        self.assertEqual(len(provenance), 3)
        self.assertEqual([o["ts"] for o in usable], [2.0, 5.0])
        self.assertTrue(all(o["status"] != "refused" for o in usable))
        refused = [o for o in provenance if o["status"] == "refused"]
        self.assertEqual(len(refused), 1)

    def test_merge_drops_redundant_lowinfo_runs(self):
        pass1 = [
            {"ts": 1.0, "description": "Same as the previous frame."},
            {"ts": 5.0, "description": "Same as the previous frame."},
            {"ts": 30.0, "description": "A new screen with a form."},
        ]
        provenance, usable = merge_observations(pass1, [])
        self.assertEqual([o["ts"] for o in usable], [1.0, 30.0])

    def test_merge_collapses_doubt_cluster_within_1_5s(self):
        doubt = [
            {"ts": float(t), "source": "doubt", "description": f"CAPTCHA checked at {t}."}
            for t in (39.0, 40.0, 41.0)
        ] + [
            {"ts": 50.0, "source": "doubt", "description": "SMS field appears."}
        ]
        _, usable = merge_observations([], doubt)
        self.assertEqual([o["ts"] for o in usable], [39.0, 50.0])

    def test_merge_keeps_pass1_interleaved_in_doubt_cluster(self):
        pass1 = [{"ts": 40.0, "source": "pass1", "description": "The dashboard is open."}]
        doubt = [
            {"ts": 39.0, "source": "doubt", "description": "CAPTCHA box."},
            {"ts": 40.5, "source": "doubt", "description": "CAPTCHA still there."},
            {"ts": 42.0, "source": "doubt", "description": "SMS prompt appears."},
        ]
        _, usable = merge_observations(pass1, doubt)
        # doubt@39 kept, doubt@40.5 dropped (within 1.5s of doubt@39),
        # pass1@40 kept, doubt@42 kept
        self.assertEqual([o["ts"] for o in usable], [39.0, 40.0, 42.0])


class TestDeterministicFill(unittest.TestCase):
    def test_fills_refusals_and_coverage_holes(self):
        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp) / "dut.mp4"
            _make_video(video, 10)
            backend = FakeBackend()
            observations = [
                {"ts": 1.0, "description": "I cannot fulfill this request."},
                {"ts": 9.0, "description": "Final screen."},
            ]
            doubt = deterministic_fill(
                observations,
                video,
                Path(tmp) / "doubt",
                "vid",
                backend,
                duration=10.0,
                budget=8,
                window_seconds=30,
            )
            # refusal at 1.0 gets captured; gap 1->9 is 8 (< 60) so no midpoint
            self.assertTrue(doubt)
            self.assertTrue(all(o["source"] == "doubt" for o in doubt))


class TestOptionWiring(unittest.TestCase):
    def test_pipeline_options_include_doubt_fields(self):
        opts = PipelineOptions()
        self.assertTrue(opts.vision_doubt)
        self.assertEqual(opts.vision_fill_budget, 60)
        opts.vision_doubt = False
        self.assertFalse(opts.vision_doubt)

    def test_api_options_model_exposes_doubt_fields(self):
        model = PipelineOptionsModel()
        self.assertTrue(model.vision_doubt)
        self.assertEqual(model.vision_fill_budget, 60)
        dumped = PipelineOptionsModel(vision_doubt=False).model_dump()
        self.assertFalse(dumped["vision_doubt"])


if __name__ == "__main__":
    unittest.main()