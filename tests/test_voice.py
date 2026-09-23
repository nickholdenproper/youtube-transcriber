"""Tests for the voice analysis tool (audio measurements + report wiring)."""

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from yt_transcriber.api import app
from yt_transcriber.context import analyze
from yt_transcriber.tools import TOOLS, list_tools
from yt_transcriber.voice import (
    analyze_audio,
    loudness_curve,
    silence_gaps,
    summarize_voice,
    volume_stats,
)

api_client = TestClient(app)


class FakeClient:
    def __init__(self):
        self.prompts = []

    def complete(self, messages, images=None):
        self.prompts.append(messages[0]["content"])
        return "# CHAPTER\nTest chapter\n\n## WHAT HAPPENED\nSomething happened on screen.\n"


def _ffmpeg():
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise unittest.SkipTest("ffmpeg not installed")
    return ffmpeg


def _make_wav(path: Path, seconds: int = 2, delay_ms: int = 0) -> None:
    ffmpeg = _ffmpeg()
    cmd = [
        ffmpeg,
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"sine=frequency=440:sample_rate=44100:duration={seconds}",
        "-ac",
        "1",
    ]
    if delay_ms:
        cmd += ["-af", f"adelay={delay_ms}|{delay_ms}"]
    cmd += [str(path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"test wav creation failed: {result.stderr[-300:]}")


class TestParsers(unittest.TestCase):
    def test_volume_stats(self):
        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp) / "tone.wav"
            _make_wav(wav)
            stats = volume_stats(wav, _ffmpeg())
            self.assertIn("mean_volume_db", stats)
            self.assertIn("max_volume_db", stats)
            self.assertLess(stats["mean_volume_db"], 0)
            self.assertGreater(stats["max_volume_db"], -50)
            self.assertGreater(stats["max_volume_db"], stats["mean_volume_db"])

    def test_loudness_curve(self):
        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp) / "tone.wav"
            _make_wav(wav, seconds=3)
            curve = loudness_curve(wav, _ffmpeg())
            self.assertGreater(len(curve), 5)
            for t, lufs in curve:
                self.assertGreater(lufs, -100)
                self.assertLess(t, 4)

    def test_silence_gaps_detect_lead_silence(self):
        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp) / "delayed.wav"
            _make_wav(wav, seconds=2, delay_ms=2000)
            gaps = silence_gaps(wav, _ffmpeg())
            self.assertTrue(gaps)
            first = gaps[0]
            self.assertLess(first["start"], 1.0)
            self.assertGreater(first["end"], 1.5)


class TestAnalyzeAudio(unittest.TestCase):
    def test_produces_stats_and_profiles(self):
        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp) / "tone.wav"
            _make_wav(wav, seconds=4)
            segs = [
                {"start": 2.0, "end": 4.0, "text": "word word word word word word word word"},
            ]
            result = analyze_audio(wav, segs, duration=4.0)
            stats = result["stats"]
            self.assertEqual(stats["duration"], 4.0)
            self.assertIn("mean_lufs", stats)
            self.assertIn("pace", stats)
            self.assertIn("n_energy_bursts", stats)
            profile = result["segments"][0]
            self.assertIn("wpm", profile)
            self.assertGreater(profile["words"], 0)
            self.assertIn("pause_ratio", profile)

    def test_speech_ratio_drops_with_lead_silence(self):
        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp) / "delayed.wav"
            _make_wav(wav, seconds=2, delay_ms=2000)
            result = analyze_audio(wav, [], duration=4.0)
            self.assertGreater(result["stats"]["n_silences"], 0)
            self.assertLess(result["stats"]["speech_ratio"], 1.0)

    def test_missing_ffmpeg_raises(self):
        from unittest import mock

        with mock.patch("yt_transcriber.voice.ffmpeg_bin", return_value=None):
            with self.assertRaises(Exception):
                analyze_audio(Path("nonexistent.mp3"), [])


class TestSummarizeVoice(unittest.TestCase):
    def test_groups_profiles_into_windows(self):
        client = FakeClient()
        profiles = [
            {"start": 0.0, "end": 5.0, "wpm": 150.0, "mean_lufs": -20.0, "pause_ratio": 0.0},
            {"start": 6.0, "end": 10.0, "wpm": 180.0, "mean_lufs": -15.0, "pause_ratio": 0.0},
            {"start": 50.0, "end": 60.0, "wpm": 100.0, "mean_lufs": -28.0, "pause_ratio": 0.5},
        ]
        notes = summarize_voice(profiles, client, window_seconds=30)
        self.assertEqual([n["ts"] for n in notes], [0.0, 50.0])
        self.assertTrue(all(n["description"] for n in notes))
        self.assertIn("wpm", client.prompts[0])

    def test_empty_profiles(self):
        self.assertEqual(summarize_voice([], FakeClient()), [])


class TestAnalyzeVoiceInjection(unittest.TestCase):
    def test_voice_notes_land_in_chunks_and_report(self):
        client = FakeClient()
        segments = [{"start": 0.0, "end": 30.0, "text": "The host explains the setup quickly."}]
        voice_notes = [
            {"ts": 5.0, "end": 15.0, "description": "Speaks fast, high energy, no pauses."}
        ]
        analyze(segments, client, voice_notes=voice_notes)
        chunk_prompt = client.prompts[0]
        synth_prompt = client.prompts[-1]
        self.assertIn("VOICE CONTEXT", chunk_prompt)
        self.assertIn("Speaks fast, high energy", chunk_prompt)
        self.assertIn("VOICE EVIDENCE", synth_prompt)

    def test_report_gets_fallback_voice_section(self):
        client = FakeClient()
        segments = [{"start": 0.0, "end": 30.0, "text": "Calm, measured delivery."}]
        voice_notes = [{"ts": 0.0, "end": 30.0, "description": "Calm pace, moderate loudness."}]
        result = analyze(segments, client, voice_notes=voice_notes)
        self.assertIn("# How it sounds", result["report"])
        self.assertIn("Calm pace", result["report"])

    def test_no_voice_section_when_no_notes(self):
        client = FakeClient()
        segments = [{"start": 0.0, "end": 30.0, "text": "Plain transcript only."}]
        result = analyze(segments, client)
        self.assertNotIn("# How it sounds", result["report"])


class TestToolsRegistry(unittest.TestCase):
    def test_registered_tools(self):
        ids = {t["id"] for t in list_tools()}
        self.assertEqual(ids, {"vision", "voice", "derive"})
        self.assertIn("--no-voice", TOOLS["voice"]["cli_flag"])

    def test_api_lists_tools(self):
        r = api_client.get("/v1/tools")
        self.assertEqual(r.status_code, 200)
        ids = {t["id"] for t in r.json()}
        self.assertIn("voice", ids)

    def test_openapi_documents_tools(self):
        r = api_client.get("/openapi.json")
        self.assertIn("/v1/tools", r.json()["paths"])


if __name__ == "__main__":
    unittest.main()