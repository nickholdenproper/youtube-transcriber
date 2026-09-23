"""Voice analysis: how a video *sounds*, measured from the audio.

The transcript tells us what is said; this module measures how it is said.
Using ffmpeg only (no heavy ML dependencies), it extracts:

* overall loudness (``mean_volume_db`` / ``max_volume_db`` via
  ``volumedetect``) and a loudness curve (momentary LUFS from ``ebur128``),
* pauses (``silencedetect``) -> speech ratio, pause-heavy segments,
* a per-segment voice profile: speaking pace (words per minute), mean
  loudness, pause share and delivery flags.

Profiles are then compressed into window narrations by the language model
(plain text work), producing ``voice_notes`` - the same shape as the visual
timeline - which the context report folds into a "# How it sounds (voice
analysis)" section. Raw measurements are written to ``voice_analysis.json``
so nothing is lost.

This is a fully offline, dependency-free analysis: only ffmpeg and the
transcript segments are needed.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Callable, Optional

from .context import VOICE_SUMMARY_PROMPT, OllamaClient
from .vision import ffmpeg_bin, probe_duration

PACE_FAST_WPM = 160
PACE_SLOW_WPM = 115
PAUSE_HEAVY_RATIO = 0.4
ENERGY_BURST_DELTA = 6  # LU above the mean is an emphatic moment

SILENCE_NOISE_DB = "-35dB"
SILENCE_MIN_SECS = 0.8


class VoiceError(Exception):
    """Raised when voice/audio analysis cannot be produced."""


def _run(ffmpeg, args: list[str]) -> subprocess.CompletedProcess:
    try:
        result = subprocess.run(
            [ffmpeg, "-hide_banner", "-i", str(args[0])] + args[1:],
            capture_output=True,
            text=True,
            timeout=600,
        )
    except subprocess.TimeoutExpired as exc:
        raise VoiceError("Voice analysis timed out.") from exc
    return result


def loudness_curve(audio: Path, ffmpeg: str) -> list[tuple[float, float]]:
    """Return ``[(time, momentary_lufs), ...]`` via the ebur128 loudness meter."""
    result = _run(
        ffmpeg,
        [
            str(audio),
            "-filter_complex",
            "ebur128",
            "-f",
            "null",
            "-",
        ],
    )
    curve: list[tuple[float, float]] = []
    for line in result.stderr.splitlines():
        m = re.search(r"t:\s+([\d.]+)\s+TARGET:.*?M:\s*(-?[\d.]+)", line)
        if m:
            t = float(m.group(1))
            lufs = float(m.group(2))
            if lufs > -100:  # pre-gate frames report ~-120.7; ignore them
                curve.append((t, lufs))
    return curve


def silence_gaps(audio: Path, ffmpeg: str) -> list[dict]:
    """Return ``[{"start", "end"}, ...]`` pauses longer than the threshold."""
    result = _run(
        ffmpeg,
        [
            str(audio),
            "-af",
            f"silencedetect=noise={SILENCE_NOISE_DB}:d={SILENCE_MIN_SECS}",
            "-f",
            "null",
            "-",
        ],
    )
    gaps: list[dict] = []
    start: Optional[float] = None
    for line in result.stderr.splitlines():
        m_start = re.search(r"silence_start:\s*([\d.]+)", line)
        m_end = re.search(r"silence_end:\s*([\d.]+)", line)
        if m_start:
            start = float(m_start.group(1))
        elif m_end and start is not None:
            gaps.append({"start": start, "end": float(m_end.group(1))})
            start = None
    return gaps


def volume_stats(audio: Path, ffmpeg: str) -> dict:
    """Return ``{"mean_volume_db", "max_volume_db"}`` via volumedetect."""
    result = _run(
        ffmpeg,
        [
            str(audio),
            "-af",
            "volumedetect",
            "-f",
            "null",
            "-",
        ],
    )
    stats: dict[str, Optional[float]] = {"mean_volume_db": None, "max_volume_db": None}
    for line in result.stderr.splitlines():
        m_mean = re.search(r"mean_volume:\s*(-?[\d.]+)\s*dB", line)
        m_max = re.search(r"max_volume:\s*(-?[\d.]+)\s*dB", line)
        if m_mean:
            stats["mean_volume_db"] = float(m_mean.group(1))
        elif m_max:
            stats["max_volume_db"] = float(m_max.group(1))
    if stats["mean_volume_db"] is None:
        raise VoiceError("volumedetect produced no volume info.")
    return stats


def _overlap(a: float, b: float, gaps: list[dict]) -> float:
    total = 0.0
    for gap in gaps:
        total += max(0.0, min(b, gap["end"]) - max(a, gap["start"]))
    return total


def _profile_segment(seg: dict, duration: float, curve: list[tuple[float, float]], gaps: list[dict]) -> dict:
    start = float(seg.get("start", 0.0))
    end = float(seg.get("end", start))
    if end <= start:
        end = start + 1.0
    words = len(str(seg.get("text", "")).split())
    mins = max((end - start) / 60.0, 1e-6)
    wpm = round(words / mins, 1)

    seg_curve = [lufs for t, lufs in curve if start <= t <= end]
    mean_lufs = round(sum(seg_curve) / len(seg_curve), 1) if seg_curve else None

    pause_secs = _overlap(start, end, gaps)
    pause_ratio = round(min(1.0, pause_secs / (end - start)), 2)

    flags: list[str] = []
    if wpm >= PACE_FAST_WPM:
        flags.append("fast-paced")
    elif wpm <= PACE_SLOW_WPM:
        flags.append("slow-paced")
    if pause_ratio >= PAUSE_HEAVY_RATIO:
        flags.append("pause-heavy")
    if mean_lufs is not None and mean_lufs <= -24:
        flags.append("soft-spoken")

    return {
        "start": round(start, 2),
        "end": round(end, 2),
        "words": words,
        "wpm": wpm,
        "mean_lufs": mean_lufs,
        "pause_ratio": pause_ratio,
        "flags": flags,
    }


def analyze_audio(
    audio: Path,
    segments: list[dict],
    duration: Optional[float] = None,
) -> dict:
    """Measure how the video sounds and return stats + per-segment profiles."""
    ffmpeg = ffmpeg_bin()
    if not ffmpeg:
        raise VoiceError("ffmpeg not found on PATH (install ffmpeg).")

    dur = duration or probe_duration(audio)
    if dur <= 0:
        dur = max((float(s.get("end", 0.0)) for s in segments), default=0.0)
    if dur <= 0:
        raise VoiceError("Audio has no measurable duration.")

    try:
        curve = loudness_curve(audio, ffmpeg)
        gaps = silence_gaps(audio, ffmpeg)
        vol = volume_stats(audio, ffmpeg)
    except VoiceError:
        raise
    except Exception as exc:  # noqa: BLE001 - ffmpeg failure, surface as VoiceError
        raise VoiceError(f"Audio analysis failed: {exc}") from exc

    lufs_values = [lufs for _t, lufs in curve]
    mean_lufs = round(sum(lufs_values) / len(lufs_values), 1) if lufs_values else None
    peak_lufs = round(max(lufs_values), 1) if lufs_values else None

    silence_total = sum(g["end"] - g["start"] for g in gaps)
    speech_ratio = round(max(0.0, min(1.0, 1.0 - silence_total / dur)), 2)

    profiles = [
        dict(seg, **(_profile_segment(seg, dur, curve, gaps))) for seg in segments
    ]

    fast_share = (
        sum(1 for p in profiles if p["wpm"] >= PACE_FAST_WPM) / len(profiles)
        if profiles
        else 0.0
    )
    slow_share = (
        sum(1 for p in profiles if p["wpm"] <= PACE_SLOW_WPM) / len(profiles)
        if profiles
        else 0.0
    )

    if fast_share >= 0.4:
        pace = "fast"
    elif slow_share >= 0.4:
        pace = "slow"
    else:
        pace = "measured"

    if mean_lufs is None:
        energy = "unknown"
    elif mean_lufs >= -18:
        energy = "high"
    elif mean_lufs <= -26:
        energy = "low"
    else:
        energy = "moderate"

    raw_bursts = [
        {"t": round(t, 1), "lufs": round(l, 1)}
        for t, l in curve
        if mean_lufs is not None and l >= mean_lufs + ENERGY_BURST_DELTA
    ]
    bursts = raw_bursts[:50]

    return {
        "stats": {
            "duration": round(dur, 2),
            "mean_volume_db": vol["mean_volume_db"],
            "max_volume_db": vol["max_volume_db"],
            "mean_lufs": mean_lufs,
            "peak_lufs": peak_lufs,
            "n_silences": len(gaps),
            "silence_total": round(silence_total, 2),
            "speech_ratio": speech_ratio,
            "pace": pace,
            "energy": energy,
            "n_energy_bursts": len(raw_bursts),
            "energy_bursts": bursts,
        },
        "segments": profiles,
    }


def summarize_voice(
    profiles: list[dict],
    client: OllamaClient,
    window_seconds: float = 30,
    on_log: Optional[Callable[[str], None]] = None,
) -> list[dict]:
    """Compress per-segment voice profiles into per-window delivery narrations.

    Mirrors the visual timeline: consecutive profiles are grouped into
    ``window_seconds`` windows and each window is narrated in 3-6 bullets that
    describe pacing, energy shifts, pauses and tone. Returns
    ``[{"ts", "end", "description"}, ...]``.
    """
    emit = on_log if on_log is not None else lambda msg: None
    if not profiles:
        return []

    windows: list[list[dict]] = []
    current = [profiles[0]]
    for prof in profiles[1:]:
        if prof["start"] < current[0]["start"] + window_seconds:
            current.append(prof)
        else:
            windows.append(current)
            current = [prof]
    windows.append(current)

    notes: list[dict] = []
    for window in windows:
        start = window[0]["start"]
        end = window[-1]["end"]
        lines = []
        for p in window:
            lufs_note = f"loudness {p['mean_lufs']} LU" if p["mean_lufs"] is not None else "loudness n/a"
            pause_note = f"{p['pause_ratio'] * 100:.0f}% paused" if p["pause_ratio"] else "no pauses"
            lines.append(
                f"- {_ts(p['start'])} - {p['wpm']} wpm, {lufs_note}, {pause_note}"
            )
        prompt = VOICE_SUMMARY_PROMPT.format(
            start=_ts(start), end=_ts(end), descriptions="\n".join(lines)
        )
        summary = client.complete([{"role": "user", "content": prompt}]).strip()
        if summary:
            notes.append({"ts": start, "end": end, "description": summary})

    emit(f"  Voice timeline: {len(notes)} window narrations from {len(profiles)} segments")
    return notes


def _ts(seconds: float) -> str:
    return f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"