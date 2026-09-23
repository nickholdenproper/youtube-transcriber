"""Visual context: understand what *happens on screen*, not just in audio.

Ads, demos, slides, body language and on-screen text are invisible to speech
transcription. This module downloads a low-resolution copy of the video,
extracts frames with ffmpeg at regular intervals, and asks a vision model
(e.g. ``gemma4`` locally or ``gemma4:31b-cloud`` on Ollama Cloud) to describe
each frame.

The resulting timeline feeds :func:`yt_transcriber.context.analyze` so the
final "what happened" section can mention visual actions too.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Optional

from .context import FRAME_PROMPT, OllamaClient
from .fetch import FetchError, download_video


class VisionError(Exception):
    """Raised when visual context cannot be produced."""


def ffmpeg_bin() -> Optional[str]:
    return shutil.which("ffmpeg")


def probe_duration(video: Path) -> float:
    """Get video duration in seconds via ffprobe."""
    probe = shutil.which("ffprobe")
    if not probe:
        raise VisionError("ffprobe not found on PATH (install ffmpeg).")
    result = subprocess.run(
        [
            probe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    try:
        return float(result.stdout.strip())
    except ValueError as exc:
        raise VisionError(f"Could not read video duration: {result.stdout.strip()}") from exc


def extract_frames(
    video: Path,
    out_dir: Path,
    video_id: str,
    interval_seconds: float = 30,
    max_frames: int = 10,
) -> list[dict]:
    """Extract evenly spaced frames with ffmpeg.

    Returns ``[{"ts": float, "path": Path}, ...]`` sorted by timestamp.
    """
    ffmpeg = ffmpeg_bin()
    if not ffmpeg:
        raise VisionError("ffmpeg not found on PATH (install ffmpeg).")

    duration = probe_duration(video)
    if duration <= 0:
        raise VisionError("Video has no measurable duration.")

    out_dir.mkdir(parents=True, exist_ok=True)
    step = max(interval_seconds, duration / max_frames)
    frames: list[dict] = []
    ts = 1.0  # skip the first second; it is often an intro card

    while ts < duration and len(frames) < max_frames:
        out_path = out_dir / f"frame_{int(ts):05d}.jpg"
        result = subprocess.run(
            [
                ffmpeg,
                "-y",
                "-ss",
                f"{ts:.1f}",
                "-i",
                str(video),
                "-frames:v",
                "1",
                "-vf",
                "scale=480:-2",
                "-q:v",
                "3",
                str(out_path),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode == 0 and out_path.exists():
            frames.append({"ts": round(ts, 1), "path": out_path})
        ts += step

    return frames


def analyze_frames(
    frames: list[dict],
    client: OllamaClient,
    model: str,
    prompt: str = FRAME_PROMPT,
) -> list[dict]:
    """Ask a vision model to describe each frame.

    Returns ``[{"ts": float, "description": str}, ...]``.
    """
    vision_client = OllamaClient(
        base_url=client.base_url,
        model=model,
        api_key=client.api_key,
    )
    observations = []
    for frame in frames:
        description = vision_client.complete(
            [{"role": "user", "content": prompt}],
            images=[str(frame["path"])],
        )
        observations.append(
            {
                "ts": frame["ts"],
                "path": str(frame["path"]),
                "description": description.strip(),
            }
        )
    return observations


def build_visual_timeline(
    url: str,
    work_dir: Path,
    video_id: str,
    client: OllamaClient,
    vision_model: str,
    interval_seconds: float = 30,
    max_frames: int = 10,
    on_log=None,
) -> list[dict]:
    """Run the whole visual step: download video, extract frames, describe them."""
    emit = on_log if on_log is not None else lambda msg: None
    emit("  Downloading low-res video for visual analysis ...")
    out_dir = work_dir / video_id
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        video = download_video(url, out_dir, video_id)
        emit(f"  Extracting up to {max_frames} frames ...")
        frames = extract_frames(video, out_dir / "frames", video_id, interval_seconds, max_frames)
        if not frames:
            raise VisionError("No frames could be extracted.")
        emit(f"  Describing {len(frames)} frames with {vision_model} ...")
        observations = analyze_frames(frames, client, vision_model)
        timeline = [
            {"ts": obs["ts"], "description": obs["description"]}
            for obs in observations
            if obs["description"]
        ]
        if not timeline:
            raise VisionError("Vision model returned no usable descriptions.")
        return timeline
    except (FetchError, VisionError) as exc:
        raise VisionError(f"Visual context failed ({exc})") from exc