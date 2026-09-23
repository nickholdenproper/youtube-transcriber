"""Visual context: understand what *happens on screen*, not just in audio.

Ads, demos, slides, body language and on-screen text are invisible to speech
transcription. This module downloads a 720p copy of the video,
extracts **one frame per second** across the *whole* video with a single ffmpeg
pass, optionally drops frames that barely change (``mpdecimate``-style dedupe,
so cost scales with *actual* screen change, never video length), and asks a
vision model to describe what is on screen.

Three ``mode``s are supported:

* ``per-frame`` (default) - every unique frame is described individually, in
  parallel, at full quality. Guaranteed timestamp binding.
* ``batch`` - up to ``batch_size`` frames per request (faster wall-clock;
  slightly looser timestamp binding for 8+ similar frames).
* ``mosaic`` - frames are stitched into adaptive grid montages (up to
  ``mosaic_cells``, near-square) and each montage is described in one call:
  a cheap overview for users who want speed over detail.

Per-frame / per-batch / per-grid descriptions are then compressed into
~30s window summaries (the whole video fits in context) and the raw
descriptions are kept in ``visual_frames.json`` so nothing is lost.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Optional

from .context import (
    BATCH_FRAME_PROMPT,
    DOUBTS_PROMPT,
    FRAME_PROMPT,
    GRID_FRAME_PROMPT,
    STAMP_NOTE,
    WINDOW_SUMMARY_PROMPT,
    ContextError,
    OllamaClient,
)
from .fetch import FetchError, download_video

MODES = ("per-frame", "batch", "mosaic")
MIN_GRID_CELLS = 4
MAX_GRID_CELLS = 144

WINDOWS_FONTS = (
    r"C:\Windows\Fonts\arial.ttf",
    r"C:\Windows\Fonts\segoeui.ttf",
    r"C:\Windows\Fonts\consola.ttf",
    r"C:\Windows\Fonts\calibril.ttf",
    r"C:\Windows\Fonts\tahoma.ttf",
)


def _pick_font() -> Optional[str]:
    """Return a known Windows font file for drawtext overlays, if any exists."""
    for candidate in WINDOWS_FONTS:
        if os.path.exists(candidate):
            return candidate
    return None


def _stamp_hms(seconds: float) -> str:
    """Render seconds as a fixed-width HH:MM:SS reference stamp."""
    total = max(0, int(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _stamp_filter(font: str, text: str) -> str:
    """Build a ffmpeg ``drawtext`` filter burning ``text`` into the top-left corner.

    ffmpeg's filtergraph parser and drawtext's own option parser each split on
    colons, so both the Windows drive colon in ``fontfile=C\\\\\\:/...`` and any
    colon inside the rendered text must be backslash-escaped (verified against
    ffmpeg 8.1).
    """
    font_esc = font.replace("\\", "/").replace(":", "\\\\\\:")
    text_esc = text.replace(":", r"\:")
    return (
        f"drawtext=fontfile={font_esc}:text='{text_esc}':"
        "fontcolor=white:fontsize=30:box=1:boxcolor=black@0.55:x=16:y=16"
    )


class VisionError(Exception):
    """Raised when visual context cannot be produced."""


class VisionBackend:
    """Interface for a vision-capable model backend.

    Only one concrete backend ships today (:class:`OllamaVisionBackend`), but
    anything implementing ``complete(messages, images=None) -> str`` can be
    plugged in (e.g. a transformers-based describer) without touching the
    pipeline, API or GUI.
    """

    def complete(self, messages: list[dict], images: Optional[list[str]] = None) -> str:
        raise NotImplementedError


class OllamaVisionBackend(VisionBackend):
    """Ollama-backed vision model (local or Ollama Cloud)."""

    def __init__(self, client: OllamaClient, model: str):
        self.client = OllamaClient(
            base_url=client.base_url,
            model=model,
            api_key=client.api_key,
        )

    def complete(self, messages: list[dict], images: Optional[list[str]] = None) -> str:
        return self.client.complete(messages, images=images)


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


def fmt_ts(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _frame_change_scores(
    video: Path,
    ffmpeg: str,
    interval_seconds: float = 1,
) -> tuple[list[float], int]:
    """Decode the video to a tiny grayscale 1-frame/sec stream and return the
    mean absolute difference between consecutive seconds.

    ``scores[i]`` is the change between second ``i-1`` and ``i`` (0-based), in
    0-255 grayscale units. Used to drop frames that barely change, so the
    number of vision calls tracks *actual* screen change, not length.
    """
    result = subprocess.run(
        [
            ffmpeg,
            "-y",
            "-i",
            str(video),
            "-vf",
            f"scale=16:9,format=gray,fps={1.0 / interval_seconds:g}",
            "-f",
            "rawvideo",
            "-",
        ],
        capture_output=True,
        timeout=600,
    )
    if result.returncode != 0:
        last_line = next(
            (line for line in reversed(result.stderr.splitlines()) if line.strip()),
            "unknown ffmpeg error",
        )
        raise VisionError(f"Could not measure frame changes: {last_line}")

    data = result.stdout
    frame_bytes = 16 * 9
    count = len(data) // frame_bytes
    scores = [0.0] * count
    previous = data[:frame_bytes]
    for i in range(1, count):
        current = data[i * frame_bytes : (i + 1) * frame_bytes]
        diff = sum(abs(a - b) for a, b in zip(previous, current)) / frame_bytes
        scores[i] = diff
        previous = current
    return scores, count


def extract_frames(
    video: Path,
    out_dir: Path,
    video_id: str,
    interval_seconds: float = 1,
    max_frames: int = 600,
    dedupe: bool = True,
    dedupe_max_gap: float = 10,
    dedupe_threshold: float = 8,
) -> list[dict]:
    """Extract a frame every ``interval_seconds`` across the whole video.

    One ffmpeg pass (``fps`` filter) keeps this fast even for long videos.
    When ``dedupe`` is on, frames that barely change are dropped so the caller
    only pays to describe what is actually new - but ``dedupe_max_gap``
    guarantees a frame is never skipped for longer than that many seconds
    (slow pans, progress bars and subtle animations are never lost).
    Finally, if more frames remain than ``max_frames`` they are evenly
    down-sampled. Returns ``[{"ts": float, "path": Path}, ...]`` sorted by
    timestamp.
    """
    ffmpeg = ffmpeg_bin()
    if not ffmpeg:
        raise VisionError("ffmpeg not found on PATH (install ffmpeg).")
    if interval_seconds <= 0:
        raise VisionError("Frame interval must be > 0.")

    duration = probe_duration(video)
    if duration <= 0:
        raise VisionError("Video has no measurable duration.")

    out_dir.mkdir(parents=True, exist_ok=True)
    fps_expr = f"{1.0 / interval_seconds:g}"
    pattern = out_dir / "f_%06d.jpg"
    vf = f"fps={fps_expr},scale=960:-2"
    font = _pick_font()
    if font:
        vf += "," + _stamp_filter(font, r"%{pts:hms}")
    result = subprocess.run(
        [
            ffmpeg,
            "-y",
            "-i",
            str(video),
            "-vf",
            vf,
            "-q:v",
            "3",
            "-threads",
            "1",
            str(pattern),
        ],
        capture_output=True,
        text=True,
        timeout=600,
    )
    if result.returncode != 0:
        last_line = next(
            (line for line in reversed(result.stderr.splitlines()) if line.strip()),
            "unknown ffmpeg error",
        )
        raise VisionError(f"ffmpeg frame extraction failed: {last_line}")

    files = sorted(out_dir.glob("f_*.jpg"))
    if not files:
        raise VisionError("No frames could be extracted.")

    kept_idx = list(range(len(files)))

    if dedupe:
        scores, scored = _frame_change_scores(video, ffmpeg, interval_seconds)
        kept_idx = [0] if files else []
        last_kept = 0
        for i in range(1, min(len(files), scored)):
            if scores[i] >= dedupe_threshold or (i - last_kept) * interval_seconds >= dedupe_max_gap:
                kept_idx.append(i)
                last_kept = i
        # Never drop frames we couldn't measure (rawvideo count may differ
        # from the jpg count due to rounding); keep that unscored tail.
        kept_idx.extend(range(scored, len(files)))
        keep_set = set(kept_idx)
        for idx, path in enumerate(files):
            if idx not in keep_set:
                path.unlink(missing_ok=True)
        files = [files[i] for i in kept_idx]

    if len(files) > max_frames:
        keep = sorted(
            {int(round(i * (len(files) - 1) / (max_frames - 1))) for i in range(max_frames)}
        )
        keep_set = set(keep)
        for idx, path in enumerate(files):
            if idx not in keep_set:
                path.unlink(missing_ok=True)
        kept_idx = [kept_idx[i] for i in keep]

    return [
        {"ts": round(idx * float(interval_seconds), 1), "path": files[i]}
        for i, idx in enumerate(kept_idx)
    ]


def analyze_frames(
    frames: list[dict],
    backend: VisionBackend,
    prompt: str = FRAME_PROMPT,
    workers: int = 4,
) -> list[dict]:
    """Describe every frame individually with a vision backend (parallel).

    Returns ``[{"ts": float, "description": str}, ...]`` in input order.
    """
    def _one(frame: dict) -> dict:
        content = prompt.format(stamp=STAMP_NOTE) if "{stamp" in prompt else prompt
        description = backend.complete(
            [{"role": "user", "content": content}],
            images=[str(frame["path"])],
        )
        return {"ts": frame["ts"], "description": description.strip()}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        observations = list(pool.map(_one, frames))
    return observations


def describe_frames_batch(
    frames: list[dict],
    backend: VisionBackend,
    batch_size: int = 8,
    workers: int = 4,
) -> list[dict]:
    """Describe frames several-at-a-time (faster wall-clock, fewer requests).

    Each batch sends up to ``batch_size`` frames in one request - the model
    sees each at full quality - with an explicit image-number -> timestamp map
    so descriptions bind to the right second. Returns one record per batch:
    ``{"ts", "end", "description", "frames"}``.
    """
    if batch_size < 1:
        batch_size = 1
    batches = [frames[i : i + batch_size] for i in range(0, len(frames), batch_size)]

    def _one(batch: list[dict]) -> dict:
        lines = "\n".join(
            f"- Image {i + 1} = {fmt_ts(f['ts'])}" for i, f in enumerate(batch)
        )
        prompt = BATCH_FRAME_PROMPT.format(
            stamp=STAMP_NOTE,
            count=len(batch),
            start=fmt_ts(batch[0]["ts"]),
            end=fmt_ts(batch[-1]["ts"]),
            lines=lines,
        )
        description = backend.complete(
            [{"role": "user", "content": prompt}],
            images=[str(f["path"]) for f in batch],
        )
        return {
            "ts": batch[0]["ts"],
            "end": batch[-1]["ts"],
            "description": description.strip(),
            "frames": len(batch),
        }

    with ThreadPoolExecutor(max_workers=workers) as pool:
        observations = list(pool.map(_one, batches))
    return observations


def choose_grid(
    count: int,
    target_grids: int = 24,
    max_cells: int = MAX_GRID_CELLS,
) -> tuple[int, int, int]:
    """Pick a near-square grid shape for ``count`` frames.

    Cell count is derived from a target number of vision calls
    (``target_grids``), clamped to ``[MIN_GRID_CELLS, max_cells]`` so a 10-min
    video uses ~24 calls and a 2-hour video stays within the model's
    practical per-image cell limit. Returns ``(cols, rows, cells)`` where
    ``cols * rows >= cells`` (the grid is padded to a full rectangle).
    """
    count = max(1, count)
    if max_cells < 4:
        max_cells = 4
    cells = max(4, min(max_cells, -(-count // max(1, target_grids))))
    cols = int(cells**0.5)
    if cols * cols < cells:
        cols += 1
    rows = -(-cells // cols)
    return cols, rows, cols * rows


def _concat_list(paths: list[Path], list_file: Path) -> None:
    def _line(p: Path) -> str:
        raw = str(p.resolve()).replace("\\", "\\\\").replace("'", "\\'")
        return f"file '{raw}'"

    list_file.write_text("\n".join(_line(p) for p in paths) + "\n", encoding="utf-8")


def compose_grids(
    frames: list[dict],
    out_dir: Path,
    cols: int,
    rows: int,
    cells: int,
    on_log: Optional[Callable[[str], None]] = None,
) -> list[dict]:
    """Stitch consecutive frames into grid montages (one per window).

    Uses a single ffmpeg ``concat`` + ``tile`` pass per grid - no image
    libraries needed. Cells keep the full size of the extracted frames (no
    per-cell downscale), so the montage is large and on-screen text stays
    readable even in big grids. The last partial window is padded by repeating
    its final frame. Returns records ``{"ts", "end", "frames", "path"}`` whose
    ``path`` is a ``visual_NNNN.png`` montage saved next to the frames.
    """
    emit = on_log if on_log is not None else lambda msg: None
    out_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg = ffmpeg_bin()
    if not ffmpeg:
        raise VisionError("ffmpeg not found on PATH (install ffmpeg).")
    if cells < 1:
        cells = 4

    grids: list[dict] = []
    total = cols * rows
    for idx in range(0, len(frames), cells):
        window = frames[idx : idx + cells]
        start, end = window[0]["ts"], window[-1]["ts"]
        padded = [f["path"] for f in window]
        while len(padded) < total:
            padded.append(padded[-1])

        list_file = out_dir / f"_grid_{idx // cells + 1:04d}.txt"
        out_png = out_dir / f"visual_{idx // cells + 1:04d}.png"
        _concat_list(padded, list_file)
        result = subprocess.run(
            [
                ffmpeg,
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(list_file),
                "-vf",
                f"tile={cols}x{rows}",
                "-frames:v",
                "1",
                str(out_png),
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )
        list_file.unlink(missing_ok=True)
        if result.returncode != 0 or not out_png.exists():
            last_line = next(
                (line for line in reversed(result.stderr.splitlines()) if line.strip()),
                "unknown ffmpeg error",
            )
            raise VisionError(f"Grid composition failed: {last_line}")
        grids.append(
            {
                "ts": start,
                "end": end,
                "frames": len(window),
                "cols": cols,
                "rows": rows,
                "path": str(out_png),
            }
        )
    emit(f"  Composed {len(grids)} grid montages ({cols}x{rows}) into {out_dir.name}")
    return grids


def describe_grids(
    grids: list[dict],
    backend: VisionBackend,
    workers: int = 4,
) -> list[dict]:
    """Describe each grid montage with a layout-aware prompt (parallel)."""

    def _one(grid: dict) -> dict:
        prompt = GRID_FRAME_PROMPT.format(
            stamp=STAMP_NOTE,
            cols=grid["cols"],
            rows=grid["rows"],
            start=fmt_ts(grid["ts"]),
            end=fmt_ts(grid["end"]),
        )
        description = backend.complete(
            [{"role": "user", "content": prompt}],
            images=[grid["path"]],
        )
        return {
            "ts": grid["ts"],
            "end": grid["end"],
            "description": description.strip(),
            "frames": grid["frames"],
            "cols": grid["cols"],
            "rows": grid["rows"],
            "path": grid["path"],
        }

    with ThreadPoolExecutor(max_workers=workers) as pool:
        observations = list(pool.map(_one, grids))
    return observations


REFUSAL_PATTERNS = (
    r"\b(can'?t|cannot|cannot fulfill|cannot provide|unable|not able|i won'?t|"
    r"i refuse|refus(e|es|ed)|i'?m not able|as an ai)\b",
)
BLANK_PATTERNS = (
    r"\b(blank|black frame|mostly black|all black|off-?air|nothing (is )?(visible|shown|there|on screen)|"
    r"no (visible )?(content|information|detail)|empty frame)\b",
)
LOWINFO_PATTERNS = (
    r"\bsame as (the )?(previous|above|earlier|prior)\b",
    r"\bno (visible )?(change|difference|variation)\b",
)


def score_description(description: str) -> str:
    """Classify an observation as ``"refused"``, ``"blank"``, ``"lowinfo"`` or ``"ok"``."""
    text = (description or "").strip().lower()
    if not text:
        return "blank"
    for pattern in REFUSAL_PATTERNS:
        if re.search(pattern, text):
            return "refused"
    for pattern in BLANK_PATTERNS:
        if re.search(pattern, text):
            return "blank"
    for pattern in LOWINFO_PATTERNS:
        if re.search(pattern, text):
            return "lowinfo"
    return "ok"


def score_observations(observations: list[dict]) -> list[dict]:
    """Attach ``status`` + ``source`` provenance to each observation (in place)."""
    for obs in observations:
        obs.setdefault("source", "pass1")
        obs["status"] = score_description(obs.get("description", ""))
        if "end" not in obs:
            obs["end"] = obs["ts"]
    return observations


def request_doubt_frames(
    client: OllamaClient,
    title: str,
    duration: float,
    transcript_segments: list[dict],
    voice_notes: list[dict],
    voice_stats: Optional[dict],
    observations: list[dict],
    max_requests: int = 40,
    on_log: Optional[Callable[[str], None]] = None,
) -> list[dict]:
    """Ask the (text) model which exact timestamps to re-capture.

    Returns ``[{"t": float, "reason": str, "priority": str}, ...]`` clamped to
    ``[0, duration]``, de-duplicated within 2 seconds and capped. Raises
    :class:`VisionError` when the reply cannot be parsed, so callers can drop
    back to a deterministic fill.
    """
    emit = on_log if on_log is not None else lambda msg: None
    emit("  Doubt pass: asking the model which frames to re-capture ...")

    transcript = "\n".join(
        f"- {fmt_ts(s['start'])} ({fmt_ts(s.get('end', s['start']))}) {s['text'][:160]}"
        for s in transcript_segments
    ) or "- (none)"
    voice = "\n".join(
        f"- {fmt_ts(n['ts']) if isinstance(n.get('ts'), (int, float)) else n.get('ts', '?')} - {n['description'][:160]}"
        for n in (voice_notes or [])
    )
    if not voice and voice_stats:
        voice = (
            f"- overall pace: {voice_stats.get('pace', 'unknown')}, "
            f"energy: {voice_stats.get('energy', 'unknown')}, "
            f"speech ratio: {voice_stats.get('speech_ratio', '?')}"
        )
    if not voice:
        voice = "- (no voice analysis)"
    visual = "\n".join(
        f"- {fmt_ts(o['ts'])} [{o.get('status', 'ok').upper()}]{o.get('description', '')[:220]}"
        for o in observations
    ) or "- (no observations)"

    prompt = DOUBTS_PROMPT.format(
        title=title,
        duration=duration,
        max=max_requests,
        transcript=transcript,
        voice=voice,
        visual=visual,
    )
    reply = client.complete([{"role": "user", "content": prompt}]).strip()

    requests: list[dict] = []
    for line in reply.splitlines():
        m = re.match(r"t\s*=\s*([\d.]+)\s*\|\s*(\w+)\s*\|\s*(.+)", line.strip())
        if not m:
            continue
        t = float(m.group(1))
        if 0 <= t <= duration:
            requests.append(
                {
                    "t": round(t, 1),
                    "priority": (m.group(2) or "mid").lower()[:3],
                    "reason": m.group(3).strip(),
                }
            )
    if reply.strip().upper() == "NONE":
        requests = []
    elif not requests:
        raise VisionError("The doubt pass reply was not in the expected format.")

    # de-duplicate within 2 seconds, keep the highest-priority first in time order
    requests.sort(key=lambda r: r["t"])
    keep: list[dict] = []
    for req in requests:
        if not keep or req["t"] - keep[-1]["t"] >= 2:
            keep.append(req)
    requests = keep[:max_requests]

    if not requests:
        emit("  Doubt pass: model reported no doubts")
        return []
    emit(f"  Doubt pass: {len(requests)} timestamps requested")
    return requests


def extract_at_timestamps(
    video: Path,
    out_dir: Path,
    video_id: str,
    timestamps: list[dict],
    duration: float,
    budget: int = 60,
) -> list[dict]:
    """Capture exact ``-ss`` frames at the requested timestamps (accurate decode).

    Each request expands to ``t-1, t, t+1`` (clamped to the video) so fast
    cursor moves between frames are not missed. Returns
    ``[{"ts", "path", "reason"}, ...]`` honoring ``budget``.
    """
    if not timestamps:
        return []
    ffmpeg = ffmpeg_bin()
    if not ffmpeg:
        raise VisionError("ffmpeg not found on PATH (install ffmpeg).")

    candidates: list[dict] = []
    for req in timestamps:
        for dt in (-1.0, 0.0, 1.0):
            t = round(min(duration, max(0.0, req["t"] + dt)), 1)
            if not candidates or abs(t - candidates[-1]["t"]) >= 0.5:
                candidates.append({"t": t, "reason": req["reason"]})
    candidates = candidates[:budget]

    out_dir.mkdir(parents=True, exist_ok=True)
    frames: list[dict] = []
    font = _pick_font()
    for i, cand in enumerate(candidates):
        out_jpg = out_dir / f"doubt_{i + 1:04d}.jpg"
        vf = "scale=960:-2"
        if font:
            vf += "," + _stamp_filter(font, _stamp_hms(cand["t"]))
        result = subprocess.run(
            [
                ffmpeg,
                "-y",
                "-i",
                str(video),
                "-ss",
                f"{cand['t']:g}",
                "-frames:v",
                "1",
                "-vf",
                vf,
                "-q:v",
                "3",
                str(out_jpg),
            ],
            capture_output=True,
            text=True,
            timeout=600,
        )
        if result.returncode != 0 or not out_jpg.exists():
            continue
        frames.append({"ts": cand["t"], "path": out_jpg, "reason": cand["reason"]})
    return frames


def describe_doubt_frames(
    frames: list[dict],
    backend: VisionBackend,
    batch_size: int = 4,
    workers: int = 4,
) -> list[dict]:
    """Describe doubt frames in small batches; re-shoot refused/blank batches per-frame.

    Every returned observation carries ``source="doubt"`` and a fresh ``status``.
    """
    if not frames:
        return []
    batch_size = max(1, min(batch_size, len(frames)))
    batch_obs = describe_frames_batch(frames, backend, batch_size, workers)
    batches = [frames[i : i + batch_size] for i in range(0, len(frames), batch_size)]

    final: list[dict] = []
    for batch, obs in zip(batches, batch_obs):
        status = score_description(obs.get("description", ""))
        if status in ("refused", "blank"):
            singles = analyze_frames(batch, backend)
            final.extend(singles)
        else:
            final.append(obs)
    for obs in final:
        obs["source"] = "doubt"
        obs["status"] = score_description(obs.get("description", ""))
    return final


def merge_observations(
    pass1: list[dict],
    doubt: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Merge pass-1 and doubt observations into (provenance list, usable list).

    The provenance list keeps everything (refusals included, flagged) for
    ``visual_frames.json``; the usable list drops refused entries and redundant
    low-info runs, and is what the window summarizer consumes.
    """
    pass1 = score_observations(list(pass1))
    doubt = score_observations(list(doubt))

    provenance = sorted(pass1 + doubt, key=lambda o: o["ts"])

    usable = [o for o in provenance if o["status"] != "refused"]
    kept: list[dict] = []
    last_lowinfo_ts: Optional[float] = None
    last_doubt_ts: Optional[float] = None
    for obs in usable:
        if (
            obs["status"] == "lowinfo"
            and last_lowinfo_ts is not None
            and obs["ts"] - last_lowinfo_ts <= 12
        ):
            continue  # drop redundant low-information run-ins
        if (
            obs.get("source") == "doubt"
            and last_doubt_ts is not None
            and obs["ts"] - last_doubt_ts <= 2.0
        ):
            continue  # ±1s expansion already covers a doubt cluster's frames
        kept.append(obs)
        if obs["status"] == "lowinfo":
            last_lowinfo_ts = obs["ts"]
        if obs.get("source") == "doubt":
            last_doubt_ts = obs["ts"]
    return provenance, sorted(kept, key=lambda o: o["ts"])


def deterministic_fill(
    observations: list[dict],
    video: Path,
    out_dir: Path,
    video_id: str,
    backend: VisionBackend,
    duration: float,
    budget: int = 60,
    window_seconds: float = 30,
    on_log: Optional[Callable[[str], None]] = None,
) -> list[dict]:
    """Fallback when the LLM doubt call fails: fill refusals + coverage holes.

    Re-captures exact frames at refused/blank timestamps and the midpoints of
    any usable gap wider than two summary windows, then describes them
    per-frame. Used so obvious gaps still get closed without the LLM.
    """
    for obs in observations:
        obs.setdefault("source", "pass1")
        obs.setdefault("status", score_description(obs.get("description", "")))

    targets: list[dict] = []
    for obs in observations:
        if obs["status"] in ("refused", "blank"):
            targets.append({"t": obs["ts"], "reason": f"retry {obs['status']}"})

    usable = sorted(
        (o for o in observations if o["status"] not in ("refused", "blank")),
        key=lambda o: o["ts"],
    )
    for a, b in zip(usable, usable[1:]):
        if b["ts"] - a["ts"] > window_seconds * 2:
            targets.append({"t": round((a["ts"] + b["ts"]) / 2, 1), "reason": "coverage gap"})

    targets.sort(key=lambda t: t["t"])
    deduped: list[dict] = []
    for target in targets:
        if not deduped or target["t"] - deduped[-1]["t"] >= 2:
            deduped.append(target)
    targets = deduped[:budget]

    if not targets:
        return []
    frames = extract_at_timestamps(video, out_dir, video_id, targets, duration, budget)
    for obs in frames:
        obs["source"] = "doubt"
    doubt = analyze_frames(frames, backend)
    for obs in doubt:
        obs["source"] = "doubt"
        obs["status"] = score_description(obs.get("description", ""))
    emit = on_log if on_log is not None else lambda msg: None
    emit(f"  Doubt pass: deterministic fill captured {len(doubt)} frames")
    return doubt


def summarize_timeline(
    observations: list[dict],
    client: OllamaClient,
    window_seconds: float = 30,
    on_log: Optional[Callable[[str], None]] = None,
) -> list[dict]:
    """Compress per-frame observations into per-window summaries.

    Groups consecutive observations into windows of ``window_seconds`` and asks
    the model for a dense 3-6 bullet summary of each window, preserving exact
    on-screen text, names and actions. The return value is what the final
    report consumes, so the whole video fits in context. Runs as plain *text*
    work, so the passed client's default (text) model is used.
    """
    emit = on_log if on_log is not None else lambda msg: None
    if not observations:
        return []

    windows: list[list[dict]] = []
    current = [observations[0]]
    for obs in observations[1:]:
        if obs["ts"] < current[0]["ts"] + window_seconds:
            current.append(obs)
        else:
            windows.append(current)
            current = [obs]
    windows.append(current)

    summaries: list[dict] = []
    for window in windows:
        start = window[0]["ts"]
        end = window[-1]["ts"]
        blob = "\n".join(f"- {fmt_ts(obs['ts'])} - {obs['description']}" for obs in window)
        prompt = WINDOW_SUMMARY_PROMPT.format(
            start=fmt_ts(start),
            end=fmt_ts(end),
            descriptions=blob,
        )
        summary = client.complete([{"role": "user", "content": prompt}]).strip()
        if summary:
            summaries.append({"ts": start, "end": end, "description": summary})

    emit(f"  Visual timeline: {len(summaries)} window summaries from {len(observations)} frames")
    return summaries


def build_visual_timeline(
    url: str,
    work_dir: Path,
    video_id: str,
    client: OllamaClient,
    vision_model: str,
    mode: str = "per-frame",
    interval_seconds: float = 1,
    max_frames: int = 600,
    window_seconds: float = 30,
    dedupe: bool = True,
    dedupe_max_gap: float = 10,
    dedupe_threshold: float = 8,
    batch_size: int = 8,
    mosaic_cells: Optional[int] = None,
    target_grids: int = 24,
    on_log=None,
    doubt: bool = True,
    fill_budget: int = 60,
    title: str = "",
    transcript_segments: Optional[list[dict]] = None,
    voice_notes: Optional[list[dict]] = None,
    voice_stats: Optional[dict] = None,
    doubt_client: Optional[OllamaClient] = None,
    duration: Optional[float] = None,
) -> dict:
    """Run the whole visual step and return ``{"timeline", "frames", "mode"}``.

    ``timeline`` is the compressed per-window view consumed by the context
    report; ``frames`` keeps the raw observations (per-frame, per-batch or
    per-grid, depending on ``mode``) written to ``visual_frames.json``.

    When ``doubt`` is enabled (and a transcript is available) a second,
    LLM-directed pass re-captures exact frames at timestamps the model flags as
    doubtful, merges them back into the timeline, and reports the fill stats
    under the ``"fill"`` key.
    """
    if mode not in MODES:
        raise VisionError(f"Unknown vision mode '{mode}' (expected {', '.join(MODES)}).")
    emit = on_log if on_log is not None else lambda msg: None
    emit("  Downloading 720p video for visual analysis ...")
    out_dir = work_dir / video_id
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        video = download_video(url, out_dir, video_id)
        emit(f"  Extracting a frame every {interval_seconds:g}s ...")
        frames = extract_frames(
            video,
            out_dir / "frames",
            video_id,
            interval_seconds,
            max_frames,
            dedupe=dedupe,
            dedupe_max_gap=dedupe_max_gap,
            dedupe_threshold=dedupe_threshold,
        )
        if not frames:
            raise VisionError("No frames could be extracted.")

        backend = OllamaVisionBackend(client, vision_model)

        if mode == "per-frame":
            emit(f"  Describing {len(frames)} frames with {vision_model} ...")
            observations = analyze_frames(frames, backend)
        elif mode == "batch":
            emit(
                f"  Describing {len(frames)} frames with {vision_model}, "
                f"{batch_size} per request ..."
            )
            observations = describe_frames_batch(frames, backend, batch_size)
        else:  # mosaic
            cols, rows, cells = choose_grid(len(frames), target_grids, mosaic_cells or MAX_GRID_CELLS)
            emit(f"  Composing {len(frames)} frames into {cols}x{rows} grids ...")
            grids = compose_grids(frames, out_dir, cols, rows, cells, on_log=emit)
            observations = describe_grids(grids, backend)

        observations = [obs for obs in observations if obs["description"]]
        if not observations:
            raise VisionError("Vision model returned no usable descriptions.")
        if len(observations) >= 100:
            emit(f"  {len(observations)} observations is a lot; this may take a while ...")

        pass1 = score_observations(observations)

        fill = {"requested": 0, "reason": "disabled", "captured": 0, "refused_after_retry": 0}
        doubt_obs: list[dict] = []
        do_doubt = doubt and transcript_segments and doubt_client is not None
        if do_doubt:
            dur = duration or probe_duration(video)
            try:
                requested = request_doubt_frames(
                    doubt_client,
                    title,
                    dur,
                    transcript_segments,
                    voice_notes,
                    voice_stats,
                    pass1,
                    on_log=emit,
                )
                if requested:
                    fill["requested"] = len(requested)
                    d_frames = extract_at_timestamps(
                        video, out_dir / "doubt", video_id, requested, dur, fill_budget
                    )
                    fill["captured"] = len(d_frames)
                    doubt_obs = describe_doubt_frames(d_frames, backend)
                    refused = [o for o in doubt_obs if o["status"] == "refused"]
                    fill["refused_after_retry"] = len(refused)
                    fill["reason"] = "doubt pass"
                    emit(
                        f"  Doubt pass: {len(doubt_obs)} observations merged "
                        f"(fill budget {fill_budget})"
                    )
                else:
                    fill["reason"] = "no doubts reported"
            except (VisionError, ContextError) as exc:
                emit(f"  Doubt pass unavailable ({exc}) - using deterministic fill")
                try:
                    doubt_obs = deterministic_fill(
                        pass1,
                        video,
                        out_dir / "doubt",
                        video_id,
                        backend,
                        dur,
                        budget=fill_budget,
                        window_seconds=window_seconds,
                        on_log=emit,
                    )
                    if doubt_obs:
                        fill["requested"] = fill["captured"] = len(doubt_obs)
                        fill["reason"] = "deterministic fill"
                except VisionError as exc:  # noqa: BLE001 - keep pass-1 results
                    emit(f"  Deterministic fill skipped ({exc})")
            if doubt_obs:
                provenance, usable = merge_observations(pass1, doubt_obs)
            else:
                provenance, usable = merge_observations(pass1, [])
        else:
            provenance, usable = merge_observations(pass1, [])

        if mode == "mosaic":
            timeline = usable
        else:
            timeline = summarize_timeline(usable, client, window_seconds, on_log=emit)
        if not timeline:
            raise VisionError("Visual summarization returned nothing.")
        return {
            "timeline": timeline,
            "frames": provenance,
            "mode": mode,
            "fill": fill,
        }
    except (FetchError, VisionError) as exc:
        raise VisionError(f"Visual context failed ({exc})") from exc