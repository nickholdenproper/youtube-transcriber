"""Download and extraction layer built on yt-dlp.

Given a YouTube URL this module produces a :class:`FetchResult` containing
video metadata plus either:

* timestamped caption segments (manual subtitles preferred, then automatic
  captions), or
* a downloaded audio file ready for local Whisper transcription.

Video frames for visual context are also downloaded low-res here.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yt_dlp


class FetchError(Exception):
    """Raised when a YouTube video cannot be fetched."""


PREFERRED_SUBTITLE_LANGS = ["en", "en-US", "en-GB"]


@dataclass(frozen=True)
class VideoMeta:
    id: str
    title: str
    channel: str
    url: str
    duration: Optional[float] = None
    upload_date: Optional[str] = None
    view_count: Optional[int] = None
    thumbnail: Optional[str] = None


@dataclass
class FetchResult:
    meta: VideoMeta
    segments: list[dict]
    source: str
    audio_path: Optional[Path] = None
    captions_path: Optional[Path] = None
    warnings: list[str] = field(default_factory=list)


def extract_info(url: str) -> dict:
    """Return raw yt-dlp info dict for a URL."""
    opts = {"quiet": True, "no_warnings": True}
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            return ydl.extract_info(url, download=False)
    except yt_dlp.utils.DownloadError as exc:
        raise FetchError(f"Could not fetch video info: {exc}") from exc


def _to_meta(info: dict) -> VideoMeta:
    return VideoMeta(
        id=info.get("id") or "video",
        title=info.get("title") or "Untitled",
        channel=info.get("channel") or info.get("uploader") or "Unknown",
        url=info.get("webpage_url") or info.get("original_url") or "",
        duration=info.get("duration"),
        upload_date=info.get("upload_date"),
        view_count=info.get("view_count"),
        thumbnail=info.get("thumbnail"),
    )


def pick_caption(info: dict) -> Optional[tuple[str, str]]:
    """Choose the best caption track: manual subtitles before auto captions.

    Returns ``(section, lang)`` or ``None`` when no captions exist.
    """
    preferred = PREFERRED_SUBTITLE_LANGS
    for section in ("subtitles", "automatic_captions"):
        tracks = info.get(section) or {}
        for lang in preferred:
            if lang in tracks and tracks[lang]:
                return section, lang
        if tracks:
            return section, next(iter(tracks))
    return None


def download_captions(
    url: str,
    out_dir: Path,
    video_id: str,
    lang: str,
    automatic: bool,
    attempts: int = 3,
    backoff: tuple = (3, 6, 12),
) -> Path:
    """Download one caption track with yt-dlp and return the VTT file path.

    Retries with backoff on transient errors (rate limits etc).
    """
    opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "subtitleslangs": [lang],
        "subtitlesformat": "vtt",
        "outtmpl": str(out_dir / f"{video_id}.%(ext)s"),
    }
    if automatic:
        opts["writeautomaticsub"] = True
    else:
        opts["writesubtitles"] = True

    last_error: Optional[str] = None
    for attempt in range(attempts):
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                ydl.download([url])
            candidates = sorted(out_dir.glob(f"{video_id}.*.vtt"))
            if candidates:
                return candidates[0]
            last_error = f"no VTT file produced for lang '{lang}'"
        except yt_dlp.utils.DownloadError as exc:
            last_error = str(exc)
        if attempt < attempts - 1:
            time.sleep(backoff[min(attempt, len(backoff) - 1)])

    raise FetchError(
        f"Caption download failed after {attempts} attempts: {last_error}"
    )


def _to_seconds(ts: str) -> float:
    parts = ts.split(":")
    hours, minutes = int(parts[0]), int(parts[1])
    seconds = float(parts[2])
    return hours * 3600 + minutes * 60 + seconds


def parse_vtt(text: str) -> list[dict]:
    """Parse a VTT subtitle file into ``[{start, end, text}, ...]`` dicts."""
    segments: list[dict] = []
    for block in re.split(r"\r?\n\r?\n+", text):
        lines = [ln.strip() for ln in block.splitlines()]
        if not lines or "-->" not in lines[0]:
            continue
        match = re.match(
            r"(\d{2}:\d{2}:\d{2}\.\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}\.\d{3})",
            lines[0],
        )
        if not match:
            continue
        body = []
        for ln in lines[1:]:
            if not ln or ln.startswith("<") and ln.endswith(">"):
                continue
            body.append(re.sub(r"<[^>]+>", "", ln).strip())
        text_body = " ".join(b for b in body if b).strip()
        if text_body:
            segments.append(
                {
                    "start": _to_seconds(match.group(1)),
                    "end": _to_seconds(match.group(2)),
                    "text": text_body,
                }
            )
    return segments


def download_audio(url: str, out_dir: Path, video_id: str) -> Path:
    """Download the best audio track and convert to MP3 with ffmpeg."""
    opts = {
        "format": "bestaudio/best",
        "outtmpl": str(out_dir / f"{video_id}.%(ext)s"),
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],
        "quiet": True,
        "no_warnings": True,
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([url])
    except yt_dlp.utils.DownloadError as exc:
        raise FetchError(f"Audio download failed: {exc}") from exc

    mp3 = out_dir / f"{video_id}.mp3"
    if not mp3.exists():
        raise FetchError("Audio download completed but the MP3 was not produced (is ffmpeg installed?)")
    return mp3


def download_video(url: str, out_dir: Path, video_id: str) -> Path:
    """Download a low-resolution MP4 for frame extraction (visual context)."""
    opts = {
        "format": "bv*[height<=240][ext=mp4]/bv*[height<=360][ext=mp4]/b",
        "outtmpl": str(out_dir / f"{video_id}.mp4"),
        "max_filesize": 250 * 1024 * 1024,
        "quiet": True,
        "no_warnings": True,
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([url])
    except yt_dlp.utils.DownloadError as exc:
        raise FetchError(f"Video download failed: {exc}") from exc

    mp4 = out_dir / f"{video_id}.mp4"
    if not mp4.exists():
        raise FetchError(
            "Video download produced no file. The video may not offer an MP4 format."
        )
    return mp4


def fetch_video(url: str, work_dir: Path, force_whisper: bool = False) -> FetchResult:
    """Fetch a YouTube video: captions first, audio file as fallback."""
    info = extract_info(url)
    meta = _to_meta(info)
    out_dir = Path(work_dir) / meta.id
    out_dir.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []

    if not force_whisper:
        chosen = pick_caption(info)
        if chosen is not None:
            section, lang = chosen
            try:
                captions_path = download_captions(
                    url, out_dir, meta.id, lang, automatic=(section == "automatic_captions")
                )
                segments = parse_vtt(
                    captions_path.read_text(encoding="utf-8", errors="replace")
                )
                if segments:
                    return FetchResult(
                        meta=meta,
                        segments=segments,
                        source="captions",
                        captions_path=captions_path,
                        warnings=warnings,
                    )
                warnings.append("Caption track existed but contained no usable text.")
            except FetchError as exc:
                warnings.append(f"Captions unavailable ({exc}) - falling back to audio.")

    audio_path = download_audio(url, out_dir, meta.id)
    return FetchResult(
        meta=meta,
        segments=[],
        source="audio",
        audio_path=audio_path,
        warnings=warnings,
    )