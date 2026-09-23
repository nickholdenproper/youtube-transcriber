"""Local speech-to-text transcription with faster-whisper.

Used as a fallback when a video has no captions: transcribes a downloaded
audio file into the same timestamped segment format the captions path
produces, so the rest of the pipeline stays identical.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .fetch import FetchError

MODEL_SIZES = ("tiny", "base", "small", "medium", "large-v3", "turbo")


def transcribe(
    audio_path: Path,
    model_size: str = "base",
    device: str = "cpu",
    language: Optional[str] = None,
    beam_size: int = 5,
    vad_filter: bool = True,
) -> dict:
    """Transcribe an audio file into ``{segments, language, ...}``."""
    if model_size not in MODEL_SIZES:
        raise FetchError(
            f"Unknown whisper model '{model_size}'. Choose from: {', '.join(MODEL_SIZES)}"
        )

    from faster_whisper import WhisperModel

    compute_type = "float16" if device == "cuda" else "int8"

    try:
        model = WhisperModel(model_size, device=device, compute_type=compute_type)
        segments_iter, info = model.transcribe(
            str(audio_path),
            beam_size=beam_size,
            language=language,
            vad_filter=vad_filter,
        )
        segments = [
            {"start": float(seg.start), "end": float(seg.end), "text": seg.text.strip()}
            for seg in segments_iter
            if seg.text.strip()
        ]
    except Exception as exc:  # noqa: BLE001 - surface a friendly message
        raise FetchError(f"Whisper transcription failed: {exc}") from exc

    return {
        "segments": segments,
        "language": info.language,
        "language_probability": round(float(info.language_probability), 3),
        "duration": info.duration,
    }