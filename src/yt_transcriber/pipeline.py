"""Shared pipeline: fetch -> transcribe -> [vision] -> context.

This is the single code path used by the CLI, the web GUI and the REST API,
so every interface behaves identically.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from .context import (
    DEFAULT_CLOUD_VISION_MODEL,
    DEFAULT_VISION_MODEL,
    ContextError,
    analyze,
    resolve_client,
)
from .fetch import VideoMeta, fetch_video
from .output import write_outputs
from .transcribe import transcribe
from .vision import VisionError, build_visual_timeline


@dataclass
class PipelineOptions:
    out: str = "output"
    skip_context: bool = False
    provider: str = "auto"
    model: Optional[str] = None
    whisper_model: str = "base"
    device: str = "cpu"
    force_whisper: bool = False
    language: Optional[str] = None
    no_vad: bool = False
    vision: bool = True
    vision_model: Optional[str] = None
    frame_interval: float = 30
    max_frames: int = 10


@dataclass
class PipelineResult:
    video_id: str
    meta: VideoMeta
    segments: list[dict]
    source: str
    context_report: Optional[str]
    context_chunks: list[dict]
    files: list[Path]
    logs: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    visual_timeline: list[dict] = field(default_factory=list)


def run_pipeline(
    url: str,
    opts: PipelineOptions,
    log: Optional[Callable[[str], None]] = None,
    on_progress: Optional[Callable[[float, str], None]] = None,
) -> PipelineResult:
    """Run the full transcriber pipeline and return everything.

    ``log`` receives human-readable status lines. ``on_progress`` receives
    ``(percent 0-100, stage label)`` at well-defined points so UIs can render
    a live progress bar.
    """
    emitter = log if log is not None else lambda msg: None
    report = on_progress if on_progress is not None else lambda pct, stage: None
    work_dir = Path(opts.out)

    report(4, "Fetching video info")
    emitter(f"Fetching video info: {url}")
    result = fetch_video(url, work_dir, force_whisper=opts.force_whisper)
    meta = result.meta
    emitter(f"  {meta.title}")
    emitter(f"  Channel: {meta.channel}")
    emitter(f"  Source: {result.source}")

    segments = list(result.segments)
    warnings = list(result.warnings)

    if not segments and result.audio_path:
        report(30, "Transcribing audio with whisper")
        emitter(f"Transcribing audio with whisper ({opts.whisper_model}, {opts.device}) ...")
        tr = transcribe(
            result.audio_path,
            model_size=opts.whisper_model,
            device=opts.device,
            language=opts.language,
            vad_filter=not opts.no_vad,
        )
        segments = tr["segments"]
        report(58, "Transcription complete")
        emitter(f"  Got {len(segments)} segments ({tr['language']})")
    else:
        report(30, "Captions downloaded")

    if not segments:
        raise RuntimeError("No transcript could be produced for this video.")

    context_report: Optional[str] = None
    context_chunks: list[dict] = []

    visual_timeline: list[dict] = []

    if opts.vision and not opts.skip_context:
        try:
            vision_client = resolve_client(opts.provider)
            vision_model = opts.vision_model or (
                DEFAULT_CLOUD_VISION_MODEL
                if vision_client.is_cloud
                else DEFAULT_VISION_MODEL
            )
            report(62, f"Analyzing video frames ({vision_model})")
            emitter(f"Visual context: {vision_model} ...")
            visual_timeline = build_visual_timeline(
                url,
                work_dir,
                meta.id,
                vision_client,
                vision_model,
                interval_seconds=opts.frame_interval,
                max_frames=opts.max_frames,
                on_log=emitter,
            )
            report(74, "Frames described")
            emitter(f"  Got {len(visual_timeline)} frame descriptions")
        except VisionError as exc:
            warnings.append(str(exc))
            emitter(f"Visual context skipped: {exc}")
        except Exception as exc:  # noqa: BLE001 - never break the transcript for vision
            warnings.append(f"Visual context skipped: {exc}")
            emitter(f"Visual context skipped: {exc}")
    else:
        report(62, "Skipping visual analysis")

    if not opts.skip_context:
        try:
            client = resolve_client(opts.provider)
            if opts.model:
                client.model = opts.model
            report(76, f"Analyzing context with {client.model}")
            emitter(
                f"Analyzing context with {client.model} "
                f"({'cloud' if client.is_cloud else 'local'}) ..."
            )
            analysis = analyze(
                segments,
                client,
                title=meta.title,
                channel=meta.channel,
                visual_timeline=visual_timeline or None,
            )
            context_report = analysis["report"]
            context_chunks = analysis["chunks"]
            report(92, "Context analysis complete")
        except ContextError as exc:
            warnings.append(str(exc))
            emitter(f"Context step skipped: {exc}")
        except Exception as exc:  # noqa: BLE001 - keep transcript on any context failure
            warnings.append(f"Context step skipped (unexpected): {exc}")
            emitter(f"Context step skipped: {exc}")
    else:
        report(92, "Context skipped")

    report(95, "Writing output files")
    out_dir = work_dir / meta.id
    written = write_outputs(
        out_dir,
        meta,
        segments,
        result.source,
        context_report,
        context_chunks,
        visual_timeline,
    )

    report(100, "Done")
    return PipelineResult(
        video_id=meta.id,
        meta=meta,
        segments=segments,
        source=result.source,
        context_report=context_report,
        context_chunks=context_chunks,
        files=written,
        warnings=warnings,
        visual_timeline=visual_timeline,
    )