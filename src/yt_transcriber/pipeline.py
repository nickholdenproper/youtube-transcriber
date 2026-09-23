"""Shared pipeline: fetch -> transcribe -> [vision] -> context.

This is the single code path used by the CLI, the web GUI and the REST API,
so every interface behaves identically.
"""

from __future__ import annotations

import os
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
from .derive import DERIVE_TYPES, DeriveError, derive as derive_doc
from .fetch import VideoMeta, download_audio, fetch_video
from .output import write_outputs
from .transcribe import transcribe
from .vision import VisionError, build_visual_timeline
from .voice import VoiceError, analyze_audio, summarize_voice


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
    vision_mode: str = "per-frame"
    frame_interval: float = 1
    max_frames: int = 600
    vision_window: float = 30
    dedupe: bool = True
    dedupe_max_gap: float = 10
    batch_size: int = 8
    mosaic_cells: Optional[int] = None
    target_grids: int = 24
    voice: bool = True
    voice_window: float = 30
    vision_doubt: bool = True
    vision_fill_budget: int = 60
    derive: list[str] = field(default_factory=list)
    derive_model: Optional[str] = None


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
    visual_frames: list[dict] = field(default_factory=list)
    voice_notes: list[dict] = field(default_factory=list)
    voice_analysis: Optional[dict] = None
    vision_fill: Optional[dict] = None
    derived: dict = field(default_factory=dict)


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
    visual_frames: list[dict] = []
    visual: Optional[dict] = None

    voice_notes: list[dict] = []
    voice_analysis: Optional[dict] = None
    vision_fill: Optional[dict] = None

    main_client = None
    if not opts.skip_context:
        main_client = resolve_client(opts.provider)
        if opts.model:
            main_client.model = opts.model

    if opts.voice and not opts.skip_context:
        try:
            client = resolve_client(opts.provider)
            audio_path = result.audio_path
            if audio_path is None or not Path(audio_path).exists():
                report(60, "Downloading audio for voice analysis")
                emitter("Voice analysis: downloading audio track ...")
                audio_path = download_audio(url, work_dir / meta.id, meta.id)
            report(62, "Measuring audio for voice analysis")
            emitter("Voice analysis: measuring pacing, loudness and pauses ...")
            voice_analysis = analyze_audio(
                Path(audio_path),
                segments,
                duration=meta.duration,
            )
            report(64, "Summarizing voice delivery")
            emitter("Voice analysis: summarizing delivery with LLM ...")
            voice_notes = summarize_voice(
                voice_analysis["segments"],
                client,
                window_seconds=opts.voice_window,
                on_log=emitter,
            )
            report(66, "Voice analysis complete")
            emitter(
                f"  Voice timeline: {len(voice_notes)} window narrations "
                f"({voice_analysis['stats']['pace']} pace, "
                f"{voice_analysis['stats']['energy']} energy)"
            )
        except VoiceError as exc:
            warnings.append(str(exc))
            emitter(f"Voice analysis skipped: {exc}")
        except Exception as exc:  # noqa: BLE001 - never break the transcript for voice
            warnings.append(f"Voice analysis skipped: {exc}")
            emitter(f"Voice analysis skipped: {exc}")
    else:
        report(64, "Skipping voice analysis")

    if opts.vision and not opts.skip_context:
        try:
            vision_client = resolve_client(opts.provider)
            vision_model = (
                opts.vision_model
                or os.getenv("OLLAMA_VISION_MODEL")
                or (
                    DEFAULT_CLOUD_VISION_MODEL
                    if vision_client.is_cloud
                    else DEFAULT_VISION_MODEL
                )
            )
            report(70, f"Analyzing video frames ({vision_model})")
            emitter(f"Visual context: {vision_model} ({opts.vision_mode}) ...")
            visual = build_visual_timeline(
                url,
                work_dir,
                meta.id,
                vision_client,
                vision_model,
                mode=opts.vision_mode,
                interval_seconds=opts.frame_interval,
                max_frames=opts.max_frames,
                window_seconds=opts.vision_window,
                dedupe=opts.dedupe,
                dedupe_max_gap=opts.dedupe_max_gap,
                batch_size=opts.batch_size,
                mosaic_cells=opts.mosaic_cells,
                target_grids=opts.target_grids,
                on_log=emitter,
                doubt=opts.vision_doubt,
                fill_budget=opts.vision_fill_budget,
                title=meta.title,
                transcript_segments=segments,
                voice_notes=voice_notes,
                voice_stats=(voice_analysis or {}).get("stats"),
                doubt_client=main_client,
                duration=meta.duration,
            )
            visual_timeline = visual["timeline"]
            visual_frames = visual["frames"]
            vision_fill = visual.get("fill")
            report(78, "Frames described")
            emitter(
                f"  Got {len(visual_frames)} frame descriptions "
                f"in {len(visual_timeline)} window summaries"
            )
        except VisionError as exc:
            warnings.append(str(exc))
            emitter(f"Visual context skipped: {exc}")
        except Exception as exc:  # noqa: BLE001 - never break the transcript for vision
            warnings.append(f"Visual context skipped: {exc}")
            emitter(f"Visual context skipped: {exc}")
    else:
        report(70, "Skipping visual analysis")

    if main_client is not None:
        try:
            report(82, f"Analyzing context with {main_client.model}")
            emitter(
                f"Analyzing context with {main_client.model} "
                f"({'cloud' if main_client.is_cloud else 'local'}) ..."
            )
            analysis = analyze(
                segments,
                main_client,
                title=meta.title,
                channel=meta.channel,
                visual_timeline=visual_timeline or None,
                voice_notes=voice_notes or None,
            )
            context_report = analysis["report"]
            context_chunks = analysis["chunks"]
            for warn in analysis.get("warnings", []):
                warnings.append(warn)
                emitter(f"  Warning: {warn}")
            report(96, "Context analysis complete")
        except ContextError as exc:
            warnings.append(str(exc))
            emitter(f"Context step skipped: {exc}")
        except Exception as exc:  # noqa: BLE001 - keep transcript on any context failure
            warnings.append(f"Context step skipped (unexpected): {exc}")
            emitter(f"Context step skipped: {exc}")
    else:
        report(96, "Context skipped")

    derived: dict[str, str] = {}
    if main_client is not None and context_report and opts.derive:
        derive_client = main_client
        if opts.derive_model:
            derive_client = resolve_client(opts.provider)
            derive_client.model = opts.derive_model
        for i, kind in enumerate(opts.derive, start=1):
            if kind not in DERIVE_TYPES:
                warnings.append(f"Unknown derived document type: {kind} (skipped)")
                emitter(f"Derive skipped: unknown type '{kind}'")
                continue
            report(96 + (1.5 * i / len(opts.derive)), f"Deriving {kind}")
            emitter(f"Deriving {kind} from context digest ({derive_client.model}) ...")
            try:
                derived[kind] = derive_doc(
                    kind,
                    derive_client,
                    segments,
                    title=meta.title,
                    channel=meta.channel,
                    report=context_report,
                    visual_timeline=visual_timeline or None,
                    voice_notes=voice_notes or None,
                    chunks=context_chunks,
                )
                emitter(f"  Derived {kind}: {len(derived[kind])} chars")
            except DeriveError as exc:
                warnings.append(f"Derived '{kind}' skipped: {exc}")
                emitter(f"Derive '{kind}' skipped: {exc}")
            except Exception as exc:  # noqa: BLE001 - derive must never break the run
                warnings.append(f"Derived '{kind}' skipped (unexpected): {exc}")
                emitter(f"Derive '{kind}' skipped: {exc}")

    report(98, "Writing output files")
    out_dir = work_dir / meta.id
    written = write_outputs(
        out_dir,
        meta,
        segments,
        result.source,
        context_report,
        context_chunks,
        visual_timeline,
        visual_frames,
        visual.get("mode") if visual else None,
        visual.get("fill") if visual else None,
        voice_notes,
        voice_analysis,
        derived=derived,
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
        visual_frames=visual_frames,
        voice_notes=voice_notes,
        voice_analysis=voice_analysis,
        vision_fill=vision_fill,
        derived=derived,
    )