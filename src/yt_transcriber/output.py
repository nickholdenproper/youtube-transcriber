"""Write transcript and context reports to disk as Markdown + JSON."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from .context import fmt_ts


def build_transcript_md(meta, segments: list[dict], source: str) -> str:
    """Render a timestamped markdown transcript for a video."""
    lines = [
        f"# {meta.title}",
        "",
        f"- Channel: {meta.channel}",
        f"- URL: {meta.url}",
        f"- Duration: {round(meta.duration or 0)}s",
        f"- Source: {source}",
        "",
        "## Transcript",
        "",
    ]
    for seg in segments:
        lines.append(f"[{fmt_ts(seg['start'])}] {seg['text']}")
    return "\n".join(lines)


def write_outputs(
    out_dir: Path,
    meta,
    segments: list[dict],
    source: str,
    context_report: Optional[str] = None,
    context_chunks: Optional[list[dict]] = None,
    visual_timeline: Optional[list[dict]] = None,
) -> list[Path]:
    """Write all output files and return the list of created paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    transcript_md = out_dir / "transcript.md"
    transcript_md.write_text(build_transcript_md(meta, segments, source), encoding="utf-8")
    written.append(transcript_md)

    transcript_json = out_dir / "transcript.json"
    payload = {"meta": asdict(meta), "source": source, "segments": segments}
    transcript_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    written.append(transcript_json)

    if context_report:
        context_md = out_dir / "context.md"
        context_md.write_text(context_report, encoding="utf-8")
        written.append(context_md)

        context_json = out_dir / "context.json"
        ctx_payload = {
            "meta": asdict(meta),
            "report": context_report,
            "chunks": context_chunks or [],
            "visual_timeline": visual_timeline or [],
        }
        context_json.write_text(
            json.dumps(ctx_payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        written.append(context_json)

    return written