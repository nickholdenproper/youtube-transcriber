"""Derived-document generation from the context digest.

The reconciliation report is a dense, structured digest of the whole video
(verbatim transcript + chunk analyses + visual and voice evidence + the report
itself). An optional second LLM call rewrites that digest into other useful
documents -- step-by-step how-to guides, articles, timestamped FAQs, action
checklists and quizzes -- without any extra vision/audio work. Each derived
document is one deterministic prompt over the digest, so long-running pipelines
stay cheap and free-model friendly.
"""

from __future__ import annotations

from typing import Optional

from .context import ContextError, OllamaClient, _render_evidence, fmt_ts

__all__ = ["DERIVE_TYPES", "DeriveError", "build_digest", "derive"]

DERIVE_TYPES = ("how-to", "article", "faq", "checklist", "quiz")

_STYLE = """\
You are rewriting a video into a new document. Everything you write must come
from the material below: the transcript is the verbatim ground truth, and the
segment/visual/voice/report sections are supporting evidence about the same
video. Never invent facts, names, steps, UI elements or quotes that are not in
the material, and never contradict the transcript. Use the real timestamps
(MM:SS) exactly as given. A burned-in timestamp stamp at the top-left of frame
descriptions is a tool reference for the analysis -- not video content -- so it
must never be described or quoted.

Output plain Markdown only, with no preamble, commentary or trailing notes.
"""

_PROMPTS = {
    "how-to": """\
{style}

{digest}

## Your task

Write a step-by-step how-to guide to what the video teaches.

- Number every step in strict order, giving each a short bold title.
- Start each step with the moment it happens: **N. Title** `[MM:SS]`.
- When a step depends on something on screen (buttons, forms, fields, links),
  name the UI element exactly as the visual evidence spells it.
- Keep each step to the observable sequence in the video; do not add generic
  advice the video never gives.
- If the video supplies prerequisites or a wrap-up (e.g. "once done, your
  account is fully verified"), reflect those as a short final note.
""",
    "article": """\
{style}

{digest}

## Your task

Write a reader-friendly article (headline, a short lead, a few clearly headed
sections, a one-paragraph conclusion) that faithfully covers what the video
teaches.

- Every factual claim must be traceable to the material -- never embellish.
- Cite the moment each part happens with inline `[MM:SS]` references.
- Match the video's tone where the material makes it clear.
- Keep the article self-contained: a reader who did not watch the video should
  still understand the full procedure and its purpose.
""",
    "faq": """\
{style}

{digest}

## Your task

List the questions this video answers as a short FAQ.

- One heading per question, formatted as `## Question`.
- Under each heading: a 2-4 sentence answer built strictly from the transcript
  and visual evidence, plus the `[MM:SS]` moment the video addresses it.
- Use the question wording the video implies; invent nothing.
""",
    "checklist": """\
{style}

{digest}

## Your task

Produce a flat, ordered action checklist of everything someone should do to
follow the video.

- One actionable item per line, prefixed with `- [ ] `.
- Append the relevant `[MM:SS]` at the end of each line.
- Keep every item actionable and grounded in the video's observed sequence;
  skip anything the video merely mentions without showing or instructing.
""",
    "quiz": """\
{style}

{digest}

## Your task

Create a study quiz of exactly 10 flashcard pairs from the video.

- Each pair is `**Q:** <question>` immediately followed by `**A:** <answer>`.
- After every answer, add the `[MM:SS]` where the video covers it.
- Questions must be answerable from the transcript or visual evidence alone,
  with the answer given verbatim or near-verbatim in the material.
""",
}


class DeriveError(ContextError):
    """Raised when a derived document cannot be generated."""


def build_digest(
    segments: list[dict],
    title: str = "Untitled",
    channel: str = "",
    report: str = "",
    visual_timeline: Optional[list[dict]] = None,
    voice_notes: Optional[list[dict]] = None,
    chunks: Optional[list[dict]] = None,
) -> str:
    """Render the whole-video digest a second LLM will consume."""
    transcript = "\n".join(
        f"[{fmt_ts(s['start'])}] {s.get('text', '').strip()}" for s in segments
    )
    duration = fmt_ts(segments[-1]["end"]) if segments else "00:00"

    if chunks:
        analyses = "\n\n---\n\n".join(
            f"[Segment {i}, ~{fmt_ts(c['start'])}-{fmt_ts(c['end'])}:]\n{c.get('analysis', '')}"
            for i, c in enumerate(chunks, start=1)
        )
    else:
        analyses = "None."

    visual = (
        _render_evidence(visual_timeline)
        if visual_timeline
        else "None - no video frames were analyzed."
    )
    voice = (
        _render_evidence(voice_notes)
        if voice_notes
        else "None - no voice analysis was performed."
    )

    return (
        f"# {title}\n"
        f"- Channel: {channel}\n"
        f"- Duration: {duration}\n\n"
        "## Verbatim transcript (ground truth)\n"
        f"{transcript}\n\n"
        "## Segment analyses\n"
        f"{analyses}\n\n"
        "## Visual evidence\n"
        f"{visual}\n\n"
        "## Voice evidence (delivery)\n"
        f"{voice}\n\n"
        "## Existing context report\n"
        f"{report}"
    )


def derive(
    kind: str,
    client: OllamaClient,
    segments: list[dict],
    title: str = "Untitled",
    channel: str = "",
    report: str = "",
    visual_timeline: Optional[list[dict]] = None,
    voice_notes: Optional[list[dict]] = None,
    chunks: Optional[list[dict]] = None,
) -> str:
    """Generate one derived document of ``kind`` with a single LLM call."""
    if kind not in _PROMPTS:
        raise DeriveError(f"Unknown derived document type: {kind}")
    if not segments:
        raise DeriveError("Nothing to derive: the transcript is empty.")

    digest = build_digest(
        segments,
        title=title,
        channel=channel,
        report=report,
        visual_timeline=visual_timeline,
        voice_notes=voice_notes,
        chunks=chunks,
    )
    prompt = _PROMPTS[kind].format(style=_STYLE, digest=digest)
    out = client.complete([{"role": "user", "content": prompt}]).strip()
    if not out:
        raise DeriveError(f"Derived document '{kind}' came back empty.")
    return out