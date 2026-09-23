"""AI context analysis using Ollama (cloud API or local server).

The transcript is analyzed with a chunked map-reduce approach so that even
very long videos stay inside the model's context window:

1. The transcript is split into chunks.
2. Each chunk is summarized independently (map). Optionally each chunk also
   receives frame descriptions from :mod:`yt_transcriber.vision` so the model
   sees what happened *on screen*, not just in the audio.
3. All chunk summaries are combined into one context report (reduce).
"""

from __future__ import annotations

import base64
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests

from .fetch import FetchError

CHUNK_MAX_CHARS = 4000

DEFAULT_LOCAL_MODEL = "llama3.2"
DEFAULT_CLOUD_MODEL = "gpt-oss:20b-cloud"
DEFAULT_VISION_MODEL = "gemma4"
DEFAULT_CLOUD_VISION_MODEL = "gemma4:31b-cloud"
LOCAL_BASE_URL = "http://localhost:11434"
CLOUD_BASE_URL = "https://ollama.com"

CHUNK_PROMPT = """You are a meticulous forensic analyst building an EXHAUSTIVE record of a \
segment of a YouTube video (from {start} to {end} seconds). The transcript may be imperfect \
speech-to-text - infer intended meaning, but never invent facts.

The transcript lines below are each tagged with their REAL, absolute timestamp, like \
"[00:12] ". Those timestamps are ground truth from the video - use them exactly as given in \
your bullets. NEVER invent, re-normalize, compress or round the clock: a fast demo still \
happened at its real timestamps, so do not squeeze events into fewer seconds than the \
timestamps show.

Write with enough detail that a reader could follow along without watching the video. \
Reply with exactly these Markdown sections:

## CHAPTER
A short descriptive title for this portion, like a YouTube chapter name.

## WHAT HAPPENED
An exhaustive bullet-by-bullet account of EVERY event in this segment, in order: exactly what \
the speaker says, every action they take, every step they show, every tool / site / number / \
name they mention, every decision. Use one idea per bullet, each starting with a real segment \
timestamp (e.g. "00:12 - "). Write as many bullets as the content needs (10-30+) - never \
summarise away concrete details.

## QUOTES
Up to 5 short notable quotes taken verbatim from the transcript when possible.

Transcript segment:
{chunk}"""

VISUAL_PREFIX = """
VISUAL CONTEXT for this time window (observations from video frames by a vision
model - describe what is visibly happening and include these in WHAT HAPPENED,
but treat uncertain details carefully):
{visual}"""

VOICE_PREFIX = """
VOICE CONTEXT for this time window (delivery measurements from the audio: pace,
loudness, pauses). Where it matters, reflect HOW the speaker delivers the lines
(pacing, energy, hesitations) in WHAT HAPPENED:
{voice}"""

STAMP_NOTE = ("Note: the capture tool burned a white-on-black digital timestamp into the top-left "
    "corner of each frame. That stamp is NOT part of the video - use it only to identify which "
    "second the frame shows, and never describe it as video content or on-screen UI.\n\n")

FRAME_PROMPT = """{stamp}Describe this video frame in great detail: who or what is present, the setting \
and background, peoples' appearance and actions, any on-screen text or UI (type out exact names, \
labels and numbers you can read), and objects. Capture everything that helps a viewer understand \
the video without watching it. Reply in 2-4 detailed sentences."""

BATCH_FRAME_PROMPT = """{stamp}This request contains {count} video frames, presented in chronological \
order, each covering roughly one second of the video (from {start} to {end}). \
Each image corresponds to one second, in this order:

{lines}

Describe WHAT IS ON SCREEN in great detail for the video as a whole: people and their actions, \
setting, objects, and any on-screen text or UI (type out exact names, labels and numbers you can \
read). Use the burned-in corner timestamps to bind each detail to its exact second, but never \
describe the stamps themselves as video content. Then list the notable changes between the frames \
with timestamps, e.g. "- 01:03 - A popup dialog appears labeled 'Sign in'". Cover every frame; \
never invent details."""

GRID_FRAME_PROMPT = """{stamp}This image is a {cols}x{rows} grid (top-left to bottom-right, left to right, \
row by row) of frames covering the video from {start} to {end}. Reading left-to-right top-to-bottom, \
cell 1 is the first captured second, cell 2 the next, and so on - consecutive cells are consecutive \
seconds of the video.

Describe what the video shows across this window in great detail: people and their actions, the \
setting, objects, scene changes between cells, and any on-screen text or UI (type out exact names, \
labels and numbers you can read). Use the burned-in corner timestamps to bind each detail to its \
exact second, but never describe the stamps themselves as video content. Use timestamped bullets \
like "- 00:07 - The host clicks 'Start Recording'". Cover every cell; never invent details."""

WINDOW_SUMMARY_PROMPT = """You are watching a video by reading second-by-second frame descriptions \
for the window {start} to {end}. Below are the observations.

Merge them into a dense summary of what is happening on screen and what the person(s) are doing \
in this window. Preserve specific details: on-screen text/UI, exact names and numbers, objects, \
scene changes, actions. Do not invent anything the observations do not support.

Reply with 3-6 concise bullet points, each starting with a timestamp like "01:23 - ":

{descriptions}"""

VOICE_SUMMARY_PROMPT = """The following are voice measurements for the video window \
{start} to {end}, taken from the audio (each line: speaking pace in words per minute, \
a loudness reading in LU and the share of the segment spent paused).

{descriptions}

Merge these into an engaging narration of HOW the speaker delivers this section: overall \
pace, energy shifts, notable pauses, emphatic moments, and what that suggests about tone \
(excited, calm, urgent, hesitant...). Only state what the measurements support.

Reply with 3-6 concise bullet points, each starting with a timestamp like "01:23 - ".
"""

DOUBTS_PROMPT = """You are planning the second, targeted pass of a two-pass visual analysis of \
the video "{title}" (total length {duration:g}s).

You have: the transcript, voice delivery notes and the FIRST visual pass. Find every moment where \
the visual record is NOT trustworthy - where you are unsure what is on screen, an earlier \
description was a refusal, blank or guess, what the speaker says implies a UI state no observation \
confirms, a quick action the transcript references but frames do not show, or a stretch with no \
visual coverage at all. These are your DOUBTS.

Reply with exact timestamps at which new frames should be captured to resolve each doubt, one per \
line, in EXACTLY this format:

t=<seconds> | <high|mid|low> | <one-line reason>

Rules:
- Seconds must be numbers within [0, {duration:g}].
- At most {max:d} lines, one doubt per line, ordered by priority.
- If you have NO doubts, reply with the single line: NONE

--- TRANSCRIPT (timestamped) ---
{transcript}

--- VOICE NOTES ---
{voice}

--- VISUAL OBSERVATIONS (PASS 1) ---
{visual}"""

REPORT_PROMPT = """You are a meticulous re-conciliator creating the definitive written record of the \
YouTube video "{title}" by {channel}.

The GROUND TRUTH TRANSCRIPT below is the authoritative record of the video's timeline and spoken \
facts - it has real, absolute timestamps. EVERY other source below (segment analyses, visual \
evidence, voice evidence) is secondary evidence to be CHECKED AGAINST that transcript: anything \
that contradicts it, strays from its timestamps, or cannot be supported must be treated as junk \
and DROPPED - never guessed at, never marked down.

Combine the *verified* evidence into one complete Markdown report so detailed that a reader who \
never watched the video understands EVERYTHING: every action, every step executed, every \
on-screen element, every decision, every number, every name. Never drop concrete details that are \
supported.

Your report MUST contain exactly these sections, and nothing after them:

# TL;DR
3-5 sentences: what the video is about, who is in it, and its main takeaway.

# What happened
An exhaustive, chronological account of the whole video. Use bullet points with timestamps. Every \
timestamp MUST come from the GROUND TRUTH TRANSCRIPT - never invent, re-normalize, compress or \
round the clock; the events happened at their real transcript seconds. Describe every action \
taken, every step performed, everything shown or built, every decision made, and every topic \
covered. This is the core of the report.

# What the video shows
Only include this section when visual evidence was supplied. A vivid description of what appears \
ON SCREEN, using ONLY that evidence (scenes, people and objects, on-screen text/UI, actions \
visible but not spoken). Label each item with its real coverage range (e.g. "00:04-00:29"), never \
a window's first second alone. When two observations read the same on-screen element \
differently, use the reading that matches the transcript (if any), otherwise the observation \
nearest the relevant transcript timestamp, and drop the others. If visual evidence is empty, \
write "No visual observations were captured."

# How it sounds (voice analysis)
ONLY include this section when voice evidence was supplied. Summarise how the speaker delivers \
the video (pace, energy, pauses, tone) using it, with timestamped bullets.

# Chapters
Markdown bullets: `- MM:SS - Chapter title`. Chapters must be DISTINCT and content-driven, \
summarising what actually changes at each point (e.g. "Logging in and starting creator \
verification"), not repeated variations of the video's own title. Chapter timestamps must come \
from the transcript.

# Key topics, people and tools
Every topic, person, tool, website and term that matters, as short bullets.

# Key quotes
Short bullets - ONLY lines that appear VERBATIM in the GROUND TRUTH TRANSCRIPT, each with its \
real transcript timestamp. Never fabricate, paraphrase, or cite analysis-line timestamps like \
"(quoted in analysis)".

GROUND RULES:
- The GROUND TRUTH TRANSCRIPT is the sole source of timestamps and spoken facts. Secondary \
evidence is only used to enrich the report; if it contradicts the transcript, the transcript wins \
and the claim is DROPPED.
- NEVER invent topics, people, tools, websites, names or numbers. If something is not in the \
transcript and not supported by the visual/voice evidence, it does not go in the report. If \
on-screen text is illegible in the evidence, do not guess - omit it entirely.
- The video is titled "{title}". Use its real product/brand names exactly (for example write \
"Fanvue", not a phonetically-misspoken "Fan View").
- The whole video spans {duration}. Timestamps must be absolute and together span it.

GROUND TRUTH TRANSCRIPT:
{transcript}

SEGMENT ANALYSES:
{analyses}

VISUAL EVIDENCE:
{visual}

VOICE EVIDENCE:
{voice}"""


class ContextError(Exception):
    """Raised when the AI context step fails."""


@dataclass
class OllamaClient:
    base_url: str
    model: str
    api_key: Optional[str] = None

    @property
    def is_cloud(self) -> bool:
        return self.base_url.startswith("https://")

    def complete(
        self,
        messages: list[dict],
        images: Optional[list[str]] = None,
    ) -> str:
        """Send a chat request and return the assistant's text reply.

        ``images`` is a list of file paths, sent to the model as inline
        images (requires a vision-capable model such as ``gemma4``).
        """
        if images:
            return self._complete_vision(messages, images)
        return self._complete_text(messages)

    def _complete_text(self, messages: list[dict]) -> str:
        """A multi-part content message given to ``ollama run`` style clients."""
        if self.is_cloud:
            resp = requests.post(
                f"{self.base_url}/v1/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"model": self.model, "messages": messages, "stream": False},
                timeout=180,
            )
            if resp.status_code == 401:
                raise ContextError(
                    "Ollama cloud rejected the API key. Check OLLAMA_API_KEY."
                )
            if resp.status_code != 200:
                raise ContextError(
                    f"Ollama cloud error {resp.status_code}: {resp.text[:300]}"
                )
            return resp.json()["choices"][0]["message"]["content"]

        resp = requests.post(
            f"{self.base_url}/api/chat",
            json={"model": self.model, "messages": messages, "stream": False},
            timeout=180,
        )
        if resp.status_code == 404 and "no such model" in resp.text:
            raise ContextError(
                f"Local Ollama model '{self.model}' is not pulled. "
                f"Run: ollama pull {self.model}"
            )
        if resp.status_code != 200:
            raise ContextError(
                f"Local Ollama error {resp.status_code}: {resp.text[:300]}"
            )
        return resp.json()["message"]["content"]

    def _complete_vision(self, messages: list[dict], images: list[str]) -> str:
        text = messages[-1]["content"]

        # Ollama's OpenAI-compatible endpoint rejects images for many models
        # (and 500s on others); the native /api/chat endpoint handles images
        # reliably for both local and cloud models, so we always use it here.
        if self.is_cloud:
            encoded = [
                base64.b64encode(Path(path).read_bytes()).decode() for path in images
            ]
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": text, "images": encoded}],
                "stream": False,
            }
            response = requests.post(
                f"{self.base_url}/api/chat",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=payload,
                timeout=180,
            )
            if response.status_code == 401:
                raise ContextError(
                    "Ollama cloud rejected the API key. Check OLLAMA_API_KEY."
                )
            if response.status_code == 404:
                raise ContextError(
                    f"Cloud vision model '{self.model}' is not available. "
                    "Try e.g. 'gemma4:31b-cloud' or 'qwen3-vl:235b'."
                )
            if response.status_code != 200:
                raise ContextError(
                    f"Ollama cloud vision error {response.status_code}: {response.text[:300]}"
                )
            return response.json()["message"]["content"]

        encoded = [
            base64.b64encode(Path(path).read_bytes()).decode() for path in images
        ]
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": text, "images": encoded}],
            "stream": False,
        }
        response = requests.post(
            f"{self.base_url}/api/chat", json=payload, timeout=180
        )
        if response.status_code == 404 and "no such model" in response.text:
            raise ContextError(
                f"Local Ollama model '{self.model}' is not pulled. Run: ollama pull {self.model}"
            )
        if response.status_code != 200:
            raise ContextError(
                f"Local Ollama vision error {response.status_code}: {response.text[:300]}"
            )
        return response.json()["message"]["content"]


def resolve_client(provider: str = "auto") -> OllamaClient:
    """Build a client for cloud (API key present) or local Ollama."""
    api_key = os.getenv("OLLAMA_API_KEY")
    if provider == "cloud":
        return OllamaClient(
            base_url=os.getenv("OLLAMA_BASE_URL", CLOUD_BASE_URL),
            model=os.getenv("OLLAMA_MODEL", DEFAULT_CLOUD_MODEL),
            api_key=api_key,
        )
    if provider == "local":
        return OllamaClient(
            base_url=os.getenv("OLLAMA_BASE_URL", LOCAL_BASE_URL),
            model=os.getenv("OLLAMA_MODEL", DEFAULT_LOCAL_MODEL),
        )
    if api_key:
        return OllamaClient(
            base_url=os.getenv("OLLAMA_BASE_URL", CLOUD_BASE_URL),
            model=os.getenv("OLLAMA_MODEL", DEFAULT_CLOUD_MODEL),
            api_key=api_key,
        )
    return OllamaClient(
        base_url=os.getenv("OLLAMA_BASE_URL", LOCAL_BASE_URL),
        model=os.getenv("OLLAMA_MODEL", DEFAULT_LOCAL_MODEL),
    )


def fmt_ts(seconds: float) -> str:
    """Format seconds as H:MM:SS / MM:SS."""
    total = int(seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


TS_RE = re.compile(r"(?<!\d)(?:\d{1,2}:)?\d{1,2}:\d{2}(?!\d)")
_WS_RE = re.compile(r"\s+")
_START_NEXT_HEADING_RE = re.compile(r"^# ", re.MULTILINE)


def _norm(text: str) -> str:
    """Lowercase + collapse whitespace for verbatim-ish comparison."""
    return _WS_RE.sub(" ", (text or "").strip().lower())


def _ts_to_seconds(token: str) -> Optional[float]:
    """Parse a ``MM:SS`` or ``H:MM:SS`` timestamp token to seconds."""
    parts = token.split(":")
    try:
        nums = [int(p) for p in parts]
    except ValueError:
        return None
    if len(nums) == 3:
        return nums[0] * 3600 + nums[1] * 60 + nums[2]
    if len(nums) == 2:
        return nums[0] * 60 + nums[1]
    return None


def _extract_quote(line: str) -> Optional[str]:
    """Pull the quoted text out of a ``- "..." (time)`` Key quotes line."""
    body = line.strip()
    if not body.startswith("- "):
        return None
    body = body[2:].strip()
    for open_q, close_q in (("“", "”"), ("\u201c", "\u201d"), ('"', '"')):
        start = body.find(open_q)
        if start != -1:
            end = body.find(close_q, start + 1)
            if end != -1:
                return body[start + 1 : end]
    return None


def _section_after(report: str, heading: str) -> tuple[str, str]:
    """Return ``(section_body, tail_from_next_heading)`` for a ``# heading``."""
    m = re.search(rf"^{heading}\s*$", report, re.MULTILINE)
    if not m:
        return "", ""
    rest = report[m.end() :]
    nxt = _START_NEXT_HEADING_RE.search(rest)
    body = rest[: nxt.start()] if nxt else rest
    tail = rest[nxt.start() :] if nxt else ""
    return body, tail


def verify_report(
    report: str,
    segments: list[dict],
    duration: float,
) -> tuple[str, list[str]]:
    """Deterministically audit a generated report against the transcript.

    Catches junk that LLM rules alone cannot: a compressed / invented clock
    and fabricated quotes. Quotes that do not appear verbatim in the transcript
    are *removed* from the report (strict drop, per the reconciliation
    contract). Returns ``(report, warnings)`` - the (possibly filtered) report
    and human-readable warnings. Does not raise - the report is usable even if
    the audit disagrees with it.
    """
    if not report:
        return report, []
    warnings: list[str] = []

    transcript_norm = _norm(" ".join(s.get("text", "") for s in segments))

    qm = re.search(r"^# Key quotes\s*$", report, re.MULTILINE)
    if qm:
        rest = report[qm.end() :]
        nxt = _START_NEXT_HEADING_RE.search(rest)
        end = qm.end() + (nxt.start() if nxt else len(rest))
        section = report[qm.end() : end]
        raw_lines = [l for l in section.splitlines() if l.strip()]
        kept = []
        for line in raw_lines:
            quote = _extract_quote(line)
            if quote and _norm(quote) in transcript_norm:
                kept.append(line)
        if len(kept) != len(raw_lines):
            dropped = len(raw_lines) - len(kept)
            report = (
                report[: qm.end()]
                + "\n\n"
                + "\n".join(kept)
                + ("\n" if kept else "")
                + report[end:]
            )
            warnings.append(
                f"Dropped {dropped} fabricated quote(s) not verbatim in the transcript."
            )

    body, _ = _section_after(report, "# Chapters")
    chapter_secs = [
        s
        for s in (_ts_to_seconds(t) for t in TS_RE.findall(body))
        if s is not None
    ]
    if chapter_secs:
        max_chapter = max(chapter_secs)
        if duration and max_chapter < 0.5 * duration:
            warnings.append(
                f"Chapters only reach {fmt_ts(max_chapter)} of a "
                f"{fmt_ts(duration)} video - timeline may be compressed/invented."
            )

    report_secs = [
        s
        for s in (_ts_to_seconds(t) for t in TS_RE.findall(report))
        if s is not None
    ]
    over = [s for s in report_secs if duration and s > duration + 30]
    if over:
        warnings.append(
            f"Context report has timestamps past the video length "
            f"(e.g. {fmt_ts(min(over))} > {fmt_ts(duration)})."
        )

    return report, warnings


def chunk_segments(segments: list[dict], max_chars: int = CHUNK_MAX_CHARS) -> list[dict]:
    """Group segments into chunks of roughly ``max_chars`` characters.

    Each chunk keeps ``segments`` (the original records) so callers can render
    the chunk with real per-line timestamps instead of bare joined text.
    """
    chunks: list[dict] = []
    current: list[dict] = []
    current_len = 0

    for seg in segments:
        text = seg["text"]
        if current and current_len + len(text) > max_chars:
            chunks.append(
                {
                    "start": current[0]["start"],
                    "end": current[-1]["end"],
                    "segments": list(current),
                    "text": " ".join(s["text"] for s in current),
                }
            )
            current = []
            current_len = 0
        current.append(seg)
        current_len += len(text) + 1

    if current:
        chunks.append(
            {
                "start": current[0]["start"],
                "end": current[-1]["end"],
                "segments": list(current),
                "text": " ".join(s["text"] for s in current),
            }
        )
    return chunks


def _render_chunk(chunk: dict) -> str:
    """Render a chunk with real per-segment timestamps for the model."""
    segs = chunk.get("segments")
    if segs:
        return "\n".join(f"[{fmt_ts(s['start'])}] {s.get('text', '').strip()}" for s in segs)
    return chunk.get("text", "")


def _render_evidence(entries: list[dict]) -> str:
    """Render evidence entries with their real coverage ranges."""
    lines = []
    for v in entries:
        label = fmt_ts(v["ts"])
        end = v.get("end")
        if end is not None and end != v["ts"]:
            label += f"\u2013{fmt_ts(end)}"
        lines.append(f"- {label} - {v['description']}")
    return "\n".join(lines)


def analyze(
    segments: list[dict],
    client: OllamaClient,
    title: str = "Untitled",
    channel: str = "",
    visual_timeline: Optional[list[dict]] = None,
    voice_notes: Optional[list[dict]] = None,
) -> dict:
    """Run chunked map-reduce analysis and return the report + intermediate results.

    ``visual_timeline`` is an optional list of ``{"ts": float, "description": str}``
    entries describing what the camera shows. Overlapping frames are pasted into
    each chunk and the full timeline is fed to the final synthesis, which adds
    a dedicated "What the video shows" section so on-screen actions are never
    lost. A deterministic per-frame section is appended as a fallback if the
    model did not already include one.

    ``voice_notes`` has the same shape and describes how the speaker delivers
    the lines (pace, energy, pauses). Overlapping notes are pasted into chunks
    and the full set is fed to the final synthesis as a "# How it sounds (voice
    analysis)" section, with a deterministic fallback as well.

    The final report is a single reconciliation call: the verbatim transcript
    (ground truth) is compared against every secondary source, then audited by
    :func:`verify_report`. Audit results are returned under ``"warnings"``.
    """
    if not segments:
        raise ContextError("Nothing to analyze: the transcript is empty.")

    timeline = visual_timeline or []
    voice = voice_notes or []
    chunks = chunk_segments(segments)
    chunk_results: list[dict] = []

    for i, chunk in enumerate(chunks, start=1):
        prompt = CHUNK_PROMPT.format(
            start=fmt_ts(chunk["start"]),
            end=fmt_ts(chunk["end"]),
            chunk=_render_chunk(chunk),
        )
        nearby = [
            v["description"]
            for v in timeline
            if chunk["start"] - 5 <= v["ts"] <= chunk["end"] + 30
        ]
        nearby_voice = [
            v["description"]
            for v in voice
            if chunk["start"] <= v["ts"] <= chunk["end"] + 30
        ]
        messages: list[dict] = []
        if nearby:
            prompt += VISUAL_PREFIX.format(visual="\n".join(f"- {d}" for d in nearby))
        if nearby_voice:
            prompt += VOICE_PREFIX.format(voice="\n".join(f"- {d}" for d in nearby_voice))
        messages.append({"role": "user", "content": prompt})

        analysis = client.complete(messages)
        chunk_results.append(
            {
                "start": chunk["start"],
                "end": chunk["end"],
                "analysis": analysis.strip(),
            }
        )

    analyses_blob = "\n\n---\n\n".join(
        f"[Segment {i}, ~{fmt_ts(c['start'])}-{fmt_ts(c['end'])}:]\n{c['analysis']}"
        for i, c in enumerate(chunk_results, start=1)
    )
    if timeline:
        visual_blob = _render_evidence(timeline)
    else:
        visual_blob = "None - no video frames were analyzed."
    if voice:
        voice_blob = _render_evidence(voice)
    else:
        voice_blob = "None - no voice analysis was performed."

    transcript_blob = "\n".join(
        f"[{fmt_ts(s['start'])}] ({fmt_ts(s.get('end', s['start']))}) {s['text']}"
        for s in segments
    )
    duration = segments[-1]["end"]

    report_prompt = REPORT_PROMPT.format(
        title=title,
        channel=channel,
        duration=fmt_ts(duration),
        transcript=transcript_blob,
        analyses=analyses_blob,
        visual=visual_blob,
        voice=voice_blob,
    )
    report = client.complete([{"role": "user", "content": report_prompt}]).strip()

    if timeline and "# What the video shows" not in report:
        fallback = (
            "\n\n# What the video shows (frame analysis)\n"
            + _render_evidence(timeline)
        )
        report += fallback

    if voice and "# How it sounds" not in report:
        fallback = (
            "\n\n# How it sounds (voice analysis)\n" + _render_evidence(voice)
        )
        report += fallback

    report, warnings = verify_report(report, segments, duration)

    return {
        "report": report,
        "chunks": chunk_results,
        "n_chunks": len(chunks),
        "warnings": warnings,
    }


def find_ollama_url() -> bool:
    """Cheap probe for a running local Ollama server."""
    try:
        return requests.get(f"{os.getenv('OLLAMA_BASE_URL', LOCAL_BASE_URL)}/", timeout=2).ok
    except requests.RequestException:
        return False


def sanitize_title(title: str) -> str:
    """Return a filesystem-safe video title."""
    safe = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", title).strip()
    return safe[:80] or "video"