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

Write with enough detail that a reader could follow along without watching the video. \
Reply with exactly these Markdown sections:

## CHAPTER
A short descriptive title for this portion, like a YouTube chapter name.

## WHAT HAPPENED
An exhaustive bullet-by-bullet account of EVERY event in this segment, in order: exactly what \
the speaker says, every action they take, every step they show, every tool / site / number / \
name they mention, every decision. Use one idea per bullet, each starting with a timestamp from \
the segment (e.g. "00:12 - "). Write as many bullets as the content needs (10-30+) - never \
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

FRAME_PROMPT = """Describe this video frame in great detail: who or what is present, the setting \
and background, peoples' appearance and actions, any on-screen text or UI (type out exact names, \
labels and numbers you can read), and objects. Capture everything that helps a viewer understand \
the video without watching it. Reply in 2-4 detailed sentences."""

SYNTH_PROMPT = """You are compiling the definitive, exhaustive written record of the YouTube \
video titled "{title}" by {channel}. Combine the per-segment analyses below into one complete \
Markdown report so detailed that a reader who never watched the video understands EVERYTHING: \
every action, every step executed, every on-screen element, every decision, every number, every \
name. Never drop concrete details.

Your report MUST contain exactly these sections:

# TL;DR
3-5 sentences: what the video is about, who is in it, and its main takeaway.

# What happened
An exhaustive, chronological account of the whole video. Use bullet points with timestamps \
(e.g. "- 01:23 - Clicked 'Verify and start earning')") describing every action taken, every step \
performed, everything shown or built, every decision made, and every topic covered. This is the \
core of the report - make it as long and detailed as the video's content deserves.

# What the video shows
A vivid description of what appears ON SCREEN, using the visual observations below (scenes, \
people and objects, on-screen text/UI, actions visible but not spoken). Bullet points with \
timestamps. If visual observations are empty, write "No visual observations were captured."

# Chapters
Markdown bullets: `- MM:SS - Chapter title`

# Key topics, people and tools
Every topic, person, tool, website and term that matters, as short bullets.

# Key quotes
Short bullets, each with a rough timestamp when possible.

Segment analyses:
{analyses}

Visual observations by timestamp:
{visual}"""


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


def chunk_segments(segments: list[dict], max_chars: int = CHUNK_MAX_CHARS) -> list[dict]:
    """Group segments into chunks of roughly ``max_chars`` characters."""
    chunks: list[dict] = []
    current: list[dict] = []
    current_len = 0

    for seg in segments:
        text = seg["text"]
        if current and current_len + len(text) > max_chars:
            start = current[0]["start"]
            end = current[-1]["end"]
            chunks.append(
                {
                    "start": start,
                    "end": end,
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
                "text": " ".join(s["text"] for s in current),
            }
        )
    return chunks


def analyze(
    segments: list[dict],
    client: OllamaClient,
    title: str = "Untitled",
    channel: str = "",
    visual_timeline: Optional[list[dict]] = None,
) -> dict:
    """Run chunked map-reduce analysis and return the report + intermediate results.

    ``visual_timeline`` is an optional list of ``{"ts": float, "description": str}``
    entries describing what the camera shows. Overlapping frames are pasted into
    each chunk and the full timeline is fed to the final synthesis, which adds
    a dedicated "What the video shows" section so on-screen actions are never
    lost. A deterministic per-frame section is appended as a fallback if the
    model did not already include one.
    """
    if not segments:
        raise ContextError("Nothing to analyze: the transcript is empty.")

    timeline = visual_timeline or []
    chunks = chunk_segments(segments)
    chunk_results: list[dict] = []

    for i, chunk in enumerate(chunks, start=1):
        prompt = CHUNK_PROMPT.format(
            start=fmt_ts(chunk["start"]), end=fmt_ts(chunk["end"]), chunk=chunk["text"]
        )
        nearby = [
            v["description"]
            for v in timeline
            if chunk["start"] - 5 <= v["ts"] <= chunk["end"] + 30
        ]
        messages: list[dict] = []
        if nearby:
            prompt += VISUAL_PREFIX.format(visual="\n".join(f"- {d}" for d in nearby))
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
        f"[Segment {i}, ~{fmt_ts(c['start'])}:]\n{c['analysis']}"
        for i, c in enumerate(chunk_results, start=1)
    )
    if timeline:
        visual_blob = "\n".join(
            f"- {fmt_ts(v['ts'])} - {v['description']}" for v in timeline
        )
    else:
        visual_blob = "None - no video frames were analyzed."
    synth_prompt = SYNTH_PROMPT.format(
        title=title, channel=channel, analyses=analyses_blob, visual=visual_blob
    )
    report = client.complete([{"role": "user", "content": synth_prompt}]).strip()

    if timeline and "# What the video shows" not in report:
        fallback = (
            "\n\n# What the video shows (frame analysis)\n"
            + "\n".join(f"- **{fmt_ts(v['ts'])}** - {v['description']}" for v in timeline)
        )
        report += fallback

    return {"report": report, "chunks": chunk_results, "n_chunks": len(chunks)}


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