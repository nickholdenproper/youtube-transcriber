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

CHUNK_PROMPT = """You are a meticulous analyst examining a segment of a video transcript \
(from {start} to {end} seconds). The transcript may be imperfect speech-to-text.

Reply with exactly these Markdown sections:

## CHAPTER
A short descriptive title for this portion, like a YouTube chapter name.

## WHAT HAPPENED
Concise bullet points describing what the speaker(s) said or did in this segment. \
Preserve concrete details: names, tools, decisions, numbers, order of actions.

## QUOTES
Up to 3 short notable quotes taken verbatim from the transcript if possible.

Transcript segment:
{chunk}"""

VISUAL_PREFIX = """
VISUAL CONTEXT for this time window (observations from video frames by a vision
model - use them to describe what is visibly happening, but treat uncertain
details carefully):
{visual}"""

FRAME_PROMPT = """Look at this video frame and describe what is happening on screen.
Focus on: actions the person(s) are doing, visible tools / screens / objects /
text, who is present, and anything important for understanding the video.
Reply in 1-2 short sentences."""

SYNTH_PROMPT = """Combine these per-segment analyses of a YouTube video into one complete \
context report written in Markdown. The video is titled "{title}" by {channel}.

Your report must contain exactly these sections:

# TL;DR
2-3 sentences summarizing what the whole video is about and its main takeaway.

# What happened
An ordered narrative of what the person(s) actually did across the video: actions taken, \
things built or shown, decisions made, topics covered. Write it for someone who has not \
watched the video. Add approximate timestamps where helpful (e.g. "around 3:20").

# Chapters
Markdown bullets: `- MM:SS - Chapter title`

# Key topics, people and tools
Short markdown bullets.

# Key quotes
Short bullets, each with a rough timestamp when possible.

Segment analyses:
{analyses}"""


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
    entries describing what the camera shows; overlapping frames are pasted into
    each chunk so the report can describe visual actions too.
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
    synth_prompt = SYNTH_PROMPT.format(title=title, channel=channel, analyses=analyses_blob)
    report = client.complete([{"role": "user", "content": synth_prompt}])

    return {"report": report.strip(), "chunks": chunk_results, "n_chunks": len(chunks)}


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