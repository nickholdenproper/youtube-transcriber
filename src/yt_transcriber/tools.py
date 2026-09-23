"""Registry of the analysis *tools* the transcriber ships with.

Each tool answers a different question about the video and produces its own
output files. Additional tools can be registered here (and surfaced through
``GET /v1/tools``) without touching the pipeline signature - they just plug
their implementation into :mod:`yt_transcriber.pipeline`.
"""

from __future__ import annotations

TOOLS: dict[str, dict] = {
    "vision": {
        "name": "Vision",
        "description": (
            "Reads what the camera shows from video frames: on-screen text and "
            "UI, scenes, people and unspoken actions, compressed into a "
            "described visual timeline."
        ),
        "default_on": True,
        "cli_flag": "--no-vision",
        "outputs": ["visual_frames.json", "visual_*.png"],
    },
    "voice": {
        "name": "Voice",
        "description": (
            "Measures how the video sounds from its audio: speaking pace "
            "(words per minute), loudness (LUFS), pauses, energy bursts and "
            "delivery tone, plus a per-window narration of the speaker's "
            "delivery."
        ),
        "default_on": True,
        "cli_flag": "--no-voice",
        "outputs": ["voice_analysis.json"],
    },
    "derive": {
        "name": "Derive",
        "description": (
            "Rewrites the whole-video context digest into extra documents via "
            "one LLM call per type: step-by-step how-to guides, articles, "
            "timestamped FAQs, action checklists and quizzes. Uses the main "
            "model unless --derive-model overrides it."
        ),
        "default_on": False,
        "cli_flag": "--derive",
        "outputs": ["derived_*.md"],
    },
}


def list_tools() -> list[dict]:
    """Return the registered tools as a metadata list for API consumers."""
    return [
        {
            "id": tool_id,
            **{k: v for k, v in fields.items() if k != "outputs"},
            "outputs": list(fields.get("outputs", [])),
        }
        for tool_id, fields in TOOLS.items()
    ]