"""Command-line entry point for yt-transcriber."""

from __future__ import annotations

import threading
import webbrowser
from pathlib import Path
from typing import List, Optional

import typer
from dotenv import load_dotenv

from .pipeline import PipelineOptions, run_pipeline

load_dotenv()

app = typer.Typer(
    help="Transcribe YouTube videos into timestamped text and generate an "
    "AI context report of what happened in the video.",
    no_args_is_help=True,
)


@app.command()
def transcribe(
    url: str = typer.Argument(..., help="YouTube video URL"),
    out: str = typer.Option("output", "--out", "-o", help="Output directory"),
    skip_context: bool = typer.Option(
        False, "--skip-context", help="Skip the AI context report"
    ),
    provider: str = typer.Option(
        "auto",
        "--provider",
        help="auto (cloud if OLLAMA_API_KEY is set, else local), cloud or local",
    ),
    model: str = typer.Option(
        None, "--model", help="Ollama model override (default depends on provider)"
    ),
    whisper_model: str = typer.Option(
        "base", "--whisper-model", help="faster-whisper model size (tiny/base/small/medium/large-v3)"
    ),
    device: str = typer.Option(
        "cpu", "--device", help="Whisper device: cpu or cuda"
    ),
    force_whisper: bool = typer.Option(
        False,
        "--force-whisper",
        help="Ignore captions and always transcribe the audio locally",
    ),
    language: str = typer.Option(
        None, "--language", help="Whisper language hint (e.g. en, es)"
    ),
    no_vad: bool = typer.Option(
        False,
        "--no-vad",
        help="Disable voice-activity detection (helps with music / ambient audio)",
    ),
    vision: bool = typer.Option(
        True,
        "--vision/--no-vision",
        help="Analyze video frames so the report includes on-screen actions "
        "(on by default; disable with --no-vision)",
    ),
    vision_model: Optional[str] = typer.Option(
        None,
        "--vision-model",
        help="Ollama vision model for frames: gemma4 (local) / gemma4:31b-cloud (cloud), "
        "or free alternatives like qwen2.5-vl:7b, moondream2",
    ),
    vision_mode: str = typer.Option(
        "per-frame",
        "--vision-mode",
        help="per-frame (each unique frame described, default), batch (up to "
        "--batch-size frames per request), or mosaic (adaptive grid montages)",
    ),
    frame_interval: float = typer.Option(
        1, "--frame-interval", help="Seconds between frames for visual analysis (1 = one frame per second)"
    ),
    max_frames: int = typer.Option(
        600, "--max-frames", help="Maximum number of frames to analyze (long videos are down-sampled evenly)"
    ),
    vision_window: float = typer.Option(
        30,
        "--vision-window",
        help="Group frame descriptions into N-second windows for summarization",
    ),
    dedupe: bool = typer.Option(
        True,
        "--dedupe/--no-dedupe",
        help="Drop frames that barely change so cost tracks actual screen change (on by default)",
    ),
    dedupe_max_gap: float = typer.Option(
        10,
        "--dedupe-max-gap",
        help="Never skip more than N seconds between kept frames, even if unchanged",
    ),
    batch_size: int = typer.Option(
        8, "--batch-size", help="Images per request in batch mode (<=8 keeps timestamp binding safe)"
    ),
    mosaic_cells: Optional[int] = typer.Option(
        None, "--mosaic-cells", help="Cells per grid in mosaic mode (auto if unset, max 144)"
    ),
    target_grids: int = typer.Option(
        24, "--target-grids", help="Target number of vision calls; drives auto grid sizing"
    ),
    voice: bool = typer.Option(
        True,
        "--voice/--no-voice",
        help="Analyze how the video sounds (pace, loudness, pauses) and add a "
        "'How it sounds' section to the report (on by default; disable with --no-voice)",
    ),
    voice_window: float = typer.Option(
        30,
        "--voice-window",
        help="Group voice measurements into N-second windows for summarization",
    ),
    vision_doubt: bool = typer.Option(
        True,
        "--vision-doubts/--no-vision-doubts",
        help="Run the second, LLM-directed vision pass: re-capture exact frames at "
        "timestamps the model flags as doubtful, then merge (on by default)",
    ),
    vision_fill_budget: int = typer.Option(
        60,
        "--vision-fill-budget",
        help="Maximum doubt frames the second vision pass may capture",
    ),
    derive: List[str] = typer.Option(
        [],
        "--derive",
        help="Rewrite the video digest into extra documents via a second LLM call: "
        "how-to, article, faq, checklist, quiz (comma-separated)",
    ),
    derive_model: Optional[str] = typer.Option(
        None,
        "--derive-model",
        help="Ollama model for the derived-document step (default: the main model)",
    ),
):
    """Transcribe a YouTube video and analyze its content."""
    opts = PipelineOptions(
        out=out,
        skip_context=skip_context,
        provider=provider,
        model=model,
        whisper_model=whisper_model,
        device=device,
        force_whisper=force_whisper,
        language=language,
        no_vad=no_vad,
        vision=vision,
        vision_model=vision_model,
        vision_mode=vision_mode,
        frame_interval=frame_interval,
        max_frames=max_frames,
        vision_window=vision_window,
        dedupe=dedupe,
        dedupe_max_gap=dedupe_max_gap,
        batch_size=batch_size,
        mosaic_cells=mosaic_cells,
        target_grids=target_grids,
        voice=voice,
        voice_window=voice_window,
        vision_doubt=vision_doubt,
        vision_fill_budget=vision_fill_budget,
        derive=derive,
        derive_model=derive_model,
    )

    try:
        result = run_pipeline(url, opts, log=typer.echo)
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        typer.echo(f"Failed: {exc}", err=True)
        raise typer.Exit(1) from exc

    for warning in result.warnings:
        typer.echo(f"Warning: {warning}", err=True)

    out_dir = Path(out) / result.video_id
    typer.echo(f"Wrote {len(result.files)} files to {out_dir}:")
    for path in result.files:
        typer.echo(f"  {path}")


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host", help="Bind address"),
    port: int = typer.Option(8000, "--port", "-p", help="Port to listen on"),
    open_browser: bool = typer.Option(
        False, "--open", help="Open the GUI in the default browser"
    ),
):
    """Run the local web GUI + REST API server."""
    import uvicorn

    from .api import app as api_app

    if open_browser:
        threading.Timer(1.2, webbrowser.open, args=[f"http://{host}:{port}/"]).start()

    uvicorn.run(api_app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    app()