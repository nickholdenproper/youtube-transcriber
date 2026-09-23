"""Command-line entry point for yt-transcriber."""

from __future__ import annotations

import threading
import webbrowser
from pathlib import Path
from typing import Optional

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
        help="Ollama vision model for frames: gemma4 (local) / gemma4:31b-cloud (cloud)",
    ),
    frame_interval: float = typer.Option(
        30, "--frame-interval", help="Seconds between frames for visual analysis"
    ),
    max_frames: int = typer.Option(
        10, "--max-frames", help="Maximum number of frames to analyze"
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
        frame_interval=frame_interval,
        max_frames=max_frames,
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