"""Command-line entry point for yt-transcriber."""

from __future__ import annotations

from pathlib import Path

import typer
from dotenv import load_dotenv

from .context import ContextError, analyze, resolve_client
from .fetch import fetch_video
from .output import write_outputs
from .transcribe import transcribe

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
):
    """Transcribe a YouTube video and analyze its content."""
    work_dir = Path(out)
    result = None

    try:
        typer.echo(f"Fetching video info: {url}")
        result = fetch_video(url, work_dir, force_whisper=force_whisper)
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        typer.echo(f"Failed to fetch video: {exc}", err=True)
        raise typer.Exit(1) from exc

    meta = result.meta
    typer.echo(f"  {meta.title}")
    typer.echo(f"  Channel: {meta.channel}")
    typer.echo(f"  Source: {result.source}")

    segments = list(result.segments)

    if not segments and result.audio_path:
        typer.echo(f"Transcribing audio with whisper ({whisper_model}, {device}) ...")
        try:
            tr = transcribe(
                result.audio_path,
                model_size=whisper_model,
                device=device,
                language=language,
                vad_filter=not no_vad,
            )
        except Exception as exc:  # noqa: BLE001 - CLI boundary
            typer.echo(f"Transcription failed: {exc}", err=True)
            raise typer.Exit(1) from exc
        segments = tr["segments"]
        typer.echo(f"  Got {len(segments)} segments ({tr['language']})")

    if not segments:
        typer.echo("No transcript could be produced for this video.", err=True)
        raise typer.Exit(1)

    context_report = None
    context_chunks = None

    if not skip_context:
        try:
            client = resolve_client(provider)
            if model:
                client.model = model
            typer.echo(
                f"Analyzing context with {client.model} "
                f"({'cloud' if client.is_cloud else 'local'}) ..."
            )
            analysis = analyze(segments, client, title=meta.title, channel=meta.channel)
            context_report = analysis["report"]
            context_chunks = analysis["chunks"]
        except ContextError as exc:
            typer.echo(f"Context step skipped: {exc}", err=True)
        except Exception as exc:  # noqa: BLE001
            typer.echo(f"Context step skipped (unexpected error): {exc}", err=True)

    out_dir = work_dir / meta.id
    written = write_outputs(
        out_dir, meta, segments, result.source, context_report, context_chunks
    )

    typer.echo(f"Wrote {len(written)} files to {out_dir}:")
    for path in written:
        typer.echo(f"  {path}")


if __name__ == "__main__":
    app()