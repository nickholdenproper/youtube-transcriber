"""Command-line entry point for yt-transcriber.

Scaffold stub - real pipeline lands in the feature branches.
"""

import typer

app = typer.Typer(
    help="Transcribe YouTube videos and generate AI context analysis.",
)


@app.command()
def transcribe(url: str = typer.Argument(..., help="YouTube video URL")):
    """Transcribe a YouTube video and analyze its content."""
    typer.echo(f"Not implemented yet - received URL: {url}")


if __name__ == "__main__":
    app()