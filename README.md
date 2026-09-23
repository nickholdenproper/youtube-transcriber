# yt-transcriber

CLI tool that turns YouTube videos into **timestamped transcripts** plus an **AI context report** — a summary of what happened, chapters with timestamps, key topics, people and tools mentioned, and notable quotes.

Not just speech-to-text: it tells you *what the person actually did* in the video.

## How it works

1. **Fetch** — `yt-dlp` grabs video metadata and, when available, the existing captions (manual → auto). Captions-first means most videos get a transcript in seconds.
2. **Transcribe** — if the video has no captions, audio is downloaded and transcribed locally with `faster-whisper`.
3. **Context** — the transcript is analyzed by Ollama (cloud API by default, local Ollama as fallback) using chunked map-reduce, producing long videos safely.
4. **Output** — Markdown + JSON files for the transcript and the context report.

## Requirements

- Python 3.10+
- ffmpeg (for the audio fallback)
- An Ollama account + API key for cloud analysis (free tier) — or local Ollama running

## Install

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -e .
```

## Usage

```bash
# Full pipeline: transcript + context report
yt-transcribe "https://www.youtube.com/watch?v=..."

# Transcript only (no AI context)
yt-transcribe "URL" --skip-context

# Force local Whisper transcription (ignore captions)
yt-transcribe "URL" --force-whisper --whisper-model small
```

Outputs are written to `output/<video-id>/`.

## Configuration

| Option | Default | Purpose |
| --- | --- | --- |
| `OLLAMA_API_KEY` | unset | Enables Ollama cloud analysis |
| `--provider` | `auto` | `auto` uses cloud if an API key is set, else local |
| `--model` | cloud: `gpt-oss:20b-cloud`, local: `llama3.2` | Ollama model to use |
| `--whisper-model` | `base` | faster-whisper model size (transcription fallback) |
| `--skip-context` | off | Skip the AI context step |

## Branch workflow

Developed feature-by-feature on branches, merged into `main` when proven:

```
main ───▓ feature/scaffold
        ▓ feature/fetch
        ▓ feature/transcribe
        ▓ feature/context
        ▓ feature/polish
```

## Troubleshooting

- **YouTube blocks yt-dlp** — update `yt-dlp` (`pip install -U yt-dlp`); if needed, pass cookies via `.env`.
- **Local Ollama** — run `ollama serve`, then `ollama pull llama3.2`.
- **Context step skipped** — no API key and no local Ollama running; transcript still works.