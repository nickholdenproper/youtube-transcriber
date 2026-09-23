# yt-transcriber

YouTube transcriber with **AI context** — turns any video into a timestamped transcript plus a report of *what happened*: TL;DR, chapters, key topics, people, tools and quotes. Not just speech-to-text.

Three interfaces, one shared pipeline:

- **CLI** — `yt-transcribe "URL"`
- **Local GUI** — double-click `run.bat`, use it in the browser (live progress bar + step log)
- **REST API** — build it into news sites, editors, RSS readers, anything

## How it works

1. **Fetch** — `yt-dlp` grabs video metadata and, when available, the existing captions (manual → auto). Captions-first means most videos get a transcript in seconds. If captions are missing *or* YouTube rate-limits them (HTTP 429), the pipeline retries with backoff and then falls back to downloading audio and transcribing locally — a caption error never blocks the run.
2. **Transcribe** — no captions? Audio is downloaded and transcribed locally with `faster-whisper`.
3. **Context** — the transcript is analyzed by Ollama (cloud API by default, local Ollama as fallback) using chunked map-reduce, so long videos stay safe.
   **Visual context (on by default)** — what the camera *shows* (demos, slides, on-screen text/UI, body language) that the audio never mentions. A low-res copy of the video is downloaded, frames are extracted with ffmpeg and described by a vision model (`gemma4` locally, `gemma4:31b-cloud` on Ollama Cloud), then merged with the transcript into one report with a dedicated **What the video shows** section. Disable with `--no-vision`.
4. **Output** — Markdown + JSON files for the transcript and the context report.

## Requirements

- Python 3.10+
- ffmpeg (for the audio fallback + visual frame extraction)
- An Ollama account + API key for cloud analysis (free tier) — or local Ollama running
- **For local vision**: pull a vision model first — `ollama pull gemma4`. With an API key set, cloud vision (`gemma4:31b-cloud`) works with no local model needed.

## Quick start (Windows)

Double-click **`run.bat`**. It creates the virtual environment on first run,
starts the server and opens the GUI at `http://127.0.0.1:8000`.

You can also pass CLI arguments through it:

```bat
run.bat "https://www.youtube.com/watch?v=..."
run.bat serve
```

## CLI usage

```bash
# Full pipeline: transcript + context report
yt-transcribe "https://www.youtube.com/watch?v=..."

# Transcript only (no AI context)
yt-transcribe "URL" --skip-context

# Force local Whisper transcription (ignore captions)
yt-transcribe "URL" --force-whisper --whisper-model small

# Use local Ollama explicitly (even with an API key set)
yt-transcribe "URL" --provider local

# Music / ambient audio where silence detection hurts
yt-transcribe "URL" --no-vad

# Detailed frame analysis (more frames = more detail)
yt-transcribe "URL" --frame-interval 15 --max-frames 20

# Skip visual/frame analysis (audio + captions only)
yt-transcribe "URL" --no-vision

# Start the GUI + API server
yt-transcribe serve --open
```

## REST API

Start the server (`run.bat` or `yt-transcribe serve`), then docs are alive at
`http://127.0.0.1:8000/docs`.

| Endpoint | Description |
| --- | --- |
| `POST /v1/transcribe` | Synchronous: transcribe a video, returns everything as JSON |
| `POST /v1/jobs` | Asynchronous: returns `{"job_id": ...}` for long videos |
| `GET  /v1/jobs/{id}` | Job status: `pending / running / done / error` + log + result |
| `GET  /v1/results` | List previously produced outputs |
| `GET  /v1/results/{video_id}/{file}` | Download a saved file (`transcript.md`, `context.json`, ...) |
| `GET  /v1/health` | Health check |

Example for a news site:

```bash
curl -X POST http://127.0.0.1:8000/v1/transcribe \
  -H "Content-Type: application/json" \
  -d '{"url": "https://www.youtube.com/watch?v=...", "options": {"skip_context": false}}'
```

```json
{
  "video_id": "dQw4w9WgXcQ",
  "meta": { "title": "...", "channel": "...", "url": "..." },
  "source": "captions",
  "segments": [ { "start": 0.0, "end": 3.0, "text": "..." } ],
  "context_report": "# TL;DR\\n...",
  "files": [ { "name": "transcript.md", "url": "/v1/results/dQw4w9WgXcQ/transcript.md" } ]
}
```

Long workflows should use `POST /v1/jobs` and poll `GET /v1/jobs/{id}` — the
worker runs in the background (2 concurrent jobs max).

Outputs are written to `output/<video-id>/`:

| File | Contents |
| --- | --- |
| `transcript.md` | Timestamped full transcript |
| `transcript.json` | Structured transcript + video metadata |
| `context.md` | AI context report (TL;DR, what happened, what the video shows, chapters, topics, quotes) |
| `context.json` | Same report in structured form |

## Configuration

| Option | Default | Purpose |
| --- | --- | --- |
| `OLLAMA_API_KEY` | unset | Enables Ollama cloud analysis |
| `OLLAMA_BASE_URL` | cloud: `https://ollama.com`, local: `http://localhost:11434` | Ollama endpoint |
| `OLLAMA_MODEL` | cloud: `gpt-oss:20b-cloud`, local: `llama3.2` | Ollama model |
| `YT_TRANSCRIBER_OUT` | `output` | Where results are stored |
| `--provider` | `auto` | `auto` uses cloud if an API key is set, else local |
| `--model` | cloud: `gpt-oss:20b-cloud`, local: `llama3.2` | Ollama model to use |
| `--whisper-model` | `base` | faster-whisper model size (transcription fallback) |
| `--device` | `cpu` | Whisper inference device (`cpu` or `cuda`) |
| `--language` | unset | Whisper language hint (e.g. `en`, `es`) |
| `--no-vad` | off | Disable voice-activity detection (music / ambient audio) |
| `--vision/--no-vision` | on | Analyze video frames so the report includes on-screen actions |
| `--vision-model` | `gemma4` (local) / `gemma4:31b-cloud` (cloud) | Vision model for frame analysis |
| `--frame-interval` | `30` | Seconds between extracted frames |
| `--max-frames` | `10` | Maximum number of frames to analyze |
| `--skip-context` | off | Skip the AI context step (also disables vision) |

The API mirrors these via the `options` object in the JSON body.

## Branch workflow

Developed feature-by-feature on branches, merged into `main` when proven:

```
main ───▓ feature/scaffold
        ▓ feature/fetch
        ▓ feature/transcribe
        ▓ feature/context
        ▓ feature/polish
        ▓ feature/api
        ▓ feature/ui
```

## Troubleshooting

- **YouTube blocks yt-dlp** — update `yt-dlp` (`pip install -U yt-dlp`); if needed, pass cookies via `.env`.
- **Local Ollama** — run `ollama serve`, then `ollama pull llama3.2`.
- **Vision step skipped** — local mode: pull the model first (`ollama pull gemma4`); cloud mode: make sure the model is a vision-capable `:cloud` model like `gemma4:31b-cloud`.
- **Context step skipped** — no API key and no local Ollama running; transcript still works.
- **Server won't start (port busy)** — `yt-transcribe serve --port 8001`.