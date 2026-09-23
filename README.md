# yt-transcriber

YouTube transcriber with **AI context** — turns any video into a timestamped transcript plus a report of *what happened*: TL;DR, chapters, key topics, people, tools and quotes. Not just speech-to-text. The report doubles as a structured digest you can re-feed to another LLM to auto-rewrite the video into step-by-step how-to guides, articles, timestamped FAQs, action checklists and quizzes (`--derive`).

Three interfaces, one shared pipeline:

- **CLI** — `yt-transcribe "URL"`
- **Local GUI** — double-click `run.bat`, use it in the browser (live progress bar + step log)
- **REST API** — build it into news sites, editors, RSS readers, anything

## How it works

1. **Fetch** — `yt-dlp` grabs video metadata and, when available, the existing captions (manual → auto). Captions-first means most videos get a transcript in seconds. If captions are missing *or* YouTube rate-limits them (HTTP 429), the pipeline retries with backoff and then falls back to downloading audio and transcribing locally — a caption error never blocks the run.
2. **Transcribe** — no captions? Audio is downloaded and transcribed locally with `faster-whisper`.
3. **Context** — the transcript is analyzed by Ollama (cloud API by default, local Ollama as fallback) using chunked map-reduce, so long videos stay safe. The final report is a **reconciliation**: the verbatim audio transcript (with its real timestamps) is the ground truth that every secondary source — chunk analyses, visual observations, voice notes — is checked against. Contradicting or unsupported visual timestamps are dropped, and *Key quotes* are only kept if they appear verbatim in the transcript; quotes the model invented are removed. Built-in gates flag a report with warnings (e.g. chapters that cover less than half the video — a sign of a compressed/invented clock — or timestamps past the video's length) without letting them into the output.
   **Visual context (on by default)** — what the camera *shows* (demos, slides, on-screen text/UI, body language) that the audio never mentions. A 720p copy of the video is downloaded and **one frame per second of the whole video** is extracted in a single ffmpeg pass, with `mpdecimate`-style **dedupe** that drops barely-changed frames (so cost tracks *actual* screen change, never video length — with a `--dedupe-max-gap` guarantee that slow pans and progress bars are never lost). Every extracted frame gets a **burned-in on-screen timestamp stamp** (in the top-left corner) as a reference; the vision prompts explicitly tell the model this stamp is a tool-injected overlay belonging to the task, never video content, so it is never described or mistaken for on-screen UI. A vision model (`gemma4` locally, `gemma4:31b-cloud` on Ollama Cloud) then describes frames in three selectable modes: **`per-frame`** (every unique frame, full detail, parallel), **`batch`** (up to 8 frames per request for speed), or **`mosaic`** (adaptive grid montages, up to 144 cells, for a cheap overview). A second, **LLM-directed "doubt pass"** then reviews the transcript + voice notes + first-pass visuals and re-captures *exact* frames at the timestamps it flags as doubtful (e.g. refusals, blanks, UI the host references but nothing confirms), merging them back before summarization — so guesses never leak into the report. Descriptions are then compressed into ~30s window summaries so the entire video fits into the final report's context; every raw frame description (with `source: pass1|doubt` and `status` provenance) is kept in `visual_frames.json`. Disable with `--no-vision` or `--no-vision-doubts`.

   **Voice analysis (on by default)** — how the video *sounds*, straight from ffmpeg (no ML dependency): speaking pace (words per minute per subtitle segment), loudness (LUFS via `ebur128`), pauses (`silencedetect`) and energy bursts, plus a per-window narration of the speaker's delivery. The report gains a *"How it sounds"* section. Disable with `--no-voice`. Output goes to `voice_analysis.json`.
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

# Tune visual detail (default: 1 frame per second, whole video)
yt-transcribe "URL" --frame-interval 5 --max-frames 40 --vision-window 20

# Faster / cheaper visual pass: batch several frames per request
yt-transcribe "URL" --vision-mode batch --batch-size 8

# Cheapest possible overview of a long video: adaptive grid montages
yt-transcribe "URL" --vision-mode mosaic --target-grids 24

# Skip visual/frame analysis (audio + captions only)
yt-transcribe "URL" --no-vision

# Skip the second vision pass (keep only the first-pass scan)
yt-transcribe "URL" --no-vision-doubts

# Bound the doubt pass to fewer re-captured frames
yt-transcribe "URL" --vision-fill-budget 20

# Skip voice analysis (captions + visual only)
yt-transcribe "URL" --no-voice

# Coarser voice windows (fewer LLM calls for long videos)
yt-transcribe "URL" --voice-window 60

# Reverse the context digest into extra documents with a second LLM call
yt-transcribe "URL" --derive how-to,article,faq,checklist,quiz

# Use a different model just for the derived documents
yt-transcribe "URL" --derive how-to --derive-model gemma4

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
| `GET  /v1/results/{video_id}/{file}` | Download a saved file (`transcript.md`, `context.json`, `visual_1.png`, `doubt_0001.jpg`, ...) |
| `GET  /v1/health` | Health check |
| `GET  /v1/tools` | Metadata for the built-in analysis tools (vision, voice) |

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
| `visual_frames.json` | Raw frame / batch / grid descriptions (when vision is on; `mode` + `count` + `fill` fields, each frame tagged `source: pass1|doubt` and `status`) |
| `visual_NNNN.png` | Grid montages written in **mosaic** mode |
| `doubt_NNNN.jpg` | Exact-timestamp frames re-captured by the doubt pass |
| `voice_analysis.json` | Voice measurements: overall pace/energy stats + per-segment profiles (when voice is on) |
| `derived_<kind>.md` | Extra documents rewritten from the context digest by a second LLM call (when `--derive` is used): `how-to`, `article`, `faq`, `checklist`, `quiz` |

## Configuration

| Option | Default | Purpose |
| --- | --- | --- |
| `OLLAMA_API_KEY` | unset | Enables Ollama cloud analysis |
| `OLLAMA_BASE_URL` | cloud: `https://ollama.com`, local: `http://localhost:11434` | Ollama endpoint |
| `OLLAMA_MODEL` | cloud: `gpt-oss:20b-cloud`, local: `llama3.2` | Ollama model |
| `OLLAMA_VISION_MODEL` | cloud: `gemma4:31b-cloud`, local: `gemma4` | Vision model (overridden by `--vision-model`) |
| `YT_TRANSCRIBER_OUT` | `output` | Where results are stored |
| `--provider` | `auto` | `auto` uses cloud if an API key is set, else local |
| `--model` | cloud: `gpt-oss:20b-cloud`, local: `llama3.2` | Ollama model to use |
| `--whisper-model` | `base` | faster-whisper model size (transcription fallback) |
| `--device` | `cpu` | Whisper inference device (`cpu` or `cuda`) |
| `--language` | unset | Whisper language hint (e.g. `en`, `es`) |
| `--no-vad` | off | Disable voice-activity detection (music / ambient audio) |
| `--vision/--no-vision` | on | Analyze video frames so the report includes on-screen actions |
| `--vision-model` | `gemma4` (local) / `gemma4:31b-cloud` (cloud) | Vision model for frame analysis |
| `--vision-mode` | `per-frame` | `per-frame` (every frame, full detail), `batch` (≤8 frames/request, faster), or `mosaic` (grids, cheap overview) |
| `--frame-interval` | `1` | Seconds between extracted frames (1 = one frame per second) |
| `--max-frames` | `600` | Maximum frames analyzed (longer videos are down-sampled evenly) |
| `--vision-window` | `30` | Group frame descriptions into N-second summary windows |
| `--dedupe/--no-dedupe` | on | Drop frames that barely change; cost tracks screen change, not length |
| `--dedupe-max-gap` | `10` | Never skip more than N seconds of unchanged footage between kept frames |
| `--batch-size` | `8` | Frames per request in `batch` mode (8 max keeps timestamps reliable) |
| `--mosaic-cells` | auto | Cells per grid in `mosaic` mode (max 144, e.g. 12×12) |
| `--target-grids` | `24` | Target number of vision calls; drives auto grid sizing |
| `--vision-doubts/--no-vision-doubts` | on | Second vision pass: an LLM flags doubtful timestamps, exact frames are re-captured and merged |
| `--vision-fill-budget` | `60` | Maximum doubt frames the second vision pass may capture |
| `--voice/--no-voice` | on | Analyze speech delivery (pace, loudness, pauses) so the report includes a "How it sounds" section |
| `--voice-window` | `30` | Group voice measurements into N-second windows for LLM narration |
| `--derive` | off | Rewrite the context digest into extra documents (one LLM call each): `how-to`, `article`, `faq`, `checklist`, `quiz` |
| `--derive-model` | main model | Ollama model for the derived-document step only |
| `--skip-context` | off | Skip the AI context step (also disables vision & voice) |

### Vision models: free options

Everything runs on **Ollama**, so both local and cloud inference are free for
you. Point `--vision-model` (or `OLLAMA_VISION_MODEL`) at any of these:

| Model | Notes |
| --- | --- |
| `gemma4` / `gemma4:31b-cloud` | Default local / cloud pair |
| `qwen2.5-vl:7b` | Best at reading on-screen text/UI labels |
| `moondream2` | Tiny and fast, good for quick per-frame scans |
| `gemma3` | Balanced local general-purpose vision |
| `llama3.2-vision` | Reliable local general-purpose vision |

**Cost & rate limits.** Vision is priced by the pixels a model actually reads,
so the real cost levers are `--dedupe` (default on) and choosing `mosaic` over
`per-frame` — batching mostly buys wall-clock speed. The doubt pass costs one
plain-text call on the main context model plus at most `--vision-fill-budget`
re-described frames. Local inference is
unlimited but slower; the free Ollama **cloud** tier is rate-limited, and the
CLI/API/GUI already retry with backoff when that happens (429), degrading
gracefully instead of failing.

The API mirrors these via the `options` object in the JSON body.

## Analysis tools

The transcriber ships as a set of **tools**, each answering a different
question and producing its own outputs (metadata available via
`GET /v1/tools`):

| Tool | What it measures | Output |
| --- | --- | --- |
| **vision** | What the camera shows (frames → on-screen text/UI, scenes, actions) | `visual_frames.json`, `visual_*.png` |
| **voice** | How the video sounds (pace, loudness, pauses, energy bursts, delivery tone) | `voice_analysis.json` |

Both are on by default and independent: `--no-vision` keeps voice, and
vice-versa. They never break the transcript — an analysis failure just leaves a
warning in the log and the report loses that section.

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
- **Voice step skipped** — needs ffmpeg (for `ebur128`/`silencedetect`); the warning in the log explains why.
- **Context step skipped** — no API key and no local Ollama running; transcript still works.
- **Server won't start (port busy)** — `yt-transcribe serve --port 8001`.