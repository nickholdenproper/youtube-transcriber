"""REST API + local web GUI for yt-transcriber.

Runs the same :mod:`yt_transcriber.pipeline` used by the CLI. Designed as a
building block for other tools (news sites, editors, RSS readers...):

* ``POST /v1/transcribe`` - synchronous, blocks until the result is ready.
* ``POST /v1/jobs``      - asynchronous, returns a job id to poll.
* ``GET  /v1/jobs/{id}`` - job status + result.
* ``GET  /v1/results``   - previously produced outputs.
* ``GET  /v1/results/{video_id}/{file}`` - download a saved file.

The web GUI lives at ``/`` and calls exactly these endpoints.
"""

from __future__ import annotations

import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .pipeline import PipelineOptions, PipelineResult, run_pipeline
from .tools import list_tools

STATIC_DIR = Path(__file__).parent / "static"
OUTPUT_DIR = Path(os.getenv("YT_TRANSCRIBER_OUT", "output"))
ALLOWED_FILES = {
    "transcript.md",
    "transcript.json",
    "context.md",
    "context.json",
    "visual_frames.json",
    "voice_analysis.json",
}


def _is_allowed(filename: str) -> bool:
    if filename in ALLOWED_FILES:
        return True
    return (
        filename.startswith("visual_") and filename.endswith(".png")
    ) or filename.startswith("doubt_") and filename.endswith(".jpg")
MAX_CONCURRENT_JOBS = 2

app = FastAPI(
    title="YT Transcriber API",
    version="0.4.0",
    description="Transcribe YouTube videos and get an AI context report of what happened.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PipelineOptionsModel(BaseModel):
    out: str = "output"
    skip_context: bool = False
    provider: str = "auto"
    model: Optional[str] = None
    whisper_model: str = "base"
    device: str = "cpu"
    force_whisper: bool = False
    language: Optional[str] = None
    no_vad: bool = False
    vision: bool = True
    vision_model: Optional[str] = None
    vision_mode: str = "per-frame"
    frame_interval: float = 1
    max_frames: int = 600
    vision_window: float = 30
    dedupe: bool = True
    dedupe_max_gap: float = 10
    batch_size: int = 8
    mosaic_cells: Optional[int] = None
    target_grids: int = 24
    voice: bool = True
    voice_window: float = 30
    vision_doubt: bool = True
    vision_fill_budget: int = 60
    derive: List[str] = Field(
        default_factory=list,
        description="Rewrite the context digest into extra documents via a second "
        "LLM call: how-to, article, faq, checklist, quiz",
    )
    derive_model: Optional[str] = Field(
        default=None, description="Ollama model for the derived-document step"
    )


class TranscribeRequest(BaseModel):
    url: str = Field(..., description="YouTube video URL to transcribe")
    options: PipelineOptionsModel = Field(default_factory=PipelineOptionsModel)


class Job:
    def __init__(self, url: str, opts: PipelineOptions):
        self.id = uuid.uuid4().hex[:8]
        self.url = url
        self.opts = opts
        self.status = "pending"  # pending / running / done / error
        self.created = datetime.now(timezone.utc).isoformat()
        self.progress: float = 0.0
        self.stage: str = "Queued"
        self.log: list[str] = []
        self.result: Optional[dict] = None
        self.error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "url": self.url,
            "status": self.status,
            "created": self.created,
            "progress": self.progress,
            "stage": self.stage,
            "log": list(self.log),
            "result": self.result,
            "error": self.error,
        }


_jobs: dict[str, Job] = {}
_jobs_lock = threading.Lock()
_jobs_slots = threading.Semaphore(MAX_CONCURRENT_JOBS)


def _options_from_model(m: PipelineOptionsModel) -> PipelineOptions:
    return PipelineOptions(**m.model_dump())


def _result_to_dict(res: PipelineResult) -> dict:
    return {
        "video_id": res.video_id,
        "meta": {
            "id": res.meta.id,
            "title": res.meta.title,
            "channel": res.meta.channel,
            "url": res.meta.url,
            "duration": res.meta.duration,
        },
        "source": res.source,
        "segments": res.segments,
        "context_report": res.context_report,
        "context_chunks": res.context_chunks,
        "visual_timeline": res.visual_timeline,
        "visual_frames": res.visual_frames,
        "voice_notes": res.voice_notes,
        "voice_analysis": res.voice_analysis,
        "vision_fill": res.vision_fill,
        "derived": res.derived,
        "files": [
            {
                "name": path.name,
                "url": f"/v1/results/{res.video_id}/{path.name}",
            }
            for path in res.files
        ],
        "warnings": res.warnings,
    }


def _run_job(job: Job) -> None:
    def on_log(msg: str) -> None:
        job.log.append(msg)

    def on_progress(pct: float, stage: str) -> None:
        job.progress = round(pct)
        job.stage = stage

    _jobs_slots.acquire()  # blocks until a slot is free
    try:
        job.status = "running"
        result = run_pipeline(job.url, job.opts, log=on_log, on_progress=on_progress)
        job.result = _result_to_dict(result)
        job.status = "done"
    except Exception as exc:  # noqa: BLE001 - job boundary
        job.error = str(exc)
        job.status = "error"
    finally:
        _jobs_slots.release()


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/v1/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/v1/transcribe", summary="Transcribe a video (synchronous)")
def transcribe_sync(req: TranscribeRequest) -> dict:
    try:
        result = run_pipeline(req.url, _options_from_model(req.options))
    except Exception as exc:  # noqa: BLE001 - surface as 500
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return _result_to_dict(result)


@app.post("/v1/jobs", summary="Start a transcription job (asynchronous)")
def create_job(req: TranscribeRequest) -> dict:
    job = Job(req.url, _options_from_model(req.options))
    with _jobs_lock:
        _jobs[job.id] = job
    threading.Thread(target=_run_job, args=(job,), daemon=True).start()
    return {"job_id": job.id, "status": job.status}


@app.get("/v1/tools", summary="List available analysis tools")
def tools() -> list[dict]:
    return list_tools()


@app.get("/v1/jobs/{job_id}", summary="Get job status and result")
def get_job(job_id: str) -> dict:
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job.to_dict()


@app.get("/v1/results", summary="List previously produced results")
def list_results() -> list[dict]:
    results = []
    if OUTPUT_DIR.is_dir():
        for video_dir in OUTPUT_DIR.iterdir():
            if not video_dir.is_dir():
                continue
            meta_file = video_dir / "transcript.json"
            if not meta_file.exists():
                continue
            import json

            try:
                payload = json.loads(meta_file.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                continue
            meta = payload.get("meta", {})
            results.append(
                {
                    "video_id": video_dir.name,
                    "title": meta.get("title"),
                    "channel": meta.get("channel"),
                    "date": meta.get("upload_date"),
                    "files": [
                        {"name": f, "url": f"/v1/results/{video_dir.name}/{f}"}
                        for f in sorted(
                            p.name for p in video_dir.iterdir() if _is_allowed(p.name)
                        )
                    ],
                }
            )
    results.sort(key=lambda r: r["video_id"])
    return results


@app.get("/v1/results/{video_id}/{filename}", summary="Download a saved output file")
def get_result_file(video_id: str, filename: str) -> FileResponse:
    if not _is_allowed(filename):
        raise HTTPException(status_code=400, detail="Unrecognised filename")
    path = OUTPUT_DIR / video_id / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"{filename} not found for {video_id}")
    return FileResponse(path, filename=filename)


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Programmatic entry point for tests / other tooling."""
    import uvicorn

    uvicorn.run(app, host=host, port=port)