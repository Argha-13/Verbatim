import uuid
import os
import tempfile

from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import (
    ProcessYouTubeRequest,
    JobStatus,
    JobResultResponse,
    ChatRequest,
    ChatResponse,
    ALLOWED_CONTENT_TYPES,
    ALLOWED_LANGUAGES,
)
from app.jobs import create_job, get_job, get_rag_chain, run_job, delete_job, start_cleanup_thread
from core.rag_engine import ask_question


app = FastAPI(title="AI Video/Meeting Assistant API")

# Allow a separate frontend (e.g. React/Streamlit) to call this API during dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def _on_startup():
    start_cleanup_thread()

@app.post("/process/youtube", response_model=JobStatus, status_code=202)
def process_youtube(request: ProcessYouTubeRequest, background_tasks: BackgroundTasks):
    """Kick off processing for a YouTube URL. Returns immediately with a job_id."""
    job_id = uuid.uuid4().hex
    create_job(job_id)

    background_tasks.add_task(
        run_job, job_id, request.url, request.content_type, request.language
    )

    return JobStatus(job_id=job_id, status="pending")


@app.post("/process/upload", response_model=JobStatus, status_code=202)
def process_upload(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    content_type: str = Form(default="meeting"),
    language: str = Form(default="english"),
):
    """Kick off processing for an uploaded audio/video file."""
    content_type = content_type.lower().strip()
    language = language.lower().strip()

    if content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=422, detail=f"content_type must be one of {ALLOWED_CONTENT_TYPES}")
    if language not in ALLOWED_LANGUAGES:
        raise HTTPException(status_code=422, detail=f"language must be one of {ALLOWED_LANGUAGES}")

    # Save uploaded file to a temp path so process_input() can read it
    suffix = os.path.splitext(file.filename or "")[1] or ".tmp"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file.file.read())
        tmp_path = tmp.name

    job_id = uuid.uuid4().hex
    create_job(job_id)

    background_tasks.add_task(run_job, job_id, tmp_path, content_type, language)

    return JobStatus(job_id=job_id, status="pending")


@app.get("/status/{job_id}", response_model=JobStatus)
def get_status(job_id: str):
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    return JobStatus(job_id=job_id, status=job["status"], error=job["error"])


@app.get("/result/{job_id}", response_model=JobResultResponse)
def get_result(job_id: str):
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    if job["status"] == "failed":
        raise HTTPException(status_code=500, detail=job["error"])

    if job["status"] != "completed":
        raise HTTPException(status_code=409, detail=f"Job is still '{job['status']}', not ready yet")

    return JobResultResponse(**job["result"])


@app.post("/chat/{job_id}", response_model=ChatResponse)
def chat(job_id: str, request: ChatRequest):
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    if job["status"] != "completed":
        raise HTTPException(status_code=409, detail=f"Job is '{job['status']}', cannot chat yet")

    rag_chain = get_rag_chain(job_id)
    if rag_chain is None:
        raise HTTPException(status_code=500, detail="RAG chain not available for this job")

    try:
        answer = ask_question(rag_chain, request.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to answer: {e}")

    return ChatResponse(job_id=job_id, question=request.question, answer=answer)


@app.get("/")
def root():
    return {"message": "AI Video/Meeting Assistant API is running"}

@app.delete("/job/{job_id}", status_code=204)
def delete_job_route(job_id: str):
    if get_job(job_id) is None:
        raise HTTPException(status_code=404, detail="Job not found")
    delete_job(job_id)