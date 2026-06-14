import time
from core.vector_store import delete_collection
import threading
from typing import Optional

from pipeline import run_pipeline


# job_id -> {"status": ..., "error": ..., "result": {...}}
_jobs: dict = {}
# job_id -> rag_chain (kept separate, not JSON-serializable)
_rag_chains: dict = {}

_lock = threading.Lock()


JOB_TTL_SECONDS = 60 * 60       # delete jobs older than 1 hour
SWEEP_INTERVAL_SECONDS = 15 * 60  # check every 15 minutes

def create_job(job_id: str):
    with _lock:
        _jobs[job_id] = {
            "status": "pending",
            "error": None,
            "result": None,
            "created_at": time.time(),  
        }

def get_job(job_id: str) -> Optional[dict]:
    with _lock:
        return _jobs.get(job_id)


def get_rag_chain(job_id: str):
    with _lock:
        return _rag_chains.get(job_id)
    
def delete_job(job_id: str):
    """Deletes a job's vector store collection and removes it from memory."""
    with _lock:
        _jobs.pop(job_id, None)
        _rag_chains.pop(job_id, None)
    delete_collection(job_id)


def run_job(job_id: str, source: str, content_type: str, language: str):
    """
    Runs the pipeline synchronously in a background thread.
    Updates the job store with progress/result/error.
    """
    with _lock:
        _jobs[job_id]["status"] = "processing"

    try:
        result = run_pipeline(source, content_type, language)

        with _lock:
            _jobs[job_id]["status"] = "completed"
            _jobs[job_id]["result"] = {
                "job_id": result["job_id"],
                "title": result["title"],
                "summary": result["summary"],
                "content_type": result["content_type"],
                "insights": result["insights"],
                "transcript": result["transcript"],
            }
            _rag_chains[job_id] = result["rag_chain"]

    except Exception as e:
        with _lock:
            _jobs[job_id]["status"] = "failed"
            _jobs[job_id]["error"] = str(e)

def _sweep_old_jobs():
    while True:
        time.sleep(SWEEP_INTERVAL_SECONDS)
        now = time.time()
        with _lock:
            stale = [
                jid for jid, j in _jobs.items()
                if now - j.get("created_at", now) > JOB_TTL_SECONDS
            ]
        for jid in stale:
            print(f"🧹 Sweeping stale job {jid}")
            delete_job(jid)


def start_cleanup_thread():
    thread = threading.Thread(target=_sweep_old_jobs, daemon=True)
    thread.start()