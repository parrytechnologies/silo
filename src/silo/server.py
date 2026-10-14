"""The local web interface and JSON API."""

from __future__ import annotations

import json
import mimetypes
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from silo import __version__
from silo._offline import network_allowed
from silo.config import Settings
from silo.engine import open_engine
from silo.indexer import Indexer
from silo.llm import LLMError, egress, stream_answer
from silo.search import MODES, reading_order

WEB_DIR = Path(__file__).parent / "web"

# The interface is fully self-contained. This policy makes the browser enforce it:
# no script, style, font, image or request may go anywhere but this server.
CSP = (
    "default-src 'self'; img-src 'self' data:; style-src 'self'; script-src 'self'; "
    "font-src 'self'; connect-src 'self'; frame-src 'self'; object-src 'self'; "
    "base-uri 'none'; form-action 'self'; frame-ancestors 'self'"
)


class AskRequest(BaseModel):
    question: str
    mode: str = "hybrid"


def create_app(settings: Settings) -> FastAPI:
    engine = open_engine(settings)
    store, embedder, searcher = engine.store, engine.embedder, engine.searcher

    def _safe_warm() -> None:
        try:
            searcher.warm()
        except Exception:
            pass

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        threading.Thread(target=_safe_warm, daemon=True).start()
        yield

    app = FastAPI(title="SILO", version=__version__, docs_url=None, redoc_url=None,
                  openapi_url="/api/openapi.json", lifespan=lifespan)
    job_lock = threading.Lock()
    job: dict = {"running": False, "current": "", "done": 0, "total": 0, "last": None,
                 "error": ""}

    @app.middleware("http")
    async def security_headers(request, call_next):
        response = await call_next(request)
        response.headers["Content-Security-Policy"] = CSP
        if not request.url.path.startswith("/api/"):
            # Always revalidate the interface so an upgrade never serves stale assets.
            response.headers["Cache-Control"] = "no-cache"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        return response

    @app.get("/api/status")
    def status() -> dict:
        stats = store.stats()
        return {
            "version": __version__,
            "offline": not network_allowed(),
            "stats": stats,
            "roots": store.roots(),
            "index": {
                "path": str(settings.index_dir),
                "read_only": store.read_only,
                "store_id": store.get_meta("store_id"),
                "size_bytes": store.size_bytes(),
            },
            "models": {
                "embedding": settings.embedding_model,
                "reranker": settings.reranker_model if settings.rerank else None,
                "device": "cpu",
                "threads": settings.num_threads,
            },
            "llm": {
                "enabled": settings.llm_enabled,
                "model": settings.llm_model or None,
                "egress": egress(settings),
            },
            "job": job,
        }

    @app.get("/api/search")
    def search(q: str = Query("", max_length=2000), mode: str = "hybrid",
               k: int = Query(10, ge=1, le=50), rerank: bool | None = None) -> dict:
        if mode not in MODES:
            raise HTTPException(400, f"mode must be one of {', '.join(MODES)}")
        return searcher.search(q, k=k, mode=mode, rerank=rerank).to_dict()

    @app.get("/api/sources")
    def sources(q: str = Query("", max_length=2000), mode: str = "hybrid",
                k: int = Query(50, ge=1, le=50), rerank: bool | None = None) -> dict:
        if mode not in MODES:
            raise HTTPException(400, f"mode must be one of {', '.join(MODES)}")
        return searcher.sources(q, k=k, mode=mode, rerank=rerank).to_dict()

    @app.get("/api/documents")
    def documents() -> dict:
        keys = ("doc_id", "rel_path", "title", "suffix", "size", "pages", "chunks", "status",
                "error", "parse_seconds", "indexed_at")
        return {"documents": [
            {**{k: row[k] for k in keys}, "file_available": Path(row["path"]).is_file()}
            for row in store.documents()
        ]}

    @app.get("/api/documents/{doc_id}/file")
    def document_file(doc_id: str):
        # Only files in the catalogue can be served; the client never supplies a path.
        row = store.document(doc_id)
        if row is None:
            raise HTTPException(404, "document not found")
        if not Path(row["path"]).is_file():
            raise HTTPException(
                404, "The source file is not available on this machine. The index still "
                     "holds its text; use `silo store relocate` if the files have moved.")
        path = Path(row["path"])
        media = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        return FileResponse(path, media_type=media, content_disposition_type="inline",
                            filename=path.name)

    @app.get("/api/documents/{doc_id}/passages")
    def document_passages(doc_id: str, start: int = Query(0, ge=0),
                          count: int = Query(10, ge=1, le=50)) -> dict:
        row = store.document(doc_id)
        if row is None:
            raise HTTPException(404, "Unknown document.")
        keys = ("id", "headings", "page_start", "page_end", "text")
        return {
            "doc_id": doc_id, "rel_path": row["rel_path"], "title": row["title"],
            "total": row["chunks"],
            "passages": [{**{k: p[k] for k in keys}, "position": p["chunk_index"]}
                         for p in store.passages(doc_id, start, start + count - 1)],
        }

    @app.post("/api/ask")
    def ask(request: AskRequest):
        if not settings.llm_enabled:
            raise HTTPException(409, "No LLM is configured for this instance.")
        question = request.question.strip()
        if not question:
            raise HTTPException(400, "question is empty")
        result = reading_order(
            searcher.search(question, k=settings.llm_context_chunks, mode=request.mode)
        )

        def events():
            yield _sse("sources", result.to_dict())
            if not result.hits:
                yield _sse("token", {"text": "Nothing in the index matches that question."})
                yield _sse("done", {})
                return
            started = time.perf_counter()
            try:
                for token in stream_answer(settings, question, result.hits):
                    yield _sse("token", {"text": token})
            except LLMError as exc:
                yield _sse("error", {"message": str(exc)})
            yield _sse("done", {"seconds": round(time.perf_counter() - started, 1)})

        return StreamingResponse(events(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache"})

    @app.post("/api/index")
    def reindex():
        if store.read_only:
            raise HTTPException(403, "This index store is open read-only.")
        roots = store.roots()
        if not roots:
            raise HTTPException(409, "Nothing indexed yet. Run `silo index <folder>` first.")
        if not job_lock.acquire(blocking=False):
            return JSONResponse({"started": False, "reason": "already running"}, status_code=202)

        def run() -> None:
            job.update(running=True, current="", done=0, total=0, error="")
            try:
                indexer = Indexer(settings, store, embedder)
                summary = {"indexed": 0, "unchanged": 0, "failed": 0, "removed": 0, "empty": 0}

                def on_event(event, position, total):
                    job.update(current=Path(event.path).name, done=position, total=total)

                for root in roots:
                    if not Path(root).exists():
                        continue
                    report = indexer.index(Path(root), on_event=on_event)
                    for key in summary:
                        summary[key] += getattr(report, key)
                job["last"] = {**summary, "finished_at": time.time()}
            except Exception as exc:
                job["error"] = f"{type(exc).__name__}: {exc}"
            finally:
                job.update(running=False, current="")
                job_lock.release()

        threading.Thread(target=run, daemon=True).start()
        return {"started": True}

    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
    return app


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"
