"""
main.py
=======

FastAPI service around RAGPipeline.

Endpoints
---------
    GET  /health            liveness + corpus stats (no auth)
    GET  /sources           what has been ingested
    POST /ingest            multipart: files=<pdf>... and/or urls="url1\nurl2"
    POST /chat              JSON body; streams Server-Sent Events (or JSON if stream=false)

SSE event types from /chat:
    sources -> {"sources": [...], "standalone_question": "..."}
    token   -> {"text": "..."}            (many)
    done    -> {"cached": bool}
    error   -> {"detail": "..."}

Run ONE worker (the index is in-process memory):
    uvicorn main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal, Protocol

from cachetools import TTLCache
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from rag_pipeline import RAGPipeline

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("rag.api")

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
API_KEY = os.getenv("API_KEY", "")                      # empty = auth disabled (dev only)
REDIS_URL = os.getenv("REDIS_URL", "")                  # empty = in-memory cache
CACHE_TTL = int(os.getenv("CACHE_TTL_SECONDS", "3600"))
CACHE_MAX = int(os.getenv("CACHE_MAX_ENTRIES", "1024"))
RATE_DEFAULT = os.getenv("RATE_LIMIT_DEFAULT", "100/minute")
RATE_CHAT = os.getenv("RATE_LIMIT_CHAT", "20/minute")
RATE_INGEST = os.getenv("RATE_LIMIT_INGEST", "5/minute")
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "25"))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
MAX_URLS_PER_REQUEST = int(os.getenv("MAX_URLS_PER_REQUEST", "10"))
HISTORY_IN_CACHE_KEY = 8

# Behind a reverse proxy, get_remote_address sees the proxy IP. Put the proxy's
# X-Forwarded-For handling in front (e.g. uvicorn --proxy-headers) or swap key_func.
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[RATE_DEFAULT],
    storage_uri=REDIS_URL or "memory://",
)


# ----------------------------------------------------------------------------
# Cache layer (Redis if configured, otherwise in-memory TTL)
# ----------------------------------------------------------------------------
class Cache(Protocol):
    async def get(self, key: str) -> dict[str, Any] | None: ...
    async def set(self, key: str, value: dict[str, Any]) -> None: ...
    async def close(self) -> None: ...


class MemoryCache:
    def __init__(self, maxsize: int, ttl: int) -> None:
        self._store: TTLCache[str, dict[str, Any]] = TTLCache(maxsize=maxsize, ttl=ttl)

    async def get(self, key: str) -> dict[str, Any] | None:
        return self._store.get(key)

    async def set(self, key: str, value: dict[str, Any]) -> None:
        self._store[key] = value

    async def close(self) -> None:
        self._store.clear()


class RedisCache:
    """Cache failures must never break chat: every error degrades to a cache miss."""

    def __init__(self, url: str, ttl: int) -> None:
        import redis.asyncio as aioredis

        self._r = aioredis.from_url(url, decode_responses=True)
        self._ttl = ttl

    async def get(self, key: str) -> dict[str, Any] | None:
        try:
            raw = await self._r.get(key)
            return json.loads(raw) if raw else None
        except Exception as exc:
            logger.warning("Redis get failed: %s", exc)
            return None

    async def set(self, key: str, value: dict[str, Any]) -> None:
        try:
            await self._r.set(key, json.dumps(value), ex=self._ttl)
        except Exception as exc:
            logger.warning("Redis set failed: %s", exc)

    async def close(self) -> None:
        await self._r.aclose()


def make_cache_key(version: str, question: str, history: list["ChatMessage"]) -> str:
    blob = json.dumps(
        {
            "v": version,  # corpus fingerprint: ingesting new docs invalidates old answers
            "q": " ".join(question.lower().split()),
            "h": [[m.role, m.content] for m in history[-HISTORY_IN_CACHE_KEY:]],
        },
        sort_keys=True,
    )
    return "rag:chat:" + hashlib.sha256(blob.encode()).hexdigest()


# ----------------------------------------------------------------------------
# App lifecycle
# ----------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Model loading is blocking and slow; keep it off the event loop.
    app.state.pipeline = await asyncio.to_thread(RAGPipeline)
    app.state.cache = RedisCache(REDIS_URL, CACHE_TTL) if REDIS_URL else MemoryCache(CACHE_MAX, CACHE_TTL)
    logger.info(
        "Ready: %d chunks, cache=%s, auth=%s",
        app.state.pipeline.chunk_count,
        "redis" if REDIS_URL else "memory",
        "on" if API_KEY else "OFF",
    )
    yield
    await app.state.cache.close()


app = FastAPI(title="Multi-Document RAG API", version="1.0.0", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


async def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    if not API_KEY:
        return
    if not x_api_key or not hmac.compare_digest(x_api_key, API_KEY):
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key header.")


# ----------------------------------------------------------------------------
# Schemas
# ----------------------------------------------------------------------------
class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=8000)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    history: list[ChatMessage] = Field(default_factory=list, max_length=40)
    stream: bool = True


def sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


# ----------------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------------
STATIC_DIR = Path(__file__).parent / "static"


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    """Built-in web UI (single HTML file). Open http://localhost:8000 after starting uvicorn."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
async def health(request: Request) -> dict[str, Any]:
    p: RAGPipeline = request.app.state.pipeline
    return {"status": "ok", "chunks": p.chunk_count, "corpus_version": p.version}


@app.get("/sources", dependencies=[Depends(require_api_key)])
@limiter.limit(RATE_DEFAULT)
async def sources(request: Request) -> dict[str, Any]:
    return {"sources": request.app.state.pipeline.list_sources()}


async def _save_pdf_upload(upload: UploadFile) -> str:
    """Stream an upload to a temp file with a hard size cap and a magic-byte check."""
    fd, path = tempfile.mkstemp(suffix=".pdf")
    size = 0
    try:
        with os.fdopen(fd, "wb") as out:
            while chunk := await upload.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise ValueError(f"File exceeds the {MAX_UPLOAD_MB} MB limit.")
                out.write(chunk)
        with open(path, "rb") as f:
            if f.read(5) != b"%PDF-":
                raise ValueError("File is not a valid PDF.")
        return path
    except Exception:
        if os.path.exists(path):
            os.unlink(path)
        raise


@app.post("/ingest", dependencies=[Depends(require_api_key)])
@limiter.limit(RATE_INGEST)
async def ingest(
    request: Request,
    files: list[UploadFile] = File(default=[]),
    urls: str = Form(default=""),
) -> JSONResponse:
    pipeline: RAGPipeline = request.app.state.pipeline
    url_list = [u.strip() for u in urls.replace(",", "\n").splitlines() if u.strip()]
    if not files and not url_list:
        raise HTTPException(status_code=400, detail="Provide at least one PDF file or URL.")
    if len(url_list) > MAX_URLS_PER_REQUEST:
        raise HTTPException(status_code=400, detail=f"Max {MAX_URLS_PER_REQUEST} URLs per request.")

    results: list[dict[str, Any]] = []

    for upload in files:
        name = os.path.basename(upload.filename or "upload.pdf")
        tmp_path = None
        try:
            tmp_path = await _save_pdf_upload(upload)
            added = await asyncio.to_thread(pipeline.ingest_pdf, tmp_path, name)
            results.append({"source": name, "status": "ok", "chunks_added": added})
        except Exception as exc:
            logger.warning("PDF ingest failed for %s: %s", name, exc)
            results.append({"source": name, "status": "error", "detail": str(exc)})
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    for url in url_list:
        try:
            added = await asyncio.to_thread(pipeline.ingest_url, url)
            results.append({"source": url, "status": "ok", "chunks_added": added})
        except Exception as exc:
            logger.warning("URL ingest failed for %s: %s", url, exc)
            results.append({"source": url, "status": "error", "detail": str(exc)})

    ok = any(r["status"] == "ok" for r in results)
    return JSONResponse(
        status_code=200 if ok else 400,
        content={"results": results, "total_chunks": pipeline.chunk_count},
    )


@app.post("/chat", dependencies=[Depends(require_api_key)])
@limiter.limit(RATE_CHAT)
async def chat(request: Request, payload: ChatRequest):
    pipeline: RAGPipeline = request.app.state.pipeline
    cache: Cache = request.app.state.cache
    key = make_cache_key(pipeline.version, payload.question, payload.history)
    cached = await cache.get(key)
    history = [m.model_dump() for m in payload.history]

    # ---------------- non-streaming JSON mode ----------------
    if not payload.stream:
        if cached:
            return {**cached, "cached": True}
        answer_parts: list[str] = []
        sources: list[dict[str, Any]] = []
        try:
            async for ev in pipeline.astream_answer(payload.question, history):
                if ev["type"] == "sources":
                    sources = ev["sources"]
                else:
                    answer_parts.append(ev["text"])
        except Exception as exc:
            logger.exception("chat failed")
            raise HTTPException(status_code=502, detail=f"Generation failed: {exc}") from exc
        result = {"answer": "".join(answer_parts), "sources": sources}
        if sources:
            await cache.set(key, result)
        return {**result, "cached": False}

    # ---------------- SSE streaming mode ----------------
    async def event_stream():
        if cached:
            yield sse("sources", {"sources": cached["sources"]})
            text = cached["answer"]
            for i in range(0, len(text), 24):  # replay cached answer in small slices
                yield sse("token", {"text": text[i : i + 24]})
            yield sse("done", {"cached": True})
            return

        parts: list[str] = []
        sources: list[dict[str, Any]] = []
        try:
            async for ev in pipeline.astream_answer(payload.question, history):
                if ev["type"] == "sources":
                    sources = ev["sources"]
                    yield sse(
                        "sources",
                        {"sources": sources, "standalone_question": ev["standalone_question"]},
                    )
                else:
                    parts.append(ev["text"])
                    yield sse("token", {"text": ev["text"]})
        except asyncio.CancelledError:
            raise  # client disconnected; do not cache a partial answer
        except Exception as exc:
            logger.exception("streaming chat failed")
            yield sse("error", {"detail": f"Generation failed: {exc}"})
            return

        if sources and parts:
            await cache.set(key, {"answer": "".join(parts), "sources": sources})
        yield sse("done", {"cached": False})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), workers=1)
