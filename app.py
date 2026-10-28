import time
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

import support_rag
from support_rag import BackendError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("app.log")],
)
logger = logging.getLogger("api")

app = FastAPI(title="Customer Support RAG API", version="1.0")


class AskRequest(BaseModel):
    question: str = Field(..., max_length=1000)

    @field_validator("question")
    @classmethod
    def not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("question must not be empty")
        return v


class AskResponse(BaseModel):
    answer: str
    intent: str
    grounded: bool
    sources: list[str]
    retries: int
    latency_ms: float


@app.get("/health")
def health():
    # Only shows the process is up - NOT that the LLM backend is reachable.
    return {"status": "ok", "backend": support_rag.BACKEND}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    start = time.perf_counter()
    try:
        result = support_rag.answer_question(req.question)
    except BackendError:
        ms = (time.perf_counter() - start) * 1000
        logger.error("ask FAILED backend_error latency_ms=%.1f q_len=%d", ms, len(req.question))
        raise HTTPException(status_code=503, detail="The service is temporarily unavailable. Please try again later.")
    except Exception:
        ms = (time.perf_counter() - start) * 1000
        logger.exception("ask FAILED unexpected latency_ms=%.1f", ms)
        raise HTTPException(status_code=500, detail="Internal server error.")
    ms = (time.perf_counter() - start) * 1000
    logger.info("ask OK grounded=%s latency_ms=%.1f q_len=%d", result["grounded"], ms, len(req.question))
    return AskResponse(**result, latency_ms=round(ms, 1))
