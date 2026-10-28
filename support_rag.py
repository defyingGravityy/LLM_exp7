"""RAG pipeline from Experiment 6 (LangChain + FAISS + LangGraph),
with the LLM replaced by a pluggable backend (Ollama / cloud / mock)."""
import os
import logging
from typing import TypedDict, List

import requests
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langgraph.graph import StateGraph, START, END

load_dotenv()
logger = logging.getLogger("support_rag")

BACKEND = os.getenv("BACKEND", "ollama").lower()
MIN_SIMILARITY = float(os.getenv("MIN_SIMILARITY", "0.30"))


class BackendError(Exception):
    """LLM backend failed. Details are logged server-side, never returned."""


# ------------------------------------------------------------------
# LLM backends  (replaces the transformers `generator` pipeline)
# ------------------------------------------------------------------
def _call_ollama(prompt: str) -> str:
    url = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/") + "/api/chat"
    payload = {
        "model": os.getenv("OLLAMA_MODEL", "qwen2.5:0.5b"),
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": 0, "num_predict": 160},
    }
    try:
        r = requests.post(url, json=payload, timeout=120)
        r.raise_for_status()
        return r.json()["message"]["content"].strip()
    except Exception as e:
        logger.error("Ollama call failed: %s", type(e).__name__)
        raise BackendError("ollama failure") from e


def _call_cloud(prompt: str) -> str:
    base, key, model = (os.getenv("CLOUD_BASE_URL"), os.getenv("CLOUD_API_KEY"),
                        os.getenv("CLOUD_MODEL"))
    if not (base and key and model):
        logger.error("Cloud backend is not fully configured")
        raise BackendError("cloud not configured")
    try:
        r = requests.post(
            base.rstrip("/") + "/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={"model": model, "temperature": 0, "max_tokens": 300,
                  "messages": [{"role": "user", "content": prompt}]},
            timeout=60,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logger.error("Cloud call failed: %s", type(e).__name__)  # type only: never log headers/key
        raise BackendError("cloud failure") from e


def _call_mock(prompt: str) -> str:
    # Echoes the first FAQ chunk: only for testing the API structure.
    ctx = prompt.split("FAQ CONTEXT:\n", 1)[1].split("\n\nCUSTOMER QUESTION:", 1)[0]
    return "(mock) " + ctx.split("\n\n")[0]


def generate(prompt: str) -> str:
    if BACKEND == "ollama":
        return _call_ollama(prompt)
    if BACKEND == "cloud":
        return _call_cloud(prompt)
    return _call_mock(prompt)


# ------------------------------------------------------------------
# Knowledge base, chunks, embeddings, FAISS  -> built ONCE at startup
# ------------------------------------------------------------------
_faq_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "support_faq.txt")
with open(_faq_path, "r", encoding="utf-8") as f:
    support_faq = f.read()

faq_doc = Document(page_content=support_faq, metadata={"source": "support_faq.txt"})
chunks = RecursiveCharacterTextSplitter(chunk_size=450, chunk_overlap=60).split_documents([faq_doc])

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={"normalize_embeddings": True},   # lets us turn L2 distance into cosine similarity
)
vector_store = FAISS.from_documents(chunks, embedding_model)
logger.info("Index ready: %d chunks, backend=%s", len(chunks), BACKEND)


# ------------------------------------------------------------------
# LangGraph workflow (your Experiment 6 nodes)
# ------------------------------------------------------------------
class SupportState(TypedDict, total=False):
    question: str
    intent: str
    search_query: str
    context: str
    sources: List[str]
    top_score: float
    draft: str
    retry_count: int
    final_answer: str


FALLBACK_TEXT = (
    "I couldn't find a reliable answer in the available FAQ. "
    "Please contact the support team through the Help section. "
    "Do not share your password, one-time code, or full payment-card details."
)


def classify_intent(state: SupportState):
    q = state["question"].lower()
    if any(w in q for w in ["refund", "return", "exchange"]):
        intent = "returns_refunds"
    elif any(w in q for w in ["track", "tracking", "shipment", "delivery", "order status"]):
        intent = "order_tracking"
    elif any(w in q for w in ["password", "sign in", "login", "account", "verification"]):
        intent = "account_access"
    elif any(w in q for w in ["charged", "payment", "billing", "invoice", "card"]):
        intent = "billing_payments"
    elif any(w in q for w in ["website", "error", "loading", "browser", "technical"]):
        intent = "technical_support"
    else:
        intent = "general_support"
    return {"intent": intent, "search_query": state["question"],
            "retry_count": state.get("retry_count", 0)}


def retrieve_context(state: SupportState):
    results = vector_store.similarity_search_with_score(state["search_query"], k=3)
    # FAISS returns L2 distance; for unit vectors cosine = 1 - d^2 / 2
    scored = [(d, 1 - float(dist) ** 2 / 2) for d, dist in results]
    top_score = max((s for _, s in scored), default=0.0)
    logger.info("retrieval top_similarity=%.3f (cut-off %.2f)", top_score, MIN_SIMILARITY)
    return {
        "context": "\n\n".join(d.page_content for d, _ in scored),
        "sources": [d.page_content.split("\n", 1)[0].strip() for d, _ in scored],
        "top_score": top_score,
    }


def route_after_retrieve(state: SupportState):
    # Nothing in the FAQ is close to the question -> don't let a small LLM guess
    return "draft" if state.get("top_score", 0.0) >= MIN_SIMILARITY else "fallback"


def draft_answer(state: SupportState):
    prompt = f"""You are a polite customer-support assistant.
Use only the FAQ CONTEXT to answer the CUSTOMER QUESTION.
Do not invent order status, refund approvals, timelines, or company policies.
If the context does not contain the answer, output exactly:
INSUFFICIENT_CONTEXT
Do not ask for passwords, one-time codes, or full payment-card details.
Keep the answer clear and concise.

FAQ CONTEXT:
{state.get("context", "")}

CUSTOMER QUESTION:
{state["question"]}

ANSWER:"""
    return {"draft": generate(prompt)}


def review_answer(state: SupportState):
    draft = state.get("draft", "").strip()
    if "INSUFFICIENT_CONTEXT" in draft:
        return {"final_answer": ""}
    return {"final_answer": draft}


def route_after_review(state: SupportState):
    if state.get("final_answer"):
        return "finish"
    if state.get("retry_count", 0) < 1:
        return "retry"
    return "fallback"


def rewrite_query(state: SupportState):
    category_terms = {
        "returns_refunds": "return eligibility refund process",
        "order_tracking": "order dispatch tracking shipment status",
        "account_access": "account sign in password reset",
        "billing_payments": "billing payment transaction",
        "technical_support": "website technical troubleshooting",
        "general_support": "customer support help contact",
    }
    expanded = state["question"] + " " + category_terms.get(
        state.get("intent", "general_support"), "customer support")
    return {"search_query": expanded, "retry_count": state.get("retry_count", 0) + 1}


def fallback_answer(state: SupportState):
    return {"final_answer": FALLBACK_TEXT}


_g = StateGraph(SupportState)
_g.add_node("classify_intent", classify_intent)
_g.add_node("retrieve", retrieve_context)
_g.add_node("draft_answer", draft_answer)
_g.add_node("review_answer", review_answer)
_g.add_node("rewrite_query", rewrite_query)
_g.add_node("fallback", fallback_answer)

_g.add_edge(START, "classify_intent")
_g.add_edge("classify_intent", "retrieve")
_g.add_conditional_edges("retrieve", route_after_retrieve,
                         {"draft": "draft_answer", "fallback": "fallback"})
_g.add_edge("draft_answer", "review_answer")
_g.add_conditional_edges("review_answer", route_after_review,
                         {"finish": END, "retry": "rewrite_query", "fallback": "fallback"})
_g.add_edge("rewrite_query", "retrieve")
_g.add_edge("fallback", END)
support_graph = _g.compile()


# ------------------------------------------------------------------
# Public function used by the API
# ------------------------------------------------------------------
def answer_question(question: str) -> dict:
    r = support_graph.invoke({"question": question, "retry_count": 0})
    answer = r.get("final_answer", FALLBACK_TEXT)
    grounded = answer != FALLBACK_TEXT
    return {
        "answer": answer,
        "intent": r.get("intent", "unknown"),
        "grounded": grounded,
        "sources": sorted(set(r.get("sources", []))) if grounded else [],
        "retries": r.get("retry_count", 0),
    }
