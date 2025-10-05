# -*- coding: utf-8 -*-
"""
RAG (compatible with your ingest.py + Qdrant)

- Uses langchain_qdrant.QdrantVectorStore with the SAME configuration as ingest:
  vector_name="text", content_payload_key="page_content", metadata_payload_key="metadata".
- L2-normalized embeddings wrapper (inherits from Embeddings) for COSINE metric.
- Retrieval:
    * TOP_K=5 "Normal" chunks (metadata.tipo == "Normal")
    * Up to MAX_SUMMARIES "summary" items (metadata.tipo == "summary") as a separate 'file_summary'
- Prints used chunks (distance + full content) and used summaries in server logs.
- LLM (Ollama) with temperature=0.2.
- Gradio UI mounted at "/", API endpoints /health and POST /chat.
- CLI "chat" mode preserved for quick debugging.

Run:
    python rag_app.py chat      # CLI chat
    python rag_app.py server    # HTTP server + Gradio UI at /
"""

from __future__ import annotations

import os
import sys
import textwrap
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

from dotenv import load_dotenv, find_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import gradio as gr
from fastapi.middleware.cors import CORSMiddleware

from langchain.schema import Document
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_qdrant import QdrantVectorStore
from langchain_core.embeddings import Embeddings

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

# ---------------- Config (aligned with ingest.py) ----------------
load_dotenv(find_dotenv())

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11435")
QDRANT_URL      = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY  = os.getenv("QDRANT_API_KEY") or None
COLLECTION      = os.getenv("QDRANT_COLLECTION", "docs_pt")
EMBED_MODEL     = os.getenv("EMBED_MODEL", "snowflake-arctic-embed2:568m")

LLM_MODEL       = os.getenv("LLM_MODEL", "llama3.3:70b-instruct-q2_K")
LLM_NUM_CTX     = int(os.getenv("LLM_NUM_CTX", "4096"))
LLM_KEEP_ALIVE  = os.getenv("LLM_KEEP_ALIVE", "10m")

TOP_K           = int(os.getenv("TOP_K", "5"))
TEMPERATURE     = float(os.getenv("LLM_TEMPERATURE", "0.2"))
MAX_SUMMARIES   = int(os.getenv("MAX_SUMMARIES", "2"))

# ---------------- L2-normalized Embeddings wrapper ----------------
def _l2_normalize(vec: List[float]) -> List[float]:
    import math
    n = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / n for x in vec]

class NormalizedEmbeddings(Embeddings):
    """
    Embeddings wrapper that L2-normalizes outputs (best for COSINE metric with Qdrant).
    IMPORTANT: inherits from langchain_core.embeddings.Embeddings (required by langchain_qdrant).
    """
    def __init__(self, base_embeddings: OllamaEmbeddings):
        self.base = base_embeddings

    def embed_query(self, text: str) -> List[float]:
        return _l2_normalize(self.base.embed_query(text))

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [_l2_normalize(v) for v in self.base.embed_documents(texts)]

# ---------------- Singletons ----------------
_VS: Optional[QdrantVectorStore] = None
_LLM: Optional[ChatOllama] = None

def _vectorstore() -> QdrantVectorStore:
    """VectorStore configured exactly like ingest.py (same payload keys and named vector 'text')."""
    global _VS
    if _VS is not None:
        return _VS

    qclient = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=30)
    base_emb = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_BASE_URL)
    embeddings = NormalizedEmbeddings(base_emb)

    _VS = QdrantVectorStore(
        client=qclient,
        collection_name=COLLECTION,
        embedding=embeddings,               # Embeddings subclass
        vector_name="text",                 # named vector present in your collection
        content_payload_key="page_content", # same as ingest
        metadata_payload_key="metadata",    # same as ingest
    )
    print(f"[QDRANT] OK: collection='{COLLECTION}', vector='text'")
    return _VS

def _llm() -> ChatOllama:
    """LLM client (temperature=0.2)."""
    global _LLM
    if _LLM is None:
        _LLM = ChatOllama(
            model=LLM_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=TEMPERATURE,
            num_ctx=LLM_NUM_CTX,
            keep_alive=LLM_KEEP_ALIVE,
        )
    return _LLM

# ---------------- Helpers ----------------
def _basename_from_doc(doc: Document) -> str:
    """Resolve a user-friendly source filename from doc.metadata."""
    meta = doc.metadata or {}
    src = meta.get("path") or meta.get("Source") or meta.get("source") or "source"
    return Path(src).name

def _print_item(tag: str, idx: int, doc: Document, score: float) -> None:
    """Print a used item (distance and full content) to server logs."""
    # With langchain_qdrant, 'score' is a distance: lower = better
    print(f"  {idx:02d}) [{tag}] distance={score:.4f} (lower=better) | {_basename_from_doc(doc)}")
    print("      CONTENT:\n", doc.page_content)

# ---------------- Retrieval ----------------
def retrieve_chunks(question: str, k: int = TOP_K) -> List[Tuple[Document, float]]:
    """
    Retrieve TOP-K 'Normal' chunks.
    Filter: metadata.tipo == 'Normal' (aligned with ingest payload).
    Returns a list of (Document, distance).
    """
    vs = _vectorstore()
    flt = qmodels.Filter(must=[
        qmodels.FieldCondition(key="metadata.tipo", match=qmodels.MatchValue(value="Normal"))
    ])
    return vs.similarity_search_with_score(question, k=k, filter=flt)  # distance: lower is better

def _is_overview_intent(q: str) -> bool:
    """Heuristic: does the question ask for overview/summary/structure/generalities?"""
    ql = (q or "").lower()
    triggers = [
        "resumo", "sumário", "sumario", "síntese", "sintese",
        "estrutura", "em geral", "visão geral", "visao geral",
        "panorama", "pontos principais", "generalidades",
        "que fala", "que falam", "o que fala", "o que falam",
        "de que trata", "do que trata", "o que trata", "o que tratam",
        "objetivo do documento", "objetivos do documento",
    ]
    return any(t in ql for t in triggers)

def retrieve_summaries(question: str, k: int = MAX_SUMMARIES) -> List[Tuple[Document, float]]:
    """
    Retrieve 'summary' items (aka 'file profiles') ONLY when the question asks for
    overview/structure/generalities.
    """
    if k <= 0 or not _is_overview_intent(question):
        return []
    vs = _vectorstore()
    flt = qmodels.Filter(must=[
        qmodels.FieldCondition(key="metadata.tipo", match=qmodels.MatchValue(value="summary"))
    ])
    return vs.similarity_search_with_score(question, k=k, filter=flt)

def build_context(chunks: List[Tuple[Document, float]]) -> str:
    """Build the Context block using ONLY chunks (no summaries here)."""
    parts: List[str] = []
    for doc, _ in chunks:
        parts.append(f"[CHUNK] {doc.page_content}\n[source: {_basename_from_doc(doc)}]")
    return "\n\n---\n\n".join(parts) if parts else ""

def build_file_summary(summaries: List[Tuple[Document, float]]) -> str:
    """
    Build the File Summary block (concatenate summaries with inline [source: ...]).
    Injected in the prompt only when the question requires generalities.
    """
    if not summaries:
        return ""
    parts: List[str] = []
    for doc, _ in summaries:
        parts.append(f"{doc.page_content}\n[source: {_basename_from_doc(doc)}]")
    return "\n\n---\n\n".join(parts)

def print_used(chunks: List[Tuple[Document, float]], summaries: List[Tuple[Document, float]]) -> None:
    """Log what was used to answer (distance + full content)."""
    if summaries:
        print("\n[FILE SUMMARY USED]")
        for i, (doc, score) in enumerate(summaries, 1):
            _print_item("SUMMARY", i, doc, score)
    print("\n[CHUNKS USED] (TOP-%d)" % len(chunks))
    if not chunks:
        print("  (empty)")
    for i, (doc, score) in enumerate(chunks, 1):
        _print_item("CHUNK", i, doc, score)

# ---------------- Prompt + LLM ----------------
def history_to_text(history: List[Tuple[str, str]], max_turns: int = 6) -> str:
    """
    Include ONLY previous user questions (not assistant answers).
    This prevents the model from copying prior answers while still helping with references.
    """
    if not history:
        return "(no history)"
    qs = [u for (u, _a) in history if u]
    qs = qs[-max_turns:]
    return "\n".join(f"Previous question: {q}" for q in qs)

# System prompt 
SYSTEM_PROMPT = (
    "- Responde sempre em Português (pt-PT).\n"
    "- Usa **apenas** o Contexto: {context} fornecido para construir a resposta.\n"
    "- O Histórico {hist} serve **apenas** para resolver referências/anáforas na pergunta atual (ex.: \"isto\", \"ele\", \"esse ponto\", \"e o 6?\").\n"
    "- **Não** uses o histórico para reutilizar respostas anteriores se a pergunta atual for diferente.\n"
    "- Se não houver contexto para a resposta, e o histórico não ajudar a responder à pergunta, diz: \"Não encontro essa informação no contexto.\".\n"
    "- Sê conciso e factual (parágrafos curtos).\n"
    "- Se tiveres de listar vários aspetos, usa bullet points.\n"
    "- Quando a pergunta pedir resumo/estrutura/generalidades, usa o Sumário: {file_summary}.\n"
)

def build_user_prompt(question: str, context: str, hist: str, file_summary: str) -> str:
    """Build the final user prompt with separate blocks for Context, History and File Summary."""
    return textwrap.dedent(f"""\
        Pergunta do utilizador:
        {question}

        Contexto:
        {context}

        Histórico:
        {hist}

        Sumário do Documento:
        {file_summary}

        Regras:
        - Responde apenas com base no contexto acima.
        - Coloca a fonte da informação no final da resposta - Exemplo: [fonte: …].
    """).strip()

# ---------------- Retrieve + LLM invoke ----------------
def answer_once(question: str, history_pairs: List[Tuple[str, str]]) -> str:
    """
    1) Retrieve chunks (always) and (optionally) summaries (only for overview intent).
    2) Log used items.
    3) Build prompt where Context = chunks-only and File Summary is separate.
    4) Invoke LLM with strict system prompt.
    """
    # 1) Retrieval
    chunks = retrieve_chunks(question, k=TOP_K)
    summaries = retrieve_summaries(question, k=MAX_SUMMARIES)  # conditional

    # 2) No useful info
    if not chunks and not summaries:
        return "Não encontro essa informação no contexto."

    # 3) Logs
    print_used(chunks, summaries)

    # 4) Build prompt parts
    context = build_context(chunks)                # chunks-only
    file_summary = build_file_summary(summaries)   # separate summary (may be empty)
    hist = history_to_text(history_pairs)

    # 5) Final prompt and LLM call
    user_prompt = build_user_prompt(question, context, hist, file_summary)
    resp = _llm().invoke([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_prompt},
    ])
    return getattr(resp, "content", str(resp))

# ---------------- API + UI (FastAPI + Gradio) ----------------
app = FastAPI(title="RAG Server", version="1.0.0")

# CORS (open for LAN use; restrict in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatItem(BaseModel):
    user: str
    assistant: Optional[str] = None

class ChatRequest(BaseModel):
    question: str
    history: List[ChatItem] = []

class ChatResponse(BaseModel):
    answer: str

@app.get("/health")
def health() -> dict:
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest) -> ChatResponse:
    """
    Stateless chat endpoint used by the Gradio UI.
    Keeps retrieval logs in the server console via answer_once().
    """
    history_pairs: List[Tuple[str, str]] = [(h.user, h.assistant or "") for h in req.history]
    answer = answer_once(req.question, history_pairs)
    return ChatResponse(answer=answer)

# ---- Gradio UI mounted at "/" ----
def _gradio_fn(message: str, history: List[Tuple[str, str]]) -> str:
    """
    Gradio passes history as a list of (user, assistant) tuples.
    We forward that to answer_once() to keep behavior consistent with CLI.
    """
    history_pairs = [(u or "", a or "") for (u, a) in (history or [])]
    return answer_once(message, history_pairs)

demo = gr.ChatInterface(
    fn=_gradio_fn,
    title="RAG Chat",
    description="Ask questions. Retrieval logs (chunks/summaries) appear in the server console.",
)

# Mount Gradio at root (/) so visiting http://IP:8000 shows the chat
app = gr.mount_gradio_app(app, demo, path="/")

# ---------------- CLI & Server ----------------
def main():
    print("Usage: python rag_app.py [chat|server]")
    sys.exit(0)

if __name__ == "__main__":
    mode = sys.argv[1].lower() if len(sys.argv) >= 2 else "server"

    if mode == "chat":
        print(f"[QDRANT] Collection='{COLLECTION}' | named vector='text'")
        print("Chat mode (CTRL+C to exit). History is in-memory and cleared on exit.")

        history: List[Tuple[str, str]] = []
        try:
            while True:
                q = input("\nUser: ").strip()
                if not q:
                    continue
                ans = answer_once(q, history)
                print("\nAssistant:\n", ans)
                history.append((q, ans))
                if len(history) > 12:
                    history[:] = history[-12:]
        except KeyboardInterrupt:
            print("\n[chat ended — history cleared]")

    elif mode == "server":
        host = os.getenv("RAG_SERVER_HOST", "0.0.0.0")
        port = int(os.getenv("RAG_SERVER_PORT", "8000"))
        uvicorn.run("rag_app:app", host=host, port=port, log_level="info")

    else:
        main()
