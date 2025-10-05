# -*- coding: utf-8 -*-
"""
RAG (compatível com o teu ingest.py + Qdrant)

- Usa langchain_qdrant.QdrantVectorStore com a MESMA configuração do ingest:
  vector_name="text", content_payload_key="page_content", metadata_payload_key="metadata".
- Embeddings L2-normalizados (NormalizedEmbeddings herda de Embeddings), ideal para COSINE.
- Recupera:
    * TOP_K=5 chunks "Normais" (metadata.tipo == "Normal")
    * até MAX_SUMMARIES "summary" (metadata.tipo == "summary") como "chunk especial"
- Imprime os chunks usados (score + conteúdo) e os summaries.
- LLM (Ollama) com temperatura=0.2.
- Histórico apenas enquanto o programa corre.

Uso:
    python rag_app.py chat
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


from langchain.schema import Document
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_qdrant import QdrantVectorStore
from langchain_core.embeddings import Embeddings  # IMPORTANTE: base class certa

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

# ---------------- Config (alinhado com ingest.py) ----------------
load_dotenv(find_dotenv())

# Mesmos defaults que usas no ingest.py
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11435")
QDRANT_URL      = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY  = os.getenv("QDRANT_API_KEY") or None
COLLECTION      = os.getenv("QDRANT_COLLECTION", "docs_pt")
EMBED_MODEL     = os.getenv("EMBED_MODEL", "snowflake-arctic-embed2:568m")

LLM_MODEL       = os.getenv("LLM_MODEL", "llama3.3:70b-instruct-q2_K")
LLM_NUM_CTX     = int(os.getenv("LLM_NUM_CTX", "4096"))
LLM_KEEP_ALIVE  = os.getenv("LLM_KEEP_ALIVE", "10m")

TOP_K           = 5
TEMPERATURE     = 0.2
MAX_SUMMARIES   = int(os.getenv("MAX_SUMMARIES", "2"))

# ---------------- System Prompt (EXATO) ----------------
SYSTEM_PROMPT = (
    "- Responde sempre em Português (pt-PT).\n"
    "- Responde usando **apenas** o contexto fornecido.\n"
    "- Se a resposta não estiver no contexto, verifica no histórico, se mesmo assim não houver contexto, diz: \"Não encontro essa informação no contexto.\".\n"
    "- Sê conciso e factual: parágrafos curtos. \n"
    "- Se tiveres que listar, enumerar os apontar vários aspectos usa bullets points.\n"
    "- Quando a pergunta solicitar resumo/estrutura/generalidades do documentos, usa o contexto de type=doc_profile.\n"
)

# ---------------- Embeddings L2-normalizados ----------------
def _l2_normalize(vec: List[float]) -> List[float]:
    import math
    n = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / n for x in vec]

class NormalizedEmbeddings(Embeddings):
    """
    Wrapper de embeddings que L2-normaliza (compatível com COSINE no Qdrant).
    **IMPORTANTE**: herda de langchain_core.embeddings.Embeddings (como no ingest.py).
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
    """VectorStore igual ao do ingest.py (mesmas chaves e vetor nomeado 'text')."""
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
        vector_name="text",                 # vetor nomeado (existe na tua coleção)
        content_payload_key="page_content", # igual ao ingest
        metadata_payload_key="metadata",    # igual ao ingest
    )
    print(f"[QDRANT] OK: collection='{COLLECTION}', vector='text'")
    return _VS

def _llm() -> ChatOllama:
    """Cliente LLM (temperatura=0.2)."""
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
    meta = doc.metadata or {}
    # No ingest as chaves são normalizadas; usamos várias possibilidades
    src = meta.get("path") or meta.get("Source") or meta.get("source") or "fonte"
    return Path(src).name

def _print_item(tag: str, idx: int, doc: Document, score: float) -> None:
    # Em langchain_qdrant o score é "distance" (menor = melhor)
    print(f"  {idx:02d}) [{tag}] distance={score:.4f} (menor=melhor) | { _basename_from_doc(doc) }")
    print("      CONTEÚDO:\n", doc.page_content)

# ---------------- Retrieve ----------------
def retrieve_chunks(question: str, k: int = TOP_K) -> List[Tuple[Document, float]]:
    """
    TOP-K chunks 'Normais'.
    Filtro pelos payloads conforme ingest: metadata.tipo == 'Normal'
    """
    vs = _vectorstore()
    flt = qmodels.Filter(must=[
        qmodels.FieldCondition(key="metadata.tipo", match=qmodels.MatchValue(value="Normal"))
    ])
    # distância: menor = melhor
    return vs.similarity_search_with_score(question, k=k, filter=flt)

def retrieve_summaries(question: str, k: int = MAX_SUMMARIES) -> List[Tuple[Document, float]]:
    """TOP summaries (o 'chunk especial'): metadata.tipo == 'summary'."""
    if k <= 0:
        return []
    vs = _vectorstore()
    flt = qmodels.Filter(must=[
        qmodels.FieldCondition(key="metadata.tipo", match=qmodels.MatchValue(value="summary"))
    ])
    return vs.similarity_search_with_score(question, k=k, filter=flt)

def build_context(chunks: List[Tuple[Document, float]], summaries: List[Tuple[Document, float]]) -> str:
    """[PERFIL] para summaries e [TRECHO] para chunks + [fonte: ficheiro]."""
    parts: List[str] = []
    for doc, _ in summaries:
        parts.append(f"[PERFIL] {doc.page_content}\n[fonte: {_basename_from_doc(doc)}]")
    for doc, _ in chunks:
        parts.append(f"[TRECHO] {doc.page_content}\n[fonte: {_basename_from_doc(doc)}]")
    return "\n\n---\n\n".join(parts) if parts else ""

def print_used(chunks: List[Tuple[Document, float]], summaries: List[Tuple[Document, float]]) -> None:
    """Imprime score + conteúdo dos elementos usados."""
    if summaries:
        print("\n[CHUNK ESPECIAL: SUMMARY]")
        for i, (doc, score) in enumerate(summaries, 1):
            _print_item("PERFIL", i, doc, score)
    print("\n[CHUNKS USADOS] (TOP-5)")
    if not chunks:
        print("  (vazio)")
    for i, (doc, score) in enumerate(chunks, 1):
        _print_item("CHUNK", i, doc, score)

# ---------------- Prompt + LLM ----------------
def history_to_text(history: List[Tuple[str, str]], max_turns: int = 6) -> str:
    if not history:
        return "(sem histórico)"
    lines: List[str] = []
    for u, a in history[-max_turns:]:
        if u: lines.append(f"Utilizador: {u}")
        if a: lines.append(f"Assistente: {a}")
    return "\n".join(lines)

def build_user_prompt(question: str, context: str, hist: str) -> str:
    return textwrap.dedent(f"""\
        Pergunta do utilizador:
        {question}

        Contexto:
        {context}

        Histórico:
        {hist}

        Regras:
        - Responde apenas com base no contexto acima.
        - Coloca a fonte da informação no final da resposta - Exemplo: [fonte: …].
    """).strip()

def answer_once(question: str, history_pairs: List[Tuple[str, str]]) -> str:
    """retrieve -> prints -> prompt -> LLM."""
    chunks = retrieve_chunks(question, k=TOP_K)
    summaries = retrieve_summaries(question, k=MAX_SUMMARIES)

    if not chunks and not summaries:
        return "Não encontro essa informação no contexto."

    print_used(chunks, summaries)
    context = build_context(chunks, summaries)
    hist = history_to_text(history_pairs)

    user_prompt = build_user_prompt(question, context, hist)
    resp = _llm().invoke([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_prompt},
    ])
    return getattr(resp, "content", str(resp))

# ---------------- CLI ----------------
def main():
    if len(sys.argv) < 2 or sys.argv[1].lower() != "chat":
        print("Uso: python rag_app.py chat")
        sys.exit(0)

    print(f"[QDRANT] Coleção='{COLLECTION}' | vetor nomeado='text'")
    print("Modo chat (CTRL+C para sair). Histórico só em memória (apagado ao terminar).")

    history: List[Tuple[str, str]] = []
    try:
        while True:
            q = input("\nUser (Eu): ").strip()
            if not q:
                continue
            ans = answer_once(q, history)
            print("\nAssistente:\n", ans)
            history.append((q, ans))
            if len(history) > 12:
                history[:] = history[-12:]
    except KeyboardInterrupt:
        print("\n[chat terminado — histórico apagado]")

if __name__ == "__main__":
    main()
