"""
Data ingest for a local RAG.

What this script does:
1) Load cleaned TXT files from /data/data_clean (or ./data/data_clean on host)
2) For each file:
   - Generate ONE special "summary" chunk via LLM with the requested JSON schema
   - Split the text into smaller chunks
   - Attach normalized metadata fields:
       Source: TEXT
       Page:   INT (unknown -> -1)
       Section: TEXT (best-effort using heading heuristics)
       tipo: "Normal" | "summary"
3) L2-normalize embeddings and store in Qdrant with COSINE metric

"""

import os
import re
import sys
import json
import uuid
from pathlib import Path
from typing import List, Tuple

from dotenv import load_dotenv, find_dotenv

# LangChain: text splitting, LLM and embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_qdrant import QdrantVectorStore
from langchain.schema import Document, HumanMessage, SystemMessage
from langchain_core.embeddings import Embeddings  # <-- important base class

# Qdrant client: to ensure collection exists with the right metric & named vector
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

# -------------------------- Paths & Config --------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_DATA = Path(os.getenv("RAG_DATA_DIR", "/data"))
if not BASE_DATA.exists() or not os.access(BASE_DATA, os.W_OK):
    BASE_DATA = PROJECT_ROOT / "data"

IN_DIR = BASE_DATA / "data_clean"   # where cleaned TXT files live
PDF_DIR = BASE_DATA / "pdfs"        # not used here, but kept for reference

load_dotenv(find_dotenv())

# Endpoints / models
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11435")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
EMBED_MODEL = os.getenv("EMBED_MODEL", "snowflake-arctic-embed2:568m")
COLLECTION = os.getenv("QDRANT_COLLECTION", "docs_pt")

# Force COSINE distance in Qdrant (paired with L2 normalization)
DISTANCE = qmodels.Distance.COSINE

# LLM to create the special "summary" chunk (one per document)
PROFILE_LLM = os.getenv("PROFILE_LLM", "llama3.3:70b-instruct-q2_K")
PROFILE_LANG = os.getenv("PROFILE_LANG", "pt-PT")
PROFILE_NUM_CTX = int(os.getenv("PROFILE_NUM_CTX", "8192"))
PROFILE_KEEP_ALIVE = os.getenv("PROFILE_KEEP_ALIVE", "10m")
MAX_PROFILE_CHARS = int(os.getenv("MAX_PROFILE_CHARS", "40000"))

# Chunking defaults
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

# Heading heuristic for sections (Portuguese technical docs, adjustable)
HEADING_RE = re.compile(
    r"^(?:CAP[IÍ]TULO\s+\d+[^\n]*|ANEXO[^\n]*|NOTA PR[ÉE]VIA[^\n]*|\d{1,3}\.\s+[^\n]{3,})$",
    re.MULTILINE
)

# -------------------------- Small utilities --------------------------

def list_txt_files(directory: Path) -> List[Path]:
    """Return all .txt files in a stable order."""
    return sorted(list(directory.glob("*.txt")) + list(directory.glob("*.TXT")))

def read_cleaned_text(txt_path: Path) -> str:
    """Read UTF-8 text with a forgiving error mode."""
    try:
        return txt_path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        print(f"[ERROR] Cannot read {txt_path.name}: {e}")
        return ""

def build_text_splitter() -> RecursiveCharacterTextSplitter:
    """Chunker tuned for technical prose; includes start_index for Section mapping."""
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", ", ", " "],
        length_function=len,
        add_start_index=True,  # we use this to infer the nearest Section
    )

def smart_truncate(text: str, max_chars: int) -> str:
    """Truncate near sentence breaks to avoid ugly cuts for LLM context."""
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    last_break = max(cut.rfind("\n\n"), cut.rfind("\n"), cut.rfind(". "))
    if last_break > max_chars * 0.6:
        return cut[:last_break].strip()
    return cut.strip()

def extract_headings_with_pos(text: str, max_items: int = 50) -> List[Tuple[int, str]]:
    """
    Detect headings and return (start_index, heading_text) pairs.
    This is a simple heuristic; if you need a different rule, just say.
    """
    positions: List[Tuple[int, str]] = []
    for m in HEADING_RE.finditer(text):
        positions.append((m.start(), m.group(0).strip()))
        if len(positions) >= max_items:
            break
    return positions

def find_section_for_offset(headings: List[Tuple[int, str]], start_index: int) -> str:
    """Find the last heading that occurs at or before start_index."""
    last_title = ""
    for pos, title in headings:
        if pos <= start_index:
            last_title = title
        else:
            break
    return last_title

# -------------------------- LLM profile generation --------------------------

def build_profile_prompt(sample: str, headings: List[str], lang: str):
    """
    Return a special chunk, a single JSON object in the given schema.
    Output MUST be JSON only (no Markdown).
    """
    system = (
        f"You are an expert technical editor. Work in {lang}. "
        "Return ONLY a single valid JSON object (no markdown, no prose, no comments). "
        "Schema:\n"
        "{\n"
        '  "summary": string,                 // <=150 words, neutral\n'
        '  "structure": string[],             // ordered list of section/chapter titles\n'
        '  "publication_date": string,        // ISO YYYY-MM-DD if possible, else YYYY or ""\n'
        '  "keywords": string[],              // 5-12 domain-relevant\n'
        '  "entity": string                   // authoring org; "" if unknown\n'
        "}\n"
        "Do not include trailing commas. Do not include extra fields."
    )
    headings_str = "\n".join(f"- {h}" for h in headings[:20]) if headings else "(none)"
    user = (
        "DOCUMENT (truncated):\n<<<\n"
        f"{sample}\n>>>\n\n"
        f"Detected headings (heuristic):\n{headings_str}\n\n"
        "Produce the JSON now."
    )
    return [SystemMessage(content=system), HumanMessage(content=user)]

def _extract_balanced_json(text: str) -> str:
    """Extract the first balanced {...} block; fallback is raw text (json.loads will fail)."""
    start = text.find("{")
    if start == -1:
        return text
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return text

def call_profile_llm(raw_text: str, llm_model: str, base_url: str, lang: str) -> dict:
    """
    Ask the LLM to produce the JSON profile. We keep this robust and deterministic.
    If anything fails, return a minimal heuristic profile.
    """
    sample = smart_truncate(raw_text, MAX_PROFILE_CHARS)
    headings_only = [h for _, h in extract_headings_with_pos(sample)]
    chat = ChatOllama(
        model=llm_model,
        base_url=base_url,
        temperature=0.0,      # deterministic JSON
        format="json",        # native JSON mode
        num_ctx=PROFILE_NUM_CTX,
        keep_alive=PROFILE_KEEP_ALIVE,
    )
    try:
        resp = chat.invoke(build_profile_prompt(sample, headings_only, lang))
        payload = _extract_balanced_json(resp.content.strip())
        data = json.loads(payload)

        # Validate & normalize keys
        for key in ["summary", "structure", "publication_date", "keywords", "entity"]:
            if key not in data:
                raise ValueError(f"Missing key: {key}")

        if not isinstance(data["structure"], list):
            data["structure"] = [str(data["structure"])]
        if not isinstance(data["keywords"], list):
            data["keywords"] = [str(data["keywords"])]

        data["summary"] = str(data["summary"]).strip()
        data["publication_date"] = str(data["publication_date"]).strip()
        data["entity"] = str(data["entity"]).strip()
        data["structure"] = [str(s).strip() for s in data["structure"] if str(s).strip()]
        data["keywords"] = [str(k).strip() for k in data["keywords"] if str(k).strip()]
        return data

    except Exception as e:
        print(f"[PROFILE][WARN] LLM profile generation failed: {e}")
        return {
            "summary": (sample[:800] + ("..." if len(sample) > 800 else "")).strip(),
            "structure": headings_only or [],
            "publication_date": "",
            "keywords": [],
            "entity": "",
        }

def make_profile_document(profile: dict, src_name: str, src_path: str) -> Document:
    """
    Create the special 'summary' chunk as requested.
    We store the JSON content itself (pretty-printed) as the chunk's text.
    """
    page_content = json.dumps(profile, ensure_ascii=False, indent=2)
    metadata = {
        # Standardized metadata fields requested by you:
        "Source": src_name,       # TEXT
        "Page": -1,               # INT (unknown from TXT)
        "Section": "",            # TEXT
        "tipo": "summary",        # "summary" chunk
        # A few extras that might help later (safe to keep):
        "path": src_path,
        "chunk_id": str(uuid.uuid4()),
    }
    return Document(page_content=page_content, metadata=metadata)

# -------------------------- Embedding wrapper (L2 normalize) --------------------------

def _l2_normalize(vec: List[float]) -> List[float]:
    """In-place safe L2 normalization (returning a new list)."""
    import math
    norm = math.sqrt(sum((x * x) for x in vec)) or 1.0
    return [x / norm for x in vec]

class NormalizedEmbeddings(Embeddings):
    """
    Wrapper around LangChain's Embeddings interface that L2-normalizes:
      - embed_query
      - embed_documents
    Use this to pair with Qdrant COSINE metric as requested.
    """
    def __init__(self, base_embeddings: OllamaEmbeddings):
        self.base = base_embeddings

    def embed_query(self, text: str) -> List[float]:
        vec = self.base.embed_query(text)
        return _l2_normalize(vec)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        vecs = self.base.embed_documents(texts)
        return [_l2_normalize(v) for v in vecs]

# -------------------------- Qdrant collection setup --------------------------

def ensure_collection(client: QdrantClient, collection: str, distance: qmodels.Distance, vector_size: int) -> None:
    """
    Create the collection if it doesn't exist yet.
    We use a NAMED VECTOR 'text' to match the vector store config.
    """
    try:
        _ = client.get_collection(collection_name=collection)
        return
    except Exception:
        pass

    client.create_collection(
        collection_name=collection,
        vectors_config={
            "text": qmodels.VectorParams(
                size=vector_size,
                distance=distance,
                on_disk=True,
            )
        },
    )

# -------------------------- Main ingest flow --------------------------

def main():
    print(f"[PATHS] IN_DIR={IN_DIR} | PDF_DIR={PDF_DIR}")
    print(f"[CFG] OLLAMA_BASE_URL={OLLAMA_BASE_URL} | QDRANT_URL={QDRANT_URL}")
    print(f"[CFG] COLLECTION={COLLECTION} | DISTANCE=COSINE | CHUNK={CHUNK_SIZE}/{CHUNK_OVERLAP}")
    print(f"[CFG] PROFILE_LLM={PROFILE_LLM} | PROFILE_LANG={PROFILE_LANG}")
    print(f"[CFG] EMBED_MODEL={EMBED_MODEL} (L2-normalized)")

    # 1) Locate input files
    txt_files = list_txt_files(IN_DIR)
    if not txt_files:
        print(f"[WARN] No cleaned TXT files found in {IN_DIR}.")
        sys.exit(0)

    # 2) Prepare embeddings (wrapped with L2 normalization)
    print("[INIT] Initializing embeddings via Ollama ...")
    base_embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_BASE_URL)
    embeddings = NormalizedEmbeddings(base_embeddings)

    # Probe dimension
    dim = len(embeddings.embed_query("dim probe"))
    print(f"[INIT] Embedding dimension: {dim}")

    # 3) Ensure Qdrant collection exists (named vector 'text', COSINE)
    qclient = QdrantClient(url=QDRANT_URL)
    ensure_collection(qclient, COLLECTION, DISTANCE, dim)
    print(f"[QDRANT] Collection ready: {COLLECTION} (metric=COSINE, dim={dim}, vector='text')")

    # 4) Create the LangChain vector store with explicit payload keys
    vstore = QdrantVectorStore(
        client=qclient,
        collection_name=COLLECTION,
        embedding=embeddings,         # now a valid Embeddings subclass
        vector_name="text",           # matches collection config
        content_payload_key="page_content",
        metadata_payload_key="metadata",
    )

    splitter = build_text_splitter()

    total_normal_chunks = 0
    total_summary_chunks = 0

    # 5) Iterate documents
    for txt in txt_files:
        raw = read_cleaned_text(txt)
        if not raw.strip():
            print(f"[SKIP] Empty text: {txt.name}")
            continue

        # Extract headings once per document for Section mapping
        headings_with_pos = extract_headings_with_pos(raw)

        # 5a) Create special "summary" chunk via LLM
        print(f"[PROFILE] Generating summary via LLM '{PROFILE_LLM}' for {txt.name} ...")
        profile = call_profile_llm(raw, PROFILE_LLM, OLLAMA_BASE_URL, PROFILE_LANG)
        profile_doc = make_profile_document(profile, txt.name, str(txt))
        vstore.add_documents([profile_doc])
        total_summary_chunks += 1
        print(f"[UPSERT][SUMMARY] {txt.name}: +1 summary chunk")

        # 5b) Chunk the document and build "Normal" chunks with requested metadata
        base_meta = {
            "Source": txt.name,  # TEXT
            # Page is unknown from plain TXT; use -1 unless you want a custom parser
            "Page": -1,          # INT
            "Section": "",       # TEXT (filled below if we detect headings)
            "tipo": "Normal",    # "Normal" chunk
            # keep path/chunk_id as helpful internal extras
            "path": str(txt),
        }

        doc_splits = splitter.create_documents([raw], metadatas=[base_meta])

        if not doc_splits:
            print(f"[SKIP] No chunks after split: {txt.name}")
            continue

        for order, d in enumerate(doc_splits):
            # Determine best-effort Section from heading positions and chunk start_index
            start_idx = d.metadata.get("start_index", 0)
            section = find_section_for_offset(headings_with_pos, start_idx) if headings_with_pos else ""

            # Update standardized metadata
            d.metadata.update({
                "Section": section,                
                "chunk_id": str(uuid.uuid4()),
                "order": order,
            })

        vstore.add_documents(doc_splits)
        total = len(doc_splits)
        total_normal_chunks += total
        print(f"[UPSERT][NORMAL] {txt.name}: +{total} chunks")

    print(f"[DONE] Total summary chunks: {total_summary_chunks}")
    print(f"[DONE] Total normal chunks: {total_normal_chunks}")
    print(f"[READY] Qdrant collection '{COLLECTION}' is ready for retrieval.")
    print("       Use filters like 'metadata.tipo == \"summary\"' or 'metadata.Source == <file>'.")

# -------------------------- Entrypoint --------------------------

if __name__ == "__main__":
    main()
