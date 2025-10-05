# RAG – LangChain (Ollama + Qdrant + FastAPI/Gradio)

**Chatbot Pt-pt** c/RAG de documentos pdf locais.
(PDFs → texto limpo → embeddings → *similarity search* → resposta com LLM), 100% local, com suporte opcional a **GPU (CUDA)**.

## Funcionalidades
- **Preparação de dados**: limpeza/OCR dos PDFs para `.txt` limpos.  
- **Ingestão de dados**: Ingestão dos chunks gea partir dos `.txt` para o **Qdrant**.
- **Chat RAG**: *Similarity search* no **Qdrant** + contexto dos documentos com um **LLM via Ollama**.  
- **Modos**: FastAPI (API REST) e/ou Gradio (UI web).  
- **GPU-ready**: EasyOCR e LLMs acelerados por CUDA (opcional).

## Arquitetura
```
    PDFs
(data/Original/) ──▶ pdf_clean.py ──▶ .txt (data/Clean/)
                                        └─▶ ingest.py ──▶ embeddings (Ollama) ──▶ Qdrant (similarity search)
                                                                                       ▲
                                                                                       │
                                               rag_app.py (FastAPI/Gradio) ───────── ──┘
                                                            ▲
                                                            │
                                                         Ollama (LLM)
```

---

## 1) Pré-requisitos

### Via Docker (recomendado)
- Docker e Docker Compose.
- GPU (opcional): NVIDIA driver instalado no **host** + `gpus: all` no serviço `rag` do `docker-compose.yml`.

### Ambiente local (sem Docker)
- Python 3.11.
- Sistema:
  - `poppler-utils` (necessário pelo `pdf2image`).
  - `libgl1` e `libglib2.0-0` (runtimes exigidos por OpenCV, mesmo em headless).
- GPU (opcional): CUDA/cuDNN compatíveis com a versão do PyTorch.

> **Para que servem:**  
> `poppler-utils` fornece `pdftoppm/pdftocairo` usados por `pdf2image` (converter PDF → imagens).  
> `libgl1` e `libglib2.0-0` são libs de runtime necessárias por OpenCV.

---

## 2) Serviços externos

- **Ollama** (LLM + embeddings) – e.g., `http://localhost:11435`  
  Instala os modelos pretendidos (os que usei.: `llama3.3:70b-instruct-q2_K`, `snowflake-arctic-embed2:568m`).

- **Qdrant** (Vector DB) – e.g., `http://localhost:6333`  
  
---

## 3) Variáveis de ambiente (`.env`)

Cria um ficheiro `.env` na raiz:

```dotenv
# ====== Paths / Data ======
RAG_DATA_DIR=/data                     # base de dados local (pdfs/txt)

# ====== Ollama ======
OLLAMA_BASE_URL=http://localhost:11435 # endpoint do Ollama

# ====== LLM da Aplicação ======
LLM_MODEL=llama3.3:70b-instruct-q2_K   # LLM para o chat
LLM_NUM_CTX=4096                       # contexto do LLM
LLM_KEEP_ALIVE=10m                     # manter modelo carregado
LLM_TEMPERATURE=0.2                    # criatividade

# ====== Embeddings ======
EMBED_MODEL=snowflake-arctic-embed2:568m # modelo de embeddings (Ollama)

# ====== Qdrant ======
QDRANT_URL=http://localhost:6333       # URL do Qdrant
QDRANT_API_KEY=                        # chave (se necessário)
QDRANT_COLLECTION=docs_pt              # coleção de vetores

# ====== Ingest – Perfil/Sumário ======
PROFILE_LLM=llama3.3:70b-instruct-q2_K # LLM p/ sumário
PROFILE_LANG=pt-PT                     # idioma
PROFILE_NUM_CTX=8192                   # contexto do LLM de perfil
PROFILE_KEEP_ALIVE=10m                 # keep-alive
MAX_PROFILE_CHARS=40000                # máx. chars p/ sumário

# ====== Ingest – Chunking ======
CHUNK_SIZE=800                         # tamanho dos chunks
CHUNK_OVERLAP=150                      # overlap

# ====== Limpeza/OCR ======
FORCE_ALL=0                            # 1=forçar recriação de TXT

# ====== Recuperação (RAG) ======
TOP_K=5                                # nº de chunks "Normal"
MAX_SUMMARIES=2                        # nº de itens "summary"

# ====== Servidor ======
RAG_SERVER_HOST=0.0.0.0                # bind
RAG_SERVER_PORT=7860                   # porta (alinhada com compose)
```

---

## 4) Instalação & Execução

### 4.1. Docker (GPU)


**Subir os serviços:**
```bash
docker compose up --build -d
```

**Aceder:**  
`http://localhost:7860`

---

### 4.2. Execução local (sem Docker)

**Sistema:**
```bash
sudo apt-get update
sudo apt-get install -y poppler-utils libgl1 libglib2.0-0
```

**Python:**
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Ambiente:** configura `.env` e garante acesso a Ollama e Qdrant.

---

## 5) Pipeline de dados (primeira utilização)

### 5.1. Preparar PDFs
- Colocar os PDFs em `./data/Original/` (ou conforme `RAG_DATA_DIR`).

### 5.2. Limpeza/OCR → `.txt`
```bash
# Converte PDFs em TXT limpos (OCR quando necessário)
python pdf_clean.py
```

### 5.3. Ingestão / Indexação (a partir dos .txt limpos)
```bash
# Lê .txt (data/Clean/), cria embeddings via Ollama e escreve no Qdrant
python chatbot/ingest.py
```

> Dicas:
> - Com **GPU**, confirma que o container/ambiente detecta a GPU (`nvidia-smi`).  
> - Sem GPU, assegura que o OCR não força `use_gpu=True` e que o PyTorch é build CPU.

---

## 6) Arranque do servidor (API/Gradio)

### 6.1. FastAPI (Uvicorn)
```bash
python -m uvicorn rag_app:app --host 0.0.0.0 --port 7860
```
ou 
```bash
python chatbot/rag_app.py server
```

> Garante que `RAG_SERVER_PORT=7860`.

---

## 7) Testes rápidos

**Qdrant:**
- `http://localhost:6333/dashboard` (se exposto): confirma a coleção `docs_pt`.

**Ollama:**
```bash
curl http://localhost:11435/api/tags
```

**API de exemplo:**
```bash
curl -X POST "http://localhost:7860/ask"   -H "Content-Type: application/json"   -d '{"question": "Qual é o sumário dos documentos carregados?"}'
```

---

## 8) Solução de problemas

- **`pdftoppm not found`** → instalar `poppler-utils`.  
- **OpenCV `libGL.so` missing`** → instalar `libgl1`.  
- **GLib warnings** → instalar `libglib2.0-0`.  
- **GPU não detetada no container** → `gpus: all` no compose e `nvidia-container-toolkit` no host.  
- **Torch em CPU apesar de GPU** → garantir wheels **CUDA (cu121)** ou usar imagem PyTorch com CUDA.  
- **Portas** → `RAG_SERVER_PORT` deve coincidir com o mapeamento do compose (7860).

---

## 9) Estrutura de pastas
```
├─ chatbot
    ├─ pdf_clean.py    # pdf -> txt
    ├─ ingest.py       # data do Vector DB
    ├─ rag_app.py      # LLM, Prompt, similarity search and API/UI
    ├─ Dockerfile
└─ data/
  ├─ Original/    # PDFs de origem
  ├─ Clean/       # TXT limpos/extraídos
├─ requirements.txt
├─ docker-compose.yml
├─ .env
```

## 10) NOTAS
- **Chunking**: `CHUNK_SIZE=800` e `CHUNK_OVERLAP=150` - ajustável como as restantes variáveis em .env; 
- **Embeddings**: `snowflake-arctic-embed2:568m` 
- **LLM**: `llama3.3:70b-instruct-q2_K`
- Tudo corre **offline** (Ollama + Qdrant locais).


