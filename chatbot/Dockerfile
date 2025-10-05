# CUDA + PyTorch já prontos (PyTorch 2.8.0, CUDA 12.1, cuDNN 9)
FROM pytorch/pytorch:2.8.0-cuda12.1-cudnn9-runtime

# Trabalhar em /app
WORKDIR /app

# System deps:
# - poppler-utils: fornece 'pdftoppm'/'pdftocairo' para o pdf2image
# - libgl1 e libglib2.0-0: runtime exigido por OpenCV (mesmo em headless)
# - build-essential: compilar extensões nativas se necessário
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    curl \
 && rm -rf /var/lib/apt/lists/*

# pip config (opcional, apenas para confiança de host)
RUN mkdir -p /etc && printf "%s\n" \
    "[global]" \
    "index-url = https://pypi.org/simple" \
    "trusted-host = pypi.org" \
    "trusted-host = files.pythonhosted.org" \
    "trusted-host = pypi.python.org" \
    > /etc/pip.conf

# Instalar deps Python
COPY requirements.txt /app/requirements.txt

# IMPORTANTE (com esta base CUDA):
# - Não forces torch/torchvision no requirements (ou remove-os),
#   porque já vêm instalados e compatíveis com a CUDA desta imagem.
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copiar o código
COPY . /app

# Porta do servidor da app (vais pôr RAG_SERVER_PORT=7860 no .env)
EXPOSE 7860
ENV PYTHONUNBUFFERED=1

# Arranque (ajusta para o que usas: uvicorn, gradio, etc.)
# Exemplo FastAPI (se tiveres app main em rag_app.py):
# CMD ["python", "-m", "uvicorn", "rag_app:app", "--host", "0.0.0.0", "--port", "7860"]
# Mantém o container vivo enquanto testas:
CMD ["bash", "-lc", "sleep infinity"]