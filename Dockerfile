FROM python:3.11-slim AS base
WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl sqlite3 && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download embedding model (ONNX)
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# Download llama.cpp binary (pre-built with AVX2)
RUN curl -L https://github.com/ggerganov/llama.cpp/releases/latest/download/llama-server-linux-x64 \
    -o /usr/local/bin/llama-server && chmod +x /usr/local/bin/llama-server

# Download SmolLM2-135M-Instruct GGUF (Q4_K_M, ~85MB)
RUN mkdir -p /app/models && \
    curl -L "https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct-GGUF/resolve/main/smollm2-135m-instruct-q4_k_m.gguf" \
    -o /app/models/smollm2-135m-instruct-q4_k_m.gguf

COPY . .

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
EXPOSE 8000 8081
CMD ["/entrypoint.sh"]
