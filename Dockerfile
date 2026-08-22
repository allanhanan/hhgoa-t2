FROM python:3.11-slim AS base
WORKDIR /app

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DATA_DIR=/app/data \
    MODELS_DIR=/app/models

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl sqlite3 && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-cache tokenizers to ensure fast container startup
RUN python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2'); AutoTokenizer.from_pretrained('deepset/minilm-uncased-squad2')"

# Ensure mount point and application directories exist
RUN mkdir -p /app/data /app/models

# Copy source code and entrypoint
COPY . /app
RUN chmod +x /app/entrypoint.sh

EXPOSE 8000
CMD ["/app/entrypoint.sh"]
