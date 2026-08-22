#!/bin/bash
set -e

# No external LLM server required.
# The pipeline uses in-process ONNX extractive QA (deepset/minilm-uncased-squad2).

echo "Starting FastAPI server with extractive QA pipeline..."
exec uvicorn app.main:app --host 0.0.0.0 --port 8000 --ws websockets --log-level info
