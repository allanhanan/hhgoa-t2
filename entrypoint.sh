#!/bin/bash
set -e

# Start llama.cpp server in background
# SmolLM2-135M only needs 2-4 threads; more threads = diminishing returns
echo "Starting llama.cpp server..."
llama-server \
  --model /app/models/smollm2-135m-instruct-q4_k_m.gguf \
  --host 0.0.0.0 --port 8081 \
  --threads 4 --ctx-size 1024 \
  --batch-size 256 \
  --repeat-penalty 1.3 &

# Wait for llama.cpp to be ready (SmolLM2-135M loads in <1s)
echo "Waiting for llama.cpp to load..."
sleep 2

# Verify llama.cpp is running
for i in $(seq 1 10); do
  if curl -sf http://localhost:8081/health > /dev/null 2>&1; then
    echo "llama.cpp server ready!"
    break
  fi
  echo "  Waiting... ($i/10)"
  sleep 1
done

# Start FastAPI with WebSocket support
echo "Starting FastAPI server..."
exec uvicorn app.main:app --host 0.0.0.0 --port 8000 --ws websockets --log-level info
