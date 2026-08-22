import numpy as np
import torch
from sentence_transformers import SentenceTransformer

MODEL = "sentence-transformers/all-MiniLM-L6-v2"

texts = [
    "What is the capital of India?",
    "தமிழ்நாட்டின் தலைநகர் எது?",
    "भारत की राजधानी क्या है?",
] * 34

print("Loading CPU model...")
cpu = SentenceTransformer(MODEL, device="cpu")

print("Loading GPU model...")
gpu = SentenceTransformer(MODEL, device="cuda" if torch.cuda.is_available() else "cpu")

cpu_emb = cpu.encode(
    texts,
    batch_size=32,
    convert_to_numpy=True,
    normalize_embeddings=True,
)

gpu_emb = gpu.encode(
    texts,
    batch_size=32,
    convert_to_numpy=True,
    normalize_embeddings=True,
)

abs_diff = np.abs(cpu_emb - gpu_emb)

cosines = np.sum(
    cpu_emb * gpu_emb,
    axis=1,
)

print()
print("CPU shape:", cpu_emb.shape)
print("GPU shape:", gpu_emb.shape)
print("Maximum absolute difference:", abs_diff.max())
print("Mean absolute difference:", abs_diff.mean())
print("Minimum cosine similarity:", cosines.min())
print("Mean cosine similarity:", cosines.mean())
