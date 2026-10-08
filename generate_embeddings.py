from pathlib import Path
import time
import json
import hashlib

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

DATA = Path("data/processed/reddit_ai_4000.parquet")
OUTPUT = Path("embeddings")
OUTPUT.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet(DATA)
assert len(df) == 4000, f"Expected 4000 documents, got {len(df)}"
assert df["post_id"].is_unique
assert df["text"].notna().all()

docs = df["text"].astype(str).tolist()

print("\n=== REDDIT SENTENCE EMBEDDINGS ===", flush=True)
print("Documents:", len(docs), flush=True)
print("Model: all-MiniLM-L6-v2", flush=True)
print("Device: CPU", flush=True)

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device="cpu"
)

start = time.time()

vectors = model.encode(
    docs,
    batch_size=16,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
)

vectors = vectors.astype(np.float32)

assert vectors.shape == (4000, 384)
assert np.isfinite(vectors).all()

np.save(OUTPUT / "reddit_ai_4000.npy", vectors)

fingerprint = hashlib.sha256(
    "\n".join(
        df["post_id"].astype(str) + ":" + df["text"].astype(str)
    ).encode("utf-8")
).hexdigest()

metadata = {
    "model": "sentence-transformers/all-MiniLM-L6-v2",
    "documents": len(docs),
    "dimensions": vectors.shape[1],
    "normalized": True,
    "dataset_sha256": fingerprint,
    "batch_size": 16
}

(OUTPUT / "metadata.json").write_text(
    json.dumps(metadata, indent=2)
)

print("\n=== RESULTS ===")
print("Embedding shape:", vectors.shape)
print("Data type:", vectors.dtype)
print("Runtime (seconds):", round(time.time() - start, 2))
print("Saved:", (OUTPUT / "reddit_ai_4000.npy").resolve())
print("Dataset fingerprint:", fingerprint)
print("\nEMBEDDING GENERATION SUCCESSFUL")
