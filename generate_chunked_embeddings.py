from pathlib import Path
import hashlib
import json
import time

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

DATA = Path("data/processed/reddit_ai_4000.parquet")
BASELINE = Path("embeddings/reddit_ai_4000.npy")
META = Path("embeddings/metadata.json")
OUTPUT = Path("embeddings/reddit_ai_4000_chunked.npy")
REPORT = Path("embeddings/chunked_metadata.json")

df = pd.read_parquet(DATA).reset_index(drop=True)
original = np.load(BASELINE)
meta = json.loads(META.read_text())

assert original.shape == (4000, 384)
assert df["post_id"].is_unique

fingerprint = hashlib.sha256(
    "\n".join(
        df["post_id"].astype(str) + ":" + df["text"].astype(str)
    ).encode("utf-8")
).hexdigest()

assert fingerprint == meta["dataset_sha256"], (
    "Dataset fingerprint mismatch. Stop: row alignment changed."
)

print("\n=== CHUNKED MINILM EMBEDDINGS ===", flush=True)

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device="cpu"
)

tokenizer = model.tokenizer

# Reserve extra space so decoded passages do not exceed
# the model limit when they are tokenized again.
content_limit = 240

documents = df["text"].astype(str).tolist()

result = original.copy().astype(np.float32)
changed_documents = []
chunk_texts = []
chunk_weights = []
chunk_doc_indices = []
chunk_counts = np.ones(len(documents), dtype=int)

start = time.time()

for start_index in range(0, len(documents), 64):
    batch = documents[start_index:start_index + 64]

    tokenized = tokenizer(
        batch,
        add_special_tokens=False,
        truncation=False,
    )["input_ids"]

    for offset, ids in enumerate(tokenized):
        doc_index = start_index + offset

        # The existing MiniLM embedding covers the whole text
        # when its token count fits the model sequence limit.
        if len(ids) + tokenizer.num_special_tokens_to_add(False) <= model.max_seq_length:
            continue

        chunks = [
            ids[j:j + content_limit]
            for j in range(0, len(ids), content_limit)
        ]

        chunk_counts[doc_index] = len(chunks)
        changed_documents.append(doc_index)

        for chunk in chunks:
            chunk_texts.append(
                tokenizer.decode(
                    chunk,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )
            )
            chunk_weights.append(len(chunk))
            chunk_doc_indices.append(doc_index)

print("Documents requiring new embeddings:", len(changed_documents), flush=True)
print("Passages to encode:", len(chunk_texts), flush=True)

assert len(changed_documents) == 1014

if chunk_texts:
    # Check that every decoded passage remains within the
    # encoder's input limit after re-tokenization.
    for i in range(0, len(chunk_texts), 64):
        encoded = tokenizer(
            chunk_texts[i:i + 64],
            add_special_tokens=True,
            truncation=False,
        )["input_ids"]

        for ids in encoded:
            if len(ids) > model.max_seq_length:
                raise ValueError(
                    "A decoded passage exceeds the MiniLM token limit."
                )

    passage_vectors = model.encode(
        chunk_texts,
        batch_size=16,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    sums = np.zeros_like(result, dtype=np.float64)
    weights = np.zeros(len(documents), dtype=np.float64)

    for i, doc_index in enumerate(chunk_doc_indices):
        weight = chunk_weights[i]
        sums[doc_index] += passage_vectors[i] * weight
        weights[doc_index] += weight

    for doc_index in changed_documents:
        vec = sums[doc_index] / weights[doc_index]
        norm = np.linalg.norm(vec)

        if norm <= 0:
            raise ValueError("Zero-length document embedding")

        result[doc_index] = (vec / norm).astype(np.float32)

assert result.shape == (4000, 384)
assert np.isfinite(result).all()

# All unchanged documents must retain their original vectors.
unchanged = np.setdiff1d(
    np.arange(len(documents)),
    np.asarray(changed_documents, dtype=int)
)

assert np.array_equal(result[unchanged], original[unchanged])

np.save(OUTPUT, result)

baseline_similarity = np.sum(original * result, axis=1)

report = {
    "embedding_model": meta["model"],
    "documents": len(documents),
    "dimensions": result.shape[1],
    "changed_documents": len(changed_documents),
    "unchanged_documents": len(unchanged),
    "passages_encoded": len(chunk_texts),
    "content_tokens_per_chunk": content_limit,
    "aggregation": "token_weighted_mean_then_l2_normalize",
    "dataset_sha256": fingerprint,
    "average_baseline_similarity_changed": (
        float(np.mean(baseline_similarity[changed_documents]))
    ),
    "runtime_seconds": round(time.time() - start, 2)
}

REPORT.write_text(json.dumps(report, indent=2))

print("\n=== CHUNKED EMBEDDING RESULTS ===")
print(json.dumps(report, indent=2))
print("Saved:", OUTPUT.resolve())
print("\nCHUNKED EMBEDDING GENERATION SUCCESSFUL")
