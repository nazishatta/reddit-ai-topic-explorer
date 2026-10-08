from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer

DATA = Path("data/processed/reddit_ai_4000.parquet")
EMBEDDINGS = Path("embeddings/reddit_ai_4000.npy")
OUTPUT = Path("artifacts/bertopic")
OUTPUT.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet(DATA)
vectors = np.load(EMBEDDINGS)
docs = df["text"].astype(str).tolist()

assert len(docs) == 4000
assert vectors.shape == (4000, 384)
assert df["post_id"].is_unique

print("\n=== BERTopic TRAINING ===", flush=True)
print("Documents:", len(docs), flush=True)
print("Embedding dimensions:", vectors.shape[1], flush=True)

umap_model = UMAP(
    n_neighbors=15,
    n_components=5,
    min_dist=0.0,
    metric="cosine",
    random_state=42,
    low_memory=True
)

hdbscan_model = HDBSCAN(
    min_cluster_size=30,
    min_samples=10,
    metric="euclidean",
    cluster_selection_method="eom",
    prediction_data=True
)

vectorizer_model = CountVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    min_df=5,
    max_df=0.90
)

model = BERTopic(
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    vectorizer_model=vectorizer_model,
    calculate_probabilities=False,
    verbose=True
)

start = time.time()

topics, probabilities = model.fit_transform(
    docs,
    embeddings=vectors
)

df["topic"] = topics

topic_info = model.get_topic_info()

assigned = int((df["topic"] != -1).sum())
unassigned = int((df["topic"] == -1).sum())
topic_count = int(df.loc[df["topic"] != -1, "topic"].nunique())

print("\n=== MODEL RESULTS ===")
print("Discovered topics:", topic_count)
print("Assigned documents:", assigned)
print("Unassigned documents:", unassigned)
print("Unassigned percentage:", round(unassigned / len(df) * 100, 2))
print("Training time (seconds):", round(time.time() - start, 2))

print("\nTOPIC INFORMATION:")
print(topic_info.head(25).to_string(index=False))

df.to_parquet(
    OUTPUT / "reddit_topic_assignments.parquet",
    index=False
)

topic_info.to_csv(
    OUTPUT / "topic_information.csv",
    index=False
)

summary = {
    "documents": len(df),
    "discovered_topics": topic_count,
    "assigned_documents": assigned,
    "unassigned_documents": unassigned,
    "unassigned_percentage": round(unassigned / len(df) * 100, 2),
    "training_seconds": round(time.time() - start, 2),
    "umap": {
        "n_neighbors": 15,
        "n_components": 5,
        "min_dist": 0.0,
        "metric": "cosine",
        "random_state": 42
    },
    "hdbscan": {
        "min_cluster_size": 30,
        "min_samples": 10
    }
}

(OUTPUT / "training_summary.json").write_text(
    json.dumps(summary, indent=2)
)

print("\nBERTopic results saved locally.")
print("BERTopic training complete.")
