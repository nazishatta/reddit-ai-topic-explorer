from pathlib import Path
import pandas as pd
import numpy as np
import json
import time

from umap import UMAP
from hdbscan import HDBSCAN
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer

df = pd.read_parquet("data/processed/reddit_ai_4000.parquet")
embeddings = np.load("embeddings/reddit_ai_4000.npy")
docs = df["text"].astype(str).tolist()

assert embeddings.shape == (len(docs), 384)

configs = [
    (30, 10),
    (30, 5),
    (50, 10),
    (50, 5)
]

results = []

print("\n=== BERTopic CLUSTERING COMPARISON ===", flush=True)

for min_cluster_size, min_samples in configs:
    print(
        f"\nTesting cluster_size={min_cluster_size}, "
        f"min_samples={min_samples}",
        flush=True
    )

    start = time.time()

    umap_model = UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
        low_memory=True
    )

    hdbscan_model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True
    )

    model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=CountVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=5,
            max_df=0.90
        ),
        calculate_probabilities=False,
        verbose=False
    )

    topics, _ = model.fit_transform(
        docs,
        embeddings=embeddings
    )

    labels = np.asarray(topics)

    topic_count = len(set(labels) - {-1})
    unassigned = int(np.sum(labels == -1))

    results.append({
        "min_cluster_size": min_cluster_size,
        "min_samples": min_samples,
        "topics": topic_count,
        "assigned": len(labels) - unassigned,
        "unassigned": unassigned,
        "unassigned_pct": round(
            unassigned / len(labels) * 100, 2
        ),
        "seconds": round(time.time() - start, 2)
    })

output = Path("artifacts/bertopic")
output.mkdir(parents=True, exist_ok=True)

report = pd.DataFrame(results)
report.to_csv(
    output / "cluster_comparison.csv",
    index=False
)

print("\n=== CONFIGURATION COMPARISON ===")
print(report.to_string(index=False))
print("\nSaved: artifacts/bertopic/cluster_comparison.csv")
print("\nCOMPARISON COMPLETE")
