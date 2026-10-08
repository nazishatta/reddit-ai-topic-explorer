from pathlib import Path
import hashlib
import json
import time
import warnings

import numpy as np
import pandas as pd
from bertopic import BERTopic
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP

DATA = Path("data/processed/reddit_ai_4000.parquet")
EMBED = Path("embeddings/reddit_ai_4000.npy")
META = Path("embeddings/metadata.json")
PROJECTION = Path("artifacts/projections/umap_2d.npy")

OUT = Path("artifacts/bertopic/final_candidate")
CHARTS = OUT / "charts"
MODEL_DIR = Path("models")

OUT.mkdir(parents=True, exist_ok=True)
CHARTS.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet(DATA).reset_index(drop=True)
X = np.load(EMBED)
xy = np.load(PROJECTION)
metadata = json.loads(META.read_text())

assert len(df) == 4000
assert X.shape == (4000, 384)
assert xy.shape == (4000, 2)
assert df["post_id"].is_unique
assert np.isfinite(X).all()
assert np.isfinite(xy).all()

fingerprint = hashlib.sha256(
    "\n".join(
        df["post_id"].astype(str) + ":" + df["text"].astype(str)
    ).encode("utf-8")
).hexdigest()

assert fingerprint == metadata["dataset_sha256"], (
    "Dataset changed since embeddings were generated. "
    "Stop and investigate document alignment."
)

docs = df["text"].astype(str).tolist()

print("\n=== FINAL CANDIDATE: BERTopic 15-TOPIC SETUP ===", flush=True)
print("Documents:", len(docs), flush=True)
print("Embeddings:", X.shape, flush=True)
print("Dataset fingerprint verified.", flush=True)

umap_model = UMAP(
    n_neighbors=15,
    n_components=5,
    min_dist=0.0,
    metric="cosine",
    random_state=42,
    low_memory=True
)

hdbscan_model = HDBSCAN(
    min_cluster_size=50,
    min_samples=5,
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

topics, _ = model.fit_transform(
    docs,
    embeddings=X
)

elapsed = time.time() - start

df["topic"] = np.asarray(topics, dtype=int)

info = model.get_topic_info()
topic_count = int(df.loc[df.topic != -1, "topic"].nunique())
outliers = int((df.topic == -1).sum())

print("\n=== TRAINING RESULTS ===")
print("Topics:", topic_count)
print("Unassigned documents:", outliers)
print("Unassigned percentage:", round(100 * outliers / len(df), 2))
print("Training seconds:", round(elapsed, 2))

print("\n=== DISCOVERED TOPICS ===")
print(info[["Topic", "Count", "Name"]].to_string(index=False))

df.to_parquet(OUT / "document_assignments.parquet", index=False)
info.to_csv(OUT / "topic_information.csv", index=False)

# Human-reviewable examples selected deterministically.
examples = []
for topic_id in sorted(df.topic.unique()):
    members = df[df.topic == topic_id]
    sample = members.sample(
        n=min(3, len(members)),
        random_state=42
    )

    for _, row in sample.iterrows():
        examples.append({
            "topic": int(topic_id),
            "subreddit": str(row["subreddit"]),
            "post_id": str(row["post_id"]),
            "text_excerpt": str(row["text"])[:600]
        })

pd.DataFrame(examples).to_csv(
    OUT / "topic_examples.csv",
    index=False
)

summary = {
    "documents": len(df),
    "topics": topic_count,
    "assigned": len(df) - outliers,
    "outliers": outliers,
    "outlier_percent": round(100 * outliers / len(df), 2),
    "training_seconds": round(elapsed, 2),
    "embedding_model": metadata["model"],
    "dataset_sha256": fingerprint,
    "umap": {
        "n_neighbors": 15,
        "n_components": 5,
        "min_dist": 0.0,
        "metric": "cosine",
        "random_state": 42
    },
    "hdbscan": {
        "min_cluster_size": 50,
        "min_samples": 5
    },
    "status": "candidate_pending_topic_quality_review"
}

(OUT / "summary.json").write_text(
    json.dumps(summary, indent=2)
)

# Local-only reusable trained model.
model_path = MODEL_DIR / "bertopic_reddit_15.pkl"
model.save(str(model_path), serialization="pickle")
print("\nSaved local BERTopic model:", model_path)

# Genuine BERTopic interactive Plotly charts.
visualizations = {
    "topic_keywords.html": lambda: model.visualize_barchart(
        top_n_topics=min(topic_count, 15),
        n_words=10
    ),
    "topic_map.html": lambda: model.visualize_topics(),
    "topic_hierarchy.html": lambda: model.visualize_hierarchy(),
    "document_map.html": lambda: model.visualize_documents(
        docs,
        reduced_embeddings=xy,
        hide_document_hover=False,
        sample=1.0
    )
}

chart_results = {}

for filename, make_chart in visualizations.items():
    print(f"\nGenerating {filename}...", flush=True)

    try:
        fig = make_chart()
        fig.write_html(
            str(CHARTS / filename),
            include_plotlyjs="cdn",
            full_html=True
        )
        chart_results[filename] = "success"
        print("Saved:", CHARTS / filename, flush=True)
    except Exception as exc:
        chart_results[filename] = f"failed: {type(exc).__name__}: {exc}"
        print("Chart generation failed:", chart_results[filename], flush=True)

(OUT / "chart_status.json").write_text(
    json.dumps(chart_results, indent=2)
)

print("\n=== FINAL CANDIDATE REPORT ===")
print(json.dumps(summary, indent=2))

print("\n=== CHART STATUS ===")
print(json.dumps(chart_results, indent=2))

print("\nMODEL AND EXPORT STEP COMPLETE")
