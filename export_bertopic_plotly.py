from pathlib import Path
import numpy as np
import pandas as pd
from bertopic import BERTopic

root = Path("artifacts/bertopic/final_candidate/charts")
root.mkdir(parents=True, exist_ok=True)

print("Loading trusted local BERTopic model...", flush=True)
model = BERTopic.load("models/bertopic_reddit_15.pkl")

df = pd.read_parquet(
    "artifacts/bertopic/final_candidate/document_assignments.parquet"
)
xy = np.load("artifacts/projections/umap_2d.npy")
docs = df["text"].astype(str).tolist()

assert len(docs) == len(xy) == 4000

charts = {
    "topic_keywords": lambda: model.visualize_barchart(
        top_n_topics=15, n_words=10
    ),
    "topic_map": lambda: model.visualize_topics(),
    "topic_hierarchy": lambda: model.visualize_hierarchy(),
    "document_map": lambda: model.visualize_documents(
        docs,
        reduced_embeddings=xy,
        sample=1.0,
        hide_document_hover=False
    )
}

for name, create in charts.items():
    print("Exporting:", name, flush=True)
    fig = create()
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#17191E",
        plot_bgcolor="#17191E",
        font=dict(color="#F2F4F8")
    )
    destination = root / f"{name}.json"
    destination.write_text(fig.to_json(), encoding="utf-8")
    print("Saved:", destination, flush=True)

print("\nPLOTLY JSON EXPORT COMPLETE")
