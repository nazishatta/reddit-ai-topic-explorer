from pathlib import Path
import pandas as pd

source = Path("data/reddit_raw/data_raw")
output = Path("data/processed")
output.mkdir(parents=True, exist_ok=True)

files = sorted(source.glob("*.parquet"))
if not files:
    raise SystemExit("ERROR: No Parquet files found.")

df = pd.concat(
    [pd.read_parquet(f) for f in files],
    ignore_index=True
)

posts = df[df["type"] == "post"].copy()

# Preserve original text for reproducibility.
posts["original_text"] = posts["text"]

# Normalize whitespace and remove URLs.
posts["text"] = (
    posts["text"].fillna("").astype(str)
    .str.replace(r"https?://\S+", " ", regex=True)
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

posts["word_count"] = posts["text"].str.split().str.len()

posts = posts[
    (posts["word_count"] >= 20)
    & (~posts["text"].str.lower().isin(
        ["[deleted]", "[removed]"]
    ))
].copy()

# Remove exact duplicates after normalization.
posts["dedup_key"] = posts["text"].str.casefold()
posts = posts.sort_values(
    ["created_at", "post_id"]
)
posts = posts.drop_duplicates(
    subset="dedup_key",
    keep="first"
)

print("\nAVAILABLE CLEAN POSTS:")
print(posts["subreddit"].value_counts())

# Deterministic balanced sample.
samples = []

for community in sorted(posts["subreddit"].unique()):
    group = posts[posts["subreddit"] == community]

    if len(group) < 1000:
        raise SystemExit(
            f"ERROR: {community} has only {len(group)} posts."
        )

    samples.append(
        group.sample(n=1000, random_state=42)
    )

sample = pd.concat(samples, ignore_index=True)
sample = sample.drop(columns=["dedup_key"])

sample.to_parquet(
    output / "reddit_ai_4000.parquet",
    index=False
)

summary = sample.groupby("subreddit").agg(
    documents=("post_id", "count"),
    average_words=("word_count", "mean"),
    median_words=("word_count", "median"),
    earliest=("created_at", "min"),
    latest=("created_at", "max")
)

summary.to_csv(output / "sampling_report.csv")

print("\nFINAL DATASET SUMMARY:")
print(summary.to_string())

print("\nTotal documents:", len(sample))
print("Unique IDs:", sample["post_id"].nunique())
print("Duplicate texts:", sample["text"].str.casefold().duplicated().sum())

print("\nSaved: data/processed/reddit_ai_4000.parquet")
print("Saved: data/processed/sampling_report.csv")
print("\nPREPROCESSING COMPLETE")
