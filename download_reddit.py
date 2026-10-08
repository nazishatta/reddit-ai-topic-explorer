
from huggingface_hub import snapshot_download

path = snapshot_download(
    repo_id="hblim/top_reddit_posts_daily",
    repo_type="dataset",
    local_dir="data/reddit_raw",
    allow_patterns=["data_raw/*.parquet"],
    max_workers=8
)

print("Download complete!")
print("Saved to:", path)
