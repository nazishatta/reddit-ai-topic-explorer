# Reddit AI Topic Explorer — Public Streamlit Release

This is a **text-minimized, aggregate public companion**, not the full-text Week 7 coursework app.

## Prepare on your Mac

Place `public_app.py` and `build_public_release.py` in the root of your `reddit-ai-topic-explorer` clone, on `deploy/public-app` branch. Your existing ignored `artifacts/` folder must include the previously prepared `artifacts/public/document_index.parquet`, `artifacts/public/word_frequencies_public.parquet`, `artifacts/public/topic_information_public.csv`, the two `artifacts/projections/umap_*.npy` files and the four `artifacts/bertopic/final_candidate/charts/*.json` files.

Run:

```bash
.venv-cloud/bin/python build_public_release.py
.venv-cloud/bin/python -m streamlit run public_app.py --server.port 8504
```

Check all four tabs and charts. The build fails closed if its allowlisted data has unexpected columns, a coordinate mismatch, basic contact patterns, or unusually long strings. This does NOT conclusively prove no private information or legal rights issue; review resulting label and aggregate content before publishing. It never copies pre-redaction backups or full document text.

## Git commit — only after a successful local test and manual review

```bash
git status --short
git add public_app.py build_public_release.py README_DEPLOY.md
# Add only the nine outputs in public_assets, after review:
git add public_assets/document_index.parquet public_assets/word_frequencies_public.parquet public_assets/topic_information_public.csv
git add public_assets/projections/umap_2d.npy public_assets/projections/umap_5d.npy
git add public_assets/charts/topic_keywords.json public_assets/charts/topic_map.json public_assets/charts/topic_hierarchy.json public_assets/charts/document_map.json
git diff --cached --stat
git diff --cached --check
```

Do not use `git add .` or `git add -f` to add the ignored original `artifacts/` directory. Do not commit the existing `document_assignments*.parquet` or the `document_map_pre_sanitization.json` backup.

After commit, push branch, open PR and merge to `main`. In Streamlit Community Cloud choose the GitHub repository, `main`, **`public_app.py`** (not `app.py`), and Python 3.11 if available.

## Interpretive limitations

- Text and original Reddit post IDs are not included in the public release.
- The retained n-gram vocabulary has a minimum term count and document-frequency threshold of 10 **per subreddit**; term counts do not support arbitrary post-level filtering and omitting rare terms changes ranking coverage.
- UMAP coordinates and BERTopic assignments were derived from the original Reddit text and can distort apparent geometry or represent uncertain topic labels.
- The public dashboard does not offer individual-document TF-IDF. The local coursework application does.
- Data source: Halston Lim, *Top Reddit Posts Daily*, Hugging Face Datasets (2025), https://huggingface.co/datasets/hblim/top_reddit_posts_daily. Original creator marks dataset MIT; underlying Reddit post rights require separate consideration.
- Disclose any AI-assisted development in accordance with your course policy.
