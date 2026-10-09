# Reddit AI Topic Explorer

An interactive NLP analytics dashboard for exploring AI discussions across Reddit communities.

## Overview

This project uses natural language processing, topic modeling, sentence embeddings, clustering, and interactive visualizations to analyze Reddit discussions about artificial intelligence.

## Technologies

- Python and Streamlit
- BERTopic and HDBSCAN
- Sentence Transformers (all-MiniLM-L6-v2)
- UMAP
- TF-IDF
- Plotly
- pandas and NumPy

## Features

- Interactive Reddit text exploration
- Word-frequency analysis and word clouds
- BERTopic topic exploration
- Sentence embedding analysis
- UMAP dimensionality reduction
- Clustering comparisons
- Interactive Plotly charts

## Project Structure

| File | Purpose |
|---|---|
| `app.py` | Streamlit dashboard |
| `download_reddit.py` | Download Reddit dataset |
| `prepare_reddit.py` | Clean and prepare dataset |
| `generate_embeddings.py` | Generate sentence embeddings |
| `generate_chunked_embeddings.py` | Generate chunked embeddings |
| `train_bertopic.py` | Train BERTopic model |
| `finalize_bertopic.py` | Prepare model outputs |
| `compare_clusters.py` | Compare clustering results |
| `prepare_interactive_projections.py` | Prepare UMAP projections |
| `export_bertopic_plotly.py` | Export interactive visualizations |

## Dataset and Reproducibility

The project processes 4,000 Reddit documents and generates 384-dimensional sentence embeddings.

The original development environment used Python 3.11.

Large datasets, trained models, embeddings, and generated visualization artifacts are intentionally excluded from the Git repository.

The Streamlit dashboard requires these generated artifacts to function correctly. A fresh clone is not yet a fully reproducible deployment.

Installation and end-to-end reproduction instructions are being validated.

## Running the Dashboard

After installing the necessary dependencies and preparing the required datasets and artifacts, start the application:

`streamlit run app.py`

## Project Status

The initial source code has been published. Documentation, environment reproducibility, and deployment packaging are being improved.