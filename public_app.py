"""Privacy-reduced public companion to Reddit AI Topic Explorer.

Uses aggregated frequencies, topic assignments, UMAP coordinates and precomputed
Plotly figures. No individual Reddit post text is loaded.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
from hdbscan import HDBSCAN
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import adjusted_rand_score

ROOT = Path(__file__).resolve().parent
PUBLIC = ROOT / "public_assets"
PROJECTIONS = PUBLIC / "projections"
CHARTS = PUBLIC / "charts"

st.set_page_config(page_title="Reddit AI Topic Explorer · Public", page_icon="🧠", layout="wide")
st.title("🧠 Reddit AI Topic Explorer")
st.caption("Public research showcase · Aggregate word frequencies · BERTopic · Interactive UMAP · Clustering")

@st.cache_data(show_spinner=False)
def load_assets():
    index = pd.read_parquet(PUBLIC / "document_index.parquet")
    freqs = pd.read_parquet(PUBLIC / "word_frequencies_public.parquet")
    topics = pd.read_csv(PUBLIC / "topic_information_public.csv")
    xy = np.load(PROJECTIONS / "umap_2d.npy")
    z5 = np.load(PROJECTIONS / "umap_5d.npy")
    assert len(index) == len(xy) == len(z5) == 4000, "Document/projection row mismatch"
    assert index["doc_id"].is_unique, "Document IDs must be unique"
    assert {"subreddit", "topic", "doc_id"}.issubset(index.columns)
    assert {"subreddit", "ngram", "remove_stopwords", "term", "count", "document_frequency"}.issubset(freqs.columns)
    assert {"Topic", "Count", "Name"}.issubset(topics.columns)
    index = index.copy()
    index["x"] = xy[:, 0]
    index["y"] = xy[:, 1]
    return index, freqs, topics, z5

try:
    documents, frequencies, info, z5 = load_assets()
except Exception as exc:
    st.error(f"Public assets missing or invalid: {exc}")
    st.info("Check the public_assets directory in the deployment package.")
    st.stop()

names = dict(zip(info["Topic"], info["Name"]))
documents["topic_label"] = documents["topic"].map(lambda t: "Unassigned (-1)" if t == -1 else f"{t}: {names.get(t, 'Topic')}")

# All navigation controls share Streamlit session state so they remain consistent
# across the BERTopic and embedding tabs.
communities = sorted(documents["subreddit"].unique().tolist())
if "sidebar_communities" not in st.session_state:
    st.session_state.sidebar_communities = communities
if "sidebar_topic" not in st.session_state:
    st.session_state.sidebar_topic = "all"
if "sidebar_outliers" not in st.session_state:
    st.session_state.sidebar_outliers = True
if "sidebar_color" not in st.session_state:
    st.session_state.sidebar_color = "subreddit"


def reset_sidebar_filters():
    st.session_state.sidebar_communities = communities
    st.session_state.sidebar_topic = "all"
    st.session_state.sidebar_outliers = True
    st.session_state.sidebar_color = "subreddit"


st.sidebar.header("🧠 Explore the corpus")
selected_communities = st.sidebar.multiselect(
    "Reddit communities", communities, key="sidebar_communities"
)
st.sidebar.divider()
st.sidebar.subheader("Topic exploration")
all_topic_ids = sorted(int(t) for t in documents.loc[documents["topic"] != -1, "topic"].unique())
topic_options = ["all"] + [str(t) for t in all_topic_ids]
selected_topic_choice = st.sidebar.selectbox(
    "Highlight a BERTopic topic",
    topic_options,
    format_func=lambda val: "All topics" if val == "all" else f"Topic {val} · {names.get(int(val), 'Topic')}",
    key="sidebar_topic",
    help="Highlights this topic in BERTopic Studio and Embedding Atlas; it does not retrain the model.",
)
include_outliers = st.sidebar.checkbox(
    "Include unassigned documents", key="sidebar_outliers"
)
st.sidebar.selectbox(
    "Color the embedding atlas by",
    ["subreddit", "topic_label"],
    format_func=lambda item: "Reddit community" if item == "subreddit" else "BERTopic topic",
    key="sidebar_color",
)
st.sidebar.button("↺ Reset filters", use_container_width=True, on_click=reset_sidebar_filters)
st.sidebar.divider()
st.sidebar.subheader("Corpus at a glance")
sc1, sc2 = st.sidebar.columns(2)
sc1.metric("Documents", f"{len(documents):,}")
sc2.metric("Topics", len(all_topic_ids))
st.sidebar.caption("Public view: aggregated statistics and synthetic document IDs only. Original Reddit post text and IDs are excluded.")

community_view = documents.loc[documents["subreddit"].isin(selected_communities)].copy()
view = community_view if include_outliers else community_view.loc[community_view["topic"] != -1].copy()
selected_topic = None if selected_topic_choice == "all" else int(selected_topic_choice)
if view.empty:
    st.warning("No documents match the sidebar filters. Select a community or enable unassigned documents.")
    st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Visible documents", f"{len(view):,}")
c2.metric("Communities", view["subreddit"].nunique())
c3.metric("Visible topics", view.loc[view["topic"] != -1, "topic"].nunique())
c4.metric("Unassigned", f"{(view['topic'] == -1).mean():.1%}")

tab1, tab2, tab3, tab4 = st.tabs([
    "01 · Honest frequency", "02 · BERTopic Studio", "03 · Clustering Lab", "04 · Embedding Atlas"
])

with tab1:
    st.subheader("Frequent terms in Reddit AI discussions")
    col1, col2, col3 = st.columns(3)
    remove_stopwords = col1.checkbox("Remove English stop words", value=True)
    ngram_max = col2.select_slider("Include n-grams up to", options=[1, 2, 3], value=1)
    limit = col3.slider("Number of terms", 5, 50, 20, 5)
    selected = frequencies.loc[
        frequencies["subreddit"].isin(selected_communities)
        & frequencies["remove_stopwords"].eq(remove_stopwords)
        & frequencies["ngram"].between(1, ngram_max)
    ]
    # Aggregate same terms across communities; counts remain literal token occurrences.
    aggregate = selected.groupby("term", as_index=False)["count"].sum().sort_values("count", ascending=False)
    top = aggregate.head(limit).sort_values("count", ascending=True)
    fig = px.bar(top, x="count", y="term", orientation="h", labels={"count": "Term occurrences", "term": ""},
                 title="Ranked term frequency (precomputed counts)", template="plotly_dark")
    fig.update_layout(height=max(430, limit * 23 + 120), margin=dict(l=20, r=20, t=60, b=30))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Reveals: which retained terms occur most frequently in the selected communities. "
               "May mislead: frequency is not importance. To reduce reidentification risk, "
               "only terms occurring at least 10 times and in at least 10 separate documents per community are included. "
               "The displayed vocabulary is therefore intentionally incomplete.")
    st.info("Public release: no individual-post search or document-level TF-IDF. Those functions remain in the local coursework app.")

with tab2:
    st.subheader("Saved BERTopic model")
    if selected_topic is None:
        st.info("Choose a topic from the sidebar to highlight its documents here and in the Embedding Atlas.")
        selected_posts = view
    else:
        selected_posts = view.loc[view["topic"] == selected_topic]
        st.metric("Visible documents assigned to highlighted topic", len(selected_posts))
        matching = info.loc[info["Topic"] == selected_topic]
        if not matching.empty and "Representation" in matching:
            st.caption(f"Topic keywords: {matching.iloc[0]['Representation']}")
    topic_plot = view.copy()
    if selected_topic is None:
        topic_plot["selection"] = topic_plot["topic_label"]
    else:
        topic_plot["selection"] = np.where(
            topic_plot["topic"].eq(selected_topic), "Selected topic", "Other topics"
        )
    fig = px.scatter(
        topic_plot, x="x", y="y", color="selection",
        hover_data=["doc_id", "subreddit"],
        color_discrete_map={"Selected topic": "#60d3c0", "Other topics": "#687586"},
        title="BERTopic documents in the 2D projection",
        template="plotly_dark", render_mode="webgl",
    )
    fig.update_traces(marker={"size": 5, "opacity": 0.7})
    fig.update_layout(height=480)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Reveals: topic assignments and their projected positions. May mislead: topics are inferred, "
               "not ground truth, and UMAP can distort distances. The topic selector highlights "
               "rather than retraining or changing the full-corpus chart below.")
    chart_options = {
        "Topic keywords (c-TF-IDF)": "topic_keywords.json",
        "Intertopic distance": "topic_map.json",
        "Hierarchical topics": "topic_hierarchy.json",
        "Document map (no post-text hover)": "document_map.json",
    }
    choice = st.selectbox("Explore precomputed full-corpus chart", list(chart_options))
    chart = CHARTS / chart_options[choice]
    if chart.exists():
        fig = pio.from_json(chart.read_text(encoding="utf-8"))
        fig.update_layout(template="plotly_dark", font=dict(color="#e8edf2"))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("This precomputed chart describes all 4,000 source documents and does not change with community filters.")
    else:
        st.warning(f"Missing chart: {chart.name}")

@st.cache_data(show_spinner=False)
def clusters(z, method, k, eps, min_samples, min_size):
    if method == "HDBSCAN":
        return HDBSCAN(min_cluster_size=min_size, min_samples=min_samples).fit_predict(z)
    if method == "DBSCAN":
        return DBSCAN(eps=eps, min_samples=min_samples).fit_predict(z)
    return KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(z)

with tab3:
    st.subheader("Experimental clustering on saved 5D UMAP coordinates")
    method = st.radio("Method", ["HDBSCAN", "KMeans", "DBSCAN"], horizontal=True)
    a, b, c = st.columns(3)
    k = a.slider("KMeans clusters", 2, 25, 10, disabled=method != "KMeans")
    eps = b.slider("DBSCAN epsilon", 0.1, 5.0, 0.8, step=0.1, disabled=method != "DBSCAN")
    min_samples = c.slider("Minimum samples", 2, 30, 5, disabled=method == "KMeans")
    min_size = st.slider("HDBSCAN minimum cluster size", 10, 150, 50, disabled=method != "HDBSCAN")
    labels = clusters(z5, method, k, eps, min_samples, min_size)
    experiment = view.copy()
    # view preserves original row index so positions align with saved coordinates.
    experiment["cluster"] = pd.Series(labels, index=documents.index).loc[experiment.index].map(
        lambda n: "Noise (-1)" if n == -1 else f"Cluster {n}"
    )
    groups = len(set(labels) - {-1})
    m1, m2, m3 = st.columns(3)
    m1.metric("Full-corpus clusters", groups)
    m2.metric("Full-corpus noise points", int(np.sum(labels == -1)))
    m3.metric("Visible documents", len(experiment))
    fig = px.scatter(experiment, x="x", y="y", color="cluster", hover_data=["doc_id", "subreddit"],
                     title=f"{method} groups displayed on the 2D map", template="plotly_dark", render_mode="webgl")
    fig.update_traces(marker={"size": 5, "opacity": 0.7})
    fig.update_layout(height=620)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Reveals: how alternate clustering algorithms group the same documents. "
               "May mislead: groups were calculated on 5D coordinates, while this picture uses 2D; "
               "topic IDs and experimental cluster IDs are not interchangeable.")

with tab4:
    st.subheader("Semantic embedding projection")
    atlas = view.copy()
    if selected_topic is None:
        color = st.session_state.sidebar_color
    else:
        atlas["highlighted_topic"] = np.where(
            atlas["topic"] == selected_topic, "Selected topic", "Other topics"
        )
        color = "highlighted_topic"
        st.caption("The sidebar topic selection takes precedence over the atlas color setting. "
                   "Reset the topic selector to 'All topics' to color by community or topic.")
    fig = px.scatter(atlas, x="x", y="y", color=color, hover_data=["doc_id", "subreddit"],
                     title="MiniLM document embeddings projected with UMAP", template="plotly_dark", render_mode="webgl",
                     color_discrete_map={"Selected topic": "#60d3c0", "Other topics": "#687586"})
    fig.update_traces(marker={"size": 5, "opacity": 0.7})
    fig.update_layout(height=720, xaxis_title="UMAP 1", yaxis_title="UMAP 2")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Reveals: local patterns in semantic similarity. May mislead: distant points, "
               "cluster shapes and cluster sizes may be distorted by the nonlinear projection.")

st.divider()
st.caption("Methodology: embeddings, BERTopic assignments and topic charts were derived from the original Reddit corpus. "
           "The public version excludes individual posts and original IDs; frequency vocabulary is filtered. "
           "Full-text coursework analysis and document-level TF-IDF are retained only in the local version. "
           "Source dataset: Halston Lim, Top Reddit Posts Daily (Hugging Face, 2025).")
