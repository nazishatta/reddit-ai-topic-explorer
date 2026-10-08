from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import streamlit as st

from sklearn.cluster import KMeans, DBSCAN
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from hdbscan import HDBSCAN

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"
CANDIDATE = ART / "bertopic" / "final_candidate"
CHARTS = CANDIDATE / "charts"

st.set_page_config(
    page_title="Reddit AI Topic Explorer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

INK = "#28748C"
GREY = "#A7AFBC"

import plotly.io as pio

pio.templates.default = "plotly_dark"

pio.templates["plotly_dark"].layout.update(
    paper_bgcolor="#17191E",
    plot_bgcolor="#17191E",
    font=dict(color="#F2F4F8"),
    xaxis=dict(
        gridcolor="#363B46",
        zerolinecolor="#363B46"
    ),
    yaxis=dict(
        gridcolor="#363B46",
        zerolinecolor="#363B46"
    )
)



@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_parquet(
        CANDIDATE / "document_assignments.parquet"
    ).reset_index(drop=True)

    xy = np.load(ART / "projections" / "umap_2d.npy")
    z5 = np.load(ART / "projections" / "umap_5d.npy")

    info = pd.read_csv(CANDIDATE / "topic_information.csv")

    if len(df) != len(xy) or len(df) != len(z5):
        raise ValueError("Document and projection rows do not align.")

    if not {"text", "topic", "subreddit", "post_id"}.issubset(df.columns):
        raise ValueError("Required dataset columns are missing.")

    df["row_id"] = np.arange(len(df))
    df["x"] = xy[:, 0]
    df["y"] = xy[:, 1]
    df["word_count"] = df["text"].str.split().str.len()

    return df, z5, info


@st.cache_data(show_spinner=False)
def get_terms(texts, remove_stopwords, ngrams):
    vectorizer = CountVectorizer(
        stop_words="english" if remove_stopwords else None,
        ngram_range=(1, ngrams),
        max_features=12000,
    )
    matrix = vectorizer.fit_transform(texts)

    scores = np.asarray(matrix.sum(axis=0)).ravel()

    return pd.DataFrame({
        "term": vectorizer.get_feature_names_out(),
        "count": scores,
    }).sort_values("count", ascending=False)


@st.cache_data(show_spinner=False)
def get_tfidf(texts, document_index, remove_stopwords, ngrams):
    vectorizer = TfidfVectorizer(
        stop_words="english" if remove_stopwords else None,
        ngram_range=(1, ngrams),
        max_features=12000,
    )
    matrix = vectorizer.fit_transform(texts)

    scores = matrix[document_index].toarray().ravel()
    terms = vectorizer.get_feature_names_out()

    result = pd.DataFrame({
        "term": terms,
        "tfidf": scores,
    })

    return (
        result[result["tfidf"] > 0]
        .sort_values("tfidf", ascending=False)
        .head(20)
    )


@st.cache_data(show_spinner="Computing clusters...")
def compute_clusters(z5, algorithm, k, eps, samples, size):
    if algorithm == "HDBSCAN":
        labels = HDBSCAN(
            min_cluster_size=size,
            min_samples=samples,
            metric="euclidean",
            cluster_selection_method="eom",
        ).fit_predict(z5)

    elif algorithm == "DBSCAN":
        labels = DBSCAN(
            eps=eps,
            min_samples=samples,
            metric="euclidean",
        ).fit_predict(z5)

    else:
        labels = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10,
        ).fit_predict(z5)

    return labels.astype(int)


def horizontal_bars(data, x, y, title, color=INK):
    plot = data.iloc[::-1]

    fig = px.bar(
        plot,
        x=x,
        y=y,
        orientation="h",
        title=title,
        color_discrete_sequence=[color],
        height=max(350, len(plot) * 27 + 100),
    )
    fig.update_layout(
        margin=dict(l=12, r=12, t=55, b=20),
        yaxis_title=None,
    )
    return fig


def display_document_map(data, color_col, chart_key, height=620):
    plot = data.reset_index(drop=True).copy()
    plot["preview"] = plot["text"].str.slice(0, 180)

    fig = px.scatter(
        plot,
        x="x",
        y="y",
        color=color_col,
        hover_name="preview",
        hover_data={
            "subreddit": True,
            "word_count": True,
            "x": False,
            "y": False,
            "row_id": False,
        },
        custom_data=["row_id"],
        opacity=0.75,
        height=height,
        render_mode="webgl",
    )

    fig.update_traces(
        marker=dict(size=7)
    )

    fig.update_layout(
        dragmode="lasso",
        margin=dict(l=8, r=8, t=15, b=10),
        xaxis_title="UMAP dimension 1",
        yaxis_title="UMAP dimension 2",
        legend_title=color_col,
    )

    event = st.plotly_chart(
        fig,
        width="stretch",
        key=chart_key,
        on_select="rerun",
        selection_mode=("points", "box", "lasso"),
    )

    selected_ids = []

    for point in event.selection.points:
        custom = point.get("customdata", [])
        if custom:
            selected_ids.append(int(custom[0]))

    # Selection works across Plotly legend traces because the stable
    # original row_id is stored in custom_data, not pointNumber.
    selected_ids = list(dict.fromkeys(selected_ids))

    st.caption(
        "Explore points with hover. To inspect posts, select the lasso "
        "or box tool on the chart toolbar and drag over points. "
        "Selected documents appear beneath the chart."
    )
    if selected_ids:
        selected = data[data["row_id"].isin(selected_ids)]
        st.write(f"**Selected documents: {len(selected):,}**")
        st.dataframe(
            selected[
                ["subreddit", "topic", "word_count", "text"]
            ],
            width="stretch",
            hide_index=True,
            height=300,
        )

    return selected_ids


st.title("🧠 Reddit AI Topic Explorer")
st.caption(
    "4,000 Reddit posts · MiniLM semantic embeddings · "
    "UMAP · BERTopic · Interactive clustering"
)

try:
    df, z5, topic_info = load_data()
except Exception as exc:
    st.error(f"Unable to load local project artifacts: {exc}")
    st.stop()

name_lookup = dict(
    zip(topic_info["Topic"], topic_info["Name"])
)

df["topic_label"] = df["topic"].map(
    lambda t: (
        "Unassigned (-1)"
        if t == -1
        else f"Topic {t}: {name_lookup.get(t, '')}"
    )
)

st.sidebar.header("Global document filters")

community_options = sorted(df["subreddit"].unique())

communities = st.sidebar.multiselect(
    "Subreddits",
    community_options,
    default=community_options,
)

minimum_words = st.sidebar.slider(
    "Minimum document words",
    min_value=0,
    max_value=300,
    value=20,
    step=10,
)

search_text = st.sidebar.text_input(
    "Search document text",
    placeholder="e.g. GPU, ChatGPT, agents",
)

filtered = df[
    df["subreddit"].isin(communities)
    & (df["word_count"] >= minimum_words)
].copy()

if search_text.strip():
    filtered = filtered[
        filtered["text"].str.contains(
            search_text.strip(),
            case=False,
            regex=False,
            na=False,
        )
    ]

st.sidebar.caption(
    "Filters affect the displayed documents and corpus statistics. "
    "They do not retrain the saved BERTopic model or the "
    "global clustering experiments."
)

if filtered.empty:
    st.warning("No documents match the filters.")
    st.stop()

m1, m2, m3, m4 = st.columns(4)

m1.metric("Visible documents", f"{len(filtered):,}")
m2.metric("Subreddits", filtered["subreddit"].nunique())
m3.metric(
    "Candidate BERTopic groups",
    filtered.loc[filtered.topic != -1, "topic"].nunique(),
)
m4.metric(
    "Visible Topic -1",
    f"{(filtered.topic == -1).mean() * 100:.1f}%",
)

tab1, tab2, tab3, tab4 = st.tabs([
    "01 · Corpus and TF–IDF",
    "02 · BERTopic Studio",
    "03 · Live Clustering Lab",
    "04 · Embedding Atlas",
])

# ============================================================
# TAB 1: WORD COUNTS AND TF-IDF
# ============================================================

with tab1:
    st.subheader("Word frequency and distinctiveness")

    c1, c2, c3 = st.columns(3)

    remove_stopwords = c1.checkbox(
        "Remove English stop words",
        value=True,
    )

    ngrams = c2.select_slider(
        "N-gram length",
        options=[1, 2, 3],
        value=1,
    )

    n_terms = c3.slider(
        "Number of terms",
        min_value=5,
        max_value=50,
        value=20,
        step=5,
    )

    texts = tuple(filtered["text"].astype(str))

    domain_stopwords = st.checkbox(
        "Exclude generic AI and conversational terms",
        value=False,
        help=(
            "Optionally remove very common Reddit and AI terms "
            "to reveal more specific discussion topics."
        ),
    )

    terms = get_terms(
        texts,
        remove_stopwords,
        ngrams,
    )

    if domain_stopwords:
        excluded = {
            "ai", "gpt", "llm", "llms", "model", "models",
            "like", "just", "use", "using", "new", "people",
            "think", "know", "don", "ve", "time", "really",
            "chatgpt", "openai"
        }

        terms = terms[
            ~terms["term"].str.lower().isin(excluded)
        ]

    top = terms.head(n_terms)

    st.markdown("### Word frequency: sorted bars versus word cloud")

    left, right = st.columns(2)

    with left:
        st.plotly_chart(
            horizontal_bars(
                top,
                "count",
                "term",
                "Sorted word-frequency bars",
            ),
            width="stretch",
        )

        st.caption(
            "Sorted bars encode frequency through position "
            "and length, making precise ranking easier."
        )

    with right:
        cloud = WordCloud(
            width=1100,
            height=750,
            background_color="#17191E",
            colormap="winter",
            max_words=n_terms,
            prefer_horizontal=0.9,
            random_state=42,
        ).generate_from_frequencies(
            dict(zip(top["term"], top["count"]))
        )

        st.image(
            cloud.to_array(),
            width="stretch",
            caption="Word cloud using exactly the same word frequencies",
        )

        st.caption(
            "Word size approximately encodes frequency, "
            "but layout and word length make precise "
            "comparisons difficult."
        )

    st.caption(
        "These bars show actual term counts, not TF–IDF or "
        "semantic importance. The corpus filters and "
        "tokenization choices affect the ranking."
    )

    st.divider()
    st.subheader("Distinctive terms in an individual document")

    st.markdown("#### Find a Reddit document")

    document_search = st.text_input(
        "Search available documents",
        placeholder="Type a keyword, such as GPU, OpenAI, or agents",
        key="tfidf_document_search",
    )

    available = filtered.reset_index(drop=True).copy()
    available["document_position"] = available.index

    if document_search.strip():
        available = available[
            available["text"].str.contains(
                document_search.strip(),
                case=False,
                regex=False,
                na=False,
            )
        ]

    # DOCUMENT SEARCH LOCAL GUARD V1
    if available.empty:
        st.info(
            "No documents match this document-only search. "
            "Clear or change the query. The BERTopic, clustering, "
            "and embedding tabs remain available."
        )
    else:
        document_positions = available["document_position"].tolist()

        document_position = st.selectbox(
            f"Choose a document ({len(document_positions):,} available)",
            options=document_positions,
            format_func=lambda i: (
                f"Post {i + 1:,} | "
                f"r/{filtered.iloc[i]['subreddit']} | "
                f"{str(filtered.iloc[i]['text'])[:95]}"
            ),
            index=0,
            key="tfidf_document_selector",
        )

        scores = get_tfidf(
            texts,
            document_position,
            remove_stopwords,
            ngrams,
        )

        if not scores.empty:
            st.plotly_chart(
                horizontal_bars(
                    scores,
                    "tfidf",
                    "term",
                    "Top document-level TF–IDF terms",
                    "#C07837",
                ),
                width="stretch",
            )

        with st.expander("View Original Reddit Post", expanded=False):
            st.text_area(
                "Original post content",
                value=str(filtered.iloc[document_position]["text"]),
                height=220,
                disabled=True,
            )

        st.caption(
            "TF–IDF is fitted to the currently filtered documents. "
            "Its weights change when the corpus selection changes."
        )

# ============================================================
# TAB 2: TRAINED BERTOPIC
# ============================================================

with tab2:
    st.subheader("BERTopic — saved semantic topic model")

    st.info(
        "Candidate model: 15 topics, 1,327 outliers (33.17%) "
        "across the complete 4,000-post corpus. "
        "These saved topics do not change when filters change."
    )

    included_topics = sorted(
        filtered["topic"].unique().tolist()
    )

    selected_topic = st.selectbox(
        "Inspect a topic",
        included_topics,
        format_func=lambda t: (
            "Unassigned (-1)"
            if t == -1
            else f"{t}: {name_lookup.get(t, '')}"
        ),
    )

    members = filtered[filtered["topic"] == selected_topic]

    st.metric(
        "Visible documents in selected topic",
        f"{len(members):,}",
    )

    if len(members):
        st.dataframe(
            members[
                ["subreddit", "word_count", "text"]
            ],
            width="stretch",
            hide_index=True,
            height=320,
        )

    st.divider()
    st.subheader("Topic-focused BERTopic explorer")
    st.caption(
        "The selected topic and global document filters control this "
        "interactive 2D UMAP map. The topic assignment itself is from "
        "the saved BERTopic model, not a new clustering run."
    )

    # Show selected-topic posts against other currently visible posts.
    # This uses existing coordinates and saved labels; no retraining.
    focused = filtered.copy()
    focused["selection_group"] = np.where(
        focused["topic"].eq(selected_topic),
        "Selected topic", "Other visible topics"
    )
    focus_fig = px.scatter(
        focused,
        x="x", y="y", color="selection_group",
        color_discrete_map={
            "Selected topic": "#69DAC8",
            "Other visible topics": "#68707D",
        },
        hover_data={
            "subreddit": True,
            "post_id": True,
            "word_count": True,
            "x": False,
            "y": False,
        },
        category_orders={"selection_group": [
            "Other visible topics", "Selected topic"
        ]},
        render_mode="webgl",
        height=500,
    )
    focus_fig.update_traces(marker=dict(size=5, opacity=0.75))
    focus_fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#17191E",
        plot_bgcolor="#17191E",
        font=dict(color="#F2F4F8"),
        title=dict(text=f"Selected topic: {selected_topic}",
                   font=dict(color="#F2F4F8")),
        legend_title_text="Document group",
        xaxis_title="UMAP dimension 1",
        yaxis_title="UMAP dimension 2",
    )
    st.plotly_chart(focus_fig, width="stretch", key="bertopic_topic_focus")
    st.caption(
        "UMAP preserves local neighborhoods imperfectly; shapes and "
        "global distances in 2D are not exact semantic distances."
    )

    st.divider()
    st.subheader("Interactive BERTopic visualizations — full model")
    st.caption(
        "These four precomputed figures show the complete 4,000-post "
        "model and are not filtered by the topic selector above. "
        "Use the focused map for a filtered view."
    )

    chart_options = {
        "Topic keywords — c-TF-IDF": "topic_keywords.json",
        "Intertopic distance map": "topic_map.json",
        "Topic hierarchy": "topic_hierarchy.json",
        "Document-level topic map": "document_map.json",
    }
    choice = st.selectbox(
        "Choose full-model visualization",
        list(chart_options),
        key="bertopic_overview_choice",
    )
    chart_path = CHARTS / chart_options[choice]

    if chart_path.exists():
        fig = pio.from_json(chart_path.read_text(encoding="utf-8"))
        # BERTopic may serialize explicit black title / annotation fonts;
        # a Plotly template alone does not override those values.
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#17191E",
            plot_bgcolor="#17191E",
            font=dict(color="#F2F4F8"),
            title_font=dict(color="#F2F4F8"),
            legend_font=dict(color="#F2F4F8"),
        )
        for annotation in fig.layout.annotations or []:
            annotation.font = dict(
                color="#F2F4F8",
                size=annotation.font.size if annotation.font and annotation.font.size else 12,
            )
        for axis in fig.select_xaxes():
            axis.update(tickfont=dict(color="#D2D7E0"),
                        title_font=dict(color="#D2D7E0"))
        for axis in fig.select_yaxes():
            axis.update(tickfont=dict(color="#D2D7E0"),
                        title_font=dict(color="#D2D7E0"))
        st.plotly_chart(fig, width="stretch", key="bertopic_full_chart")
    else:
        st.warning(f"Missing visualization: {chart_path.name}")

# ============================================================
# TAB 3: LIVE CLUSTERING
# ============================================================

with tab3:
    st.subheader("Live clustering laboratory")

    st.markdown(
        "Experiment on the **same saved 5D UMAP embedding** "
        "using three different clustering algorithms. "
        "Change the parameters to compute new global "
        "cluster assignments across all 4,000 posts."
    )

    algorithm = st.radio(
        "Clustering algorithm",
        ["HDBSCAN", "DBSCAN", "KMeans"],
        horizontal=True,
    )

    p1, p2, p3 = st.columns(3)

    k = 15
    eps = 0.8
    samples = 5
    size = 50

    if algorithm == "HDBSCAN":
        size = p1.slider(
            "Minimum cluster size",
            10, 150, 50, 5,
        )
        samples = p2.slider(
            "Minimum samples",
            1, 30, 5,
        )
        p3.caption(
            "HDBSCAN discovers cluster count and may "
            "label documents as noise (-1)."
        )

    elif algorithm == "DBSCAN":
        eps = p1.slider(
            "Neighborhood radius (eps)",
            0.05, 5.0, 0.8, 0.05,
        )
        samples = p2.slider(
            "Minimum samples",
            2, 30, 5,
        )
        p3.caption(
            "DBSCAN is highly sensitive to eps. "
            "Noise is explicitly labeled -1."
        )

    else:
        k = p1.slider(
            "Number of clusters (k)",
            2, 40, 15,
        )
        p2.caption(
            "KMeans always assigns all documents. "
            "The number of clusters is chosen by you."
        )

    # The model is fitted on the complete 4,000-row corpus.
    # The sidebar filters only control which points are shown.
    labels = compute_clusters(
        z5, algorithm, k, eps, samples, size
    )

    found = len(set(labels.tolist()) - {-1})
    outlier_count = int(np.sum(labels == -1))

    a, b, c = st.columns(3)
    a.metric("Discovered clusters", found)
    b.metric("Outliers", f"{outlier_count:,}")
    c.metric(
        "Outlier percentage",
        f"{100 * outlier_count / len(labels):.2f}%",
    )

    experiment = filtered.copy()

    experiment["cluster_id"] = labels[
        experiment["row_id"].to_numpy()
    ]

    experiment["cluster_label"] = experiment[
        "cluster_id"
    ].map(
        lambda v: "Noise (-1)" if v == -1 else f"Cluster {v}"
    )

    color_by = st.selectbox(
        "Color the embedding map by",
        [
            "cluster_label",
            "subreddit",
            "topic_label",
        ],
        format_func=lambda x: {
            "cluster_label": "Experimental cluster",
            "subreddit": "Reddit community",
            "topic_label": "Saved BERTopic topic",
        }[x],
    )

    display_document_map(
        experiment,
        color_by,
        chart_key="clustering_scatter",
    )

    st.caption(
        "The map uses a separately computed **2D UMAP** "
        "projection for visualization. Clustering is "
        "performed on the **5D UMAP** coordinates. "
        "Distances and apparent boundaries in 2D should "
        "not be interpreted as exact 5D geometry."
    )


    # ================= BASELINE VS EXPERIMENT =================

    st.divider()
    st.subheader("Baseline vs. Experiment")
    st.caption(
        "Compare the verified HDBSCAN baseline (50, 5) "
        "against the currently selected clustering configuration. "
        "Both use identical documents and 5D UMAP coordinates."
    )

    from sklearn.metrics import adjusted_rand_score

    baseline_labels = compute_clusters(
        z5, "HDBSCAN", 15, 0.8, 5, 50
    )

    ari_score = adjusted_rand_score(baseline_labels, labels)

    baseline_noise = int(np.sum(baseline_labels == -1))
    experiment_noise = int(np.sum(labels == -1))

    baseline_groups = len(set(baseline_labels.tolist()) - {-1})
    experiment_groups = len(set(labels.tolist()) - {-1})

    b1, b2, b3 = st.columns(3)
    b1.metric(
        "Baseline clusters",
        baseline_groups
    )
    b2.metric(
        "Experiment clusters",
        experiment_groups,
        delta=experiment_groups - baseline_groups
    )
    b3.metric(
        "Adjusted Rand Index",
        f"{ari_score:.4f}"
    )

    metrics = pd.DataFrame({
        "Configuration": ["Verified baseline", "Current experiment"],
        "Clusters": [baseline_groups, experiment_groups],
        "Outliers": [baseline_noise, experiment_noise],
        "Outlier percentage": [
            round(100 * baseline_noise / len(df), 2),
            round(100 * experiment_noise / len(df), 2)
        ]
    })

    st.dataframe(
        metrics,
        width="stretch",
        hide_index=True
    )

    st.caption(
        "ARI compares cluster partitions without assuming "
        "that cluster IDs match. ARI = 1 indicates identical "
        "partitions, including outlier assignments."
    )

    comparison_rows = filtered.copy()
    row_indices = comparison_rows["row_id"].to_numpy()

    comparison_rows["Baseline"] = [
        "Noise (-1)" if v == -1 else f"Cluster {v}"
        for v in baseline_labels[row_indices]
    ]

    comparison_rows["Experiment"] = [
        "Noise (-1)" if v == -1 else f"Cluster {v}"
        for v in labels[row_indices]
    ]

    comparison_rows["Preview"] = (
        comparison_rows["text"].astype(str).str.slice(0, 140)
    )

    left_comparison, right_comparison = st.columns(2)

    for target, grouping, heading in [
        (left_comparison, "Baseline", "Verified HDBSCAN baseline"),
        (right_comparison, "Experiment", f"Current {algorithm} experiment")
    ]:
        with target:
            st.markdown(f"**{heading}**")

            chart = px.scatter(
                comparison_rows,
                x="x",
                y="y",
                color=grouping,
                hover_name="Preview",
                hover_data={"subreddit": True, "x": False, "y": False},
                color_discrete_map={"Noise (-1)": "#8A909A"},
                render_mode="webgl",
                height=470,
                opacity=0.75
            )

            chart.update_traces(marker=dict(size=5))
            chart.update_layout(
                template="plotly_dark",
                paper_bgcolor="#17191E",
                plot_bgcolor="#17191E",
                showlegend=False,
                xaxis_title="UMAP 1",
                yaxis_title="UMAP 2",
                margin=dict(l=5, r=5, t=10, b=10)
            )

            st.plotly_chart(
                chart,
                width="stretch",
                key=f"comparison_{grouping.lower()}"
            )

    baseline_is_noise = baseline_labels == -1
    experiment_is_noise = labels == -1

    gained_assignment = int(np.sum(
        baseline_is_noise & ~experiment_is_noise
    ))
    lost_assignment = int(np.sum(
        ~baseline_is_noise & experiment_is_noise
    ))

    a1, a2 = st.columns(2)

    a1.metric(
        "Previously unassigned, now assigned",
        gained_assignment
    )

    a2.metric(
        "Previously assigned, now unassigned",
        lost_assignment
    )

    st.caption(
        "These last two metrics describe changes in noise status, "
        "not every change of cluster membership. "
        "The side-by-side maps share the same 2D coordinates, "
        "but colors and cluster IDs are independently assigned. "
        "Do not compare cluster numbers directly."
    )

    # =============== END BASELINE VS EXPERIMENT ===============

    st.subheader("Inspect experimental clusters")

    options = sorted(experiment["cluster_id"].unique())

    chosen = st.selectbox(
        "Cluster to inspect",
        options,
        format_func=lambda v: (
            "Noise (-1)"
            if v == -1 else f"Cluster {v}"
        ),
    )

    selected_members = experiment[
        experiment["cluster_id"] == chosen
    ]

    if len(selected_members):
        st.dataframe(
            selected_members[
                ["subreddit", "topic", "word_count", "text"]
            ],
            width="stretch",
            hide_index=True,
            height=300,
        )

    st.download_button(
        "Download current global cluster assignments",
        data=pd.DataFrame({
            "post_id": df["post_id"],
            "subreddit": df["subreddit"],
            "cluster": labels,
        }).to_csv(index=False).encode("utf-8"),
        file_name=f"{algorithm.lower()}_assignments.csv",
        mime="text/csv",
    )

    st.warning(
        "Experimental clusters are not BERTopic topics. "
        "Their cluster IDs do not inherit the saved "
        "BERTopic c-TF-IDF names or descriptions."
    )

# ============================================================
# TAB 4: EMBEDDING EXPLORER
# ============================================================

with tab4:
    st.subheader("Semantic embedding atlas")

    st.markdown(
        "All points are Reddit posts encoded with "
        "**all-MiniLM-L6-v2 (384D)** and projected "
        "to **2D using UMAP**."
    )

    atlas_color = st.selectbox(
        "Color documents by",
        ["subreddit", "topic_label"],
        format_func=lambda x: (
            "Subreddit" if x == "subreddit"
            else "Candidate BERTopic topic"
        ),
    )

    show_noise = st.checkbox(
        "Include Topic -1 documents",
        value=True,
        key="atlas_noise",
    )

    atlas = (
        filtered
        if show_noise
        else filtered[filtered["topic"] != -1]
    )

    if atlas.empty:
        st.info("No documents to display.")
    else:
        display_document_map(
            atlas,
            atlas_color,
            chart_key="atlas_scatter",
            height=700,
        )

    st.caption(
        "UMAP emphasizes neighborhood relationships. "
        "Global distances, cluster sizes, and shape "
        "can be distorted by projection. "
        "Changing the color grouping does not "
        "change the underlying coordinates."
    )

st.divider()

st.caption(
    "Methodology: MiniLM embeddings → UMAP 5D → "
    "HDBSCAN → BERTopic c-TF-IDF. "
    "Exploratory clustering uses the same saved 5D "
    "representation. AI assistance and analytical "
    "limitations should be acknowledged in the "
    "final README."
)
