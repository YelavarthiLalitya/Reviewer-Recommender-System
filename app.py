# ==========================================
# Cross-Document Reviewer Recommendation App (Light Theme, Green Style)
# ==========================================

import streamlit as st
import fitz
import numpy as np
import json
import re
import os
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

# ==========================================
# Config
# ==========================================
EMBEDDINGS_FILE = "author_embeddings_agg.npy"
AUTHOR_NAMES_FILE = "author_names.npy"
AUTHOR_PROFILES_FILE = "author_profiles.json"
TOP_K = 5
BACKUP_PER_AUTHOR = 2

# ==========================================
# Helper Functions
# ==========================================
def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = "".join(page.get_text() for page in doc)
    return text

def preprocess(text):
    return re.findall(r'\b\w+\b', text.lower())

# ==========================================
# Streamlit Page Setup
# ==========================================
st.set_page_config(
    page_title="Reviewer Recommendation System",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main {
        background-color: #fafafa;
        padding: 2rem;
        color: #222;
    }
    h1, h2, h3, h4 {
        color: #1c1c1c;
        font-family: 'Helvetica Neue', sans-serif;
        letter-spacing: -0.2px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.8rem;
        border-bottom: 1px solid #ddd;
        padding-bottom: 0.3rem;
    }
    .recommend-card {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 1rem 1.25rem;
        margin-bottom: 0.75rem;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }
    .recommend-card:hover {
        background-color: #f9f9f9;
        transition: 0.3s;
    }
    .green-text {
        color: #0b6e4f;
        font-weight: 600;
        font-size: 0.95rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ==========================================
# Load Data
# ==========================================
with st.spinner("Loading author profiles and models..."):
    author_texts = {}

    # Load single or multiple parts
    if os.path.exists(AUTHOR_PROFILES_FILE):
        with open(AUTHOR_PROFILES_FILE, "r", encoding="utf-8") as f:
            author_texts = json.load(f)
    else:
        part_idx = 1
        while True:
            part_file = f"author_profiles_part{part_idx}.json"
            if os.path.exists(part_file):
                with open(part_file, "r", encoding="utf-8") as f:
                    part_data = json.load(f)
                    author_texts.update(part_data)
                part_idx += 1
            else:
                break

    if not author_texts:
        st.error("No author profile files found. Please include 'author_profiles.json' or its split parts.")
        st.stop()

    author_names = np.load(AUTHOR_NAMES_FILE, allow_pickle=True)
    author_embeddings = np.load(EMBEDDINGS_FILE)
    author_embeddings = np.array([normalize(e.reshape(1, -1))[0] for e in author_embeddings])
    reviewer_sim_matrix = cosine_similarity(author_embeddings)

# ==========================================
# Load Models
# ==========================================
bert_model = SentenceTransformer("all-mpnet-base-v2")

tagged_docs = [TaggedDocument(preprocess(doc), [i]) for i, doc in enumerate(author_texts.values())]
d2v_model = Doc2Vec(vector_size=300, window=5, min_count=2, workers=4, epochs=40)
d2v_model.build_vocab(tagged_docs)
d2v_model.train(tagged_docs, total_examples=d2v_model.corpus_count, epochs=d2v_model.epochs)

tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words="english")
tfidf = tfidf_vectorizer.fit_transform(list(author_texts.values()))
nmf_model = NMF(n_components=20, random_state=42)
nmf_topics = nmf_model.fit_transform(tfidf)

# ==========================================
# Recommendation Methods
# ==========================================
def recommend_bert(paper_text, top_k=TOP_K, backup_per_author=BACKUP_PER_AUTHOR):
    paper_emb = bert_model.encode(paper_text, convert_to_numpy=True)
    paper_emb = normalize(paper_emb.reshape(1, -1))[0].reshape(1, -1)
    sims = cosine_similarity(paper_emb, author_embeddings)[0]
    top_indices = sims.argsort()[::-1][:top_k]

    results = []
    for i in top_indices:
        author = author_names[i]
        score = float(sims[i])
        reviewer_sims = reviewer_sim_matrix[i]
        backup_indices = reviewer_sims.argsort()[::-1]
        backup_authors = [author_names[j] for j in backup_indices if j != i][:backup_per_author]
        results.append({"author": author, "score": score, "backups": backup_authors})
    return results

def recommend_doc2vec(paper_text, top_k=TOP_K):
    paper_tokens = preprocess(paper_text)
    paper_vec = d2v_model.infer_vector(paper_tokens)
    author_vecs = np.array([d2v_model.dv[i] for i in range(len(author_names))])
    sims = cosine_similarity([paper_vec], author_vecs)[0]
    top_indices = sims.argsort()[::-1][:top_k]
    return [(author_names[i], float(sims[i])) for i in top_indices]

def jaccard_similarity(text1, text2):
    w1, w2 = set(preprocess(text1)), set(preprocess(text2))
    if not w1 or not w2:
        return 0.0
    return len(w1 & w2) / len(w1 | w2)

def recommend_jaccard(paper_text, top_k=TOP_K):
    sims = [jaccard_similarity(paper_text, author_texts[a]) for a in author_names]
    sims = np.array(sims)
    top_indices = sims.argsort()[::-1][:top_k]
    return [(author_names[i], float(sims[i])) for i in top_indices]

def recommend_topic(paper_text, top_k=TOP_K):
    paper_vec = tfidf_vectorizer.transform([paper_text])
    paper_topic = nmf_model.transform(paper_vec)
    sims = np.dot(nmf_topics, paper_topic.T).flatten()
    sims = sims / (np.linalg.norm(nmf_topics, axis=1) * np.linalg.norm(paper_topic))
    top_indices = sims.argsort()[::-1][:top_k]
    return [(author_names[i], float(sims[i])) for i in top_indices]

# ==========================================
# UI
# ==========================================
st.title("Cross-Document Reviewer Recommendation System")
st.markdown("Upload a research paper to identify suitable reviewers using multiple NLP-based approaches.")

uploaded_file = st.file_uploader("Upload a Research Paper (PDF)", type=["pdf"])

tabs = st.tabs([
    "BERT (Cross-Document)",
    "Doc2Vec",
    "Jaccard",
    "Topic Modeling",
    "Comparison"
])

if uploaded_file:
    paper_text = extract_text_from_pdf(uploaded_file)

    with st.spinner("Processing the paper and generating reviewer recommendations..."):
        bert_results = recommend_bert(paper_text)
        doc2vec_results = recommend_doc2vec(paper_text)
        jaccard_results = recommend_jaccard(paper_text)
        topic_results = recommend_topic(paper_text)

    st.success("Recommendations generated successfully.")

    # BERT
    with tabs[0]:
        st.subheader("BERT (Cross-Document Embedding) Recommendations")
        for rank, rec in enumerate(bert_results, start=1):
            st.markdown(
                f"""
                <div class="recommend-card">
                    <strong class="green-text">{rank}. {rec['author']}</strong><br>
                    <span class="green-text">Similarity: {rec['score']:.4f}</span><br>
                    <span class="green-text">Backup Reviewers: {', '.join(rec['backups'])}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Doc2Vec
    with tabs[1]:
        st.subheader("Doc2Vec Recommendations")
        for rank, (author, score) in enumerate(doc2vec_results, start=1):
            st.markdown(
                f"<div class='recommend-card'><strong class='green-text'>{rank}. {author}</strong><br><span class='green-text'>Similarity: {score:.4f}</span></div>",
                unsafe_allow_html=True,
            )

    # Jaccard
    with tabs[2]:
        st.subheader("Jaccard Similarity Recommendations")
        for rank, (author, score) in enumerate(jaccard_results, start=1):
            st.markdown(
                f"<div class='recommend-card'><strong class='green-text'>{rank}. {author}</strong><br><span class='green-text'>Score: {score:.4f}</span></div>",
                unsafe_allow_html=True,
            )

    # Topic Modeling
    with tabs[3]:
        st.subheader("Topic Modeling (NMF) Recommendations")
        for rank, (author, score) in enumerate(topic_results, start=1):
            st.markdown(
                f"<div class='recommend-card'><strong class='green-text'>{rank}. {author}</strong><br><span class='green-text'>Score: {score:.4f}</span></div>",
                unsafe_allow_html=True,
            )

    # Comparison
    with tabs[4]:
        st.subheader("Cross-Method Comparison")
        df = pd.DataFrame({
            "BERT": [r['author'] for r in bert_results],
            "Doc2Vec": [a for a, _ in doc2vec_results],
            "Jaccard": [a for a, _ in jaccard_results],
            "TopicModel": [a for a, _ in topic_results],
        })

        styled_df = df.style.set_properties(**{
            'color': '#0b6e4f',
            'font-weight': '600'
        })

        st.dataframe(styled_df, use_container_width=True)
        st.caption("Comparison of top reviewers identified by each model.")
else:
    st.info("Upload a PDF to start reviewer recommendation.")
