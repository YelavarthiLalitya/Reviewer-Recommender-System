import os
import fitz
import json
import numpy as np
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.utils import simple_preprocess
import seaborn as sns
import matplotlib.pyplot as plt

# Optional: WMD
try:
    from gensim.models import KeyedVectors
    wmd_available = True
except ImportError:
    wmd_available = False

# -----------------------------
# Config
# -----------------------------
AUTHOR_PROFILES_JSON = "author_profiles.json"
TEST_PAPERS_FOLDER = "TestPapers"
TOP_K = 5
NUM_TOPICS = 20  # for NMF topic modeling
WORD2VEC_PATH = "GoogleNews-vectors-negative300.bin.gz"

# -----------------------------
# Load author profiles
# -----------------------------
with open(AUTHOR_PROFILES_JSON, "r", encoding="utf-8") as f:
    author_texts = json.load(f)

author_names = list(author_texts.keys())
author_docs = list(author_texts.values())
num_authors = len(author_names)

# -----------------------------
# Helper: extract text from PDF
# -----------------------------
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# -----------------------------
# Preprocessing helper
# -----------------------------
def preprocess(text):
    return re.findall(r'\b\w+\b', text.lower())

# -----------------------------
# 1️⃣ BERT Embeddings Method
# -----------------------------
bert_model = SentenceTransformer('all-mpnet-base-v2')
bert_embeddings = np.array([bert_model.encode(doc, convert_to_numpy=True) for doc in author_docs])

def recommend_bert(input_text, top_k=TOP_K):
    paper_emb = bert_model.encode(input_text, convert_to_numpy=True).reshape(1, -1)
    sims = cosine_similarity(paper_emb, bert_embeddings)[0]
    top_indices = sims.argsort()[::-1][:top_k]
    return [(author_names[i], float(sims[i])) for i in top_indices]

# -----------------------------
# 2️⃣ Doc2Vec Method
# -----------------------------
tagged_docs = [TaggedDocument(preprocess(doc), [i]) for i, doc in enumerate(author_docs)]
d2v_model = Doc2Vec(vector_size=300, window=5, min_count=2, workers=4, epochs=40)
d2v_model.build_vocab(tagged_docs)
d2v_model.train(tagged_docs, total_examples=d2v_model.corpus_count, epochs=d2v_model.epochs)

def recommend_doc2vec(input_text, top_k=TOP_K):
    paper_tokens = preprocess(input_text)
    paper_vector = d2v_model.infer_vector(paper_tokens)
    author_vectors = np.array([d2v_model.dv[i] for i in range(len(author_docs))])
    sims = cosine_similarity([paper_vector], author_vectors)[0]
    top_indices = sims.argsort()[::-1][:top_k]
    return [(author_names[i], float(sims[i])) for i in top_indices]

# -----------------------------
# 3️⃣ Jaccard Similarity
# -----------------------------
def jaccard_similarity(text1, text2):
    words1 = set(preprocess(text1))
    words2 = set(preprocess(text2))
    if not words1 or not words2:
        return 0.0
    return len(words1 & words2) / len(words1 | words2)

def recommend_jaccard(input_text, top_k=TOP_K):
    sims = [jaccard_similarity(input_text, author_texts[author]) for author in author_names]
    sims = np.array(sims)
    top_indices = sims.argsort()[::-1][:top_k]
    return [(author_names[i], float(sims[i])) for i in top_indices]

# -----------------------------
# 4️⃣ Topic Modeling (NMF)
# -----------------------------
tf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
tfidf = tf_vectorizer.fit_transform(author_docs)
nmf_model = NMF(n_components=NUM_TOPICS, random_state=42)
nmf_topics = nmf_model.fit_transform(tfidf)

def recommend_topic(input_text, top_k=TOP_K):
    paper_vec = tf_vectorizer.transform([input_text])
    paper_topic = nmf_model.transform(paper_vec)
    sims = np.dot(nmf_topics, paper_topic.T).flatten()
    sims = sims / (np.linalg.norm(nmf_topics, axis=1) * np.linalg.norm(paper_topic))
    top_indices = sims.argsort()[::-1][:top_k]
    return [(author_names[i], float(sims[i])) for i in top_indices]

# -----------------------------
# 5️⃣ Word Mover's Distance (optional)
# -----------------------------
if wmd_available and os.path.exists(WORD2VEC_PATH):
    wmd_vectors = KeyedVectors.load_word2vec_format(WORD2VEC_PATH, binary=True)

    def recommend_wmd(input_text, top_k=TOP_K):
        paper_tokens = preprocess(input_text)
        sims = []
        for author in author_names:
            author_tokens = preprocess(author_texts[author])
            try:
                distance = wmd_vectors.wmdistance(paper_tokens, author_tokens)
                sim = 1 / (1 + distance)
            except:
                sim = 0.0
            sims.append(sim)
        sims = np.array(sims)
        top_indices = sims.argsort()[::-1][:top_k]
        return [(author_names[i], float(sims[i])) for i in top_indices]
else:
    recommend_wmd = None
    print("WMD embeddings not found. Skipping Word Mover's Distance method.")

# -----------------------------
# Evaluate all methods on papers
# -----------------------------
def evaluate_all_methods():
    pdf_files = [os.path.join(TEST_PAPERS_FOLDER, f) for f in os.listdir(TEST_PAPERS_FOLDER) if f.endswith(".pdf")]

    for pdf_path in pdf_files:
        print(f"\n===== Evaluating {os.path.basename(pdf_path)} =====")
        paper_text = extract_text_from_pdf(pdf_path)

        methods = {
            "BERT Embeddings": recommend_bert,
            "Doc2Vec": recommend_doc2vec,
            "Jaccard": recommend_jaccard,
            "Topic Modeling (NMF)": recommend_topic
        }

        if recommend_wmd is not None:
            methods["Word Mover's Distance"] = recommend_wmd

        for name, func in methods.items():
            try:
                top_reviewers = func(paper_text, top_k=TOP_K)
                print(f"\n{name} Top-{TOP_K}:")
                for rank, (author, score) in enumerate(top_reviewers, start=1):
                    print(f"{rank}. {author} (Score: {score:.4f})")
            except Exception as e:
                print(f"{name} failed: {e}")

# -----------------------------
# 6️⃣ Reviewer-Reviewer Similarity
# -----------------------------
def compute_reviewer_similarity(method="bert", top_n=5):
    print("\n===== Computing Reviewer-Reviewer Similarity =====")
    if method == "bert":
        embeddings = bert_embeddings
    elif method == "doc2vec":
        embeddings = np.array([d2v_model.dv[i] for i in range(len(author_docs))])
    else:
        raise ValueError("Unsupported method for reviewer similarity.")

    sim_matrix = cosine_similarity(embeddings)
    top_similar = {}

    for i, author in enumerate(author_names):
        sims = [(author_names[j], float(sim_matrix[i, j])) for j in range(len(author_names)) if j != i]
        sims_sorted = sorted(sims, key=lambda x: x[1], reverse=True)[:top_n]
        top_similar[author] = sims_sorted

    # Save full similarity to JSON
    with open("reviewer_similarity.json", "w", encoding="utf-8") as f:
        json.dump(top_similar, f, indent=2)

    # Flatten all pairs and print top 20
    all_pairs = [(a, b, s) for a, lst in top_similar.items() for b, s in lst]
    all_pairs_sorted = sorted(all_pairs, key=lambda x: x[2], reverse=True)
    print("\nTop 20 reviewer pairs by similarity:")
    for a, b, s in all_pairs_sorted[:20]:
        print(f"{a} <-> {b} (Score: {s:.4f})")

    # Heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(sim_matrix, xticklabels=author_names, yticklabels=author_names, cmap="YlGnBu")
    plt.title("Reviewer-Reviewer Similarity")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig("reviewer_similarity_heatmap.png")
    print("\nHeatmap saved as reviewer_similarity_heatmap.png")

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    evaluate_all_methods()
    compute_reviewer_similarity(method="bert", top_n=5)