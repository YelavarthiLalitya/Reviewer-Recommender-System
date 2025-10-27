import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity

AGGREGATE_METHOD = True  # True: one embedding per author

# -----------------------------
# Load author profiles from JSON
# -----------------------------
with open("author_profiles.json", "r", encoding="utf-8") as f:
    author_texts = json.load(f)

author_names = list(author_texts.keys())
author_docs = list(author_texts.values())

# -----------------------------
# Load SentenceTransformer model
# -----------------------------
model = SentenceTransformer('all-mpnet-base-v2')

# -----------------------------
# Compute embeddings
# -----------------------------
author_embeddings = {}

for author, text in zip(author_names, author_docs):
    print(f"Encoding author: {author}")
    emb = model.encode(text, convert_to_numpy=True)
    author_embeddings[author] = normalize(emb.reshape(1, -1))[0]

# -----------------------------
# Save embeddings
# -----------------------------
np.save("author_embeddings_agg.npy", np.array(list(author_embeddings.values())))
np.save("author_names.npy", np.array(author_names))
print(f"Saved embeddings for {len(author_embeddings)} authors.")