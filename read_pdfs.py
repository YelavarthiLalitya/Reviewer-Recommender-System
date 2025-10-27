import os
import fitz  # PyMuPDF
import json

# Path to your dataset folder
dataset_path = r"C:\Users\Lalitya\Desktop\Projects\aa2\Dataset"

# Dictionary to hold author profiles
# Key: author name, Value: concatenated text of all their papers
author_profiles = {}

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file using PyMuPDF."""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Iterate over authors
for author in os.listdir(dataset_path):
    author_folder = os.path.join(dataset_path, author)
    if os.path.isdir(author_folder):
        texts = []
        for file in os.listdir(author_folder):
            if file.endswith(".pdf"):
                pdf_path = os.path.join(author_folder, file)
                print(f"Parsing {pdf_path} ...")
                text = extract_text_from_pdf(pdf_path)
                texts.append(text)
        # Concatenate all texts for this author
        author_profiles[author] = " ".join(texts)

print(f"Total authors processed: {len(author_profiles)}")

# -----------------------------
# Save author_profiles to JSON
# -----------------------------
output_json = "author_profiles.json"
with open(output_json, "w", encoding="utf-8") as f:
    json.dump(author_profiles, f, ensure_ascii=False, indent=2)

print(f"Saved author profiles to {output_json}")