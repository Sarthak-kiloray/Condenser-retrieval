from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import json
from src.config import get_config
from src.data.datasets import load_corpus

def build_index(corpus_path: str, out_dir: str = "indexes"):
    config = get_config()
    corpus = load_corpus(corpus_path)
    corpus_texts = [doc["text"] for doc in corpus]

    # TF-IDF Vectorizer
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus_texts)
    np.save(f"{out_dir}/tfidf_index.npz", X.toarray())  # Save as NumPy array
    with open(f"{out_dir}/vectorizer.pkl", 'wb') as f:
        pickle.dump(vectorizer, f)

    # Save mapping file (doc_id to text)
    with open(f"{out_dir}/mapping.jsonl", "w", encoding="utf-8") as f:
        for doc in corpus:
            mapping_entry = {"doc_id": doc["id"], "text": doc["text"]}
            f.write(json.dumps(mapping_entry, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    build_index("data/corpus.jsonl", "indexes")
