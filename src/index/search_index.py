import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from src.config import get_config
from src.data.datasets import load_corpus

def search_index(query, index_dir="indexes", k=5):
    # Load the TF-IDF matrix and vectorizer
    tfidf_matrix = np.load(f"{index_dir}/tfidf_index.npz")
    with open(f"{index_dir}/vectorizer.pkl", 'rb') as f:
        vectorizer = pickle.load(f)

    # Convert the query into a TF-IDF vector
    query_vector = vectorizer.transform([query]).toarray()

    # Calculate cosine similarity between the query vector and the TF-IDF matrix
    similarities = cosine_similarity(query_vector, tfidf_matrix)
    sorted_indices = similarities[0].argsort()[::-1][:k]

    # Load corpus mapping
    mapping = []
    with open(f"{index_dir}/mapping.jsonl", "r", encoding="utf-8") as f:
        mapping = [json.loads(line) for line in f]

    # Get top k results
    results = []
    for idx in sorted_indices:
        result = mapping[idx]
        result["score"] = similarities[0][idx]
        results.append(result)

    return results

if __name__ == "__main__":
    query = "What is Python?"
    results = search_index(query)
    for res in results:
        print(res)
