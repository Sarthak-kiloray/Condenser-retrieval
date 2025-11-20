import os

def build_index(corpus_path: str, out_dir: str = "indexes"):
    config = get_config()
    corpus = load_corpus(corpus_path)
    corpus_texts = [doc["text"] for doc in corpus]

    # Ensure the output directory exists
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

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

