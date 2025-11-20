"""Script to create demo data for smoke testing."""

from src.config import get_config
from src.data.preprocess import build_demo_corpus


def create_demo_data():
    """Create demo corpus and training pairs for quick smoke testing."""
    config = get_config()

    # Create data directory
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Build demo corpus
    build_demo_corpus()

    # Load corpus to create pairs
    from src.data.datasets import load_corpus

    corpus = load_corpus(str(config.CORPUS_PATH))
    if not corpus:
        print("Warning: No corpus found. Run build_demo_corpus() first.")
        return

    # Create synthetic Q/A pairs
    train_pairs = [
        ("What is Python?", "Python is a high-level programming language known for its simplicity and readability. It is widely used in data science, web development, and artificial intelligence."),
        ("What is machine learning?", "Machine learning is a subset of artificial intelligence that enables systems to learn from data without being explicitly programmed. It powers many modern applications."),
        ("What is NLP?", "Natural language processing (NLP) is a field of AI that focuses on the interaction between computers and human language. It enables machines to understand and generate text."),
        ("What is deep learning?", "Deep learning uses neural networks with multiple layers to learn complex patterns in data. It has revolutionized image recognition, speech processing, and language understanding."),
        ("What is information retrieval?", "Information retrieval is the process of finding relevant documents from a large collection. It is fundamental to search engines and recommendation systems."),
        ("What are vector embeddings?", "Vector embeddings are dense representations of text or objects in a high-dimensional space. They capture semantic meaning and enable similarity search."),
        ("What is contrastive learning?", "Contrastive learning is a training technique that learns representations by pulling similar items together and pushing dissimilar ones apart in embedding space."),
        ("What is FAISS?", "FAISS is a library for efficient similarity search and clustering of dense vectors. It can handle billions of vectors and is widely used in production systems."),
        ("What are transformers?", "Transformers are neural network architectures that use attention mechanisms to process sequences. They have become the foundation of modern NLP models."),
        ("What is BERT?", "BERT is a bidirectional transformer model that learns contextualized word representations. It has been pre-trained on large text corpora and can be fine-tuned for various tasks."),
        ("What is dense retrieval?", "Dense retrieval uses learned embeddings to find relevant documents. It is more efficient than traditional keyword-based search and can capture semantic meaning."),
        ("What is semantic search?", "Semantic search goes beyond keyword matching to understand the meaning and intent behind queries. It uses AI to find contextually relevant results."),
        ("What is fine-tuning?", "Fine-tuning adapts pre-trained models to specific tasks by training on domain-specific data. It requires less data and computation than training from scratch."),
        ("What is tokenization?", "Tokenization is the process of breaking text into smaller units called tokens. It is the first step in most NLP pipelines and affects model performance."),
        ("What is query understanding?", "Query understanding is the ability to interpret user queries and extract intent. It is crucial for building effective search and retrieval systems."),
    ]

    # Find matching documents for queries
    train_pairs_with_docs = []
    for query, answer_text in train_pairs:
        # Find the document that contains this text
        matching_doc = None
        for doc in corpus:
            if answer_text.lower() in doc["text"].lower():
                matching_doc = doc
                break

        if matching_doc:
            train_pairs_with_docs.append((query, matching_doc["text"]))
        else:
            # Fallback: use the answer text as document
            train_pairs_with_docs.append((query, answer_text))

    # Write training pairs to TSV
    train_pairs_path = config.TRAIN_PAIRS_PATH
    with open(train_pairs_path, "w", encoding="utf-8") as f:
        for query, doc_text in train_pairs_with_docs:
            f.write(f"{query}\t{doc_text}\n")

    print(f"Created {len(train_pairs_with_docs)} training pairs at {train_pairs_path}")
    print(f"Demo data created successfully!")


if __name__ == "__main__":
    create_demo_data()

