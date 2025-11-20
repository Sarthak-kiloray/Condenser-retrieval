# Condenser Retrieval

This project implements a lightweight semantic search engine using
TF‑IDF embeddings, FastAPI and a simple web interface. It is inspired by
the original condenser dense retrieval architecture but has been
simplified to run in environments without GPU support or large deep
learning libraries. The goal of this version is to provide a working
example of semantic search that can be run locally with minimal setup.

# Objective

The original Condenser Retrieval system relied on PyTorch, HuggingFace
Transformers and FAISS to learn dense representations and build
approximate nearest neighbour indexes. Those dependencies are not
available in this environment, so this simplified fork replaces them
with scikit‑learn's TfidfVectorizer. Documents and queries are
represented as normalised TF‑IDF vectors, and similarity is computed via
cosine similarity (dot product). While this approach is not as
powerful as deep neural models, it still enables meaningful semantic
search and can be executed entirely on CPU.


# Project structure

src/data/ – utilities to create a small demo corpus and training
pairs. The demo corpus contains a handful of short documents about
natural language processing and machine learning.

src/index/ – scripts for building a TF‑IDF index (build_index.py)
and searching it (search_index.py). Running build_index.py will
vectorise all documents in the corpus and persist the embeddings,
fitted vectoriser and a mapping file.

src/api/ – a FastAPI server (server.py) that loads the TF‑IDF
index on startup and exposes /health and /search endpoints. It
returns the top k most similar documents for a given query.

src/ui/ – a simple HTML/JavaScript front‑end for interacting with
the search API. It allows you to type queries and displays ranked
results.


The training/ and models/ modules from the original repository are
retained for reference but are not used in this simplified version.


#  Prerequisites

The only runtime dependencies are fastapi, uvicorn, pydantic and
scikit‑learn, all of which are available in this environment. You do
not need PyTorch, Transformers or FAISS. To set up the project, run: