# Condenser Retrieval

A dense retrieval system implementing Condenser-style architecture for efficient semantic search and information retrieval.

## Overview

This project implements a state-of-the-art dense retriever using contrastive learning principles inspired by the Condenser architecture. The system combines transformer-based encoders with FAISS vector indexing for scalable similarity search, providing both training and inference capabilities through a FastAPI-based REST API and a simple HTML user interface.

### Key Features

- **Contrastive Training**: Train dense retrieval models using contrastive loss functions to learn high-quality representations
- **FAISS Integration**: Build and search vector indexes using Facebook AI Similarity Search for efficient nearest neighbor retrieval
- **FastAPI Inference**: RESTful API for query encoding and retrieval operations
- **Web UI**: Simple HTML interface for interactive search and evaluation
- **Modular Architecture**: Clean separation of concerns across data processing, model definition, training, indexing, and serving

### Architecture

The project is organized into several key modules:

- **data/**: Dataset loading and preprocessing utilities
- **models/**: Condenser model implementation, pooling strategies, and loss functions
- **training/**: Training scripts for contrastive learning and evaluation metrics
- **index/**: FAISS index construction and search functionality
- **api/**: FastAPI server with request/response schemas
- **ui/**: Frontend interface for querying and visualization

### Getting Started

1. Install dependencies: `make setup`
2. Train a model: `make train`
3. Build the index: `make build-index`
4. Start the API: `make run-api`
5. Open the UI in your browser

### Usage

The system supports end-to-end workflows from training to deployment, with configurable parameters for model architecture, training hyperparameters, and indexing strategies. The contrastive training objective learns to pull relevant documents closer together while pushing irrelevant ones apart in the embedding space.

