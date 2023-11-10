# Multi-Document QA with GPT-3 and Neural Reranking

## Overview
This project implements a multi-document Question-Answering (QA) system that leverages document retrieval, question decomposition, and neural reranking to provide accurate answers with supporting explanations. The process is detailed in our proposed method, Visconde.

### Pipeline Steps
1. **Sliding Window**: Segment documents into passages of N sentences.
2. **Indexing**: Create an inverted index for efficient passage retrieval.
3. **Retrieval**: Employ Pyserini's BM25 to fetch relevant passages.
4. **Reranking**: Use monoT5 for selecting highly relevant passages.
5. **Question-Answering**: Deploy GPT-3.5-turbo to generate answers based on the reranked passages.
6. **Question Decomposition**: Decompose the main question into sub-questions for a finer-grained QA approach.

### Hardware Requirements
- The system is designed to run on platforms like Colab PRO with an NVIDIA A100 GPU.

## Installation

Ensure you have the necessary packages installed:

```bash
# Pyserini for information retrieval tasks
!pip install pyserini

# Faiss-cpu for efficient similarity search and clustering
!pip install faiss-cpu

# Pygaggle for document reranking
!pip install git+https://github.com/castorini/pygaggle.git

# Hugging Face Transformers for pre-trained NLP models
!pip install transformers --upgrade

# OpenAI for accessing GPT-3.5-turbo and other models
!pip install openai

# Sentence Transformers for computing sentence embeddings
!pip install sentence-transformers
