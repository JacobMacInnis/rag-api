# RAG API: Retrieval-Augmented Generation with FastAPI

This project is a production-ready RAG (Retrieval-Augmented Generation) backend built with FastAPI, SentenceTransformers, and FLAN-T5. It demonstrates a modern approach to question-answering using both semantic search and language generation.

## Features

- Supports both static corpus (preloaded) and one-shot document Q&A
- Dense retrieval via Sentence-BERT (`intfloat/e5-small-v2`)
- Reranking with CrossEncoder (`ms-marco-MiniLM-L-6-v2`)
- Response generation via FLAN-T5
- Fully containerized and deployable via Docker or GCP Cloud Run

## 🛠️ Tech Stack

- **Backend**: FastAPI + Uvicorn
- **Models**: Hugging Face Transformers
- **Retrieval**: FAISS (dense index)
- **Vectorization**: Sentence Transformers
- **Language Model**: `google/flan-t5-small`

## API Endpoints

### `POST /ask`

Ask a question against the pre-indexed corpus.

**Payload:**

```json
{
  "question": "What is the return policy?",
  "top_k": 4,
  "top_k_raw": 20
}
POST /doc_qa
Ask a question about an uploaded .txt document.

Form fields:

file: Upload .txt

question: The question to ask

### Project Structure

├── app/
│   ├── main.py                # FastAPI app
│   └── rag_pipeline/          # RAG implementation (v3)
├── data/
│   └── companyPolicies.txt    # Pre-indexed document
├── Dockerfile
├── requirements.txt
└── README.md
```
