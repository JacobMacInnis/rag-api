# RAG API: Retrieval-Augmented Generation with FastAPI

This project is a production-ready Retrieval-Augmented Generation (RAG) backend built using FastAPI, SentenceTransformers, FAISS, and FLAN-T5. It demonstrates a modern approach to question answering using both semantic search and local language generation, suitable for secure and private deployments.

## Key Features

- Supports both pre-indexed document corpora and one-shot document Q&A
- Dense retrieval using `intfloat/e5-small-v2` Sentence-BERT
- Reranking with `cross-encoder/ms-marco-TinyBERT-L-6`
- Natural language response generation via `google/flan-t5-base`
- RAG Fusion mode: generates multiple answers from top passages and reranks the results
- Customizable prompt parameters: document type and number of sentences
- Fully local and self-contained (no external APIs or OpenAI dependency)
- Containerized for fast deployment on Docker or Google Cloud Run

## Tech Stack

- **Backend**: FastAPI + Uvicorn
- **Models**: Hugging Face Transformers
- **Retrieval**: FAISS (vector similarity search)
- **Embeddings**: Sentence Transformers
- **Language Model**: FLAN-T5 (base)
- **Reranking**: Hugging Face CrossEncoder

## API Endpoints

### `POST /ask`

Ask a question against the pre-indexed corpus using traditional RAG.

**Payload:**

```json
{
  "question": "What is the return policy?",
  "top_k": 4,
  "top_k_raw": 20,
  "num_sentences": 1
}
```

### `POST /ask_fusion`

Ask a question using RAG Fusion mode, which generates one answer per passage and reranks the outputs.

**Payload:**

```json
{
  "question": "What happens if an employee violates policy?",
  "top_k": 4,
  "top_k_raw": 20,
  "num_sentences": 2
}
```

### `POST /doc_qa`

Ask a question about an uploaded .txt document using an ad-hoc RAG pipeline (on-the-fly FAISS index).

##### Form fields:

- file: Upload a .txt document
- question: The question to ask
- doc_type: (Optional) Context type (e.g. "resume", "contract", "report"); defaults to "company-policy"
- num_sentences: (Optional) Desired number of output sentences; defaults to 1

Example:
Asking 3-sentence question about a resume:

```bash
curl -X POST http://localhost:8080/doc_qa \
  -F "file=@resume.txt" \
  -F "question=What are the candidate's key skills?" \
  -F "doc_type=resume" \
  -F "num_sentences=3"
```

### Project Structure

```graphql
├── app/
│   ├── main.py                  # FastAPI endpoints
│   └── rag_pipeline_v2.py       # Core RAG logic and models
├── data/
│   └── companyPolicies.txt      # Pre-indexed document for RAG
├── Dockerfile                   # Container definition
├── requirements.txt             # Python dependencies
├── cloudbuild.yaml              # Google Cloud Build configuration
└── README.md
```

### Deployment Options

#### Local (Docker)

```bash
docker build -t rag-api .
docker run -p 8080:8080 rag-api
```

#### Google Cloud Run

```bash
gcloud builds submit --config cloudbuild.yaml
```

Requires:

A GCP project with Cloud Run and Cloud Build APIs enabled

Sufficient memory allocation (at least 2.75Gi for FLAN-T5 base)

### Notes

- compute is very small for POC project and for real use a larger model and more powerful machine would be preferred

- This project keeps all processing local and avoids third-party model APIs, making it ideal for private or compliance-sensitive deployments.

- RAG Fusion improves response robustness by considering multiple passage-level generations.

- Future enhancements could include streaming token output (via simulated or server-backed methods), feedback loops, or model upgrades.
