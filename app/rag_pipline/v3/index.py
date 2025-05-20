# app/rag_pipeline_v2.py
import os, re, torch, faiss, nltk
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ───────────────────────────────────────────
#  Config / one-time start-up
# ───────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# sentence-piece tokenizer for chunking
nltk.download("punkt", quiet=True)

def load_docs(fp: str, chunk_tokens: int = 180, overlap: int = 30) -> list[str]:
    """Split the file into ~180-token chunks with 1-chunk overlap."""
    text = open(fp).read()
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    sents = [s for p in paras for s in nltk.sent_tokenize(p)]

    chunks, buff, cur_len = [], [], 0
    for sent in sents:
        tok_len = len(sent.split())
        if cur_len + tok_len > chunk_tokens:          # start new chunk
            chunks.append(" ".join(buff))
            buff, cur_len = [], 0
        buff.append(sent)
        cur_len += tok_len
    if buff:
        chunks.append(" ".join(buff))

    # 1-chunk overlap to keep context continuity
    final = []
    for i in range(len(chunks)):
        start = max(0, i - 1)
        final.append(" ".join(chunks[start : i + 1]))
    return final

# dense retriever
embedder = SentenceTransformer(
    "intfloat/e5-small-v2",
    device=str(DEVICE)        # "cpu" or "cuda"
)

# cross-encoder reranker
reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    device=str(DEVICE)
)

# generator
gen_tok = AutoTokenizer.from_pretrained("google/flan-t5-small")
gen_mod = AutoModelForSeq2SeqLM.from_pretrained(
    "google/flan-t5-small"
).to(DEVICE).eval()

# build index
PASSAGES = load_docs("data/companyPolicies.txt")
EMBEDS   = embedder.encode(
    PASSAGES, batch_size=64, convert_to_numpy=True, show_progress_bar=False
)
index = faiss.IndexFlatL2(EMBEDS.shape[1])
index.add(EMBEDS)

# ───────────────────────────────────────────
#  Helper functions used by FastAPI route
# ───────────────────────────────────────────
def embed_question(q: str):
    vec = embedder.encode([q], convert_to_numpy=True)
    return vec.reshape(1, -1)          # (1, dim) for FAISS

def retrieve_rerank(question: str, top_k_raw: int, final_k: int):
    q_vec = embed_question(question)
    _, idxs = index.search(q_vec, top_k_raw)
    raw_ctx = [PASSAGES[i] for i in idxs[0]]

    scores = reranker.predict([[question, p] for p in raw_ctx])
    best   = sorted(zip(raw_ctx, scores),
                    key=lambda x: x[1], reverse=True)[:final_k]
    return [p for p, _ in best]

PROMPT_TEMPLATE = (
    "You are an AI assistant answering questions about a company-policy "
    "document. Use only the provided context. "
    "If the context is insufficient, reply \"I don't know.\""
    "\n\n### Question:\n{question}"
    "\n\n### Context:\n{context}"
    "\n\n### Answer (one clear sentence):"
)

def generate(context: str, question: str) -> str:
    prompt = PROMPT_TEMPLATE.format(question=question, context=context)
    inp = gen_tok([prompt], return_tensors="pt",
                  truncation=True, padding=True).to(DEVICE)
    with torch.no_grad():
        out = gen_mod.generate(
            **inp,
            max_length=64,
            num_beams=4,
            repetition_penalty=1.15,
            length_penalty=0.9,
            early_stopping=True,
        )
    return gen_tok.decode(out[0], skip_special_tokens=True).strip()


def answer_from_text(doc_text: str, question: str,
                     top_k_raw: int = 20, final_k: int = 4) -> str:
    """Ad-hoc RAG: build index on the given text and answer the question."""
    passages = load_docs_from_string(doc_text)          # split
    embeds   = embedder.encode(
        passages, batch_size=64, convert_to_numpy=True, show_progress_bar=False
    )
    ix = faiss.IndexFlatL2(embeds.shape[1])
    ix.add(embeds)

    # retrieval + rerank
    q_vec = embed_question(question)
    _, idxs = ix.search(q_vec, top_k_raw)
    raw = [passages[i] for i in idxs[0]]

    scores = reranker.predict([[question, p] for p in raw])
    best   = sorted(zip(raw, scores), key=lambda x: x[1], reverse=True)[:final_k]
    ctx    = "\n".join(p for p, _ in best)

    return generate(ctx, question)

def load_docs_from_string(text: str, chunk_tokens: int = 180, overlap: int = 30):
    """Same splitter as load_docs() but takes raw string instead of file path."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    sents = [s for p in paras for s in nltk.sent_tokenize(p)]
    chunks, buff, cur_len = [], [], 0
    for sent in sents:
        tok_len = len(sent.split())
        if cur_len + tok_len > chunk_tokens:
            chunks.append(" ".join(buff))
            buff, cur_len = [], 0
        buff.append(sent)
        cur_len += tok_len
    if buff:
        chunks.append(" ".join(buff))
    out = []
    for i in range(len(chunks)):
        start = max(0, i - 1)
        out.append(" ".join(chunks[start : i + 1]))
    return out


# ───────────────────────────────────────────
#  Pydantic model for the /ask route
# ───────────────────────────────────────────
class Query(BaseModel):
    question: str
    top_k: int = 4        # passages after rerank
    top_k_raw: int = 20   # passages from FAISS before rerank
