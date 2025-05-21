import os, re, torch, faiss, nltk
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ───────────────────────────────────────────
#  Config / one-time start-up
# ───────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

nltk.download("punkt", quiet=True)

def load_docs(fp: str, chunk_tokens: int = 180, overlap: int = 30) -> list[str]:
    """Split file into token-limited chunks with overlap."""
    text = open(fp).read()
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    sents = [s for p in paras for s in nltk.sent_tokenize(p)]

    chunks = []
    i = 0
    while i < len(sents):
        buff = []
        cur_len = 0
        j = i
        while j < len(sents) and cur_len + len(sents[j].split()) <= chunk_tokens:
            buff.append(sents[j])
            cur_len += len(sents[j].split())
            j += 1
        chunks.append(" ".join(buff))
        i += max(1, len(buff) - overlap)
    return chunks


def load_docs_from_string(text: str, chunk_tokens: int = 180, overlap: int = 30) -> list[str]:
    """Split a string into token-limited chunks with overlap."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    sents = [s for p in paras for s in nltk.sent_tokenize(p)]

    chunks = []
    i = 0
    while i < len(sents):
        buff = []
        cur_len = 0
        j = i
        while j < len(sents) and cur_len + len(sents[j].split()) <= chunk_tokens:
            buff.append(sents[j])
            cur_len += len(sents[j].split())
            j += 1
        chunks.append(" ".join(buff))
        i += max(1, len(buff) - overlap)
    return chunks


# ───────────────────────────────────────────
#  Model Initialization
# ───────────────────────────────────────────
embedder = SentenceTransformer(
    "intfloat/e5-small-v2",
    device=str(DEVICE)
)

reranker = CrossEncoder(
    "cross-encoder/ms-marco-TinyBERT-L-6",
    device=str(DEVICE)
)

gen_tok = AutoTokenizer.from_pretrained("google/flan-t5-base")
gen_mod = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base").to(DEVICE).eval()

# ───────────────────────────────────────────
#  FAISS Index Build
# ───────────────────────────────────────────
PASSAGES = load_docs("data/companyPolicies.txt")
print(f"[INFO] Loaded {len(PASSAGES)} passages from companyPolicies.txt")

EMBEDS = embedder.encode(
    PASSAGES, batch_size=64, convert_to_numpy=True, show_progress_bar=False
)
index = faiss.IndexFlatL2(EMBEDS.shape[1])
index.add(EMBEDS)


# ───────────────────────────────────────────
#  Prompt Template
# ───────────────────────────────────────────
def build_prompt(question: str, context: str, doc_type: str = "company-policy", num_sentences: int = 1) -> str:
    return (
        f"You are an AI assistant answering questions about a {doc_type} document. "
        f"Use only the provided context. If the context is insufficient, reply \"I don't know.\""
        f"\n\n### Question:\n{question}"
        f"\n\n### Context:\n{context}"
        f"\n\n### Answer (respond in {num_sentences} clear sentence{'s' if num_sentences > 1 else ''}):"
    )

# ───────────────────────────────────────────
#  RAG Pipeline Components
# ───────────────────────────────────────────
def embed_question(q: str):
    vec = embedder.encode([q], convert_to_numpy=True)
    return vec.reshape(1, -1)

def retrieve_rerank(question: str, top_k_raw: int, final_k: int):
    q_vec = embed_question(question)
    _, idxs = index.search(q_vec, top_k_raw)
    raw_ctx = [PASSAGES[i] for i in idxs[0]]

    scores = reranker.predict([[question, p] for p in raw_ctx])
    best = sorted(zip(raw_ctx, scores), key=lambda x: x[1], reverse=True)[:final_k]
    return [p for p, _ in best]


def generate(context: str, question: str, doc_type: str = "company-policy", num_sentences: int = 1) -> str:
    prompt = build_prompt(question, context, doc_type, num_sentences)

    inp = gen_tok([prompt], return_tensors="pt", truncation=True, padding=True).to(DEVICE)
    with torch.no_grad():
        out = gen_mod.generate(
            **inp,
            max_length=160,
            num_beams=4,
            repetition_penalty=1.15,
            length_penalty=0.9,
            early_stopping=True,
        )
    return gen_tok.decode(out[0], skip_special_tokens=True).strip()


def rag_fusion(question: str, top_k_raw: int = 20, final_k: int = 4,
               doc_type: str = "company-policy", num_sentences: int = 1) -> str:
    """Answer using RAG Fusion: generate answers per passage, then rerank."""
    q_vec = embed_question(question)
    _, idxs = index.search(q_vec, top_k_raw)
    raw_ctx = [PASSAGES[i] for i in idxs[0]]

    scores = reranker.predict([[question, p] for p in raw_ctx])
    top_ctxs = sorted(zip(raw_ctx, scores), key=lambda x: x[1], reverse=True)[:final_k]

    answers = []
    for ctx, _ in top_ctxs:
        prompt = build_prompt(question, context=ctx, doc_type=doc_type, num_sentences=num_sentences)
        inp = gen_tok([prompt], return_tensors="pt", truncation=True, padding=True).to(DEVICE)
        with torch.no_grad():
            out = gen_mod.generate(
                **inp,
                max_length=160,
                num_beams=4,
                repetition_penalty=1.15,
                length_penalty=0.9,
                early_stopping=True,
            )
        ans = gen_tok.decode(out[0], skip_special_tokens=True).strip()
        answers.append((ctx, ans))

    ranked = reranker.predict([[question, a] for _, a in answers])
    best = sorted(zip(answers, ranked), key=lambda x: x[1], reverse=True)
    return best[0][0][1]


def answer_from_text(doc_text: str, question: str,
                     top_k_raw: int = 20, final_k: int = 4,
                     doc_type: str = "company-policy", num_sentences: int = 1) -> str:
    
    passages = load_docs_from_string(doc_text)
    embeds = embedder.encode(
        passages, batch_size=64, convert_to_numpy=True, show_progress_bar=False
    )
    ix = faiss.IndexFlatL2(embeds.shape[1])
    ix.add(embeds)

    q_vec = embed_question(question)
    _, idxs = ix.search(q_vec, top_k_raw)
    raw = [passages[i] for i in idxs[0]]

    scores = reranker.predict([[question, p] for p in raw])
    best = sorted(zip(raw, scores), key=lambda x: x[1], reverse=True)[:final_k]
    ctx = "\n".join(p for p, _ in best)
    return generate(ctx, question, doc_type=doc_type, num_sentences=num_sentences)


# ───────────────────────────────────────────
#  Pydantic Schema
# ───────────────────────────────────────────
class Query(BaseModel):
    question: str
    top_k: int = 4
    top_k_raw: int = 20
    num_sentences: int = 1



# Optional: define module exports
__all__ = [
    "embed_question", "retrieve_rerank", "generate",
    "rag_fusion", "answer_from_text", "load_docs",
    "load_docs_from_string", "Query"
]
