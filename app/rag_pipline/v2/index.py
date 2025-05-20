import os, torch, faiss
from pydantic import BaseModel
from transformers import (
    DPRContextEncoder, DPRContextEncoderTokenizer,
    DPRQuestionEncoder, DPRQuestionEncoderTokenizer,
    AutoTokenizer, AutoModelForSeq2SeqLM
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- one-time start-up ---------- #
def load_docs(fp: str, chunk=600, overlap=100):
    with open(fp) as f:
        txt = f.read()
    chunks = [
        txt[i : i + chunk]
        for i in range(0, len(txt), chunk - overlap)
    ]
    return chunks

def embed_passages(passages, enc, tok):
    inputs = tok(passages, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
    with torch.no_grad():
        return enc(**inputs).pooler_output.cpu().numpy()

print("🔄 Loading models …")
ctx_enc  = DPRContextEncoder.from_pretrained(
              "facebook/dpr-ctx_encoder-single-nq-base").to(DEVICE).eval()
ctx_tok  = DPRContextEncoderTokenizer.from_pretrained(
              "facebook/dpr-ctx_encoder-single-nq-base")
q_enc    = DPRQuestionEncoder.from_pretrained(
              "facebook/dpr-question_encoder-single-nq-base").to(DEVICE).eval()
q_tok    = DPRQuestionEncoderTokenizer.from_pretrained(
              "facebook/dpr-question_encoder-single-nq-base")
gen_tok  = AutoTokenizer.from_pretrained("google/flan-t5-small")
gen_mod  = AutoModelForSeq2SeqLM.from_pretrained(
              "google/flan-t5-small").to(DEVICE).eval()

print("🔄 Building index …")
PASSAGES   = load_docs("data/companyPolicies.txt")
EMBEDS     = embed_passages(PASSAGES, ctx_enc, ctx_tok)
index      = faiss.IndexFlatL2(EMBEDS.shape[1])
index.add(EMBEDS)

class Query(BaseModel):
    question: str
    top_k:   int = 4

def embed_question(q):
    inp = q_tok(q, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        return q_enc(**inp).pooler_output.cpu().numpy()

def retrieve(q_vec, k):
    _, idxs = index.search(q_vec, k)
    return [PASSAGES[i] for i in idxs[0]]

def generate(context, question):
    # text   = f"Answer the question concisely from the context.\n\nQuestion: {question}\n\nContext:\n{context}\n\nAnswer:"
    # inp    = gen_tok([text], return_tensors="pt", truncation=True, padding=True).to(DEVICE)
    # with torch.no_grad():
    #     out = gen_mod.generate(**inp, max_length=128, num_beams=2, early_stopping=True)
    # return gen_tok.decode(out[0], skip_special_tokens=True)
    
    
    prompt = (
        "You are an AI assistant answering questions about a company-policy "
        "document. Use only the provided context. "
        "Answer in one clear sentence using **only** the context."
        "If the context is insufficient, reply \"I don't know.\" "

        f"\n\n### Question:\n{question}"

        f"\n\n### Context:\n{context}"

        "\n\n### Answer (one clear sentence):"
    )

    

    inp = gen_tok([prompt], return_tensors="pt",
                  truncation=True, padding=True).to(DEVICE)

    with torch.no_grad():
        out = gen_mod.generate(
            **inp, max_length=64, num_beams=4, early_stopping=True
        )

    return gen_tok.decode(out[0], skip_special_tokens=True).strip()