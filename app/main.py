# app/main.py

from fastapi import FastAPI, UploadFile, Form, HTTPException
# from app.rag_pipeline_v1 import run_sample
# from app.rag_pipeline_v2 import Query, embed_question, retrieve, generate
import os; print("PORT ENV VAR:", os.environ.get("PORT"))

from app.rag_pipline.v3.index import Query, rag_fusion, retrieve_rerank, generate, answer_from_text

app = FastAPI()

@app.get("/")
def root():
    return {"message": "RAG API is running"}


@app.post("/ask")
def ask(q: Query):
    ctxs = retrieve_rerank(q.question, top_k_raw=q.top_k_raw, final_k=q.top_k)
    context_block = "\n".join(ctxs)
    answer = generate(context_block, q.question, num_sentences=q.num_sentences)
    return {"answer": answer}

@app.post("/ask_fusion")
def ask_fusion(q: Query):
    answer = rag_fusion(
        question=q.question,
        top_k_raw=q.top_k_raw,
        final_k=q.top_k,
        num_sentences=q.num_sentences
    )
    return {"answer": answer}


# ─── new one-shot document-Q-A endpoint ────────────────────────────────────
@app.post("/doc_qa")
async def doc_qa(
    file: UploadFile,
    question: str = Form(...),
    doc_type: str = Form("company-policy"),
    num_sentences: int = Form(1),
):
    if not file.content_type.startswith("text/"):
        raise HTTPException(status_code=400, detail="Only text files allowed.")
    doc_bytes = await file.read()
    doc_text  = doc_bytes.decode("utf-8", errors="ignore")

    answer = answer_from_text(doc_text, question, doc_type=doc_type, num_sentences=num_sentences)
    return {"answer": answer}


# ─── old one-shot question-Q-A endpoint ────────────────────────────────────
# @app.get("/ask")
# def ask(question: str = "What is this about?"):
#     answer = run_sample(question)  # You’ll adapt this function next
#     return {"question": question, "answer": answer}

# @app.post("/ask")
# def ask(q: Query):
#     q_vec   = embed_question(q.question)
#     ctxs    = retrieve(q_vec, q.top_k)
#     answer  = generate("\n".join(ctxs), q.question)
#     return {"answer": answer}