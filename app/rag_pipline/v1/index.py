import os
import torch
import numpy as np
import faiss

from transformers import (
    DPRContextEncoder, DPRContextEncoderTokenizer,
    DPRQuestionEncoder, DPRQuestionEncoderTokenizer,
    AutoTokenizer, AutoModelForSeq2SeqLM
)


# 1. Load and split documents
def load_documents(file_path, chunk_size=1000):
    with open(file_path, 'r') as f:
        text = f.read()
    # Split into chunks
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    return chunks


# 2. Embed documents using DPR Context Encoder
def embed_documents(passages, ctx_encoder, ctx_tokenizer):
    inputs = ctx_tokenizer(passages, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to("cpu") for k, v in inputs.items()}  # 🔧 Ensure CPU tensors
    ctx_encoder = ctx_encoder.to("cpu")  # 🔧 Ensure model is on CPU
    with torch.no_grad():
        embeddings = ctx_encoder(**inputs).pooler_output
    return embeddings.numpy()


# 3. Build FAISS index
def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index


# 4. Embed the user query
def embed_question(question, q_encoder, q_tokenizer):
    inputs = q_tokenizer(question, return_tensors="pt")
    inputs = {k: v.to("cpu") for k, v in inputs.items()}  # 🔧 Ensure CPU
    q_encoder = q_encoder.to("cpu")  # 🔧 Ensure model on CPU
    with torch.no_grad():
        question_embedding = q_encoder(**inputs).pooler_output
    return question_embedding.numpy()


# 5. Retrieve top-k passages
def retrieve(question_embedding, index, passages, top_k=3):
    distances, indices = index.search(question_embedding, top_k)
    return [passages[i] for i in indices[0]]


# 6. Generate answer using T5 or similar model
def generate_answer(context_passages, question, generator_model, generator_tokenizer):
    context = "\n".join(context_passages)
    input_text = f"question: {question} context: {context}"
    inputs = generator_tokenizer(
        [input_text], return_tensors="pt", padding=True, truncation=True
    )
    inputs = {k: v.to("cpu") for k, v in inputs.items()}  # 🔧 Ensure CPU
    generator_model = generator_model.to("cpu")  # 🔧 Ensure model is on CPU
    with torch.no_grad():
        outputs = generator_model.generate(
            **inputs,
            max_length=100,        # ✅ Avoid warning about truncation
            num_beams=2,
            early_stopping=True
        )
    return generator_tokenizer.decode(outputs[0], skip_special_tokens=True)


# Sample usage
def run_sample(question="What is the document about?"):
    # Load models correctly from pretrained (✅ no from_config!)
    ctx_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
    ctx_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")

    q_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
    q_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")

    generator_model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
    generator_tokenizer = AutoTokenizer.from_pretrained("t5-base")

    # Load and embed docs
    passages = load_documents("data/companyPolicies.txt")
    doc_embeddings = embed_documents(passages, ctx_encoder, ctx_tokenizer)
    index = build_faiss_index(doc_embeddings)

    # Ask a question
    q_embed = embed_question(question, q_encoder, q_tokenizer)
    top_docs = retrieve(q_embed, index, passages)

    # Generate answer
    answer = generate_answer(top_docs, question, generator_model, generator_tokenizer)
    print("Answer:", answer)
    return answer


if __name__ == "__main__":
    run_sample()
