# main.py
import argparse
import json
import os
from pathlib import Path

# ---------- CONFIG ----------
DOCS_DIR = "docs"   # folder containing extracted .rst files
TOP_K = 5
CONTEXT_LIMIT = 2000   # token budget (rough approx by words)
MODEL_NAME = "gpt-4o-mini"   # replace with TritonAI/OpenAI compatible model

# ---------- LOAD DOCUMENTS ----------
def load_documents(folder):
    docs = []
    for file in Path(folder).rglob("*.rst"):
        try:
            text = file.read_text(encoding="utf-8")
            docs.append({
                "file": file.name,
                "text": text
            })
        except:
            pass
    return docs

# ---------- SIMPLE CHUNKING ----------
def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks

# ---------- BUILD CHUNK INDEX ----------
def prepare_chunks(documents):
    chunks = []

    for doc in documents:
        doc_chunks = chunk_text(doc["text"])
        for idx, chunk in enumerate(doc_chunks):
            chunks.append({
                "file": doc["file"],
                "chunk_id": idx,
                "text": chunk
            })

    return chunks

# ---------- RETRIEVAL (KEYWORD BASELINE) ----------
def retrieve(question, chunks, top_k=TOP_K):
    q_words = set(question.lower().split())

    scored = []
    for c in chunks:
        c_words = set(c["text"].lower().split())
        score = len(q_words.intersection(c_words))
        scored.append((score, c))

    scored.sort(reverse=True, key=lambda x: x[0])
    return [x[1] for x in scored[:top_k]]

# ---------- CONTEXT BUILDER ----------
def build_context(retrieved):
    context_parts = []
    total_words = 0

    for r in retrieved:
        words = r["text"].split()

        if total_words + len(words) > CONTEXT_LIMIT:
            break

        snippet = f"[SOURCE: {r['file']}]\n{r['text']}"
        context_parts.append(snippet)
        total_words += len(words)

    return "\n\n".join(context_parts)

# ---------- GENERATION ----------
def generate_answer(question, context):
    """
    Replace this with OpenAI / TritonAI call
    """

    prompt = f"""
Use only the context below to answer.

Context:
{context}

Question:
{question}

Answer clearly and concisely:
"""

    # Placeholder baseline answer
    return "Generated answer based on retrieved context."

# ---------- MAIN PIPELINE ----------
def run_pipeline(input_file, output_file):
    docs = load_documents(DOCS_DIR)
    chunks = prepare_chunks(docs)

    with open(input_file, "r") as f:
        questions = json.load(f)

    outputs = []

    for item in questions:
        qid = item["question_id"]
        question = item["question"]

        retrieved = retrieve(question, chunks)
        context = build_context(retrieved)
        answer = generate_answer(question, context)

        sources = []
        for r in retrieved:
            sources.append({
                "file": r["file"],
                "lines": [1, 50]   # replace with real line mapping later
            })

        outputs.append({
            "question_id": qid,
            "answer": answer,
            "retrieved_context": context,
            "sources": sources
        })

    with open(output_file, "w") as f:
        json.dump(outputs, f, indent=2)

# ---------- ENTRY ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)

    args = parser.parse_args()

    run_pipeline(args.input, args.output)