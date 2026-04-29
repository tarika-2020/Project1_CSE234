import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path


DOCS_DIR = Path("sourcedocs")
TOP_K = 5
CONTEXT_LIMIT = 2000
MODEL_NAME = "gpt-4o-mini"
CHUNK_SIZE_WORDS = 220
CHUNK_OVERLAP_LINES = 3
BM25_K1 = 1.5
BM25_B = 0.75


def tokenize(text: str):
    return re.findall(r"[a-z0-9_]+", text.lower())


def load_documents(folder: Path):
    docs = []
    for file in folder.rglob("*.rst"):
        try:
            text = file.read_text(encoding="utf-8")
        except OSError:
            continue
        lines = text.splitlines()
        docs.append({"file": file.name, "text": text, "lines": lines})
    return docs


def chunk_document_lines(lines, chunk_size_words: int = CHUNK_SIZE_WORDS, overlap_lines: int = CHUNK_OVERLAP_LINES):
    chunks = []
    total_lines = len(lines)
    start_idx = 0

    while start_idx < total_lines:
        end_idx = start_idx
        word_count = 0

        while end_idx < total_lines:
            next_line_words = len(lines[end_idx].split())
            if end_idx > start_idx and word_count + next_line_words > chunk_size_words:
                break
            word_count += next_line_words
            end_idx += 1

        if end_idx == start_idx:
            end_idx += 1

        chunk_lines = lines[start_idx:end_idx]
        chunk_text = "\n".join(chunk_lines).strip()
        if chunk_text:
            chunks.append(
                {
                    "text": chunk_text,
                    "start_line": start_idx + 1,
                    "end_line": end_idx,
                }
            )

        if end_idx >= total_lines:
            break

        start_idx = max(end_idx - overlap_lines, start_idx + 1)

    return chunks


def prepare_chunks(documents):
    chunks = []
    for doc in documents:
        for idx, chunk in enumerate(chunk_document_lines(doc["lines"])):
            tokens = tokenize(chunk["text"])
            heading = next((line.strip() for line in chunk["text"].splitlines() if line.strip()), "")
            chunks.append(
                {
                    "file": doc["file"],
                    "chunk_id": idx,
                    "text": chunk["text"],
                    "start_line": chunk["start_line"],
                    "end_line": chunk["end_line"],
                    "tokens": tokens,
                    "term_freqs": Counter(tokens),
                    "doc_len": len(tokens),
                    "heading": heading.lower(),
                    "file_stem": Path(doc["file"]).stem.lower(),
                }
            )
    return chunks


def build_retrieval_stats(chunks):
    doc_freqs = Counter()
    total_doc_len = 0

    for chunk in chunks:
        total_doc_len += chunk["doc_len"]
        for token in set(chunk["tokens"]):
            doc_freqs[token] += 1

    avg_doc_len = total_doc_len / len(chunks) if chunks else 0.0
    return {"doc_freqs": doc_freqs, "avg_doc_len": avg_doc_len, "num_docs": len(chunks)}


def score_chunk(question_tokens, chunk, stats):
    if not chunk["tokens"]:
        return 0.0

    score = 0.0
    num_docs = stats["num_docs"]
    avg_doc_len = stats["avg_doc_len"] or 1.0

    for token in question_tokens:
        tf = chunk["term_freqs"].get(token, 0)
        if tf == 0:
            continue

        df = stats["doc_freqs"].get(token, 0)
        idf = math.log(1.0 + (num_docs - df + 0.5) / (df + 0.5))
        denom = tf + BM25_K1 * (1.0 - BM25_B + BM25_B * chunk["doc_len"] / avg_doc_len)
        score += idf * ((tf * (BM25_K1 + 1.0)) / denom)

    heading_tokens = set(tokenize(chunk["heading"]))
    question_token_set = set(question_tokens)
    if heading_tokens:
        score += 0.35 * len(question_token_set.intersection(heading_tokens))
    if chunk["file_stem"] in question_token_set:
        score += 1.0

    return score


def retrieve(question: str, chunks, stats, top_k: int = TOP_K):
    question_tokens = tokenize(question)
    scored = []
    for chunk in chunks:
        score = score_chunk(question_tokens, chunk, stats)
        scored.append((score, chunk))

    scored.sort(
        reverse=True,
        key=lambda item: (item[0], item[1]["file"], -item[1]["start_line"]),
    )
    return [item[1] for item in scored[:top_k]]


def build_context(retrieved):
    context_parts = []
    total_words = 0

    for item in retrieved:
        words = item["text"].split()
        if total_words + len(words) > CONTEXT_LIMIT:
            break

        context_parts.append(
            f"[SOURCE: {item['file']} lines {item['start_line']}-{item['end_line']}]\n{item['text']}"
        )
        total_words += len(words)

    return "\n\n".join(context_parts)


def generate_answer(question: str, context: str):
    return "Generated answer based on retrieved context."


def run_pipeline(input_file: str, output_file: str):
    docs = load_documents(DOCS_DIR)
    chunks = prepare_chunks(docs)
    stats = build_retrieval_stats(chunks)

    with open(input_file, "r", encoding="utf-8") as handle:
        questions = json.load(handle)

    outputs = []
    for item in questions:
        retrieved = retrieve(item["question"], chunks, stats)
        context = build_context(retrieved)
        answer = generate_answer(item["question"], context)
        sources = [
            {"file": r["file"], "lines": [r["start_line"], r["end_line"]]}
            for r in retrieved
        ]

        outputs.append(
            {
                "question_id": item["question_id"],
                "answer": answer,
                "retrieved_context": context,
                "sources": sources,
            }
        )

    with open(output_file, "w", encoding="utf-8") as handle:
        json.dump(outputs, handle, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run_pipeline(args.input, args.output)
