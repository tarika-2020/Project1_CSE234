import argparse
import hashlib
import json
import math
import os
import re
from collections import Counter
from pathlib import Path

import tiktoken
from openai import OpenAI


DOCS_DIR = Path("sourcedocs")
MODEL_NAME = "gpt-4o-mini"
CHUNK_SIZE_WORDS = int(os.environ.get("RAG_CHUNK_SIZE_WORDS", "160"))
CHUNK_OVERLAP_LINES = int(os.environ.get("RAG_CHUNK_OVERLAP_LINES", "2"))
BM25_K1 = 1.5
BM25_B = 0.75
TOTAL_LLM_PROMPT_BUDGET = int(os.environ.get("RAG_TOTAL_LLM_PROMPT_BUDGET", "2000"))
PROMPT_BUDGET_SAFETY_MARGIN = int(os.environ.get("RAG_PROMPT_BUDGET_SAFETY_MARGIN", "64"))
TOKEN_ENCODING_NAME = "cl100k_base"
TOKEN_FALLBACK_RATIO = 1.3
DEFAULT_TRITONAI_BASE_URL = "https://tritonai-api.ucsd.edu/v1"
DEFAULT_API_KEY_PATH = Path.home() / "api-key.txt"
GENERATION_MAX_TOKENS = int(os.environ.get("RAG_GENERATION_MAX_TOKENS", "550"))
HEADING_BOOST = float(os.environ.get("RAG_HEADING_BOOST", "0.25"))
FILENAME_BOOST = float(os.environ.get("RAG_FILENAME_BOOST", "1.0"))
FINAL_TOP_K = int(os.environ.get("RAG_FINAL_TOP_K", "4"))
EMBEDDING_MODEL = os.environ.get("RAG_EMBEDDING_MODEL", "api-tgpt-embeddings").strip()
EMBEDDING_WEIGHT = float(os.environ.get("RAG_EMBEDDING_WEIGHT", "0.4"))
LEXICAL_CANDIDATES = int(os.environ.get("RAG_LEXICAL_CANDIDATES", "10"))
EMBEDDING_CANDIDATES = int(os.environ.get("RAG_EMBEDDING_CANDIDATES", "10"))
RERANK_CANDIDATES = int(os.environ.get("RAG_RERANK_CANDIDATES", "8"))
RERANK_MODEL = os.environ.get("RAG_RERANK_MODEL", "api-mistral-small-3.2-2506").strip()
RERANK_MAX_CHARS = int(os.environ.get("RAG_RERANK_MAX_CHARS", "1000"))
EMBEDDING_CACHE_PATH = Path(os.environ.get("RAG_EMBEDDING_CACHE_PATH", ".embedding_cache_api_tgpt.json"))
MAX_SAME_FILE_CHUNKS = int(os.environ.get("RAG_MAX_SAME_FILE_CHUNKS", "2"))
MAX_OVERLAP_LINES = int(os.environ.get("RAG_MAX_OVERLAP_LINES", "6"))

EMBEDDING_CACHE = None
EMBEDDING_DIRTY = False


try:
    ENCODER = tiktoken.get_encoding(TOKEN_ENCODING_NAME)
except Exception:
    ENCODER = None


def tokenize(text: str):
    return re.findall(r"[a-z0-9_]+", text.lower())


def load_embedding_cache():
    global EMBEDDING_CACHE
    if EMBEDDING_CACHE is not None:
        return EMBEDDING_CACHE
    if EMBEDDING_CACHE_PATH.exists():
        try:
            EMBEDDING_CACHE = json.loads(EMBEDDING_CACHE_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            EMBEDDING_CACHE = {}
    else:
        EMBEDDING_CACHE = {}
    return EMBEDDING_CACHE


def save_embedding_cache():
    global EMBEDDING_DIRTY
    if not EMBEDDING_DIRTY or EMBEDDING_CACHE is None:
        return
    EMBEDDING_CACHE_PATH.write_text(json.dumps(EMBEDDING_CACHE), encoding="utf-8")
    EMBEDDING_DIRTY = False


def cache_key_for_text(model: str, text: str):
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return f"{model}:{digest}"


def get_api_client():
    cfg = get_generator_config()
    if not cfg["api_key"]:
        return None
    return OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"])


def batch_get_embeddings(texts, model: str):
    if not texts:
        return []
    cache = load_embedding_cache()
    uncached = []
    uncached_positions = []
    outputs = [None] * len(texts)

    for idx, text in enumerate(texts):
        key = cache_key_for_text(model, text)
        if key in cache:
            outputs[idx] = cache[key]
        else:
            uncached.append(text)
            uncached_positions.append((idx, key))

    if uncached:
        client = get_api_client()
        if client is None:
            return []
        response = client.embeddings.create(
            input=uncached,
            model=model,
            encoding_format="float",
        )
        global EMBEDDING_DIRTY
        for (idx, key), item in zip(uncached_positions, response.data):
            cache[key] = item.embedding
            outputs[idx] = item.embedding
        EMBEDDING_DIRTY = True
        save_embedding_cache()

    return outputs


def cosine_similarity(vec_a, vec_b):
    if not vec_a or not vec_b:
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def reciprocal_rank_fusion(ranks, k: int = 60):
    return sum(1.0 / (k + rank) for rank in ranks if rank is not None)


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
                    "embedding_text": f"{doc['file']}\n{heading}\n{chunk['text']}",
                }
            )
    return chunks


def attach_embeddings(chunks):
    texts = [chunk["embedding_text"] for chunk in chunks]
    embeddings = batch_get_embeddings(texts, EMBEDDING_MODEL)
    if not embeddings or len(embeddings) != len(chunks):
        return False
    for chunk, embedding in zip(chunks, embeddings):
        chunk["embedding"] = embedding
    return True


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
        score += HEADING_BOOST * len(question_token_set.intersection(heading_tokens))
    if chunk["file_stem"] in question_token_set:
        score += FILENAME_BOOST

    return score


def rerank_with_llm(question: str, candidates):
    client = get_api_client()
    if client is None or not candidates:
        return None

    candidate_blocks = []
    for idx, chunk in enumerate(candidates, start=1):
        snippet = chunk["text"][:RERANK_MAX_CHARS]
        candidate_blocks.append(
            f"[{idx}] {chunk['file']} lines {chunk['start_line']}-{chunk['end_line']}\n{snippet}"
        )

    prompt = (
        "Rank the candidate documentation chunks by how useful they are for answering the question.\n"
        "Prefer chunks that directly answer the question, define the requested concept, or contain exact parameter names and defaults.\n"
        "Return JSON only in the form {\"ranked_ids\": [best_id, ...]} using candidate ids.\n\n"
        f"Question:\n{question}\n\nCandidates:\n" + "\n\n".join(candidate_blocks)
    )

    response = client.chat.completions.create(
        model=RERANK_MODEL,
        messages=[
            {"role": "system", "content": "You rank retrieval candidates for a documentation QA system."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=200,
    )
    content = response.choices[0].message.content or ""
    content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip(), flags=re.MULTILINE).strip()
    try:
        parsed = json.loads(content)
        ranked_ids = parsed.get("ranked_ids", [])
    except json.JSONDecodeError:
        return None

    chosen = []
    seen = set()
    for item in ranked_ids:
        try:
            idx = int(item) - 1
        except (TypeError, ValueError):
            continue
        if 0 <= idx < len(candidates) and idx not in seen:
            seen.add(idx)
            chosen.append(candidates[idx])

    return chosen if chosen else None


def retrieve(question: str, chunks, stats, top_k: int):
    question_tokens = tokenize(question)
    lexical_scored = []
    for chunk in chunks:
        score = score_chunk(question_tokens, chunk, stats)
        lexical_scored.append((score, chunk))

    lexical_scored.sort(
        reverse=True,
        key=lambda item: (item[0], item[1]["file"], -item[1]["start_line"]),
    )

    if not chunks or "embedding" not in chunks[0]:
        return [item[1] for item in lexical_scored[:top_k]]

    query_embeddings = batch_get_embeddings([question], EMBEDDING_MODEL)
    if not query_embeddings:
        return [item[1] for item in lexical_scored[:top_k]]
    query_embedding = query_embeddings[0]

    embedding_scored = []
    for chunk in chunks:
        similarity = cosine_similarity(query_embedding, chunk.get("embedding"))
        embedding_scored.append((similarity, chunk))

    embedding_scored.sort(
        reverse=True,
        key=lambda item: (item[0], item[1]["file"], -item[1]["start_line"]),
    )

    candidate_pool = {}

    for rank, (score, chunk) in enumerate(lexical_scored[:LEXICAL_CANDIDATES], start=1):
        key = (chunk["file"], chunk["start_line"], chunk["end_line"])
        entry = candidate_pool.setdefault(
            key,
            {"chunk": chunk, "lex_rank": None, "emb_rank": None, "lex_score": 0.0, "emb_score": 0.0},
        )
        entry["lex_rank"] = rank
        entry["lex_score"] = score

    for rank, (score, chunk) in enumerate(embedding_scored[:EMBEDDING_CANDIDATES], start=1):
        key = (chunk["file"], chunk["start_line"], chunk["end_line"])
        entry = candidate_pool.setdefault(
            key,
            {"chunk": chunk, "lex_rank": None, "emb_rank": None, "lex_score": 0.0, "emb_score": 0.0},
        )
        entry["emb_rank"] = rank
        entry["emb_score"] = score

    merged = []
    for entry in candidate_pool.values():
        fused = reciprocal_rank_fusion([entry["lex_rank"], entry["emb_rank"]])
        fused += 0.02 * entry["lex_score"] + EMBEDDING_WEIGHT * entry["emb_score"]
        merged.append((fused, entry["chunk"]))

    merged.sort(
        reverse=True,
        key=lambda item: (item[0], item[1]["file"], -item[1]["start_line"]),
    )

    top_candidates = [item[1] for item in merged[:RERANK_CANDIDATES]]
    reranked = rerank_with_llm(question, top_candidates)
    if reranked:
        return reranked[:top_k]

    return top_candidates[:top_k]


def count_tokens(text: str):
    if ENCODER is not None:
        return len(ENCODER.encode(text))
    return math.ceil(len(tokenize(text)) * TOKEN_FALLBACK_RATIO)


def render_context_chunk(item):
    return f"[SOURCE: {item['file']} lines {item['start_line']}-{item['end_line']}]\n{item['text']}"


def chunks_overlap_too_much(chunk_a, chunk_b):
    if chunk_a["file"] != chunk_b["file"]:
        return False
    overlap_start = max(chunk_a["start_line"], chunk_b["start_line"])
    overlap_end = min(chunk_a["end_line"], chunk_b["end_line"])
    return overlap_end >= overlap_start and (overlap_end - overlap_start + 1) > MAX_OVERLAP_LINES


def select_context_chunks(retrieved):
    selected = []
    per_file_counts = Counter()

    for chunk in retrieved:
        if per_file_counts[chunk["file"]] >= MAX_SAME_FILE_CHUNKS:
            continue
        if any(chunks_overlap_too_much(chunk, kept) for kept in selected):
            continue
        selected.append(chunk)
        per_file_counts[chunk["file"]] += 1

    return selected if selected else retrieved


def prompt_overhead_tokens(question: str):
    messages = build_generation_messages(question, "")
    return sum(count_tokens(message["content"]) for message in messages)


def build_context(question: str, retrieved):
    context_parts = []
    used_chunks = []
    total_tokens = 0
    available_context_budget = max(
        0,
        TOTAL_LLM_PROMPT_BUDGET - prompt_overhead_tokens(question) - PROMPT_BUDGET_SAFETY_MARGIN,
    )

    for item in select_context_chunks(retrieved):
        rendered = render_context_chunk(item)
        rendered_tokens = count_tokens(rendered)
        separator_tokens = 2 if context_parts else 0

        if context_parts and total_tokens + separator_tokens + rendered_tokens > available_context_budget:
            break
        if not context_parts and rendered_tokens > available_context_budget:
            continue

        context_parts.append(rendered)
        used_chunks.append(item)
        total_tokens += separator_tokens + rendered_tokens

    return "\n\n".join(context_parts), used_chunks, total_tokens


def load_api_key():
    env_key = os.environ.get("OPENAI_API_KEY")
    if env_key:
        return env_key.strip()

    if DEFAULT_API_KEY_PATH.exists():
        try:
            return DEFAULT_API_KEY_PATH.read_text(encoding="utf-8").strip()
        except OSError:
            return None

    return None


def get_generator_config():
    return {
        "api_key": load_api_key(),
        "base_url": os.environ.get("OPENAI_BASE_URL", DEFAULT_TRITONAI_BASE_URL).strip(),
        "model": os.environ.get("RAG_MODEL", MODEL_NAME).strip(),
    }


def build_generation_messages(question: str, context: str):
    system_prompt = (
        "You are answering questions about RapidFire AI documentation. "
        "Use only the provided retrieved context. "
        "Give the most complete answer supported by the context. "
        "If the context includes names, defaults, constraints, supported options, or steps, include them explicitly. "
        "If the context does not contain enough information, say so briefly instead of guessing."
    )
    user_prompt = (
        f"Retrieved context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Write a concise but complete answer grounded only in the retrieved context. "
        "Prefer exact terminology from the docs when available. "
        "If multiple details are asked for, answer all of them."
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def generate_answer(question: str, context: str):
    if not context.strip():
        return "I could not find enough relevant context to answer this question."

    generator_cfg = get_generator_config()
    if not generator_cfg["api_key"]:
        return "Generated answer based on retrieved context."

    client = OpenAI(api_key=generator_cfg["api_key"], base_url=generator_cfg["base_url"])
    response = client.chat.completions.create(
        model=generator_cfg["model"],
        messages=build_generation_messages(question, context),
        temperature=0.0,
        max_tokens=GENERATION_MAX_TOKENS,
    )
    answer = response.choices[0].message.content or ""
    return answer.strip() or "I could not produce an answer from the retrieved context."


def run_pipeline(input_file: str, output_file: str):
    docs = load_documents(DOCS_DIR)
    chunks = prepare_chunks(docs)
    attach_embeddings(chunks)
    stats = build_retrieval_stats(chunks)

    with open(input_file, "r", encoding="utf-8") as handle:
        questions = json.load(handle)

    outputs = []
    for item in questions:
        retrieved = retrieve(item["question"], chunks, stats, top_k=FINAL_TOP_K)
        context, used_chunks, _ = build_context(item["question"], retrieved)
        answer = generate_answer(item["question"], context)
        sources = [
            {"file": r["file"], "lines": [r["start_line"], r["end_line"]]}
            for r in used_chunks
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
