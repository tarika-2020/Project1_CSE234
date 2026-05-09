#!/usr/bin/env python3
"""Run Project 1 RAG config sweeps and log them to RapidFire/MLflow.

Start RapidFire first:
    conda activate cse234
    rapidfireai start

Then run this script:
    python run_rapidfire.py

By default this runs retrieval-only experiments, which are fast and enough to
compare chunking, top-k, embedding, and reranker settings. Add
``--full-generation`` when you also want generated answers in the output files.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import main as rag_main
from Metrics.project1_eval import f1_at_k, precision_at_k, recall_at_k, to_spans

try:
    import mlflow
except ImportError:  # pragma: no cover - handled at runtime.
    mlflow = None


@dataclass(frozen=True)
class RagConfig:
    name: str
    chunk_size_words: int = 160
    chunk_overlap_lines: int = 2
    final_top_k: int = 4
    use_embeddings: bool = True
    use_reranker: bool = True
    lexical_candidates: int = 10
    embedding_candidates: int = 10
    rerank_candidates: int = 8
    embedding_weight: float = 0.4
    heading_boost: float = 0.25
    filename_boost: float = 1.0
    max_same_file_chunks: int = 2
    max_overlap_lines: int = 6
    total_prompt_budget: int = 2000
    prompt_budget_safety_margin: int = 64


CONFIGS = [
    RagConfig(name="final_default_160_top4"),
    RagConfig(name="smaller_chunks_120_top4", chunk_size_words=120),
    RagConfig(name="larger_chunks_220_top4", chunk_size_words=220),
    RagConfig(name="final_top3", final_top_k=3),
    RagConfig(name="final_top5", final_top_k=5),
    RagConfig(name="lexical_only_top4", use_embeddings=False, use_reranker=False),
    RagConfig(name="hybrid_no_reranker_top4", use_reranker=False),
    RagConfig(
        name="broader_candidates_top4",
        lexical_candidates=16,
        embedding_candidates=16,
        rerank_candidates=12,
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", default="validation-set-golden-qa-pairs.json")
    parser.add_argument("--corpus-dir", default="sourcedocs")
    parser.add_argument("--api-key-txt", default=str(Path.home() / "api-key.txt"))
    parser.add_argument("--generation-model", default=rag_main.MODEL_NAME)
    parser.add_argument("--output-dir", default="logs/rapidfire_runs")
    parser.add_argument("--experiment-name", default="project1-rag-config-sweep")
    parser.add_argument("--tracking-uri", default=os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:8852"))
    parser.add_argument("--limit", type=int, default=0, help="Run only the first N validation questions.")
    parser.add_argument("--full-generation", action="store_true", help="Also call the LLM to generate answers.")
    parser.add_argument("--no-mlflow", action="store_true", help="Write local JSON only; skip MLflow logging.")
    return parser.parse_args()


@contextmanager
def patched_main(config: RagConfig):
    original_values = {
        "CHUNK_SIZE_WORDS": rag_main.CHUNK_SIZE_WORDS,
        "CHUNK_OVERLAP_LINES": rag_main.CHUNK_OVERLAP_LINES,
        "FINAL_TOP_K": rag_main.FINAL_TOP_K,
        "LEXICAL_CANDIDATES": rag_main.LEXICAL_CANDIDATES,
        "EMBEDDING_CANDIDATES": rag_main.EMBEDDING_CANDIDATES,
        "RERANK_CANDIDATES": rag_main.RERANK_CANDIDATES,
        "EMBEDDING_WEIGHT": rag_main.EMBEDDING_WEIGHT,
        "HEADING_BOOST": rag_main.HEADING_BOOST,
        "FILENAME_BOOST": rag_main.FILENAME_BOOST,
        "MAX_SAME_FILE_CHUNKS": rag_main.MAX_SAME_FILE_CHUNKS,
        "MAX_OVERLAP_LINES": rag_main.MAX_OVERLAP_LINES,
        "TOTAL_LLM_PROMPT_BUDGET": rag_main.TOTAL_LLM_PROMPT_BUDGET,
        "PROMPT_BUDGET_SAFETY_MARGIN": rag_main.PROMPT_BUDGET_SAFETY_MARGIN,
        "chunk_document_lines": rag_main.chunk_document_lines,
        "rerank_with_llm": rag_main.rerank_with_llm,
    }

    def configured_chunker(lines, chunk_size_words=None, overlap_lines=None):
        return original_values["chunk_document_lines"](
            lines,
            chunk_size_words=config.chunk_size_words,
            overlap_lines=config.chunk_overlap_lines,
        )

    try:
        rag_main.CHUNK_SIZE_WORDS = config.chunk_size_words
        rag_main.CHUNK_OVERLAP_LINES = config.chunk_overlap_lines
        rag_main.FINAL_TOP_K = config.final_top_k
        rag_main.LEXICAL_CANDIDATES = config.lexical_candidates
        rag_main.EMBEDDING_CANDIDATES = config.embedding_candidates
        rag_main.RERANK_CANDIDATES = config.rerank_candidates
        rag_main.EMBEDDING_WEIGHT = config.embedding_weight
        rag_main.HEADING_BOOST = config.heading_boost
        rag_main.FILENAME_BOOST = config.filename_boost
        rag_main.MAX_SAME_FILE_CHUNKS = config.max_same_file_chunks
        rag_main.MAX_OVERLAP_LINES = config.max_overlap_lines
        rag_main.TOTAL_LLM_PROMPT_BUDGET = config.total_prompt_budget
        rag_main.PROMPT_BUDGET_SAFETY_MARGIN = config.prompt_budget_safety_margin
        rag_main.chunk_document_lines = configured_chunker
        if not config.use_reranker:
            rag_main.rerank_with_llm = lambda question, candidates: None
        yield
    finally:
        for name, value in original_values.items():
            setattr(rag_main, name, value)


def load_validation(path: Path, limit: int) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        rows = json.load(handle)
    return rows[:limit] if limit > 0 else rows


def build_outputs(config: RagConfig, questions: list[dict[str, Any]], full_generation: bool) -> list[dict[str, Any]]:
    docs = rag_main.load_documents(rag_main.RUNTIME_CONFIG["corpus_dir"])
    chunks = rag_main.prepare_chunks(docs)

    if config.use_embeddings:
        embedded = rag_main.attach_embeddings(chunks)
        if not embedded:
            print(f"[{config.name}] embeddings unavailable; falling back to lexical retrieval")

    stats = rag_main.build_retrieval_stats(chunks)
    outputs = []
    for item in questions:
        retrieved = rag_main.retrieve(item["question"], chunks, stats, top_k=config.final_top_k)
        context, used_chunks, _ = rag_main.build_context(item["question"], retrieved)
        if full_generation:
            answer = rag_main.generate_answer(item["question"], context)
        else:
            answer = "Retrieval-only RapidFire experiment; run with --full-generation to generate answers."

        outputs.append(
            {
                "question_id": item["question_id"],
                "answer": answer,
                "retrieved_context": context,
                "sources": [
                    {"file": chunk["file"], "lines": [chunk["start_line"], chunk["end_line"]]}
                    for chunk in used_chunks
                ],
            }
        )
    return outputs


def score_outputs(validation_rows: list[dict[str, Any]], outputs: list[dict[str, Any]], k: int = 5) -> dict[str, float]:
    output_by_id = {int(row["question_id"]): row for row in outputs}
    f1s = []
    precisions = []
    recalls = []

    for row in validation_rows:
        question_id = int(row["question_id"])
        gold = to_spans(row["source_evidence"])
        retrieved = to_spans(output_by_id[question_id]["sources"])
        f1s.append(f1_at_k(retrieved, gold, k))
        precisions.append(precision_at_k(retrieved, gold, k))
        recalls.append(recall_at_k(retrieved, gold, k))

    mean_f1 = sum(f1s) / len(f1s)
    mean_precision = sum(precisions) / len(precisions)
    mean_recall = sum(recalls) / len(recalls)
    return {
        f"F1_at_{k}": mean_f1,
        f"Precision_at_{k}": mean_precision,
        f"Recall_at_{k}": mean_recall,
        "Retrieval_Score": (mean_f1 + mean_precision + mean_recall) / 3,
    }


def config_params(config: RagConfig) -> dict[str, Any]:
    return {
        "chunk_size_words": config.chunk_size_words,
        "chunk_overlap_lines": config.chunk_overlap_lines,
        "final_top_k": config.final_top_k,
        "use_embeddings": config.use_embeddings,
        "use_reranker": config.use_reranker,
        "lexical_candidates": config.lexical_candidates,
        "embedding_candidates": config.embedding_candidates,
        "rerank_candidates": config.rerank_candidates,
        "embedding_weight": config.embedding_weight,
        "heading_boost": config.heading_boost,
        "filename_boost": config.filename_boost,
        "max_same_file_chunks": config.max_same_file_chunks,
        "max_overlap_lines": config.max_overlap_lines,
        "total_prompt_budget": config.total_prompt_budget,
        "prompt_budget_safety_margin": config.prompt_budget_safety_margin,
    }


def log_run(args: argparse.Namespace, config: RagConfig, metrics: dict[str, float], output_path: Path) -> None:
    if args.no_mlflow:
        return
    if mlflow is None:
        print("mlflow is not installed; skipping RapidFire/MLflow logging")
        return

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)
    with mlflow.start_run(run_name=config.name):
        mlflow.log_params(config_params(config))
        mlflow.log_param("full_generation", args.full_generation)
        mlflow.log_param("generation_model", args.generation_model)
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(output_path))


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rag_main.configure_runtime(args.corpus_dir, args.api_key_txt, args.generation_model)
    validation_rows = load_validation(Path(args.validation), args.limit)

    summary = []
    for config in CONFIGS:
        print(f"Running {config.name}...")
        started = time.time()
        with patched_main(config):
            outputs = build_outputs(config, validation_rows, args.full_generation)

        metrics = score_outputs(validation_rows, outputs)
        elapsed = time.time() - started
        metrics["Elapsed_Seconds"] = elapsed

        output_path = output_dir / f"{config.name}.json"
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(outputs, handle, indent=2)

        log_run(args, config, metrics, output_path)
        row = {"config": config.name, **metrics, **config_params(config)}
        summary.append(row)
        print(
            f"  Retrieval_Score={metrics['Retrieval_Score']:.4f} "
            f"F1@5={metrics['F1_at_5']:.4f} "
            f"P@5={metrics['Precision_at_5']:.4f} "
            f"R@5={metrics['Recall_at_5']:.4f}"
        )

    summary.sort(key=lambda row: row["Retrieval_Score"], reverse=True)
    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("\nBest configs:")
    for row in summary[:3]:
        print(f"  {row['config']}: Retrieval_Score={row['Retrieval_Score']:.4f}")
    print(f"\nWrote outputs and summary to {output_dir}")


if __name__ == "__main__":
    main()
