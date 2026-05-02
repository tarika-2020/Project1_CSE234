# Project 1: RAG and Context Engineering

This repository contains our CSE/DSC 234 Project 1 submission work for a
single-turn RAG question-answering system over the RapidFire AI documentation
corpus.

## Repository Layout

- `main.py`: submission entrypoint. The TAs should be able to run it as
  `python main.py --input <input.json> --output <output.json>`.
- `sourcedocs/`: the documentation corpus for retrieval. This is the canonical
  corpus root for the project.
- `Metrics/`: released evaluation utilities, including retrieval scoring and
  LLM-as-judge scripts.
- `logs/`: experiment logs and metrics artifacts that should be committed during
  the project.
- `validation-set-golden-qa-pairs.json`: released validation set with reference
  answers and source evidence.
- `validation-set-question-type-distribution.json`: question-type distribution
  for the released validation set.
- `validation-output.json`: a saved output file for the released validation set.

## Current System

The current `main.py` is our tuned submission pipeline. It:

- loads the `.rst` documentation corpus from `sourcedocs/`
- builds line-tracked chunks for evidence extraction
- runs hybrid retrieval using lexical BM25-style scoring plus Triton embeddings
- reranks the merged retrieval candidates with a chat model
- enforces the full per-query 2,000-token prompt budget from the project
  statement, including prompt overhead
- filters redundant chunks before context packing
- generates answers through the TritonAI OpenAI-compatible API
- emits the required output schema with `question_id`, `answer`,
  `retrieved_context`, and `sources`

## Environment and Configuration

Expected Python packages for the final system:

- `openai`
- `tiktoken`

Expected environment variables for TritonAI or OpenAI-compatible generation and
judge workflows:

- `OPENAI_API_KEY`: required for TritonAI or OpenAI calls
- `OPENAI_BASE_URL`: optional for the generator client; defaults to TritonAI
- `JUDGE_BASE_URL`: optional for the released judge script
- `JUDGE_MODEL`: optional override for the released judge script; default to
  `claude-sonnet-4-6-aws`, with `api-gpt-oss-120b` as a fallback if needed
- `RAG_MODEL`: optional override for the generator model in `main.py`
- `RAG_FINAL_TOP_K`: optional number of chunks kept after ranking/reranking
- `RAG_CHUNK_SIZE_WORDS`: optional chunk size override
- `RAG_CHUNK_OVERLAP_LINES`: optional chunk overlap override
- `RAG_HEADING_BOOST`: optional BM25 heading-match boost override
- `RAG_FILENAME_BOOST`: optional filename-match boost override
- `RAG_EMBEDDING_MODEL`: optional embedding model override; current default is
  `api-tgpt-embeddings`
- `RAG_RERANK_MODEL`: optional reranker model override; current default is
  `api-mistral-small-3.2-2506`
- `RAG_LEXICAL_CANDIDATES`: optional lexical candidate pool size
- `RAG_EMBEDDING_CANDIDATES`: optional embedding candidate pool size
- `RAG_RERANK_CANDIDATES`: optional reranking pool size
- `RAG_EMBEDDING_CACHE_PATH`: optional local cache path for chunk/query
  embeddings
- `RAG_GENERATION_MAX_TOKENS`: optional generation token cap
- `RAG_TOTAL_LLM_PROMPT_BUDGET`: optional total prompt budget; current default
  is `2000`
- `RAG_PROMPT_BUDGET_SAFETY_MARGIN`: optional reserved prompt budget margin
- `RAG_MAX_SAME_FILE_CHUNKS`: optional maximum selected chunks per file
- `RAG_MAX_OVERLAP_LINES`: optional overlap threshold used for context
  deduplication

For the course TritonAI gateway, the base URL is expected to be:

`https://tritonai-api.ucsd.edu/v1`

On Datahub, `main.py` also supports the course convention of reading the API key
from `~/api-key.txt` if `OPENAI_API_KEY` is not set.

## TA Run Instructions

The TAs should be able to run the full pipeline end to end as:

```bash
python main.py --input input_filename.json --output output_filename.json
```

If TritonAI generation is enabled, `OPENAI_API_KEY` must be set or the API key
must be present in `~/api-key.txt`. The script uses the course TritonAI base URL
by default, so no extra base URL argument is needed for `main.py`.

The input file must be a JSON list of objects with:

- `question_id`
- `question`

The output file will be a JSON list of objects with:

- `question_id`
- `answer`
- `retrieved_context`
- `sources`

## Running the Pipeline

Example command:

```bash
python main.py --input validation-set-golden-qa-pairs.json --output validation-output.json
```

This command can be run from an SSH session on Datahub. SSH itself is not part
of the Python pipeline, but the code is configured to work cleanly in that
environment by defaulting to the TritonAI endpoint and by reading `~/api-key.txt`
when available.

The current default tuned configuration in `main.py` is:

```powershell
$env:RAG_CHUNK_SIZE_WORDS="160"
$env:RAG_CHUNK_OVERLAP_LINES="2"
$env:RAG_FINAL_TOP_K="4"
$env:RAG_HEADING_BOOST="0.25"
$env:RAG_FILENAME_BOOST="1.0"
$env:RAG_EMBEDDING_MODEL="api-tgpt-embeddings"
$env:RAG_RERANK_MODEL="api-mistral-small-3.2-2506"
$env:RAG_LEXICAL_CANDIDATES="10"
$env:RAG_EMBEDDING_CANDIDATES="10"
$env:RAG_RERANK_CANDIDATES="8"
$env:RAG_GENERATION_MAX_TOKENS="550"
$env:RAG_TOTAL_LLM_PROMPT_BUDGET="2000"
$env:RAG_PROMPT_BUDGET_SAFETY_MARGIN="64"
$env:RAG_MAX_SAME_FILE_CHUNKS="2"
$env:RAG_MAX_OVERLAP_LINES="6"
```

## Running the Released Evaluators

Retrieval evaluation:

```bash
python Metrics/evaluate_retrieval.py --output validation-output.json --validation validation-set-golden-qa-pairs.json
```

Judge evaluation:

```bash
python Metrics/run_judge.py --output validation-output.json --validation validation-set-golden-qa-pairs.json --base-url https://tritonai-api.ucsd.edu/v1 --model claude-sonnet-4-6-aws
```

## Experiment Summary

Below is a compact record of the main experiments run during tuning on the
released validation set.

1. Original lexical baseline from earlier `main.py`.
Retrieval:
`F1@5=0.4959`, `P@5=0.3663`, `R@5=0.8944`, `Retrieval Score=0.5856`
Generation with `api-mistral-small-3.2-2506` judge:
`Correctness=0.9778`, `Faithfulness=1.0000`, `Completeness=4.1333/5`,
`Partial Generation Score=0.9348`
Average of retrieval and partial generation:
`0.7602`

2. Section-aware refined retriever experiment.
Retrieval:
`F1@5=0.4989`, `P@5=0.3556`, `R@5=0.9611`, `Retrieval Score=0.6052`
Generation with `api-mistral-small-3.2-2506`:
`Correctness=0.9778`, `Faithfulness=0.9778`, `Completeness=3.4667/5`,
`Partial Generation Score=0.8830`
Average:
`0.7441`

3. Older lexical baseline restored from git history as `main_old5856.py`.
Retrieval:
`F1@5=0.5057`, `P@5=0.3685`, `R@5=0.9056`, `Retrieval Score=0.5932`
Generation with `api-mistral-small-3.2-2506`:
`Correctness=0.9556`, `Faithfulness=1.0000`, `Completeness=4.0222/5`,
`Partial Generation Score=0.9200`
Average:
`0.7566`

4. Tuned hybrid retrieval + embedding + reranking pipeline.
Retrieval:
`F1@5=0.5749`, `P@5=0.4296`, `R@5=0.9611`, `Retrieval Score=0.6552`
Generation with `api-mistral-small-3.2-2506`:
`Correctness=1.0000`, `Faithfulness=1.0000`, `Completeness=4.1778/5`,
`Partial Generation Score=0.9452`
Average:
`0.8002`

5. Current final pipeline in `main.py` with true 2,000-token total prompt
budget enforcement, stronger completeness prompting, moderate generation budget
increase, and context deduplication.
Retrieval:
`F1@5=0.6122`, `P@5=0.4796`, `R@5=0.9611`, `Retrieval Score=0.6843`
Generation with `api-mistral-small-3.2-2506`:
`Correctness=1.0000`, `Faithfulness=1.0000`, `Completeness=4.4667/5`,
`Partial Generation Score=0.9644`
Average:
`0.8244`

## Submission Checklist

- `main.py` exists and runs from the CLI
- `README.md` explains setup and usage
- `logs/` exists and contains experiment artifacts
- released validation output JSON is saved
- golden Q&A file is included
- project report PDF is included
- git history reflects iterative development
