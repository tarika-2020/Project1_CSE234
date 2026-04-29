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

## Current Baseline

The current `main.py` is only a baseline scaffold. It:

- reads `.rst` files from `sourcedocs/`
- chunks with line-aware span tracking
- retrieves with a deterministic BM25-style lexical baseline
- serializes context under a token-based budget
- emits the required output schema

It does **not** yet provide:

- a production-quality retriever
- TritonAI generation integration
- final submission-quality metrics

## Environment and Configuration

Expected Python packages for the final system:

- `openai`
- `tiktoken`
- `rapidfireai`

Expected environment variables for TritonAI or OpenAI-compatible generation and
judge workflows:

- `OPENAI_API_KEY`: required for TritonAI or OpenAI calls
- `OPENAI_BASE_URL`: optional for the generator client
- `JUDGE_BASE_URL`: optional for the released judge script
- `JUDGE_MODEL`: optional override for the released judge script

For the course TritonAI gateway, the base URL is expected to be:

`https://tritonai-api.ucsd.edu/v1`

## Running the Pipeline

Example command:

```bash
python main.py --input validation-set-golden-qa-pairs.json --output validation-output.json
```

The input file must be a JSON list of objects with:

- `question_id`
- `question`

The output file must be a JSON list of objects with:

- `question_id`
- `answer`
- `retrieved_context`
- `sources`

## Running the Released Evaluators

Retrieval evaluation:

```bash
python Metrics/evaluate_retrieval.py --output validation-output.json --validation validation-set-golden-qa-pairs.json
```

Judge evaluation:

```bash
python Metrics/run_judge.py --output validation-output.json --validation validation-set-golden-qa-pairs.json --base-url https://tritonai-api.ucsd.edu/v1 --model <chat-model>
```

## Submission Checklist

- `main.py` exists and runs from the CLI
- `README.md` explains setup and usage
- `logs/` exists and contains experiment artifacts
- released validation output JSON is saved
- golden Q&A file is included
- project report PDF is included
- git history reflects iterative development

## Next Planned Milestones

- integrate TritonAI generation
- wire local evaluation and RapidFire experiments
