# RapidFire Experiment Writeup

## Goal

We used RapidFire to compare retrieval configurations for the Project 1 RAG
pipeline while keeping `main.py` as the final single-config submission script.
The experiment focused on the retriever, because retrieval quality directly
controls which documentation evidence is available to the answer generator.

The baseline for comparison was the pre-sweep final `main.py` configuration:

- chunk size: 160 words
- chunk overlap: 2 lines
- final top-k: 4 before the sweep; changed to 3 after the follow-up comparison
- lexical candidates: 10
- embedding candidates: 10
- rerank candidates: 8
- embeddings enabled
- LLM reranker enabled
- embedding weight: 0.4
- heading boost: 0.25
- filename boost: 1.0
- max chunks per file: 2
- max overlap lines: 6
- total prompt budget: 2000 tokens
- prompt budget safety margin: 64 tokens

## Experiment Setup

RapidFire services were started on the server, and the sweep was run with:

```bash
conda activate cse234
rapidfireai start
python run_rapidfire.py
```

The script tested 8 configurations against
`validation-set-golden-qa-pairs.json`. For each configuration, it wrote a
Project-1-style output JSON to `logs/rapidfire_runs/`, evaluated the retrieved
source spans with the released retrieval metrics, and logged the run to the
RapidFire/MLflow tracking server.

The retrieval score is the mean of:

- F1@5
- Precision@5
- Recall@5

The sweep was retrieval-only by default, so the output answers are placeholders.
Generation was evaluated separately for the best retrieval variant by running
`main.py` with the selected setting.

## Configurations Tested

| Configuration | Main Change |
| --- | --- |
| `final_default_160_top4` | Current final/default config |
| `smaller_chunks_120_top4` | Reduced chunk size from 160 to 120 |
| `larger_chunks_220_top4` | Increased chunk size from 160 to 220 |
| `final_top3` | Reduced final selected chunks from top 4 to top 3 |
| `final_top5` | Increased final selected chunks from top 4 to top 5 |
| `lexical_only_top4` | Disabled embeddings and LLM reranking |
| `hybrid_no_reranker_top4` | Kept embeddings, disabled LLM reranking |
| `broader_candidates_top4` | Increased lexical/embedding candidates to 16 and rerank candidates to 12 |

## Retrieval Results

| Rank | Configuration | Retrieval Score | F1@5 | Precision@5 | Recall@5 |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | `final_top3` | 0.7408 | 0.6871 | 0.5741 | 0.9611 |
| 2 | `smaller_chunks_120_top4` | 0.7279 | 0.6801 | 0.5815 | 0.9222 |
| 3 | `broader_candidates_top4` | 0.7157 | 0.6435 | 0.5148 | 0.9889 |
| 4 | `final_default_160_top4` | 0.6928 | 0.6211 | 0.4852 | 0.9722 |
| 5 | `larger_chunks_220_top4` | 0.6576 | 0.5820 | 0.4519 | 0.9389 |
| 6 | `final_top5` | 0.6442 | 0.5552 | 0.4163 | 0.9611 |
| 7 | `hybrid_no_reranker_top4` | 0.6189 | 0.5383 | 0.4019 | 0.9167 |
| 8 | `lexical_only_top4` | 0.5997 | 0.5158 | 0.3778 | 0.9056 |

## Key Findings

The best retrieval-only configuration was `final_top3`. It kept the same chunk
size, embeddings, candidate pools, and reranker as the final default, but
reduced the final selected context from 4 chunks to 3 chunks. This improved the
retrieval score from 0.6928 to 0.7408 in the RapidFire sweep. The main gain came
from higher precision, while recall remained high.

Smaller chunks also helped. `smaller_chunks_120_top4` scored 0.7279, suggesting
that shorter evidence spans can make retrieved sources more precise. However,
its recall dropped compared with the default and with `final_top3`.

Expanding candidate pools helped recall but did not beat `final_top3`.
`broader_candidates_top4` reached the highest recall in the sweep, 0.9889, but
its lower precision kept its overall retrieval score below the top two configs.

The ablations confirmed that embeddings and reranking matter. The lexical-only
configuration scored 0.5997, and the hybrid configuration without reranking
scored 0.6189. Both were clearly worse than the default with embeddings and the
LLM reranker.

Increasing the final number of selected chunks from 4 to 5 hurt retrieval score.
`final_top5` had the same recall as `final_top3`, but much lower precision. For
this validation set, adding more context tended to add noise rather than useful
evidence.

## Follow-up Generation Check

After the RapidFire sweep, `main.py` was run with `RAG_FINAL_TOP_K=3` to test
whether the best retrieval variant also improved the final end-to-end score.

```bash
RAG_FINAL_TOP_K=3 python main.py \
  --input validation-set-golden-qa-pairs.json \
  --output logs/final_top3_validation_output.json \
  --corpus-dir sourcedocs \
  --apikey-txt ~/api-key.txt \
  --generation-model api-mistral-small-3.2-2506
```

The released retrieval evaluator gave:

- Retrieval Score: 0.7447
- F1@5: 0.6915
- Precision@5: 0.5815
- Recall@5: 0.9611

The official `claude-sonnet-4-6-aws` judge could not produce a reliable
generation score because the API hit a daily token limit. The run had 28 judge
failures out of 45 questions, and failed judge calls are scored as zero by the
released evaluator.

Using the documented fallback judge, `api-gpt-oss-120b`, the `top3` output
scored:

- Correctness pass rate: 0.8889
- Faithfulness pass rate: 1.0000
- Completeness: 3.9333 / 5
- Partial Generation Score: 0.8919

This fallback judge score is useful for comparison, but it is not directly
equivalent to the official judge score.

## Decision

We changed `main.py` to use `RAG_FINAL_TOP_K=3` as the default.

The first comparison against the historical README score was not perfectly
apples-to-apples because the older generation score had been judged with a
different model. To make the decision cleaner, we reran the current `top4`
default through the same fallback judge used for `top3`.

Using the same fallback judge, `api-gpt-oss-120b`, the comparison was:

| Configuration | Retrieval Score | Partial Generation Score | Average |
| --- | ---: | ---: | ---: |
| `final_top3` | 0.7447 | 0.8919 | 0.8183 |
| `final_default_160_top4` | 0.6928 | 0.8637 | 0.7783 |

Under this same-judge comparison, `top3` improves both retrieval and the
released partial generation score. The final submission script now defaults to
top 3 selected context chunks, while still allowing the value to be overridden
with the `RAG_FINAL_TOP_K` environment variable.

## Artifacts

- Experiment runner: `run_rapidfire.py`
- RapidFire/MLflow experiment name: `project1-rag-config-sweep`
- Local run outputs: `logs/rapidfire_runs/`
- Summary table: `logs/rapidfire_runs/summary.json`
- Top-3 validation output: `logs/final_top3_validation_output.json`
