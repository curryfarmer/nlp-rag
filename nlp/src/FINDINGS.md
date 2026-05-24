# NLP Regex-Cheese — Findings (Phases 1-3)

BrainHack TIL-AI 2026 NLP track. RAG QA over 296-doc fictional cyberpunk corpus, 883 eval questions (`nlp.jsonl`). Goal: beat the old RAG pipeline's **0.759** leaderboard score with little/no ML ("cheese it").

## Scoring model (what we're optimising for)

`test/test_nlp.py` per-question:
- Retrieval gate: ≥1 of our top-3 predicted docs ∈ `source_docs` → else **0.0**.
- Retrieval hit + answer equivalent (BEM-style model, threshold 0.9) → **1.0**.
- Retrieval hit + answer not equivalent → **0.4** (`RETRIEVAL_ONLY_SCORE`).
- Final score = mean across questions.

So: nail retrieval + return *any* string → floor ~0.4. Each question that clears the 0.9 equivalence threshold adds 0.6/N.

**Key unknown**: we have no working local copy of the real equivalence model. `nlp/src/nlp_eval_512/` is a friend's stand-in (config+tokenizer only, no weights) — NOT the competition scorer. All our scores are *proxy estimates*; the real number is only knowable on GCP.

## Dataset facts

- 883 rows; all `answerable: true`; all exactly 1 `source_docs`. L1=592, L2=291.
- L1 answers: median 3 words, 84% appear verbatim in source doc.
- L2 answers: median 8 words; 90% are multi-fact composition / arithmetic / comparison → ~0% extractable by pure regex.
- Frozen 80/20 split at `nlp/src/eval_split.json` (seed 42): 706 dev / 177 held-out.

## Phase 1 — Retrieval (DONE, shipped)

Pure-regex rarity-weighted inverted index in `nlp/src/nlp_manager.py`. No ML.
- Tokeniser preserves codes (`EA-76-088`), PCE dates, caps entities; IDF scoring; df=1 unique-token shortcut; verbatim phrase + content-bigram boosters; distinct-token coverage bonus.
- **Result: top-3 recall = 0.953 dev / 0.961 held-out.** Retrieval is solved.

## Phase 2 — Answer extraction (DONE, shipped, default)

Hint-gated bucket-router in `nlp/src/nlp_answer.py`. Pure regex. Buckets: money, percent, date_pce, count (noun-anchored), codename (designation/caps-list/whole-doc), entity, years_delta (date arithmetic), bold-label (`**Label:** value`), sentence-trim fallback.

Iteration log in `nlp/src/test_logs/phase2.md` (v0→v7b). **Best: v7b, est 0.447 strict / 0.516 loose (dev), 0.427 / 0.532 (held-out).**

Per-bucket loose pass (dev): percent 0.46, count 0.42, money 0.36, date 0.25, codename 0.25 — fallback (348/706 rows) stuck at ~0.10, L2 ~0%.

Plateau confirmed: <1pp gain per iteration after v3.

## Phase 3 — Cross-encoder reranker (DONE, did NOT help, behind flag)

Idea: regex enumerates ~30 candidate spans, `cross-encoder/ms-marco-MiniLM-L6-v2` picks the best. Code in `nlp/src/nlp_rerank.py` + `generate_candidates()`/`extract_answer_reranked()` in `nlp_answer.py`. Gated by `NLP_USE_RERANK=1` (default 0).

- **Candidate-pool recall ceiling = 0.61** (gold ∈ 30 candidates for 61% of dev) → regex generates good candidates.
- v8.0 (CE over full pool): **0.44 loose — disaster.** CE biases toward verbose passages, destroys tight numeric (percent 0.46→0.09).
- v8.1 (specialist first, CE on fallback only): 0.516 loose — flat, CE rarely fires.
- v8.2 (always CE, tight-only pool): 0.42 loose — CE picks the wrong same-shape span (no sentence context to disambiguate "which 4.5M Credits is the penalty").

**Conclusion**: MS-MARCO CE is the wrong tool. It scores (query, passage) relevance, not span-level answerhood. Our isolated-span candidates lack the context CE needs; with context it over-prefers length. **Default reverted to regex-only v7b.**

## Where things live

- Inference path: pure regex, **zero model weights**.
- Optional CE: `~/.cache/huggingface/hub/models--cross-encoder--ms-marco-MiniLM-L6-v2` (88MB), download-on-first-use. Would need baking into Dockerfile if used (eval box may lack internet).
- `nlp/requirements.txt`: fastapi+uvicorn (always) + sentence-transformers/transformers/torch/numpy (only for CE path). Local env needs `numpy<2` to match torch 2.2.2.

## Honest state

- Proxy estimate: **~0.52 loose / ~0.45 strict** vs target 0.759.
- Real BEM likely more lenient than strict proxy (asymmetric, accepts gold-substring) → real number plausibly 0.50-0.60, maybe higher. **Unverified.**
- Pure-regex realistic ceiling ~0.60-0.65 (candidate recall 0.61 is the hard cap unless candidate generation broadens).

## Recommended next steps (ranked)

1. **Calibrate on GCP first.** Push branch, run `python test/test_nlp.py` with the real BEM weights. Everything downstream depends on the true number. We're optimising blind.
2. **Swap CE for a span-QA model** if reranking is still wanted: `cross-encoder/qnli-distilroberta-base` (entailment-trained, closer to answerhood) via `NLP_RERANK_MODEL` env var — one-line test. Or a real extractive reader `distilbert-base-uncased-distilled-squad` (207MB) as the *fallback-bucket* extractor (not a reranker).
3. **Broaden candidate generation** to lift the 0.61 recall ceiling (more sweep patterns, sentence-context candidates) — only worth it if a working reranker exists.
4. **Accept regex-only** and ship if the real GCP number is close enough to 0.759.

## How to run things

```bash
cd nlp/src
python eval_answers.py --by-bucket                 # regex-only dev proxy
python eval_answers.py --rerank --by-bucket        # + CE (needs sentence-transformers, numpy<2)
python eval_answers.py --candidate-coverage        # recall ceiling of candidate pool
python eval_answers.py --heldout                   # held-out (run sparingly)
python eval_retrieval.py                           # Phase 1 retrieval recall
# Real scorer (GCP, has BEM weights):
#   NLP_DATA_DIR=$(pwd)/nlp/src NLP_RESULTS_DIR=/tmp/res python test/test_nlp.py
```
