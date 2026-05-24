# Phase 4 — Small-LLM reader (grounded reading comprehension)

Thesis: regex plateaued (~0.45 strict / 0.52 loose proxy). The unsolved mass is
the **fallback bucket (~49% of rows)** and **L2 (33%)** — both need
*comprehension*, not more patterns. Research (2026) + dataset analysis: 68.7% of
L2 is reading/comparison/counting a small instruct LLM handles when fed the
source doc; only 7.6% is hard arithmetic. So: small instruct LLM reads the
retrieved doc(s) and answers.

## Two structural findings (drive the design)

1. **Retrieval@1 = 0.763, @3 = 0.950 (dev).** Feeding the LLM only top-1 caps
   correctness at 0.76 regardless of comprehension. **Feed top-k (k=3).** Lifts
   the answer-in-context ceiling to 0.95. Biggest single lever.
2. **The scorer's 64-token candidate cap is NOT enforced.** `test_nlp.py:184`
   uses `tokenizer.tokenize(..., max_length=64, truncation=True)`, but HF
   `tokenize()` ignores `max_length`/`truncation` (only `__call__`/`encode` honor
   them). Real truncation is the 512-cap on the whole `Q+Ref+Candidate` string
   (line 233). So long composed L2 answers survive — but keep terse anyway; BEM
   @0.9 penalizes verbosity/rewording.

## Levers (combined config = "1-2-3-4")

1. top-k=3 doc feeding (`NLP_LLM_TOP_K=3`)
2. bigger model (1.5B -> 3B -> 7B; GPU box only for 7B)
3. few-shot, gold-style hedged exemplars (`NLP_LLM_FEWSHOT=1`)
4. CoT-then-extract for arithmetic/aggregation (`NLP_LLM_COT=1`)

## Code

- `nlp_llm.py` — lazy-singleton reader (Qwen2.5-1.5B-Instruct default,
  Apache-2.0). Env: `NLP_USE_LLM`, `NLP_LLM_MODEL`, `NLP_LLM_FEWSHOT`,
  `NLP_LLM_COT`, `NLP_LLM_TOP_K`. GPU/fp16 auto, CPU/fp32 fallback.
- `nlp_answer.py::extract_answer_llm(question, docs)` — delegates to reader;
  regex `extract_answer` fallback if reader unavailable/empty.
- `nlp_manager.py::qa_batch` — `NLP_USE_LLM=1` branch feeds top-k docs.
- `eval_answers.py` — `--llm --llm-fewshot --llm-cot --llm-top-k --llm-model`.
- `scratch/try_llm.py` — standalone sample harness (proxy + dump for manual read).

## Local constraint

Dev box is CPU-only, 16GB. 1.5B fp32 runs (~slow, tens of s/q). 3B borderline,
7B does NOT fit. Local = smoke test + tiny samples only. **Real sweep + real BEM
scoring happens on the GCP GPU box.**

## GCP runbook (GPU + real BEM weights)

```bash
cd nlp/src
# 0. FINALLY get the real regex-only baseline (no LLM):
#    run server regex-only, then test_nlp.py against it (real BEM @0.9).
# 1. Proxy sweep (fast on GPU) — pick the knee. Ablate model size AND cot:
#    (cot needs a bigger token budget: NLP_LLM_MAX_NEW=256)
python eval_answers.py --llm --llm-top-k 3 --llm-fewshot --by-bucket                         # 1.5B, no cot
python eval_answers.py --llm --llm-top-k 3 --llm-fewshot --llm-model Qwen/Qwen2.5-3B-Instruct --by-bucket
python eval_answers.py --llm --llm-top-k 3 --llm-fewshot --llm-model Qwen/Qwen2.5-7B-Instruct --by-bucket
NLP_LLM_MAX_NEW=256 python eval_answers.py --llm --llm-top-k 3 --llm-fewshot --llm-cot --llm-model Qwen/Qwen2.5-7B-Instruct --by-bucket  # cot only worth it at 7B
# 2. Real BEM on the winning config: launch server with the env, run test_nlp.py.
#    NLP_USE_LLM=1 NLP_LLM_TOP_K=3 NLP_LLM_FEWSHOT=1 [NLP_LLM_COT=1 NLP_LLM_MAX_NEW=256] NLP_LLM_MODEL=... <launch server>
#    then: NLP_SERVER_URL=... python test/test_nlp.py
# Also get the regex-only baseline first (NLP_USE_LLM unset) for the real delta.
```

## Iteration log

| ver | config | sample | retr@3 | F1 | gold_sub | est_strict | est_loose | note |
|-----|--------|--------|--------|-----|----------|------------|-----------|------|
| v9.0 | 1.5B top1 (baseline) | — | — | — | — | — | — | killed; CPU too slow, ceiling-capped @0.76 |
| v9.1 | 1.5B top3+fewshot+cot | 10 (5L1+5L2) | 1.00 | 0.58 | 0.50 | 0.70 | 0.76 | L1 5/5 by eye; L2 ~0.5/5; CoT misfires |

### v9.1 smoke read (manual — proxy undercounts semantics)

- **L1 = 5/5 correct.** 1.5B reads L1 cleanly in gold style ("63 PCE (2105 CE)",
  "Every 3 years", "Class III, permit number SH-EV-00714"). proxy est_strict
  1.00. Huge vs regex (L1-in-fallback was ~0.10). **L1 is solved by the reader.**
- **L2 ≈ 0.5/5.** 1.5B too weak for multi-step composition/arithmetic.
- **CoT is net-negative at 1.5B:**
  - `max_new=128` truncates reasoning *before* the `Answer:` line (key745 →
    garbage). CoT needs >=256.
  - Model often omits the `Answer:` line entirely → `parse_cot` returns the
    verbose reasoning blob (key247, key515).
  - Wrong intermediate math / wrong count (key745, key693 "11" vs gold "3").
- **Speed: 83.6 s/q on CPU.** Local sweep infeasible — confirmed GCP-only.

### Next (GCP, GPU)

1. Sweep **3B and 7B** — L2 is a model-capacity problem.
2. Ablate **CoT on vs off** (suspect off is better below 7B; verbosity + format
   miss hurt). If on, set `--max-new 256`.
3. Keep top-3 + fewshot (both clearly helping L1 style + ceiling).
4. Then real BEM via test_nlp.py on the winning config.
