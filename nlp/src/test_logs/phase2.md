# Phase 2 — answer extraction iteration log

Proxy metric is `LIKELY` (F1 ≥ 0.7 + tight length). Real scorer is BEM-style at threshold 0.9 → numbers will shift, deltas hold.

`est equiv_rate = retrieval@3 × (likely × 1.0 + (1-likely) × 0.4)`.

Target ≥ 0.759 (the current RAG baseline).

| Version | Change | DEV n | retr@3 | EM | F1 | likely | est equiv | L1 likely | L2 likely | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| v0 | Phase 1 retrieval + empty answer | 706 | 0.953 | 0.000 | 0.000 | 0.000 | 0.381 | 0.000 | 0.000 | Floor from retrieval-only (0.4 × 0.953). |
| v1 | Cascade extractors (codename/percent/date/money/count/entity) + 25-word fallback | 706 | 0.950 | 0.072 | 0.121 | 0.088 | 0.433 | 0.125 | 0.009 | percent=0.39, money=0.27, count=0.09. date_pce=0 (bug), codename=0 (bug), fallback=0.02 (sentences too long). |
| v2 | Anchored hints (gate not order). Longer date alts first + (CE) suffix. Codename = caps-list + designation. Fallback = best-overlap clause, ≤15 words. | 706 | 0.955 | 0.068 | 0.166 | 0.088 | 0.435 | 0.127 | 0.004 | F1 up across board (0.12→0.17). LIKELY threshold (F1≥0.7) too strict to register — answers got closer but didn't cross. date_pce 0→0.04, entity_what 0.17→0.33, fallback gold_sub 0.083→0.115. |
| v3 | Count = noun-anchored window. Number-word regex. Strip md noise from fallback. Fallback ≤10 words. | 706 | 0.953 | 0.095 | 0.183 | 0.109 | 0.447 | 0.158 | 0.004 | count 0.10→0.20 ✓ date 0.04→0.17 ✓ L1 0.13→0.16. Fallback regressed (10-word too tight). |
| v4 | Sliding-window fallback over whole doc (density-scored). | 706 | 0.956 | 0.095 | 0.161 | 0.109 | 0.448 | 0.158 | 0.004 | Reverted: density picks repeated phrases, not answers. Fallback gold_sub dropped 0.078→0.029. |
| v5 | Reverted to clause-trim fallback ≤16 words + anchor (number/caps/code) bonus. Added LOOSE proxy. | 706 | 0.952 | 0.092 | 0.194 | 0.109 | 0.446 (strict) / 0.513 (loose) | 0.158 | 0.004 | Loose pass: count=0.42, money=0.36, percent=0.46, date=0.21. Strict-vs-loose gap = padded answers — real BEM probably between. |
| v6 | Codename whole-doc scan (designation regex first). years_delta arithmetic calculator (2 PCE years near q-keywords → "approximately N years"). | 706 | 0.950 | 0.096 | 0.199 | 0.113 | **0.448 (strict) / 0.514 (loose)** | 0.164 | 0.004 | codename 0→0.25, date 0.08→0.17, years_delta=0 strict / 0.06 loose. Plateau without AE model weights to calibrate. |
| v7 | 3 corpus-reading agents + stdlib miner. Added bold-label extractor (`**Label:** value`), Q→doc synonym map, expanded codename blocklist (CLASSIFICATION/DISTRIBUTION/etc). Bold-label FIRST. | 706 | 0.948 | 0.084 | 0.179 | 0.099 | 0.439 / 0.504 | 0.143 | 0.004 | Bold-label fired too aggressively, hurt money 0.250→0.163. Reverted ordering in v7b. |
| v7b | Bold-label moved AFTER specialists, BEFORE fallback. Same blocklist + synonym map. | 706 | 0.950 | 0.094 | 0.194 | 0.112 | **0.447 (strict) / 0.516 (loose)** | 0.162 | 0.004 | Marginal +0.002 loose. fallback 0.017→0.020 likely. Hard plateau confirmed on proxy. |
| v8.0 | + cross-encoder/ms-marco-MiniLM-L6-v2 reranker over full 30-candidate pool (specialist + bold + clause + sweep + window). Always rerank. | 706 | 0.950 | 0.000 | 0.116 | 0.004 | 0.383 / 0.438 | 0.006 | 0.000 | DISASTER. CE biases toward verbose passages — destroys tight numeric. percent 0.45→0.09, money 0.36→0.13, count 0.42→0.15. Confirms plan-risk: MS-MARCO CE trained for passage relevance, not span QA. |
| v8.1 | Specialist regex first; CE only fires when no specialist hit. | 706 | 0.953 | 0.095 | 0.204 | 0.112 | 0.448 / 0.516 | 0.162 | 0.004 | Flat vs v7b. Few questions reach CE (specialists fire on most). CE that does fire shows mild fallback gold_sub lift (+0.029) but loose unchanged. |
| v8.2 | Always CE, but tight-only candidate pool (no clauses / sliding windows). | 706 | 0.952 | 0.017 | 0.062 | 0.021 | 0.393 / 0.422 | 0.027 | 0.009 | Worse. CE picks the wrong tight span (no sentence-context info). Confirms CE needs surrounding context to disambiguate among candidates of identical shape (e.g. multiple money figures in one doc). |
| **Conclusion** | MS-MARCO cross-encoder isn't the right tool with our isolated-span candidates. Either need span-QA-trained CE (DistilBERT-SQuAD, DeBERTa-v3-xsmall, qnli-distilroberta), or feed CE sentence-context candidates and have a separate extractor narrow within. | | | | | | | | Default reverted to regex-only `extract_answer`. CE path retained behind `NLP_USE_RERANK=1` for next experiment. |



