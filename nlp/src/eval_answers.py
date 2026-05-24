"""Phase 2 proxy scorer.

Local equivalence-model weights are missing, so we estimate per-bucket pass-rate
via four cheap proxies (stdlib only) and report what the official AE scorer
would likely award:

  - EM      : normalised exact match (lowercase, strip, drop punct/articles)
  - F1      : token-overlap F1 (SQuAD-style)
  - GS      : gold-substring  (gold ⊆ pred or pred ⊆ gold after normalisation)
  - LIKELY  : F1 >= 0.7 AND len(pred_tokens) <= 3*len(gold_tokens) — a rough
              "tight + accurate" proxy for the BEM-style 0.9 threshold.

The official equivalence model is asymmetric and strict; F1/EM are weak proxies
but track improvements well enough to iterate. Trust deltas, not absolutes.

Score estimate = retrieval_recall * (likely_pass_rate * 1.0 + (1-likely_pass_rate) * 0.4)
… mirrors the formula in test/test_nlp.py (retrieval-only credit = 0.4).

Usage:
  python eval_answers.py                 # dev split
  python eval_answers.py --heldout       # held-out (run once)
  python eval_answers.py --all
  python eval_answers.py --by-bucket
  python eval_answers.py --dump FAILS.json  # write misses for inspection
"""
from __future__ import annotations

import argparse
import json
import os
import re
import string
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from nlp_manager import NLPManager  # noqa: E402
from nlp_answer import (  # noqa: E402
    extract_answer, extract_answer_reranked, generate_candidates,
)

DOCS_DIR = HERE / "documents"
EVAL_PATH = HERE / "nlp.jsonl"
SPLIT_PATH = HERE / "eval_split.json"


_PUNCT = str.maketrans("", "", string.punctuation)
_ARTICLES = {"a", "an", "the"}


def _normalise(s: str) -> str:
    s = s.lower().translate(_PUNCT)
    return " ".join(w for w in s.split() if w not in _ARTICLES)


def _tokens(s: str) -> list[str]:
    return _normalise(s).split()


def em(pred: str, gold: str) -> bool:
    return _normalise(pred) == _normalise(gold)


def f1(pred: str, gold: str) -> float:
    p, g = _tokens(pred), _tokens(gold)
    if not p or not g:
        return 1.0 if p == g else 0.0
    common = Counter(p) & Counter(g)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(p)
    recall = overlap / len(g)
    return 2 * precision * recall / (precision + recall)


def gold_substring(pred: str, gold: str) -> bool:
    p, g = _normalise(pred), _normalise(gold)
    if not p or not g:
        return False
    return g in p or p in g


def likely_pass(pred: str, gold: str) -> bool:
    """Strict proxy for BEM 0.9 threshold: high F1 + tight length."""
    if not pred or not gold:
        return pred == gold
    p_tokens, g_tokens = _tokens(pred), _tokens(gold)
    if not p_tokens or not g_tokens:
        return False
    if len(p_tokens) > 3 * len(g_tokens) and len(p_tokens) > 5:
        return False
    return f1(pred, gold) >= 0.7


def loose_pass(pred: str, gold: str) -> bool:
    """Looser BEM proxy — F1 ≥ 0.5 OR gold is a substring of pred (bounded length).
    Real BEM is asymmetric and accepts a candidate containing gold + minor extras.
    Use this for an upper-bound score estimate."""
    if not pred or not gold:
        return pred == gold
    p_tokens, g_tokens = _tokens(pred), _tokens(gold)
    if not p_tokens or not g_tokens:
        return False
    if len(p_tokens) > 4 * max(3, len(g_tokens)):
        return False
    return f1(pred, gold) >= 0.5 or gold_substring(pred, gold)


# --- bucketing (mirrors scratch/answer_analysis.py rules, for diagnostics) ----

_BUCKET_RULES: list[tuple[str, list[str]]] = [
    ("money",      ["credits", "phi credits", "dollar", "penalty", "cost", "revenue",
                    "fine", "budget", "funding", "how much"]),
    ("percent",    ["%", "percentage", "percent", "share of"]),
    ("date_pce",   ["when ", "by what deadline", "on what date", "in what year"]),
    ("years_delta",["how many years", "how long after", "how long between", "interval"]),
    ("count",      ["how many of", "how many"]),
    ("codename",   ["codename", "designation", "code name", "name of the program"]),
    ("entity_who", ["who ", "by whom", "which person", "which official"]),
    ("entity_what",["which company", "which division", "which sector", "what industry"]),
    ("proportion_other", ["what proportion", "what fraction", "what ratio"]),
]


def bucket_of(question: str) -> str:
    q = question.lower()
    for name, keys in _BUCKET_RULES:
        if any(k in q for k in keys):
            return name
    return "fallback"


# --- corpus + eval loaders -----------------------------------------------------


def load_corpus() -> list[dict[str, str]]:
    return [{"id": p.stem, "document": p.read_text(encoding="utf-8")}
            for p in sorted(DOCS_DIR.glob("DOC-*.txt"))]


def load_eval() -> list[dict]:
    rows = []
    with EVAL_PATH.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def get_split() -> dict:
    return json.loads(SPLIT_PATH.read_text())


# --- core eval ---------------------------------------------------------------


def evaluate(rows: list[dict], mgr: NLPManager, use_proxy: bool = False) -> dict:
    preds = mgr.qa_batch([r["question"] for r in rows])

    proxy_equiv = None
    if use_proxy:
        from nlp_bem_proxy import equivalent as proxy_equiv  # noqa: F811

    per: list[dict] = []
    bucket_stats: dict[str, list[dict]] = defaultdict(list)
    difficulty_stats: dict[str, list[dict]] = defaultdict(list)
    retrieval_hits = 0
    for r, p in zip(rows, preds):
        gold_docs = set(r["source_docs"])
        retrieval_hit = bool(set(p["documents"][:3]) & gold_docs)
        retrieval_hits += int(retrieval_hit)
        row_em = em(p["answer"], r["answer"])
        row_f1 = f1(p["answer"], r["answer"])
        row_gs = gold_substring(p["answer"], r["answer"])
        row_lp = likely_pass(p["answer"], r["answer"])
        row_loose = loose_pass(p["answer"], r["answer"])
        row_proxy = (proxy_equiv(p["answer"], r["answer"], r["question"])
                     if proxy_equiv else None)
        rec = {
            "key": r.get("key"),
            "difficulty": r.get("difficulty"),
            "bucket": bucket_of(r["question"]),
            "question": r["question"],
            "gold": r["answer"],
            "pred": p["answer"],
            "gold_docs": list(gold_docs),
            "pred_docs": p["documents"],
            "retrieval_hit": retrieval_hit,
            "em": row_em,
            "f1": round(row_f1, 3),
            "gold_substring": row_gs,
            "likely_pass": row_lp,
            "loose_pass": row_loose,
            "proxy_pass": row_proxy,
        }
        per.append(rec)
        bucket_stats[rec["bucket"]].append(rec)
        difficulty_stats[rec["difficulty"]].append(rec)

    n = len(rows)
    summary = {
        "n": n,
        "retrieval_top3": round(retrieval_hits / n, 4),
        "EM": round(sum(r["em"] for r in per) / n, 4),
        "F1_mean": round(sum(r["f1"] for r in per) / n, 4),
        "gold_substring": round(sum(r["gold_substring"] for r in per) / n, 4),
        "likely_pass": round(sum(r["likely_pass"] for r in per) / n, 4),
        "loose_pass": round(sum(r["loose_pass"] for r in per) / n, 4),
    }

    def _est(pass_key: str) -> float:
        s = 0.0
        for r in per:
            if not r["retrieval_hit"]:
                s += 0.0
            elif r[pass_key]:
                s += 1.0
            else:
                s += 0.4
        return round(s / n, 4)

    summary["est_strict"] = _est("likely_pass")
    summary["est_loose"] = _est("loose_pass")
    if use_proxy:
        summary["proxy_pass"] = round(sum(bool(r["proxy_pass"]) for r in per) / n, 4)
        summary["est_proxy"] = _est("proxy_pass")

    by_bucket = {}
    for b, rs in sorted(bucket_stats.items(), key=lambda kv: -len(kv[1])):
        m = len(rs)
        by_bucket[b] = {
            "n": m,
            "EM": round(sum(r["em"] for r in rs) / m, 4),
            "F1": round(sum(r["f1"] for r in rs) / m, 4),
            "gold_sub": round(sum(r["gold_substring"] for r in rs) / m, 4),
            "likely": round(sum(r["likely_pass"] for r in rs) / m, 4),
            "loose": round(sum(r["loose_pass"] for r in rs) / m, 4),
        }
    by_difficulty = {}
    for d, rs in sorted(difficulty_stats.items()):
        m = len(rs)
        by_difficulty[d] = {
            "n": m,
            "likely_pass": round(sum(r["likely_pass"] for r in rs) / m, 4),
            "F1": round(sum(r["f1"] for r in rs) / m, 4),
        }

    return {"summary": summary, "by_bucket": by_bucket,
            "by_difficulty": by_difficulty, "per_row": per}


def _candidate_coverage(args) -> int:
    docs = load_corpus()
    rows = load_eval()
    mgr = NLPManager()
    mgr.load_corpus(docs)
    doc_lookup = {d["id"]: d["document"] for d in docs}

    if args.all:
        sel = rows
        label = "FULL"
    else:
        split = get_split()
        idx = split["heldout"] if args.heldout else split["dev"]
        sel = [rows[i] for i in idx]
        label = "HELDOUT" if args.heldout else "DEV"

    exact_in = partial_in = 0
    cand_count_sum = 0
    print(f"\n=== {label} candidate-coverage ({len(sel)} rows) ===", file=sys.stderr)
    for r in sel:
        gold = r["answer"]
        doc_id = r["source_docs"][0]
        if doc_id not in doc_lookup:
            continue
        cands = generate_candidates(r["question"], doc_lookup[doc_id])
        cand_count_sum += len(cands)
        gold_n = _normalise(gold)
        norm_cands = [_normalise(c) for c in cands]
        if any(c == gold_n for c in norm_cands):
            exact_in += 1
            partial_in += 1
        elif any(gold_n in c or (c and c in gold_n) for c in norm_cands if c):
            partial_in += 1

    n = len(sel)
    print(f"  candidate-set size (mean) = {cand_count_sum / max(1,n):.1f}")
    print(f"  gold normalised-EM hit-rate     = {exact_in / n:.4f}")
    print(f"  gold-substring (either dir)     = {partial_in / n:.4f}")
    print(f"  -> reranker ceiling on this set (loose) ≈ {partial_in / n:.4f}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--heldout", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--by-bucket", action="store_true")
    ap.add_argument("--proxy-bem", action="store_true",
                    help="Score answers with nlp_bem_proxy (BEM-style equivalence proxy).")
    ap.add_argument("--dump", help="write misses (likely_pass=False) to this json file")
    ap.add_argument("--rerank", action="store_true",
                    help="Enable cross-encoder reranker (sets NLP_USE_RERANK=1).")
    ap.add_argument("--llm", action="store_true",
                    help="Enable small-LLM reader (sets NLP_USE_LLM=1).")
    ap.add_argument("--llm-fewshot", action="store_true")
    ap.add_argument("--llm-cot", action="store_true")
    ap.add_argument("--llm-top-k", type=int, default=None)
    ap.add_argument("--llm-model", default=None)
    ap.add_argument("--candidate-coverage", action="store_true",
                    help="Skip scoring; report the fraction of dev rows that have gold "
                         "(or substring) in the regex-generated candidate set. "
                         "Diagnoses recall ceiling.")
    args = ap.parse_args()

    if args.rerank:
        os.environ["NLP_USE_RERANK"] = "1"
        print("[rerank] enabled", file=sys.stderr)

    if args.llm:
        os.environ["NLP_USE_LLM"] = "1"
        if args.llm_fewshot:
            os.environ["NLP_LLM_FEWSHOT"] = "1"
        if args.llm_cot:
            os.environ["NLP_LLM_COT"] = "1"
        if args.llm_top_k is not None:
            os.environ["NLP_LLM_TOP_K"] = str(args.llm_top_k)
        if args.llm_model:
            os.environ["NLP_LLM_MODEL"] = args.llm_model
        print(f"[llm] enabled (fewshot={args.llm_fewshot} cot={args.llm_cot} "
              f"top_k={args.llm_top_k} model={args.llm_model})", file=sys.stderr)

    if args.candidate_coverage:
        return _candidate_coverage(args)

    docs = load_corpus()
    rows = load_eval()
    mgr = NLPManager()
    mgr.load_corpus(docs)

    if args.all:
        sel = rows
        label = "FULL"
    else:
        split = get_split()
        idx = split["heldout"] if args.heldout else split["dev"]
        sel = [rows[i] for i in idx]
        label = "HELDOUT" if args.heldout else "DEV"

    out = evaluate(sel, mgr, use_proxy=args.proxy_bem)
    s = out["summary"]
    print(f"\n=== {label} ({s['n']} rows) ===")
    print(f"  retrieval@3        = {s['retrieval_top3']}")
    print(f"  EM                 = {s['EM']}")
    print(f"  F1 (mean)          = {s['F1_mean']}")
    print(f"  gold⊆pred or rev   = {s['gold_substring']}")
    print(f"  LIKELY pass (strict)= {s['likely_pass']}  -> est = {s['est_strict']}")
    print(f"  LOOSE  pass        = {s['loose_pass']}  -> est = {s['est_loose']}")
    if args.proxy_bem:
        print(f"  PROXY-BEM pass     = {s['proxy_pass']}  -> est = {s['est_proxy']}")

    print("\n  By difficulty:")
    for d, v in out["by_difficulty"].items():
        print(f"    {d:5} n={v['n']:4}  likely={v['likely_pass']:.3f}  F1={v['F1']:.3f}")

    if args.by_bucket:
        print("\n  By bucket:")
        for b, v in out["by_bucket"].items():
            print(f"    {b:18} n={v['n']:4}  likely={v['likely']:.3f}  "
                  f"loose={v['loose']:.3f}  F1={v['F1']:.3f}  "
                  f"EM={v['EM']:.3f}  gold_sub={v['gold_sub']:.3f}")

    if args.dump:
        misses = [r for r in out["per_row"] if r["retrieval_hit"] and not r["likely_pass"]]
        Path(args.dump).write_text(json.dumps(misses, indent=2))
        print(f"\n  wrote {len(misses)} misses to {args.dump}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
