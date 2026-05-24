"""Measure retrieval recall of NLPManager against the local eval set.

Usage:
    python eval_retrieval.py            # dev split (706 rows)
    python eval_retrieval.py --heldout  # held-out split (177 rows). Run ONCE.
    python eval_retrieval.py --all      # full 882 rows (sanity check only)
    python eval_retrieval.py --failures # dev split + dump miss details

Creates eval_split.json on first run (seeded 80/20 shuffle).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from nlp_manager import NLPManager  # noqa: E402

DOCS_DIR = HERE / "documents"
EVAL_PATH = HERE / "nlp.jsonl"
SPLIT_PATH = HERE / "eval_split.json"
SEED = 42
DEV_FRAC = 0.80


def load_corpus_from_disk() -> list[dict[str, str]]:
    documents = []
    for p in sorted(DOCS_DIR.glob("DOC-*.txt")):
        documents.append({"id": p.stem, "document": p.read_text(encoding="utf-8")})
    return documents


def load_eval() -> list[dict]:
    rows = []
    with EVAL_PATH.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def make_or_load_split(n_rows: int) -> dict:
    if SPLIT_PATH.exists():
        return json.loads(SPLIT_PATH.read_text())
    rng = random.Random(SEED)
    idx = list(range(n_rows))
    rng.shuffle(idx)
    cut = int(n_rows * DEV_FRAC)
    split = {"seed": SEED, "dev_frac": DEV_FRAC, "n": n_rows,
             "dev": idx[:cut], "heldout": idx[cut:]}
    SPLIT_PATH.write_text(json.dumps(split, indent=2))
    return split


def score(manager: NLPManager, rows: list[dict]) -> dict:
    top1 = top3 = 0
    misses: list[dict] = []
    ranks: list[int | None] = []
    for r in rows:
        gold = set(r["source_docs"])
        preds = manager._retrieve(r["question"], k=10)  # over-retrieve for diagnostics
        top3_preds = preds[:3]
        top1_hit = bool(preds and preds[0] in gold)
        top3_hit = bool(set(top3_preds) & gold)
        top1 += int(top1_hit)
        top3 += int(top3_hit)
        rank = next((i + 1 for i, p in enumerate(preds) if p in gold), None)
        ranks.append(rank)
        if not top3_hit:
            misses.append({
                "key": r.get("key"),
                "difficulty": r.get("difficulty"),
                "question": r["question"],
                "gold": list(gold),
                "preds_top10": preds,
                "true_rank": rank,
            })
    n = len(rows)
    return {
        "n": n,
        "top1_recall": round(top1 / n, 4),
        "top3_recall": round(top3 / n, 4),
        "mean_rank_when_found": (
            round(sum(r for r in ranks if r is not None) / max(1, sum(1 for r in ranks if r is not None)), 2)
        ),
        "misses": misses,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--heldout", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--failures", action="store_true", help="dump miss details for dev")
    ap.add_argument("--max-misses", type=int, default=20)
    args = ap.parse_args()

    print(f"Loading corpus from {DOCS_DIR} ...", file=sys.stderr)
    docs = load_corpus_from_disk()
    print(f"  {len(docs)} documents loaded.", file=sys.stderr)

    print(f"Loading eval from {EVAL_PATH} ...", file=sys.stderr)
    rows = load_eval()
    print(f"  {len(rows)} eval rows.", file=sys.stderr)

    print("Building NLPManager index ...", file=sys.stderr)
    mgr = NLPManager()
    mgr.load_corpus(docs)
    print(f"  vocab={len(mgr.df)} N={mgr.N}", file=sys.stderr)

    if args.all:
        eval_rows = rows
        label = "FULL"
    else:
        split = make_or_load_split(len(rows))
        if args.heldout:
            eval_rows = [rows[i] for i in split["heldout"]]
            label = "HELDOUT"
        else:
            eval_rows = [rows[i] for i in split["dev"]]
            label = "DEV"

    print(f"Scoring {label} ({len(eval_rows)} rows) ...", file=sys.stderr)
    summary = score(mgr, eval_rows)

    print(f"\n=== {label} retrieval recall ===")
    print(f"  n              = {summary['n']}")
    print(f"  top1_recall    = {summary['top1_recall']}")
    print(f"  top3_recall    = {summary['top3_recall']}")
    print(f"  mean_rank_hit  = {summary['mean_rank_when_found']}")
    print(f"  estimated end-to-end score floor (0.4 * top3_recall) = "
          f"{round(0.4 * summary['top3_recall'], 4)}")

    if args.failures and summary["misses"]:
        print(f"\n=== {label} miss samples (showing {min(args.max_misses, len(summary['misses']))} of {len(summary['misses'])}) ===")
        for m in summary["misses"][:args.max_misses]:
            print(json.dumps(m, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
