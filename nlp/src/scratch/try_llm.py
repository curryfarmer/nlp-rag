"""Thesis validation: small instruct LLM reader vs regex on L1+L2.

Standalone. Retrieves docs via the existing regex retriever, feeds top-k to a
small instruct LLM, proxy-scores against gold, dumps every (q,gold,pred) for
manual read.

Levers (combine freely):
  --top-k K     feed top-K retrieved docs (1 -> 0.76 ceiling, 3 -> 0.95)
  --fewshot     prepend style exemplars (terse, hedged, arithmetic-shown)
  --cot         reason first, emit "Answer: X", parse final line
  --model ID    hf model id

Blind-scoring caveat: proxy UNDERcounts semantic answers. Manual read of the
dump is the real signal.

Run:
  cd nlp/src && python3 scratch/try_llm.py --top-k 3 --fewshot --cot
  python3 scratch/try_llm.py --top-k 3 --fewshot --model Qwen/Qwen2.5-7B-Instruct  # GPU box
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent  # nlp/src
sys.path.insert(0, str(HERE))

from nlp_manager import NLPManager  # noqa: E402
from eval_answers import (  # noqa: E402
    load_corpus, load_eval, get_split, f1, gold_substring,
    likely_pass, loose_pass,
)

SYSTEM = (
    "You answer questions using the provided document(s). Reply with ONLY the "
    "answer in as few words as possible. Copy wording and numbers verbatim from "
    "the document. Do arithmetic if the question requires it. No explanation, no "
    "preamble, no restating the question."
)

SYSTEM_COT = (
    "You answer questions using the provided document(s). First think briefly "
    "in one or two sentences, doing any arithmetic the question needs. Then on a "
    "new line output 'Answer:' followed by the answer in as few words as "
    "possible, copying wording and numbers verbatim from the document."
)

# Synthetic exemplars (NOT from the eval set) mirroring gold answer style:
# terse, verbatim, hedged numbers, arithmetic shown.
FEWSHOT = [
    ("Document:\n**Financial Penalty:** 2.3 million Credits and license suspension\n\n"
     "Question: What penalty was imposed on the firm?",
     "2.3 million Credits and license suspension"),
    ("Document:\nThe Vega reactor was commissioned in 71 PCE and decommissioned in 89 PCE.\n\n"
     "Question: How many years did the reactor operate?",
     "approximately 18 years"),
    ("Document:\nDivision A reported 400 million Credits; Division B reported 520 million Credits.\n\n"
     "Question: Which division reported higher revenue?",
     "Division B"),
]

FEWSHOT_COT = [
    (q, f"The document states it directly.\nAnswer: {a}") if "year" not in q.lower()
    else (q, f"Commissioned 71 PCE, decommissioned 89 PCE; 89 - 71 = 18.\nAnswer: {a}")
    for q, a in FEWSHOT
]

_RE_ANSWER_LINE = re.compile(r"answer\s*:\s*(.+)", re.IGNORECASE | re.DOTALL)


def parse_cot(text: str) -> str:
    m = _RE_ANSWER_LINE.search(text)
    if m:
        return m.group(1).strip().splitlines()[0].strip()
    return text.strip().splitlines()[-1].strip() if text.strip() else ""


def build_messages(question: str, docs: list[str], fewshot: bool, cot: bool,
                   max_doc_chars: int) -> list[dict]:
    sys_msg = SYSTEM_COT if cot else SYSTEM
    msgs = [{"role": "system", "content": sys_msg}]
    if fewshot:
        shots = FEWSHOT_COT if cot else FEWSHOT
        for ex_user, ex_asst in shots:
            msgs.append({"role": "user", "content": ex_user})
            msgs.append({"role": "assistant", "content": ex_asst})
    joined = "\n\n---\n\n".join(d[:max_doc_chars] for d in docs)
    msgs.append({"role": "user",
                 "content": f"Document:\n{joined}\n\nQuestion: {question}"})
    return msgs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--n-l1", type=int, default=40)
    ap.add_argument("--n-l2", type=int, default=40)
    ap.add_argument("--top-k", type=int, default=1)
    ap.add_argument("--fewshot", action="store_true")
    ap.add_argument("--cot", action="store_true")
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--max-doc-chars", type=int, default=10000)
    ap.add_argument("--dump", default="scratch/llm_sample_out.json")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.float16 if device != "cpu" else torch.float32
    print(f"[load] {args.model} on {device} ({dtype}) "
          f"top_k={args.top_k} fewshot={args.fewshot} cot={args.cot}", file=sys.stderr)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    model.to(device).eval()

    docs = load_corpus()
    rows = load_eval()
    mgr = NLPManager()
    mgr.load_corpus(docs)
    doc_lookup = {d["id"]: d["document"] for d in docs}
    split = get_split()
    dev = [rows[i] for i in split["dev"]]

    l1 = [r for r in dev if r["difficulty"] == "L1"][: args.n_l1]
    l2 = [r for r in dev if r["difficulty"] == "L2"][: args.n_l2]
    sample = l1 + l2
    print(f"[sample] {len(l1)} L1 + {len(l2)} L2 = {len(sample)}", file=sys.stderr)

    out = []
    t0 = time.time()
    for i, r in enumerate(sample):
        q = r["question"]
        top = mgr._retrieve(q, k=3)
        retrieval_hit = bool(set(top) & set(r["source_docs"]))
        fed = [doc_lookup[d] for d in top[: args.top_k] if d in doc_lookup]

        msgs = build_messages(q, fed, args.fewshot, args.cot, args.max_doc_chars)
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tok([text], return_tensors="pt").to(device)
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=args.max_new,
                                 do_sample=False, pad_token_id=tok.eos_token_id)
        new = gen[0][inputs.input_ids.shape[1]:]
        raw = tok.decode(new, skip_special_tokens=True).strip()
        pred = parse_cot(raw) if args.cot else raw

        out.append({
            "key": r["key"], "difficulty": r["difficulty"],
            "question": q, "gold": r["answer"], "pred": pred, "raw": raw,
            "retrieval_hit": retrieval_hit,
            "f1": round(f1(pred, r["answer"]), 3),
            "gold_sub": gold_substring(pred, r["answer"]),
            "likely": likely_pass(pred, r["answer"]),
            "loose": loose_pass(pred, r["answer"]),
        })
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(sample)}  ({(time.time()-t0)/(i+1):.1f}s/q)", file=sys.stderr)

    Path(args.dump).write_text(json.dumps(out, indent=2))

    def est(recs, key):
        s = sum(1.0 if r[key] else 0.4 for r in recs if r["retrieval_hit"])
        return s / len(recs) if recs else 0.0

    for label, recs in [("ALL", out),
                        ("L1", [r for r in out if r["difficulty"] == "L1"]),
                        ("L2", [r for r in out if r["difficulty"] == "L2"])]:
        if not recs:
            continue
        n = len(recs)
        print(f"\n=== {label} (n={n}) ===")
        print(f"  retrieval@3 = {sum(r['retrieval_hit'] for r in recs)/n:.3f}")
        print(f"  F1 mean     = {sum(r['f1'] for r in recs)/n:.3f}")
        print(f"  gold_sub    = {sum(r['gold_sub'] for r in recs)/n:.3f}")
        print(f"  likely      = {sum(r['likely'] for r in recs)/n:.3f}  -> est {est(recs,'likely'):.3f}")
        print(f"  loose       = {sum(r['loose'] for r in recs)/n:.3f}  -> est {est(recs,'loose'):.3f}")

    print(f"\n[dump] {args.dump}  ({time.time()-t0:.0f}s total)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
