"""Phase B — build the SFT dataset for teacher finetune + student distillation.

For each eval-set question we run the SOLVED regex retrieval (NLPManager) to get
the top-k source docs, then emit a chat-format training example whose prompt is
identical to what `nlp_llm.py` feeds the reader at inference (same system prompt,
same few-shot block, same "Document:\n...\n\nQuestion: ..." layout) and whose
target is the gold answer. Train/eval split comes from `eval_split.json` — we
ONLY emit the dev rows; the 177 held-out rows stay untouched for measurement.

Output: JSONL, one obj per line: {"messages": [{role, content}, ...]} where the
final assistant turn is the gold answer. This is the format TRL's SFTTrainer and
Unsloth consume directly.

Runnable locally (stdlib + repo modules only, no torch/transformers).

  python train/prepare_data.py --out data/sft_dev.jsonl --top-k 3
  python train/prepare_data.py --out data/sft_dev_fewshot.jsonl --fewshot --top-k 3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent
sys.path.insert(0, str(SRC))

from nlp_manager import NLPManager  # noqa: E402
from nlp_llm import _SYSTEM, _FEWSHOT, _MAX_DOC_CHARS  # noqa: E402  (reuse exact prompt)

DOCS_DIR = SRC / "documents"
EVAL_PATH = SRC / "nlp.jsonl"
SPLIT_PATH = SRC / "eval_split.json"


def _load_corpus() -> list[dict[str, str]]:
    return [{"id": p.stem, "document": p.read_text(encoding="utf-8")}
            for p in sorted(DOCS_DIR.glob("DOC-*.txt"))]


def _load_eval() -> list[dict]:
    return [json.loads(l) for l in EVAL_PATH.read_text().splitlines() if l.strip()]


def _build_messages(question: str, docs: list[str], fewshot: bool) -> list[dict]:
    """Mirror nlp_llm.Reader._messages (non-CoT path) so train == inference."""
    msgs = [{"role": "system", "content": _SYSTEM}]
    if fewshot:
        for u, a in _FEWSHOT:
            msgs.append({"role": "user", "content": u})
            msgs.append({"role": "assistant", "content": a})
    joined = "\n\n---\n\n".join(d[:_MAX_DOC_CHARS] for d in docs if d)
    msgs.append({"role": "user", "content": f"Document:\n{joined}\n\nQuestion: {question}"})
    return msgs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output JSONL path")
    ap.add_argument("--top-k", type=int, default=3, help="docs retrieved per question")
    ap.add_argument("--fewshot", action="store_true", help="embed style exemplars in prompt")
    ap.add_argument("--split", choices=["dev", "heldout", "all"], default="dev")
    ap.add_argument("--synth-in", help="convert raw synth pairs "
                    "({question,answer,source_docs}) to chat format instead of the eval split")
    args = ap.parse_args()

    if args.synth_in:
        # gen_synthetic emits raw QA pairs (same fields as eval rows minus 'key');
        # process them through the identical retrieval + prompt builder.
        sel = [json.loads(l) for l in Path(args.synth_in).read_text().splitlines()
               if l.strip()]
    else:
        rows = _load_eval()
        split = json.loads(SPLIT_PATH.read_text())
        if args.split == "all":
            sel = rows
        else:
            sel = [rows[i] for i in split[args.split]]

    mgr = NLPManager()
    mgr.load_corpus(_load_corpus())

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w") as f:
        for r in sel:
            # Train-time context MUST contain the answer, else the model learns to
            # hallucinate. Retrieval@3 misses the gold doc ~5% (eval) / more (synth),
            # so force the known source_docs in, then top up with retrieved docs.
            retrieved = mgr._retrieve(r["question"], k=args.top_k)
            ordered = list(dict.fromkeys((r.get("source_docs") or []) + retrieved))
            doc_ids = ordered[:max(args.top_k, len(r.get("source_docs") or []))]
            docs = [mgr.doc_text[d] for d in doc_ids if d in mgr.doc_text]
            msgs = _build_messages(r["question"], docs, args.fewshot)
            msgs.append({"role": "assistant", "content": r["answer"]})
            f.write(json.dumps({"messages": msgs, "key": r.get("key"),
                                "difficulty": r.get("difficulty")}) + "\n")
            n += 1
    print(f"wrote {n} examples -> {out_path}  (split={args.split}, top_k={args.top_k}, "
          f"fewshot={args.fewshot})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
