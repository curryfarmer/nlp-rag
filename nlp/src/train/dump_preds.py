"""Diagnostic: dump raw student answers for a few train + held-out questions.

Run from nlp/src:
  NLP_USE_LLM=1 NLP_LLM_MODEL=train/ckpt/student python3 train/dump_preds.py
"""
import glob
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent
sys.path.insert(0, str(SRC))

from nlp_manager import NLPManager  # noqa: E402
from nlp_llm import get_reader      # noqa: E402

rows = [json.loads(l) for l in (SRC / "nlp.jsonl").read_text().splitlines() if l.strip()]
split = json.loads((SRC / "eval_split.json").read_text())
docs = [{"id": os.path.basename(p)[:-4], "document": open(p, encoding="utf-8").read()}
        for p in glob.glob(str(SRC / "documents" / "DOC-*.txt"))]
mgr = NLPManager()
mgr.load_corpus(docs)
r = get_reader()
print("reader:", r and r.model_id)
if r is None:
    print("READER IS NONE — NLP_USE_LLM not set or load failed.")
    raise SystemExit(1)
for tag, idxs in [("TRAIN", split["dev"][:3]), ("HELDOUT", split["heldout"][:3])]:
    for i in idxs:
        row = rows[i]
        q = row["question"]
        d = mgr._retrieve(q, 3)
        raw = r.answer(q, [mgr.doc_text[x] for x in d[:3]])
        print("=" * 60, tag)
        print("Q   :", q[:90])
        print("GOLD:", repr(row["answer"]))
        print("PRED:", repr(raw))
