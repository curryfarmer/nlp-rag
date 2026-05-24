"""Phase 4 small-LLM reader.

Wraps a small instruct LLM (default Qwen2.5-1.5B-Instruct, Apache-2.0) as a
grounded reading-comprehension answerer: given (question, top-k source docs) it
returns a terse answer string. Unlike the Phase 3 reranker, this READS the docs
and can compose multi-fact / arithmetic answers — the L2 segment regex can't
reach. Feeding top-k (not just top-1) lifts the answer-in-context ceiling from
~0.76 to ~0.95 (regex retrieval@1 vs @3).

Lazy-loaded so `import nlp_llm` works on boxes without transformers/torch.
Env-flag gated so the regex-only path stays default until opted in.

API:
    reader = get_reader()                 # cached singleton or None
    ans = reader.answer(question, docs)   # docs: list[str], returns str

Env:
    NLP_USE_LLM=1          enable
    NLP_LLM_MODEL=<hf-id>  override model
    NLP_LLM_FEWSHOT=1      prepend style exemplars
    NLP_LLM_COT=1          reason-then-extract (helps arithmetic/aggregation)
    NLP_LLM_TOP_K=3        docs fed (set in qa_batch)
"""
from __future__ import annotations

import logging
import os
import re
from typing import Optional

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

_SYSTEM = (
    "You answer questions using the provided document(s). Reply with ONLY the "
    "answer in as few words as possible. Copy wording and numbers verbatim from "
    "the document. Do arithmetic if the question requires it. No explanation, no "
    "preamble, no restating the question."
)

_SYSTEM_COT = (
    "You answer questions using the provided document(s). First think briefly "
    "in one or two sentences, doing any arithmetic the question needs. Then on a "
    "new line output 'Answer:' followed by the answer in as few words as "
    "possible, copying wording and numbers verbatim from the document."
)

# Synthetic exemplars (NOT from the eval set) mirroring gold answer style.
_FEWSHOT = [
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

_FEWSHOT_COT = [
    ("Document:\n**Financial Penalty:** 2.3 million Credits and license suspension\n\n"
     "Question: What penalty was imposed on the firm?",
     "The document states it directly.\nAnswer: 2.3 million Credits and license suspension"),
    ("Document:\nThe Vega reactor was commissioned in 71 PCE and decommissioned in 89 PCE.\n\n"
     "Question: How many years did the reactor operate?",
     "Commissioned 71 PCE, decommissioned 89 PCE; 89 - 71 = 18.\nAnswer: approximately 18 years"),
    ("Document:\nDivision A reported 400 million Credits; Division B reported 520 million Credits.\n\n"
     "Question: Which division reported higher revenue?",
     "520 > 400, so Division B.\nAnswer: Division B"),
]

_RE_ANSWER_LINE = re.compile(r"answer\s*:\s*(.+)", re.IGNORECASE | re.DOTALL)

# Per-doc context cap. MUST keep prompt+answer under the trainer's max_length:
# at 10000 every SFT example was ~5000+ tokens > the 2048 cap, so SFTTrainer
# right-truncated the trailing gold answer off EVERY example -> the student
# trained pure doc-continuation (token-acc 0.998, but F1 0.024 at inference).
# 4000 chars * 3 docs + prompt ~= 3200 tokens, fits a 4096 max_length with the
# answer intact, and still keeps ~82% of verbatim answer spans in-context.
# prepare_data.py imports this same constant, so train==inference stays in sync.
_MAX_DOC_CHARS = int(os.getenv("NLP_LLM_MAX_DOC_CHARS", "4000"))
_MAX_NEW_TOKENS = int(os.getenv("NLP_LLM_MAX_NEW", "128"))


def _parse_cot(text: str) -> str:
    m = _RE_ANSWER_LINE.search(text)
    if m:
        return m.group(1).strip().splitlines()[0].strip()
    return text.strip().splitlines()[-1].strip() if text.strip() else ""


class Reader:
    def __init__(self, model_id: str, fewshot: bool, cot: bool):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.model_id = model_id
        self.fewshot = fewshot
        self.cot = cot
        self.device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        dtype = torch.float16 if self.device != "cpu" else torch.float32
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
        self.model.to(self.device).eval()

    def _messages(self, question: str, docs: list[str]) -> list[dict]:
        msgs = [{"role": "system", "content": _SYSTEM_COT if self.cot else _SYSTEM}]
        if self.fewshot:
            for u, a in (_FEWSHOT_COT if self.cot else _FEWSHOT):
                msgs.append({"role": "user", "content": u})
                msgs.append({"role": "assistant", "content": a})
        joined = "\n\n---\n\n".join(d[:_MAX_DOC_CHARS] for d in docs if d)
        msgs.append({"role": "user",
                     "content": f"Document:\n{joined}\n\nQuestion: {question}"})
        return msgs

    def answer(self, question: str, docs: list[str]) -> str:
        if not docs:
            return ""
        text = self.tok.apply_chat_template(
            self._messages(question, docs), tokenize=False, add_generation_prompt=True)
        inputs = self.tok([text], return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            gen = self.model.generate(
                **inputs, max_new_tokens=_MAX_NEW_TOKENS, do_sample=False,
                pad_token_id=self.tok.eos_token_id)
        new = gen[0][inputs.input_ids.shape[1]:]
        raw = self.tok.decode(new, skip_special_tokens=True).strip()
        return _parse_cot(raw) if self.cot else raw


_SINGLETON: Optional[Reader] = None
_DISABLED: bool = False


def get_reader() -> Optional[Reader]:
    """Cached singleton. Returns None when disabled or load fails."""
    global _SINGLETON, _DISABLED
    if _DISABLED:
        return None
    if _SINGLETON is not None:
        return _SINGLETON
    if os.getenv("NLP_USE_LLM", "0") == "0":
        _DISABLED = True
        return None
    model_id = os.getenv("NLP_LLM_MODEL", _DEFAULT_MODEL)
    fewshot = os.getenv("NLP_LLM_FEWSHOT", "0") == "1"
    cot = os.getenv("NLP_LLM_COT", "0") == "1"
    try:
        _SINGLETON = Reader(model_id, fewshot, cot)
        logger.info(f"LLM reader loaded: {model_id} on {_SINGLETON.device} "
                    f"(fewshot={fewshot}, cot={cot})")
        return _SINGLETON
    except Exception as e:
        logger.warning(f"LLM reader load failed ({type(e).__name__}: {e}); regex-only fallback.")
        _DISABLED = True
        return None
