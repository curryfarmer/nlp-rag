"""Phase B (augment) — generate synthetic QA pairs over the corpus with a teacher.

The 706 dev rows are a thin SFT signal, especially for L2 (multi-fact /
arithmetic / comparison). We expand coverage by having a large teacher model read
each of the 296 docs and emit additional (question, answer) pairs in the corpus's
answer style — short literal spans for L1, complete composed answers for L2
(exploiting the un-enforced 64-token cap). These pairs feed BOTH the teacher
finetune and the student distillation set.

GPU job (Phase C box). Heavy deps (vllm/transformers) are imported lazily so this
file imports cleanly anywhere.

  python train/gen_synthetic.py \
      --teacher Qwen/Qwen2.5-7B-Instruct \
      --per-doc 6 --out data/synth.jsonl

Output: JSONL {"question","answer","source_docs":[doc_id],"difficulty","synthetic":true}
Pass through a judge/filter (e.g. nlp_bem_proxy + length sanity) before training.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent
sys.path.insert(0, str(SRC))

DOCS_DIR = SRC / "documents"

_GEN_PROMPT = """You are creating question-answer pairs for a closed-book reading test over the document below. Write {n} diverse questions a grader could verify from the text alone.

Rules:
- Mix difficulty: some single-fact (L1), some requiring combining 2+ facts or arithmetic/comparison (L2).
- Answers must be grounded VERBATIM in the document (copy numbers/codes/names exactly). For L2, give a complete composed answer, not a single token.
- Keep answers as short as fully correct allows.

Return a JSON array: [{{"question": "...", "answer": "...", "difficulty": "L1"|"L2"}}].

Document:
{doc}
"""


def _load_docs() -> list[dict[str, str]]:
    return [{"id": p.stem, "document": p.read_text(encoding="utf-8")}
            for p in sorted(DOCS_DIR.glob("DOC-*.txt"))]


def _load_teacher(model_id: str):
    """Lazy load. Prefer vLLM for throughput; fall back to transformers.

    SKIP_VLLM=1 skips the vLLM attempt entirely (use it on Blackwell where
    vLLM's flashinfer build fails on missing CUDA dev headers). VLLM_BNB=0 loads
    the HF fallback in fp16/bf16 instead of 4-bit — bitsandbytes has no sm_120
    kernels, and a 32GB card holds a 7B in fp16 (~14GB) with room to spare.
    """
    import torch
    if os.getenv("SKIP_VLLM", "0") != "1":
        try:
            from vllm import LLM  # noqa: F401
            util = float(os.getenv("VLLM_GPU_MEM_UTIL", "0.55"))
            maxlen = int(os.getenv("VLLM_MAX_LEN", "8192"))
            kw = dict(gpu_memory_utilization=util, max_model_len=maxlen, dtype="auto")
            if os.getenv("VLLM_BNB", "1") == "1":
                kw.update(quantization="bitsandbytes", load_format="bitsandbytes")
            return ("vllm", LLM(model=model_id, **kw))
        except Exception as e:  # noqa: BLE001
            print(f"[synth] vllm unavailable ({type(e).__name__}: {e}); HF fallback",
                  file=sys.stderr)
    else:
        print("[synth] SKIP_VLLM=1 -> HF backend", file=sys.stderr)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    if os.getenv("VLLM_BNB", "1") == "1":
        # T4 (16GB): nf4 4-bit brings a 7B to ~5-6GB. Needs bitsandbytes kernels.
        from transformers import BitsAndBytesConfig
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.float16)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=bnb, device_map="auto")
    else:
        # Big GPU: fp16/bf16, no bitsandbytes (no Blackwell kernels).
        bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=(torch.bfloat16 if bf16_ok else torch.float16),
            device_map="auto")
    return ("hf", (tok, model))


def _generate(backend, handle, prompt: str) -> str:
    kind = backend
    if kind == "vllm":
        from vllm import SamplingParams
        out = handle.generate([prompt], SamplingParams(temperature=0.7, max_tokens=1024))
        return out[0].outputs[0].text
    tok, model = handle
    import torch
    msgs = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = tok([text], return_tensors="pt").to(model.device)
    with torch.no_grad():
        gen = model.generate(**inp, max_new_tokens=1024, do_sample=True, temperature=0.7,
                             pad_token_id=tok.eos_token_id)
    return tok.decode(gen[0][inp.input_ids.shape[1]:], skip_special_tokens=True)


def _parse_pairs(raw: str) -> list[dict]:
    start, end = raw.find("["), raw.rfind("]")
    if start == -1 or end == -1:
        return []
    try:
        return json.loads(raw[start:end + 1])
    except json.JSONDecodeError:
        return []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True, help="HF id of teacher model")
    ap.add_argument("--per-doc", type=int, default=6)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit-docs", type=int, default=0, help="0 = all docs")
    args = ap.parse_args()

    docs = _load_docs()
    if args.limit_docs:
        docs = docs[: args.limit_docs]

    backend, handle = _load_teacher(args.teacher)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with out_path.open("w") as f:
        for d in docs:
            prompt = _GEN_PROMPT.format(n=args.per_doc, doc=d["document"][:8000])
            raw = _generate(backend, handle, prompt)
            for pair in _parse_pairs(raw):
                q, a = pair.get("question"), pair.get("answer")
                if not q or not a:
                    continue
                f.write(json.dumps({
                    "question": q, "answer": a, "source_docs": [d["id"]],
                    "difficulty": pair.get("difficulty", "L1"), "synthetic": True,
                }) + "\n")
                n += 1
    print(f"wrote {n} synthetic pairs from {len(docs)} docs -> {out_path}")
    print("NEXT: filter with nlp_bem_proxy (drop pairs where answer not grounded), "
          "then feed to prepare_data / distill_student.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
