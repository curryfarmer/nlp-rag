"""Phase E — quantize the distilled student to Q4 for the Docker submission.

Inference speed is scored, and the student must fit the container. Q4 keeps
~92-95% on narrow extractive QA. Two paths:

  gguf  (default) llama.cpp GGUF Q4_K_M — best for CPU / llama-cpp-python serving,
        smallest artifact. Converts HF -> f16 GGUF -> quantizes to Q4_K_M.

  awq   AutoAWQ 4-bit — best if serving on GPU via vLLM/transformers; keeps the
        HF runtime in nlp_llm.py unchanged.

After quantizing, re-run eval_answers.py --proxy-bem against the quantized model
to confirm retention, then point nlp_llm.py NLP_LLM_MODEL at it.

  python train/quantize.py --mode gguf --model ckpt/student-0.5b --out artifacts/student-q4.gguf
  python train/quantize.py --mode awq  --model ckpt/student-0.5b --out artifacts/student-awq
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _gguf(model: str, out: str, llama_cpp: str):
    """HF -> GGUF f16 -> Q4_K_M via llama.cpp scripts."""
    repo = Path(llama_cpp)
    convert = repo / "convert_hf_to_gguf.py"
    quant_bin = repo / "build" / "bin" / "llama-quantize"
    if not convert.exists():
        sys.exit(f"convert script not found at {convert}; clone ggerganov/llama.cpp "
                 "and pass --llama-cpp <repo>")
    f16 = str(Path(out).with_suffix(".f16.gguf"))
    subprocess.run([sys.executable, str(convert), model, "--outfile", f16,
                    "--outtype", "f16"], check=True)
    subprocess.run([str(quant_bin), f16, out, "Q4_K_M"], check=True)
    print(f"GGUF Q4_K_M -> {out}")
    print("Serve via llama-cpp-python; update nlp_llm.py to a GGUF backend, or run "
          "as a llama.cpp server and point NLP_SERVER_URL at it.")


def _awq(model: str, out: str):
    from awq import AutoAWQForCausalLM
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model)
    m = AutoAWQForCausalLM.from_pretrained(model)
    m.quantize(tok, quant_config={"zero_point": True, "q_group_size": 128,
                                  "w_bit": 4, "version": "GEMM"})
    m.save_quantized(out)
    tok.save_pretrained(out)
    print(f"AWQ 4-bit -> {out}  (set NLP_LLM_MODEL={out})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["gguf", "awq"], default="gguf")
    ap.add_argument("--model", required=True, help="HF dir of the distilled student")
    ap.add_argument("--out", required=True)
    ap.add_argument("--llama-cpp", default="llama.cpp", help="path to llama.cpp repo (gguf mode)")
    args = ap.parse_args()

    if args.mode == "gguf":
        _gguf(args.model, args.out, args.llama_cpp)
    else:
        _awq(args.model, args.out)

    print("NEXT: eval_answers.py --proxy-bem on the quantized model to confirm "
          "retention, then update nlp/Dockerfile + nlp_llm.py default.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
