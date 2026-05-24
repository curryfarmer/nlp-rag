#!/usr/bin/env bash
# One-shot env setup for an RTX 5090 (Blackwell, sm_120) box.
# Run from the repo root after `git clone`:  bash setup_5090.sh
set -euo pipefail

PY="${PY:-python3}"

echo "== 1/5 venv =="
$PY -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install -U pip wheel setuptools

echo "== 2/5 PyTorch for Blackwell (CUDA 12.8) =="
# sm_120 needs a cu128+ build. A default/CPU wheel errors 'no kernel image for sm_120'.
pip install torch --index-url https://download.pytorch.org/whl/cu128

echo "== 3/5 training deps =="
pip install -r nlp/requirements-train.txt

echo "== 4/5 vllm (optional, speeds synth ~10x) =="
# vllm may try to repin torch; on Blackwell a recent vllm uses cu128 torch and is fine.
# If it downgrades torch and breaks CUDA, re-run step 2, or train with SKIP_VLLM=1.
pip install vllm || echo "  vllm install failed -> run with SKIP_VLLM=1 (HF fallback)"

echo "== 5/5 verify GPU =="
python - <<'PY'
import torch
print("torch:", torch.__version__)
ok = torch.cuda.is_available()
print("cuda available:", ok)
if ok:
    cc = torch.cuda.get_device_capability(0)
    print("device:", torch.cuda.get_device_name(0))
    print("compute capability:", f"{cc[0]}.{cc[1]}")
    print("bf16 supported:", torch.cuda.is_bf16_supported())
    if cc[0] < 9:
        print("WARNING: expected Blackwell (sm_120); got", cc)
else:
    raise SystemExit("CUDA not available — fix torch install (cu128) before training.")
PY

echo
echo "DONE. Now run the pipeline (5090 profile is the default):"
echo "  source .venv/bin/activate"
echo "  cd nlp/src && bash train/run_gcp.sh 2>&1 | tee run.log"
echo
echo "Fast first number (skip the 7B teacher): SKIP_TEACHER=1 bash train/run_gcp.sh"
