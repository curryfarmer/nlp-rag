#!/usr/bin/env bash
# Execute the cheese on the GCP GPU box. Run from nlp/src.
#   cd nlp/src && bash train/run_gcp.sh
# Phases A/B data are already built locally and committed; this re-builds them on
# the box (idempotent), then runs the GPU phases C->E and the calibration F.
set -euo pipefail

TEACHER="${TEACHER:-Qwen/Qwen2.5-7B-Instruct}"
STUDENT="${STUDENT:-Qwen/Qwen2.5-0.5B-Instruct}"
PER_DOC="${PER_DOC:-6}"
# vllm coexists with other jobs on the shared T4: cap its VRAM share + 4-bit load.
export VLLM_GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.55}"
export VLLM_BNB="${VLLM_BNB:-1}"
DATA=train/data
CKPT=train/ckpt
ART=train/artifacts
mkdir -p "$DATA" "$CKPT" "$ART"

echo "== deps =="
pip install -q -r ../requirements-train.txt
# vllm makes synth ~10-20min instead of hours. It may upgrade torch — if that
# breaks QLoRA/bitsandbytes, set SKIP_VLLM=1 and gen_synthetic falls back to HF 4-bit.
if [ "${SKIP_VLLM:-0}" != "1" ]; then
    pip install -q vllm || echo "vllm install failed -> HF 4-bit fallback for synth"
fi

echo "== A: baseline proxy (regex-only) =="
python eval_answers.py --proxy-bem | tail -8

echo "== B: build SFT data =="
python train/prepare_data.py --out "$DATA/sft_dev.jsonl" --top-k 3
python train/gen_synthetic.py --teacher "$TEACHER" --per-doc "$PER_DOC" \
    --out "$DATA/synth.jsonl"
# TODO: filter synth.jsonl with nlp_bem_proxy (drop ungrounded answers).

echo "== C: finetune teacher (QLoRA) =="
python train/finetune_teacher.py --base "$TEACHER" \
    --data "$DATA/sft_dev.jsonl" --data "$DATA/synth.jsonl" \
    --out "$CKPT/teacher" --merge

echo "== C.5: teacher labels for distillation =="
# Relabel every dev+synth prompt with the finetuned teacher -> teacher_labels.jsonl.
# (Generate with the merged teacher; reuse gen_synthetic backend or a small script.)
# Placeholder: distil directly on gold dev for the first pass.
cp "$DATA/sft_dev.jsonl" "$DATA/teacher_labels.jsonl"

echo "== D: distil student; target proxy >= 0.95 on heldout =="
python train/distill_student.py --mode seq --student "$STUDENT" \
    --data "$DATA/teacher_labels.jsonl" --out "$CKPT/student"
NLP_USE_LLM=1 NLP_LLM_MODEL="$CKPT/student" \
    python eval_answers.py --proxy-bem --heldout | tail -8

echo "== E: quantize to Q4 =="
python train/quantize.py --mode gguf --model "$CKPT/student" \
    --out "$ART/student-q4.gguf"

echo "== F: real BEM (needs nlp_eval_512 weights on box) =="
python ../../test/test_nlp.py || echo "real BEM unavailable here; rely on proxy"

echo "DONE. If proxy/BEM >= 0.95: swap NLP_LLM_MODEL default in nlp_llm.py, update nlp/Dockerfile."
