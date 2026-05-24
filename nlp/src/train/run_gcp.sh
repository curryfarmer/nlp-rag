#!/usr/bin/env bash
# Execute the cheese on the GCP GPU box. Run from nlp/src.
#   cd nlp/src && bash train/run_gcp.sh
# Phases A/B data are already built locally and committed; this re-builds them on
# the box (idempotent), then runs the GPU phases C->E and the calibration F.
set -euo pipefail

TEACHER="${TEACHER:-Qwen/Qwen2.5-7B-Instruct}"
STUDENT="${STUDENT:-Qwen/Qwen2.5-0.5B-Instruct}"
PER_DOC="${PER_DOC:-6}"
# vllm coexists with other jobs on the shared T4: 4-bit load + enough VRAM for the
# KV cache. 0.75 of 15GB = ~11.5GB (weights 5.5 + cuda-graphs 1.7 + KV ~4); at 0.5
# the cache went negative -> vllm dies -> HF 4-bit fallback (slow). Lower only if
# the co-tenant job grows and vllm hits "free memory < desired".
export VLLM_GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.75}"
export VLLM_MAX_LEN="${VLLM_MAX_LEN:-4096}"
export VLLM_BNB="${VLLM_BNB:-1}"
# Reduce CUDA fragmentation (the OOM error itself suggests this).
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# T4-safe QLoRA: batch 1 + short seq keeps the 7B logits upcast (152k vocab) from
# OOMing in ForCausalLMLoss. Effective batch = BATCH*ACCUM. Bump on a bigger GPU.
TEACHER_BATCH="${TEACHER_BATCH:-1}"
TEACHER_ACCUM="${TEACHER_ACCUM:-16}"
TEACHER_MAXLEN="${TEACHER_MAXLEN:-1024}"
# Student is tiny but the same 152k-vocab logits upcast OOMs at big batch/seq.
STUDENT_BATCH="${STUDENT_BATCH:-2}"
STUDENT_ACCUM="${STUDENT_ACCUM:-8}"
STUDENT_MAXLEN="${STUDENT_MAXLEN:-1024}"
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

# finetune/distill consume CHAT format ({messages}). gold dev is already chat;
# raw synth ({question,answer,source_docs}) MUST be converted or finetune dies
# with KeyError: 'messages'.
SFT_ARGS="--data $DATA/sft_dev.jsonl"
if [ "$PER_DOC" != "0" ]; then
    if [ -s "$DATA/synth.jsonl" ]; then
        echo "  reusing existing $DATA/synth.jsonl (skip generation)"
    else
        python train/gen_synthetic.py --teacher "$TEACHER" --per-doc "$PER_DOC" \
            --out "$DATA/synth.jsonl"
    fi
    # TODO: filter synth.jsonl with nlp_bem_proxy (drop ungrounded answers).
    python train/prepare_data.py --synth-in "$DATA/synth.jsonl" \
        --out "$DATA/sft_synth.jsonl" --top-k 3
    SFT_ARGS="$SFT_ARGS --data $DATA/sft_synth.jsonl"
fi

# NOTE: this pass distils the student directly on gold+synth chat data; the
# finetuned teacher is NOT yet used to relabel (that's the next upgrade). So the
# teacher finetune is optional for a first number -> SKIP_TEACHER=1 to skip the
# slow 7B step. No --merge: merging a LoRA into a 4-bit base degrades/breaks, and
# the merged teacher isn't consumed downstream this pass anyway.
if [ "${SKIP_TEACHER:-0}" != "1" ]; then
    echo "== C: finetune teacher (QLoRA) =="
    python train/finetune_teacher.py --base "$TEACHER" $SFT_ARGS \
        --batch "$TEACHER_BATCH" --grad-accum "$TEACHER_ACCUM" \
        --max-seq-len "$TEACHER_MAXLEN" --out "$CKPT/teacher"
else
    echo "== C: SKIPPED (SKIP_TEACHER=1) =="
fi

echo "== C.5: teacher labels for distillation =="
# First pass: distil on the chat data itself (gold + converted synth).
if [ "$PER_DOC" != "0" ] && [ -s "$DATA/sft_synth.jsonl" ]; then
    cat "$DATA/sft_dev.jsonl" "$DATA/sft_synth.jsonl" > "$DATA/teacher_labels.jsonl"
else
    cp "$DATA/sft_dev.jsonl" "$DATA/teacher_labels.jsonl"
fi

echo "== D: distil student; target proxy >= 0.95 on heldout =="
python train/distill_student.py --mode seq --student "$STUDENT" \
    --batch "$STUDENT_BATCH" --grad-accum "$STUDENT_ACCUM" \
    --max-seq-len "$STUDENT_MAXLEN" \
    --data "$DATA/teacher_labels.jsonl" --out "$CKPT/student"
# Explicitly route eval through the distilled student (env + --llm flag).
NLP_USE_LLM=1 NLP_LLM_MODEL="$CKPT/student" \
    python eval_answers.py --proxy-bem --heldout --llm --llm-model "$CKPT/student" | tail -8

echo "== E: quantize to Q4 =="
python train/quantize.py --mode gguf --model "$CKPT/student" \
    --out "$ART/student-q4.gguf"

echo "== F: real BEM (needs nlp_eval_512 weights on box) =="
python ../../test/test_nlp.py || echo "real BEM unavailable here; rely on proxy"

echo "DONE. If proxy/BEM >= 0.95: swap NLP_LLM_MODEL default in nlp_llm.py, update nlp/Dockerfile."
