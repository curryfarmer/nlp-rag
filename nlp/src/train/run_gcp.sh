#!/usr/bin/env bash
# Execute the cheese. Run from nlp/src:  cd nlp/src && bash train/run_gcp.sh
# Builds SFT data, then GPU phases C->E + calibration F.
#
# GPU_PROFILE selects hardware defaults (any value still overridable per-knob):
#   5090 (default) — RTX 5090 / Blackwell, 32GB, dedicated. bf16, big batch/seq,
#                    fp16 vllm (no 4-bit needed), high VRAM util.
#   t4             — shared Tesla T4, 16GB. fp16-only, batch 1 + short seq + 4-bit
#                    everywhere to survive the cramped, contended card.
set -euo pipefail

TEACHER="${TEACHER:-Qwen/Qwen2.5-7B-Instruct}"
# 3B LoRA (fp16) beats the 0.5B on L2 composition and still fits 32GB; the 0.5B
# distilled to est ~0.47 (below the 0.759 regex baseline) and scored L2 = 0.000.
STUDENT="${STUDENT:-Qwen/Qwen2.5-3B-Instruct}"
PER_DOC="${PER_DOC:-6}"
GPU_PROFILE="${GPU_PROFILE:-5090}"

if [ "$GPU_PROFILE" = "5090" ]; then
    # 32GB + bf16 (auto-detected in the trainers): 7B QLoRA at batch 4/seq 2048,
    # tiny student at batch 8. vllm loads the 7B teacher in fp16 (fits easily) at
    # high util since the card is dedicated.
    # NB: Qwen2.5 has a 152k vocab -> the loss logits tensor is batch*seq*152k.
    # Big batch OOMs even 32GB (batch8*seq2048 logits+upcast ~= 9GB+). Keep
    # batch*seq modest; grad-accum recovers effective batch.
    # max_length MUST exceed prompt+answer or SFTTrainer truncates the gold
    # answer off the end (it lives last) -> student learns doc-continuation, not
    # QA. With _MAX_DOC_CHARS=4000 examples top out ~3225 tok, so 4096 keeps every
    # answer. seq doubled vs 2048 -> halve batch (logits = batch*seq*152k vocab is
    # the OOM driver) and double accum to hold effective batch.
    : "${VLLM_GPU_MEM_UTIL:=0.90}"; : "${VLLM_MAX_LEN:=8192}"; : "${VLLM_BNB:=0}"
    : "${TEACHER_BATCH:=1}";  : "${TEACHER_ACCUM:=16}"; : "${TEACHER_MAXLEN:=4096}"
    : "${STUDENT_BATCH:=2}";  : "${STUDENT_ACCUM:=8}";  : "${STUDENT_MAXLEN:=4096}"
else  # t4
    : "${VLLM_GPU_MEM_UTIL:=0.75}"; : "${VLLM_MAX_LEN:=4096}"; : "${VLLM_BNB:=1}"
    # 1024 truncated the answer off every example (same bug as 5090); 4096 keeps
    # it. 16GB -> batch 1. Teacher 7B at 4096 likely OOMs on T4; use SKIP_TEACHER=1.
    : "${TEACHER_BATCH:=1}";  : "${TEACHER_ACCUM:=16}"; : "${TEACHER_MAXLEN:=4096}"
    : "${STUDENT_BATCH:=1}";  : "${STUDENT_ACCUM:=16}"; : "${STUDENT_MAXLEN:=4096}"
fi
export VLLM_GPU_MEM_UTIL VLLM_MAX_LEN VLLM_BNB
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# Cloud containers cap processes/threads (ulimit -u / cgroup pids); OpenMP +
# tokenizers + dataloader workers blow past it -> "libgomp: Thread creation
# failed" + segfault. Cap thread spawning and bump the limit best-effort.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
ulimit -u 64000 2>/dev/null || true
echo "GPU_PROFILE=$GPU_PROFILE  teacher(b=$TEACHER_BATCH a=$TEACHER_ACCUM L=$TEACHER_MAXLEN)  student(b=$STUDENT_BATCH a=$STUDENT_ACCUM L=$STUDENT_MAXLEN)  vllm(util=$VLLM_GPU_MEM_UTIL bnb=$VLLM_BNB)"

DATA=train/data
CKPT=train/ckpt
ART=train/artifacts
mkdir -p "$DATA" "$CKPT" "$ART"

echo "== deps =="
pip install -q -r ../requirements-train.txt
# vllm speeds synth a lot. If its torch pin clashes with your Blackwell torch,
# set SKIP_VLLM=1 and gen_synthetic falls back to HF (4-bit on t4, fp16 on 5090).
if [ "${SKIP_VLLM:-0}" != "1" ]; then
    pip install -q vllm || echo "vllm install failed -> HF fallback for synth"
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
