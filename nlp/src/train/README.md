# train/ — finetune → distil → quantize pipeline

Executes the **cheese** (see `../../cheese.md`): finetune a large teacher on the
corpus, distil into a small Qwen student, quantize to Q4, deploy behind the
solved regex retrieval. Goal: answer-equivalence rate **f → ~1.0** so the total
score clears **0.95** (baseline 0.759).

**Where it runs:** Phases A–B locally; C–F on the **GCP GPU box**.
**Sacred:** train on the `dev` split only (`../eval_split.json`); keep the 177
`heldout` rows for measurement.

## Flow

```
A. proxy scorer        ../nlp_bem_proxy.py        # measurable "0.95 local" target
   eval               ../eval_answers.py --proxy-bem

B. SFT data            prepare_data.py            # gold dev rows -> chat JSONL  (local)
   synthetic           gen_synthetic.py           # teacher QA over 296 docs     (GPU)

C. finetune teacher    finetune_teacher.py        # QLoRA Qwen2.5-7B/14B          (GPU)

D. distil student      distill_student.py         # -> Qwen2.5-0.5B / Qwen3-0.6B  (GPU)
                                                    # train until proxy >= 0.95

E. quantize            quantize.py                # Q4 GGUF / AWQ                 (GPU)
   integrate           ../nlp_llm.py + ../../nlp/Dockerfile  # swap student, hybrid route

F. calibrate           ../../test/test_nlp.py     # real BEM on GCP; refit proxy threshold
```

## Commands

```bash
cd nlp/src

# A — measure (works today, regex-only baseline)
python eval_answers.py --proxy-bem --by-bucket

# B — build SFT data (local, no GPU)
python train/prepare_data.py --out train/data/sft_dev.jsonl --top-k 3
# (GPU) synthetic augmentation
python train/gen_synthetic.py --teacher Qwen/Qwen2.5-7B-Instruct --per-doc 6 \
    --out train/data/synth.jsonl

# C — finetune teacher (GPU)
python train/finetune_teacher.py --base Qwen/Qwen2.5-7B-Instruct \
    --data train/data/sft_dev.jsonl --data train/data/synth.jsonl \
    --out train/ckpt/teacher-7b --merge

# D — distil student (GPU); train until proxy >= 0.95 on heldout
python train/distill_student.py --mode seq --student Qwen/Qwen2.5-0.5B-Instruct \
    --data train/data/teacher_labels.jsonl --out train/ckpt/student-0.5b
NLP_USE_LLM=1 NLP_LLM_MODEL=train/ckpt/student-0.5b \
    python eval_answers.py --proxy-bem --heldout

# E — quantize to Q4
python train/quantize.py --mode gguf --model train/ckpt/student-0.5b \
    --out train/artifacts/student-q4.gguf

# F — real BEM on GCP, then recalibrate the proxy cutoff
python ../../test/test_nlp.py
python nlp_bem_proxy.py --calibrate train/data/bem_labels.json
```

## Notes / gotchas
- **No "Qwen 0.8B"** — use Qwen2.5-0.5B or Qwen3-0.6B.
- **64-token cap is not enforced** (`test/test_nlp.py:184`) — let the student emit
  complete composed L2 answers; the real cap is 512 chars on `Q+Ref+Candidate`.
- **Hybrid routing** at integration: keep regex for high-precision buckets
  (codename/money/percent/count), student reader for fallback + L2.
- Proxy ≠ real BEM. Trust proxy **deltas** until Phase F calibration on GCP.
- Heavy deps (torch/transformers/trl/peft/vllm/awq) are training-only — add to a
  separate `requirements-train.txt`, not the inference image.
