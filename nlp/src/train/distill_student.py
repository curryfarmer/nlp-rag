"""Phase D — distil the finetuned teacher into a small student (GCP GPU).

Target students (NB: there is NO "Qwen 0.8B"):
  - Qwen/Qwen2.5-0.5B-Instruct
  - Qwen/Qwen3-0.6B

Two distillation modes:

  seq   (default) Sequence-level KD: the teacher answers every (question, top-k
        docs) prompt; the student is SFT'd on those teacher answers (plus the
        gold dev answers). Simple, robust, framework-agnostic, usually enough to
        hit the target on a narrow domain. Run gen_teacher_labels first (or pass
        a teacher-labeled JSONL via --data).

  logit On-the-fly logit KD: KL(student || teacher) on shared-vocab logits +
        CE on gold, loss = alpha*CE + (1-alpha)*T^2*KL. Needs teacher + student
        to share a tokenizer/vocab (true within the Qwen family). +2-3% but
        slower; enable with --mode logit.

Train until eval_answers.py --proxy-bem reports proxy ≥ 0.95 on held-out.
Heavy deps lazy-imported in main().

  python train/distill_student.py --mode seq \
      --student Qwen/Qwen2.5-0.5B-Instruct \
      --data data/teacher_labels.jsonl --out ckpt/student-0.5b --epochs 4
"""
from __future__ import annotations

import argparse
from pathlib import Path


def load_jsonl(path: str):
    import json
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def _train_seq(args):
    """Sequence-level KD == SFT on teacher-generated (and gold) chat data.

    LoRA over an fp16/bf16 base (NOT 4-bit), then merge_and_unload + save full
    weights. Rationale:
      - LoRA keeps the optimizer state tiny, so a 1.5B/3B student fits 32GB where
        full fine-tune + the 152k-vocab loss logits would OOM.
      - fp16 base (vs QLoRA 4-bit) merges cleanly into deployable full weights;
        merging a LoRA into a 4-bit base degrades/breaks.
      - nlp_llm loads the merged dir via AutoModelForCausalLM directly — no PEFT
        at inference.
    """
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    rows = []
    for p in args.data:
        rows.extend(load_jsonl(p))
    # prompt/completion format -> TRL masks the prompt and trains loss on the
    # answer only (see prepare_data.py). A flat messages list would put loss on
    # the whole sequence and the doc-length prompt would drown the answer.
    ds = Dataset.from_list([{"prompt": r["prompt"], "completion": r["completion"]}
                            for r in rows])

    bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()  # T4 -> False
    dtype = torch.bfloat16 if bf16_ok else torch.float16
    tok = AutoTokenizer.from_pretrained(args.student)
    model = AutoModelForCausalLM.from_pretrained(args.student, dtype=dtype)
    if torch.cuda.is_available():
        model.enable_input_require_grads()  # required for grad-checkpoint + LoRA

    lora = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
                      bias="none", task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"])

    cfg = SFTConfig(output_dir=args.out, num_train_epochs=args.epochs,
                    per_device_train_batch_size=args.batch,
                    gradient_accumulation_steps=args.grad_accum,
                    learning_rate=args.lr, warmup_ratio=0.03, lr_scheduler_type="cosine",
                    logging_steps=10, save_strategy="no",  # merge+save once at end
                    bf16=bf16_ok, fp16=not bf16_ok,
                    dataloader_num_workers=0,  # avoid container thread/pids cap
                    gradient_checkpointing=True,
                    gradient_checkpointing_kwargs={"use_reentrant": False},
                    max_length=args.max_seq_len)
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds,
                         processing_class=tok, peft_config=lora)
    trainer.train()
    # Merge LoRA into the base and save full weights so nlp_llm loads it directly.
    merged = trainer.model.merge_and_unload()
    merged.save_pretrained(args.out)
    tok.save_pretrained(args.out)


def _train_logit(args):
    """On-policy logit KD. Teacher + student must share vocab (Qwen family).

    SCAFFOLD: wire a custom Trainer.compute_loss that runs the frozen teacher on
    the same batch and blends CE with temperature-scaled KL. Pseudo-code:

        loss = alpha * ce_loss(student_logits, labels)
             + (1 - alpha) * (T**2) * KL(
                   log_softmax(student_logits / T),
                   softmax(teacher_logits / T))
    """
    raise NotImplementedError(
        "logit-KD loop is a scaffold: implement custom Trainer.compute_loss "
        "blending CE + temperature-scaled KL against the frozen teacher "
        f"(teacher={args.teacher!r}, T={args.temperature}, alpha={args.alpha}).")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["seq", "logit"], default="seq")
    ap.add_argument("--student", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--teacher", help="teacher dir/id (required for --mode logit)")
    ap.add_argument("--data", action="append", required=True,
                    help="teacher-labeled chat JSONL (repeatable)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=2.0, help="logit-KD softmax temp")
    ap.add_argument("--alpha", type=float, default=0.5, help="logit-KD CE weight")
    args = ap.parse_args()

    if args.mode == "logit" and not args.teacher:
        ap.error("--mode logit requires --teacher")

    (_train_seq if args.mode == "seq" else _train_logit)(args)

    print(f"student saved -> {args.out}")
    print("NEXT: eval_answers.py --proxy-bem (NLP_LLM_MODEL=<out>); if proxy>=0.95 "
          "-> train/quantize.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
