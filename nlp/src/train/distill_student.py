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
    """Sequence-level KD == SFT on teacher-generated (and gold) chat data."""
    import torch
    from datasets import Dataset
    from transformers import AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    rows = []
    for p in args.data:
        rows.extend(load_jsonl(p))
    ds = Dataset.from_list([{"messages": r["messages"]} for r in rows])

    bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()  # T4 -> False
    tok = AutoTokenizer.from_pretrained(args.student)
    cfg = SFTConfig(output_dir=args.out, num_train_epochs=args.epochs,
                    per_device_train_batch_size=args.batch,
                    gradient_accumulation_steps=args.grad_accum,
                    learning_rate=args.lr, warmup_ratio=0.03, lr_scheduler_type="cosine",
                    logging_steps=10, save_strategy="epoch",
                    bf16=bf16_ok, fp16=not bf16_ok,
                    max_length=args.max_seq_len)
    trainer = SFTTrainer(model=args.student, args=cfg, train_dataset=ds, processing_class=tok)
    trainer.train()
    trainer.save_model(args.out)


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
