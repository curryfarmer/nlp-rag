"""Phase C — QLoRA finetune of the large teacher (GCP GPU).

Finetune a large instruct model (Qwen2.5-7B/14B-Instruct) on the chat-format SFT
data from prepare_data.py (+ filtered synthetic from gen_synthetic.py). The
teacher learns the corpus answer style and L2 composition; it then (a) serves as
the distillation teacher and (b) can generate more synthetic data.

Uses TRL SFTTrainer + PEFT LoRA over a 4-bit base (QLoRA) — fits a 7B on a single
24GB GPU. Heavy deps lazy-imported in main() so the file imports anywhere.

  python train/finetune_teacher.py \
      --base Qwen/Qwen2.5-7B-Instruct \
      --data data/sft_dev.jsonl --data data/synth.jsonl \
      --out ckpt/teacher-7b --epochs 3 --lr 2e-4

Evaluate the merged adapter with eval_answers.py --proxy-bem (point NLP_LLM_MODEL
at the merged dir) before distilling.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def load_jsonl_messages(paths: list[str]):
    import json
    rows = []
    for p in paths:
        for line in Path(p).read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--data", action="append", required=True,
                    help="chat-format JSONL (repeatable)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--merge", action="store_true", help="merge+save full weights after train")
    args = ap.parse_args()

    # --- lazy heavy imports ---
    import torch
    from datasets import Dataset
    from peft import LoraConfig, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    rows = load_jsonl_messages(args.data)
    ds = Dataset.from_list([{"messages": r["messages"]} for r in rows])

    # T4 (Turing) has NO bf16 — use fp16. A100/L4 (Ampere+) can flip these to bf16.
    bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if bf16_ok else torch.float16
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=compute_dtype,
                             bnb_4bit_use_double_quant=True)
    tok = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForCausalLM.from_pretrained(
        args.base, quantization_config=bnb, dtype=compute_dtype, device_map="auto")
    # Required for QLoRA: casts norms to fp32, enables input grads so gradient
    # checkpointing actually propagates through the frozen 4-bit base.
    model = prepare_model_for_kbit_training(model)

    lora = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
                      bias="none", task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"])

    cfg = SFTConfig(output_dir=args.out, num_train_epochs=args.epochs,
                    per_device_train_batch_size=args.batch,
                    gradient_accumulation_steps=args.grad_accum,
                    learning_rate=args.lr, warmup_ratio=0.03, lr_scheduler_type="cosine",
                    logging_steps=10, save_strategy="epoch",
                    bf16=bf16_ok, fp16=not bf16_ok,
                    dataloader_num_workers=0,  # avoid container thread/pids cap
                    max_length=args.max_seq_len, gradient_checkpointing=True)

    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds,
                         processing_class=tok, peft_config=lora)
    trainer.train()
    trainer.save_model(args.out)

    if args.merge:
        merged = trainer.model.merge_and_unload()
        merged.save_pretrained(f"{args.out}-merged")
        tok.save_pretrained(f"{args.out}-merged")
        print(f"merged weights -> {args.out}-merged")

    print(f"teacher adapter saved -> {args.out}")
    print("NEXT: eval with eval_answers.py --proxy-bem (NLP_LLM_MODEL=<merged dir>), "
          "then train/distill_student.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
