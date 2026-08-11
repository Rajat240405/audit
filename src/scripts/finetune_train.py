"""
LoRA fine-tuning for the INCOIS Audit Pro generator (Qwen3-4B).

Trains a small LoRA adapter on the Alpaca-format JSONL produced by
finetune_prepare_data.py. Runs on a 4GB VRAM GPU (RTX 3050 Laptop).

IMPORTANT (4GB VRAM path):
- We do NOT use 4-bit quantization: on a 4GB GPU, 4-bit + LoRA + fp16
  compute dequantizes to ~10GB and OOMs (verified). Instead we load the
  model in fp16, use gradient checkpointing, a short max-seq-len, and a
  small LoRA rank — this fits 4GB.
- Qwen3 thinking mode is disabled so answers are direct (no <think>).

Usage (create a venv first — see INSTALL below):
    python scripts/finetune_train.py \
        --data data/finetune \
        --base-model Qwen/Qwen3-4B \
        --out-dir data/finetune/adapter \
        --epochs 3

INSTALL (Windows, PowerShell, your venv):
    pip install torch --index-url https://download.pytorch.org/whl/cu121
    pip install transformers>=4.51 peft datasets accelerate
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

SYSTEM_PROMPT = (
    "You are a parliamentary research assistant for INCOIS. Answer the question "
    "using ONLY the provided retrieved context. Copy names and figures exactly "
    "as written in the context; never invent facts, names, programmes, or "
    "statistics. If the context does not contain the information, say so. "
    "Cite [Source N] for each claim."
)


def build_messages(item: dict) -> list[dict]:
    """Alpaca item -> Qwen3 chat messages (system/user/assistant)."""
    inst = item.get("instruction", "")
    inp = item.get("input", "")
    user = f"{inst}\n\n{inp}" if inp else inst
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
        {"role": "assistant", "content": item.get("output", "")},
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/finetune")
    ap.add_argument("--base-model", default="Qwen/Qwen3-4B")
    ap.add_argument("--out-dir", default="data/finetune/adapter")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-seq-len", type=int, default=512)
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--max-steps", type=int, default=-1)
    args = ap.parse_args()

    data_dir = Path(os.path.abspath(args.data))
    out_dir = Path(os.path.abspath(args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    import transformers
    print(f"[versions] transformers={transformers.__version__}, "
          f"torch={torch.__version__}, cuda={torch.cuda.is_available()}")

    print(f"== Loading base model: {args.base_model} (fp16, no 4-bit) ==")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available — cannot train. Reinstall torch with CUDA.")

    # fp16, NO quantization. 4-bit OOMs on 4GB (dequantizes to ~10GB).
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        device_map={"": 0},
        trust_remote_code=True,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    n_cuda = sum(1 for p in model.parameters() if p.device.type == "cuda")
    n_cpu = sum(1 for p in model.parameters() if p.device.type == "cpu")
    print(f"[gpu] layers on cuda: {n_cuda} | layers on cpu: {n_cpu}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # enable gradient checkpointing to fit 4GB
    model.gradient_checkpointing_enable()

    print("== Attaching LoRA adapters ==")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print(f"== Loading dataset from {data_dir} ==")
    ds = load_dataset(
        "json",
        data_files={
            "train": str(data_dir / "train.jsonl"),
            "validation": str(data_dir / "val.jsonl"),
        },
    )

    def format_example(example):
        messages = build_messages(example)
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        return {"text": text}

    train_raw = ds["train"].map(format_example)
    val_raw = ds["validation"].map(format_example)

    print("\n== Sample formatted training example ==\n")
    print(train_raw[0]["text"][:600])
    print("\n...\n")

    base_cols = train_raw.column_names

    def tokenize_fn(ex):
        return tokenizer(
            ex["text"], truncation=True, max_length=args.max_seq_len, padding=False
        )

    train_ds = train_raw.map(tokenize_fn, remove_columns=base_cols)
    val_ds = val_raw.map(tokenize_fn, remove_columns=val_raw.column_names)

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    args_train = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        eval_strategy="no",  # skip eval — saves time, avoids validation errors
        fp16=True,
        bf16=False,
        gradient_checkpointing=True,
        # CRITICAL for 4GB laptops: force CUDA, never CPU
        use_cpu=False,
        # free memory between steps
        optim="adamw_torch",
        report_to=[],
        remove_unused_columns=False,
        dataloader_pin_memory=False,
    )

    # Confirm where the model actually is, right before training.
    dev_counts: dict[str, int] = {}
    for p in model.parameters():
        dev_counts[p.device.type] = dev_counts.get(p.device.type, 0) + 1
    print(f"[gpu] trainable params by device: {dev_counts}")

    trainer = Trainer(
        model=model,
        args=args_train,
        train_dataset=train_ds,
        data_collator=collator,
        processing_class=tokenizer,
    )

    print("== Training ==")
    trainer.train()
    print(f"== Saving adapter to {out_dir} ==")
    trainer.model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    print("Done. Adapter saved. Next: merge + import to Ollama "
          "(see finetune_merge.py).")


if __name__ == "__main__":
    main()
