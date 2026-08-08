"""
LoRA fine-tuning for the INCOIS Audit Pro generator (Qwen3-4B).

Trains a small LoRA adapter on the Alpaca-format JSONL produced by
finetune_prepare_data.py. Runs on a 4GB VRAM GPU (RTX 3050 Laptop).

Qwen3-specific notes:
- Qwen3-4B defaults to THINKING mode (emits <think>...</think> blocks).
  For our evidence-extraction task we want DIRECT answers, so thinking is
  disabled at inference AND we use the non-thinking chat template during
  training so the model never learns to emit <think> noise.
- Requires transformers>=4.51 (the Qwen3 arch needs it).

Uses `transformers.Trainer` directly (NOT trl.SFTTrainer) because trl's
SFTTrainer API keeps changing between versions (tokenizer vs
processing_class) and broke on some installs. transformers.Trainer is
version-stable.

Usage (create a venv first — see INSTALL below):
    python scripts/finetune_train.py \
        --data data/finetune \
        --base-model Qwen/Qwen3-4B \
        --out-dir data/finetune/adapter \
        --epochs 3

INSTALL (Windows, PowerShell, your venv):
    pip install torch --index-url https://download.pytorch.org/whl/cu121
    pip install transformers>=4.51 peft datasets accelerate bitsandbytes
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
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
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--max-steps", type=int, default=-1)
    args = ap.parse_args()

    data_dir = Path(args.data)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import transformers
    print(f"[versions] transformers={transformers.__version__}, "
          f"torch={torch.__version__}, cuda={torch.cuda.is_available()}")

    print(f"== Loading base model: {args.base_model} ==")
    # 4-bit quantized base so LoRA fits 4GB VRAM
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = prepare_model_for_kbit_training(model)

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

    # Tokenize with the trainer (causal LM labels = input ids).
    # Drop ALL original columns (instruction/input/output/text) so the collator
    # only ever sees input_ids + attention_mask — otherwise it tries to pad the
    # leftover string columns and crashes with "too many dimensions 'str'".
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
        eval_strategy="steps" if len(val_ds) > 0 else "no",
        eval_steps=100,
        fp16=torch.cuda.is_available(),
        bf16=False,
        report_to=[],
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=args_train,
        train_dataset=train_ds,
        eval_dataset=val_ds if len(val_ds) > 0 else None,
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
