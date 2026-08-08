"""
Merge the LoRA adapter into a standalone model and import into Ollama.

LoRA training keeps the base model frozen + a small adapter. For Ollama we
must FUSE the adapter into a full model, export to GGUF, and import it.

Usage:
    python scripts/finetune_merge.py \
        --base-model Qwen/Qwen3-4B \
        --adapter data/finetune/adapter \
        --out-dir data/finetune/merged

Then import to Ollama:
    ollama create incois-qa -f data/finetune/Modelfile
    ollama run incois-qa

INSTALL (in the same venv as training):
    pip install transformers peft torch
    # for GGUF export:
    pip install llama-cpp-python   # optional — see notes below

NOTE on GGUF export: the cleanest path is to load the merged model in
transformers, then use `llama.cpp`'s convert script OR upload to a HF repo
and let Ollama/llamafile quantize. If llama-cpp-python isn't available,
this script still saves the full merged HF model — you can convert with the
llama.cpp `convert_hf_to_gguf.py` tool on the merged folder.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="Qwen/Qwen3-4B")
    ap.add_argument("--adapter", default="data/finetune/adapter")
    ap.add_argument("--out-dir", default="data/finetune/merged")
    ap.add_argument("--modelfile", default="data/finetune/Modelfile")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"== Loading base + adapter ==")
    # Qwen3: load with sdpa attention; disable thinking at generation
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base, args.adapter)
    print("== Merging ==")
    merged = model.merge_and_unload()

    print(f"== Saving merged model to {out_dir} ==")
    merged.save_pretrained(str(out_dir))
    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    tok.save_pretrained(str(out_dir))

    # write a Modelfile for Ollama (points at the merged dir)
    modelfile = Path(args.modelfile)
    modelfile.parent.mkdir(parents=True, exist_ok=True)
    modelfile.write_text(
        f"FROM {out_dir.resolve()}\n\n"
        'SYSTEM """You are a parliamentary research assistant for INCOIS. '
        'Answer using ONLY the provided retrieved context. Copy names and '
        'figures exactly as written. Never invent facts. Cite [Source N] when '
        'appropriate."""\n',
        encoding="utf-8",
    )
    print(f"\nMerged model saved. To import into Ollama:\n"
          f"  ollama create incois-qa -f {modelfile}\n"
          f"  ollama run incois-qa\n"
          f"\nIf GGUF export is needed, run llama.cpp's convert_hf_to_gguf.py "
          f"on {out_dir.resolve()} then create the Ollama model from the GGUF.")


if __name__ == "__main__":
    main()
