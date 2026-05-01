"""
export_adapter.py — Export a trained LoRA adapter for deployment.

Can merge the adapter into the base model for standalone inference,
or export just the adapter weights for portable deployment.
"""

import argparse
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def merge_and_export(base_path: str, adapter_path: str, output_path: str):
    """Merge LoRA adapter into base model and save as standalone."""

    print(f"Loading base model: {base_path}")
    model = AutoModelForCausalLM.from_pretrained(
        base_path,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(base_path)

    print(f"Loading adapter: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)

    print("Merging adapter into base model...")
    model = model.merge_and_unload()

    output = Path(output_path)
    output.mkdir(parents=True, exist_ok=True)

    print(f"Saving merged model to: {output}")
    model.save_pretrained(output)
    tokenizer.save_pretrained(output)

    print("Done. Merged model saved.")


def export_adapter_only(adapter_path: str, output_path: str):
    """Copy just the adapter weights (small, portable)."""

    import shutil
    src = Path(adapter_path)
    dst = Path(output_path)
    dst.mkdir(parents=True, exist_ok=True)

    for f in src.glob("*"):
        if f.is_file():
            shutil.copy2(f, dst / f.name)

    print(f"Adapter exported to {dst}")
    total_size = sum(f.stat().st_size for f in dst.glob("*"))
    print(f"Total size: {total_size / 1024 / 1024:.1f} MB")


def main():
    parser = argparse.ArgumentParser(description="Export trained adapter")
    parser.add_argument("--base-model", required=True, help="Path to base model")
    parser.add_argument("--adapter", required=True, help="Path to trained adapter")
    parser.add_argument("--output", required=True, help="Output path")
    parser.add_argument("--merge", action="store_true", help="Merge adapter into base (larger but standalone)")
    args = parser.parse_args()

    if args.merge:
        merge_and_export(args.base_model, args.adapter, args.output)
    else:
        export_adapter_only(args.adapter, args.output)


if __name__ == "__main__":
    main()
