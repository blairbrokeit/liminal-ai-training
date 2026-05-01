"""
download_model.py — Download and optionally quantize a base model from HuggingFace.
"""

import argparse
from pathlib import Path
from huggingface_hub import snapshot_download


def main():
    parser = argparse.ArgumentParser(description="Download a base model")
    parser.add_argument("--model", required=True, help="HuggingFace model ID (e.g. meta-llama/Llama-3.1-8B-Instruct)")
    parser.add_argument("--output", default="./models", help="Output directory")
    parser.add_argument("--quantize", default=None, choices=["q4", "q8", "none"], help="Quantization level")
    args = parser.parse_args()

    model_name = args.model.split("/")[-1]
    output_path = Path(args.output) / model_name
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {args.model} to {output_path}...")
    snapshot_download(
        repo_id=args.model,
        local_dir=str(output_path),
        ignore_patterns=["*.gguf", "*.bin"],  # prefer safetensors
    )

    print(f"Model downloaded to {output_path}")

    if args.quantize and args.quantize != "none":
        print(f"\nNote: For {args.quantize} quantization, the model will be quantized")
        print("at load time via bitsandbytes (see src/model.py).")
        print("No separate quantization step needed.")


if __name__ == "__main__":
    main()
