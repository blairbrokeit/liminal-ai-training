"""smoke_run.py — One-command smoke test for the full liminal pipeline.

Downloads Phi-3 Mini if missing, then runs 2 training loops with MMLU + GSM8K
benchmarks before and after. Designed to give you real before/after numbers in
under an hour on a consumer GPU (or a few hours on CPU).

Usage:
    python scripts/smoke_run.py
    python scripts/smoke_run.py --skip-download   # if the model is already there

Requires:
    - OPENAI_API_KEY env var (judge + NPCs)
    - ~7 GB disk for the Phi-3 Mini model
    - 8 GB+ VRAM (or patience on CPU)
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_REPO = "microsoft/Phi-3-mini-4k-instruct"
MODEL_LOCAL_DIR = REPO_ROOT / "models" / "phi-3-mini-4k-instruct"
CONFIG_PATH = REPO_ROOT / "configs" / "smoke-phi3.yaml"
TASKS_PATH = REPO_ROOT / "tasks" / "example.jsonl"


def _check_api_key() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY is not set.", file=sys.stderr)
        print("       The judge and NPCs both need it. Set it and re-run.", file=sys.stderr)
        sys.exit(1)


def _download_model() -> None:
    if MODEL_LOCAL_DIR.exists() and any(MODEL_LOCAL_DIR.iterdir()):
        print(f"[smoke] Model already present at {MODEL_LOCAL_DIR}, skipping download.")
        return
    print(f"[smoke] Downloading {MODEL_REPO} (~7 GB)...")
    subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "download_model.py"),
         "--model", MODEL_REPO],
        check=True,
        cwd=REPO_ROOT,
    )


def _run_training() -> None:
    print("[smoke] Starting 2-loop training run with MMLU+GSM8K benchmarks before/after.")
    subprocess.run(
        [sys.executable, "-m", "liminal.train",
         "--model", str(MODEL_LOCAL_DIR),
         "--tasks", str(TASKS_PATH),
         "--config", str(CONFIG_PATH),
         "--loops", "2",
         "--batch-size", "8",
         "--benchmark",
         "--regression"],
        check=True,
        cwd=REPO_ROOT,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Liminal AI Training — smoke test")
    parser.add_argument("--skip-download", action="store_true", help="Skip model download")
    args = parser.parse_args()

    _check_api_key()
    if not args.skip_download:
        _download_model()
    _run_training()
    print("[smoke] Done. Check ./metrics/benchmarks/ for the saved before/after results.")


if __name__ == "__main__":
    main()
