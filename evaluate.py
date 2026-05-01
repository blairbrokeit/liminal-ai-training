"""
evaluate.py — Compare base model vs adapted model.

Runs both on the same tasks, shows per-category improvement,
and optionally runs TruthfulQA benchmark.
"""

import argparse
import json
import yaml
from rich.console import Console
from rich.table import Table

from src.model import load_model, generate
from src.judge import Judge
from src.benchmarks import BenchmarkRunner

console = Console()


def load_tasks(path: str) -> list[dict]:
    tasks = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))
    return tasks


def main():
    parser = argparse.ArgumentParser(description="Evaluate base vs adapted model")
    parser.add_argument("--model", required=True, help="Path to base model")
    parser.add_argument("--adapter", default=None, help="Path to LoRA adapter")
    parser.add_argument("--tasks", required=True, help="Path to task JSONL file")
    parser.add_argument("--config", default="config.yaml", help="Config file")
    parser.add_argument("--truthfulqa", action="store_true", help="Also run TruthfulQA")
    parser.add_argument("--truthfulqa-limit", type=int, default=200, help="TruthfulQA sample size")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    tasks = load_tasks(args.tasks)
    judge = Judge(model=config["judge"]["model"])
    benchmarks = BenchmarkRunner()

    # Base model
    console.rule("[bold]Base Model (no adapter)[/bold]")
    base_model, tokenizer = load_model(args.model, adapter_path=None)
    base_result = benchmarks.run_custom("eval", tasks, base_model, tokenizer, judge, generate)
    benchmarks.save_result(base_result, label="base")
    del base_model

    # Adapted model
    adapted_result = None
    if args.adapter:
        console.rule("[bold]Adapted Model (with LoRA)[/bold]")
        adapted_model, tokenizer = load_model(args.model, adapter_path=args.adapter)
        adapted_result = benchmarks.run_custom("eval", tasks, adapted_model, tokenizer, judge, generate)
        benchmarks.save_result(adapted_result, label="adapted")

        console.print()
        benchmarks.compare(base_result, adapted_result)

        # TruthfulQA
        if args.truthfulqa:
            console.rule("[bold]TruthfulQA — Base[/bold]")
            base_tqa = benchmarks.run_truthfulqa(
                base_model, tokenizer, judge, generate, limit=args.truthfulqa_limit
            )

            console.rule("[bold]TruthfulQA — Adapted[/bold]")
            adapted_tqa = benchmarks.run_truthfulqa(
                adapted_model, tokenizer, judge, generate, limit=args.truthfulqa_limit
            )

            benchmarks.compare(base_tqa, adapted_tqa)
            benchmarks.save_result(base_tqa, label="base")
            benchmarks.save_result(adapted_tqa, label="adapted")

        del adapted_model
    else:
        console.print("\n[dim]No adapter specified. Run with --adapter to compare.[/dim]")


if __name__ == "__main__":
    main()
