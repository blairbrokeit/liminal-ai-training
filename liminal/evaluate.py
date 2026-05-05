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

from liminal.model import load_model, generate
from liminal.judge import build_judge_from_config
from liminal.benchmarks import BenchmarkRunner

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
    parser.add_argument("--truthfulqa-limit", type=int, default=200)
    parser.add_argument("--mmlu", action="store_true", help="Also run MMLU (exact-match, no judge)")
    parser.add_argument("--mmlu-limit", type=int, default=500)
    parser.add_argument("--gsm8k", action="store_true", help="Also run GSM8K math word problems")
    parser.add_argument("--gsm8k-limit", type=int, default=500)
    parser.add_argument("--humaneval", action="store_true", help="Also run HumanEval (executes generated code)")
    parser.add_argument("--humaneval-limit", type=int, default=164)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    tasks = load_tasks(args.tasks)
    judge = build_judge_from_config(config["judge"])
    benchmarks = BenchmarkRunner()

    def run_standard(model, tokenizer, label: str) -> dict:
        out = {}
        if args.truthfulqa:
            console.rule(f"[bold]TruthfulQA ({label})[/bold]")
            out["truthfulqa"] = benchmarks.run_truthfulqa(model, tokenizer, judge, generate, limit=args.truthfulqa_limit)
            benchmarks.save_result(out["truthfulqa"], label=label)
        if args.mmlu:
            console.rule(f"[bold]MMLU ({label})[/bold]")
            out["mmlu"] = benchmarks.run_mmlu(model, tokenizer, generate, limit=args.mmlu_limit)
            benchmarks.save_result(out["mmlu"], label=label)
        if args.gsm8k:
            console.rule(f"[bold]GSM8K ({label})[/bold]")
            out["gsm8k"] = benchmarks.run_gsm8k(model, tokenizer, generate, limit=args.gsm8k_limit)
            benchmarks.save_result(out["gsm8k"], label=label)
        if args.humaneval:
            console.rule(f"[bold]HumanEval ({label})[/bold]")
            out["humaneval"] = benchmarks.run_humaneval(model, tokenizer, generate, limit=args.humaneval_limit)
            benchmarks.save_result(out["humaneval"], label=label)
        return out

    # Base model
    console.rule("[bold]Base Model (no adapter)[/bold]")
    base_model, tokenizer = load_model(args.model, adapter_path=None)
    base_result = benchmarks.run_custom("eval", tasks, base_model, tokenizer, judge, generate)
    benchmarks.save_result(base_result, label="base")
    base_standard = run_standard(base_model, tokenizer, label="base")
    del base_model

    # Adapted model
    if args.adapter:
        console.rule("[bold]Adapted Model (with LoRA)[/bold]")
        adapted_model, tokenizer = load_model(args.model, adapter_path=args.adapter)
        adapted_result = benchmarks.run_custom("eval", tasks, adapted_model, tokenizer, judge, generate)
        benchmarks.save_result(adapted_result, label="adapted")

        console.print()
        benchmarks.compare(base_result, adapted_result)

        adapted_standard = run_standard(adapted_model, tokenizer, label="adapted")
        for name, before in base_standard.items():
            after = adapted_standard.get(name)
            if after:
                benchmarks.compare(before, after)

        del adapted_model
    else:
        console.print("\n[dim]No adapter specified. Run with --adapter to compare.[/dim]")


if __name__ == "__main__":
    main()
