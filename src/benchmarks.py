"""
benchmarks.py — Run standard benchmarks to prove the model improved.

Supports TruthfulQA, custom eval sets, and before/after comparison.
Results are saved and can be compared across training runs.
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path
from rich.console import Console
from rich.table import Table

console = Console()


@dataclass
class BenchmarkResult:
    name: str
    total: int
    correct: int
    timestamp: str
    per_category: dict

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total > 0 else 0.0


class BenchmarkRunner:
    def __init__(self, output_dir: str = "./metrics/benchmarks"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_truthfulqa(self, model, tokenizer, judge, generate_fn, limit: int = 200) -> BenchmarkResult:
        """Run TruthfulQA benchmark."""
        from datasets import load_dataset

        console.print("[bold]Running TruthfulQA benchmark...[/bold]")
        ds = load_dataset("truthful_qa", "generation", split="validation")

        correct = 0
        total = 0
        per_category = {}

        for i, row in enumerate(ds):
            if i >= limit:
                break

            question = row["question"]
            best_answer = row["best_answer"]
            category = row.get("category", "unknown")

            response = generate_fn(model, tokenizer, question)
            verdict = judge.evaluate(question, response, best_answer)

            if category not in per_category:
                per_category[category] = {"correct": 0, "total": 0}

            per_category[category]["total"] += 1
            total += 1

            if not verdict["sent_to_backrooms"]:
                correct += 1
                per_category[category]["correct"] += 1

            if (i + 1) % 50 == 0:
                console.print(f"  {i + 1}/{min(limit, len(ds))} — {correct}/{total} correct ({correct/total:.1%})")

        result = BenchmarkResult(
            name="truthfulqa",
            total=total,
            correct=correct,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            per_category=per_category,
        )

        console.print(f"  TruthfulQA: {result.accuracy:.1%} ({correct}/{total})")
        return result

    def run_custom(self, name: str, tasks: list[dict], model, tokenizer, judge, generate_fn) -> BenchmarkResult:
        """Run a custom benchmark from a task JSONL."""
        console.print(f"[bold]Running benchmark: {name}[/bold]")

        correct = 0
        total = 0
        per_category = {}

        for task_data in tasks:
            task = task_data["task"]
            correct_answer = task_data.get("correct", "")
            category = task_data.get("category", "unknown")

            response = generate_fn(model, tokenizer, task)
            verdict = judge.evaluate(task, response, correct_answer)

            if category not in per_category:
                per_category[category] = {"correct": 0, "total": 0}

            per_category[category]["total"] += 1
            total += 1

            if not verdict["sent_to_backrooms"]:
                correct += 1
                per_category[category]["correct"] += 1

        result = BenchmarkResult(
            name=name,
            total=total,
            correct=correct,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            per_category=per_category,
        )

        console.print(f"  {name}: {result.accuracy:.1%} ({correct}/{total})")
        return result

    def save_result(self, result: BenchmarkResult, label: str = ""):
        """Save benchmark result to disk."""
        filename = f"{result.name}_{label}_{result.timestamp.replace(':', '-')}.json" if label else f"{result.name}_{result.timestamp.replace(':', '-')}.json"
        data = {
            "name": result.name,
            "label": label,
            "total": result.total,
            "correct": result.correct,
            "accuracy": round(result.accuracy, 4),
            "timestamp": result.timestamp,
            "per_category": result.per_category,
        }
        with open(self.output_dir / filename, "w") as f:
            json.dump(data, f, indent=2)

    def compare(self, before: BenchmarkResult, after: BenchmarkResult):
        """Print a comparison between two benchmark runs."""
        table = Table(title=f"Benchmark: {before.name}")
        table.add_column("", style="bold")
        table.add_column("Before")
        table.add_column("After")
        table.add_column("Change")

        diff = after.accuracy - before.accuracy
        color = "green" if diff > 0 else ("red" if diff < 0 else "dim")

        table.add_row(
            "Overall",
            f"{before.accuracy:.1%} ({before.correct}/{before.total})",
            f"{after.accuracy:.1%} ({after.correct}/{after.total})",
            f"[{color}]{diff:+.1%}[/{color}]",
        )

        # Per-category
        all_cats = set(list(before.per_category.keys()) + list(after.per_category.keys()))
        for cat in sorted(all_cats):
            b = before.per_category.get(cat, {"correct": 0, "total": 0})
            a = after.per_category.get(cat, {"correct": 0, "total": 0})
            b_acc = b["correct"] / max(1, b["total"])
            a_acc = a["correct"] / max(1, a["total"])
            cat_diff = a_acc - b_acc
            cat_color = "green" if cat_diff > 0 else ("red" if cat_diff < 0 else "dim")
            table.add_row(
                f"  {cat}",
                f"{b_acc:.1%}",
                f"{a_acc:.1%}",
                f"[{cat_color}]{cat_diff:+.1%}[/{cat_color}]",
            )

        console.print(table)
