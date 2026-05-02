"""
regression.py — Regression testing for trained adapters.

After training on one category, make sure other categories didn't get worse.
Runs a full eval across all categories and flags any drops.
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path
from rich.console import Console
from rich.table import Table

console = Console()


@dataclass
class RegressionResult:
    category: str
    before: float
    after: float
    change: float
    regressed: bool
    sample_size: int


class RegressionTester:
    def __init__(self, threshold: float = 0.05):
        """
        Args:
            threshold: accuracy drop larger than this counts as regression (default 5%)
        """
        self.threshold = threshold

    def test(self, model, tokenizer, tasks: list[dict], judge, generate_fn,
             baseline_results: dict | None = None) -> list[RegressionResult]:
        """
        Run regression test across all categories.

        Args:
            model: the model to test
            tokenizer: tokenizer
            tasks: full task set (all categories)
            judge: Judge instance
            generate_fn: generate function
            baseline_results: optional dict of {category: accuracy} from before training

        Returns:
            list of RegressionResult per category
        """
        # Run eval on all tasks, grouped by category
        category_results = {}

        for task_data in tasks:
            task = task_data["task"]
            correct_answer = task_data.get("correct", "")
            category = task_data.get("category", "unknown")

            if category not in category_results:
                category_results[category] = {"correct": 0, "total": 0}

            response = generate_fn(model, tokenizer, task)
            verdict = judge.evaluate(task, response, correct_answer)

            category_results[category]["total"] += 1
            if not verdict.get("sent_to_backrooms", True):
                category_results[category]["correct"] += 1

        # Compare against baseline
        results = []
        for cat, data in sorted(category_results.items()):
            after = data["correct"] / data["total"] if data["total"] > 0 else 0.0
            before = 0.0

            if baseline_results and cat in baseline_results:
                before = baseline_results[cat]

            change = after - before
            regressed = change < -self.threshold

            results.append(RegressionResult(
                category=cat,
                before=round(before, 3),
                after=round(after, 3),
                change=round(change, 3),
                regressed=regressed,
                sample_size=data["total"],
            ))

        return results

    def print_report(self, results: list[RegressionResult]):
        """Print regression test results."""
        table = Table(title="Regression Test Results")
        table.add_column("Category")
        table.add_column("Before")
        table.add_column("After")
        table.add_column("Change")
        table.add_column("Status")
        table.add_column("Samples")

        has_regression = False
        for r in results:
            color = "green" if r.change > 0 else ("red" if r.regressed else "yellow")
            status = "[red]REGRESSION[/red]" if r.regressed else "[green]OK[/green]"

            if r.regressed:
                has_regression = True

            table.add_row(
                r.category,
                f"{r.before:.1%}",
                f"{r.after:.1%}",
                f"[{color}]{r.change:+.1%}[/{color}]",
                status,
                str(r.sample_size),
            )

        console.print(table)

        if has_regression:
            console.print("\n[red bold]⚠ REGRESSIONS DETECTED[/red bold]")
            console.print("[red]Some categories got worse after training. Consider:[/red]")
            console.print("[red]  - Rolling back to an earlier checkpoint[/red]")
            console.print("[red]  - Adding more tasks for the regressed categories[/red]")
            console.print("[red]  - Reducing the learning rate[/red]")
        else:
            console.print("\n[green]✓ No regressions detected.[/green]")

    def save_results(self, results: list[RegressionResult], path: str):
        """Save regression results to disk."""
        data = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "threshold": self.threshold,
            "results": [
                {
                    "category": r.category,
                    "before": r.before,
                    "after": r.after,
                    "change": r.change,
                    "regressed": r.regressed,
                    "sample_size": r.sample_size,
                }
                for r in results
            ],
            "has_regressions": any(r.regressed for r in results),
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
