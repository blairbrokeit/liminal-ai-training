"""
metrics.py — Training metrics tracker with logging and reporting.

Tracks everything needed to prove the model is actually improving:
per-loop accuracy, per-category accuracy, loss curves, pair counts,
before/after comparisons.
"""

import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()


@dataclass
class LoopMetrics:
    loop: int = 0
    tasks_attempted: int = 0
    tasks_correct: int = 0
    tasks_wrong: int = 0
    pairs_generated: int = 0
    training_loss: float | None = None
    category_results: dict = field(default_factory=lambda: defaultdict(lambda: {"correct": 0, "total": 0}))
    timestamp: str = ""

    @property
    def accuracy(self) -> float:
        return self.tasks_correct / self.tasks_attempted if self.tasks_attempted > 0 else 0.0


class MetricsTracker:
    def __init__(self, output_dir: str = "./metrics"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.loops: list[LoopMetrics] = []
        self.current: LoopMetrics | None = None
        self.baseline_accuracy: float | None = None
        self.start_time = time.time()

    def begin_loop(self, loop_number: int):
        """Start tracking a new loop."""
        self.current = LoopMetrics(
            loop=loop_number,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )

    def record_task(self, category: str, correct: bool):
        """Record a single task result."""
        self.current.tasks_attempted += 1
        if correct:
            self.current.tasks_correct += 1
        else:
            self.current.tasks_wrong += 1

        cat_stats = self.current.category_results[category]
        cat_stats["total"] += 1
        if correct:
            cat_stats["correct"] += 1

    def record_pairs(self, count: int):
        """Record preference pairs generated."""
        self.current.pairs_generated += count

    def record_loss(self, loss: float):
        """Record training loss for this loop."""
        self.current.training_loss = loss

    def end_loop(self):
        """Finalize and store the current loop's metrics."""
        if self.current:
            # Convert defaultdict to regular dict for serialization
            self.current.category_results = dict(self.current.category_results)
            self.loops.append(self.current)
            self.current = None

    def set_baseline(self, accuracy: float):
        """Set the baseline accuracy (before any training)."""
        self.baseline_accuracy = accuracy

    def get_improvement(self) -> float | None:
        """Get accuracy improvement over baseline."""
        if self.baseline_accuracy is None or not self.loops:
            return None
        latest = self.loops[-1].accuracy
        return latest - self.baseline_accuracy

    def get_accuracy_trend(self, window: int = 5) -> list[float]:
        """Get rolling average accuracy over last N loops."""
        accuracies = [l.accuracy for l in self.loops]
        if len(accuracies) < window:
            return accuracies

        trend = []
        for i in range(len(accuracies) - window + 1):
            avg = sum(accuracies[i:i + window]) / window
            trend.append(round(avg, 3))
        return trend

    def get_category_improvement(self) -> dict[str, dict]:
        """Compare per-category accuracy between first and last loops."""
        if len(self.loops) < 2:
            return {}

        first = self.loops[0].category_results
        last = self.loops[-1].category_results

        all_cats = set(list(first.keys()) + list(last.keys()))
        result = {}
        for cat in sorted(all_cats):
            first_acc = first.get(cat, {}).get("correct", 0) / max(1, first.get(cat, {}).get("total", 1))
            last_acc = last.get(cat, {}).get("correct", 0) / max(1, last.get(cat, {}).get("total", 1))
            result[cat] = {
                "first": round(first_acc, 3),
                "latest": round(last_acc, 3),
                "change": round(last_acc - first_acc, 3),
            }

        return result

    def print_loop_summary(self):
        """Print a summary of the most recent loop."""
        if not self.loops:
            return

        loop = self.loops[-1]
        console.print(f"\n  Loop {loop.loop}: {loop.accuracy:.1%} accuracy "
                      f"({loop.tasks_correct}/{loop.tasks_attempted}), "
                      f"{loop.pairs_generated} pairs, "
                      f"loss={loop.training_loss or 'n/a'}")

    def print_progress_report(self):
        """Print a full progress report."""
        if not self.loops:
            console.print("[dim]No loops completed yet.[/dim]")
            return

        elapsed = time.time() - self.start_time

        # Overall table
        table = Table(title="Training Progress")
        table.add_column("Metric", style="bold")
        table.add_column("Value")

        table.add_row("Loops completed", str(len(self.loops)))
        table.add_row("Elapsed time", f"{elapsed / 60:.1f} min")

        if self.baseline_accuracy is not None:
            table.add_row("Baseline accuracy", f"{self.baseline_accuracy:.1%}")

        latest = self.loops[-1]
        table.add_row("Latest accuracy", f"{latest.accuracy:.1%}")

        improvement = self.get_improvement()
        if improvement is not None:
            color = "green" if improvement > 0 else "red"
            table.add_row("Improvement", f"[{color}]{improvement:+.1%}[/{color}]")

        total_pairs = sum(l.pairs_generated for l in self.loops)
        table.add_row("Total pairs generated", str(total_pairs))

        losses = [l.training_loss for l in self.loops if l.training_loss is not None]
        if losses:
            table.add_row("Latest loss", f"{losses[-1]:.4f}")
            table.add_row("Loss trend", f"{losses[0]:.4f} → {losses[-1]:.4f}")

        console.print(table)

        # Category breakdown
        cat_improvement = self.get_category_improvement()
        if cat_improvement:
            cat_table = Table(title="Per-Category Improvement")
            cat_table.add_column("Category")
            cat_table.add_column("First Loop")
            cat_table.add_column("Latest Loop")
            cat_table.add_column("Change")

            for cat, data in cat_improvement.items():
                change = data["change"]
                color = "green" if change > 0 else ("red" if change < 0 else "dim")
                cat_table.add_row(
                    cat,
                    f"{data['first']:.1%}",
                    f"{data['latest']:.1%}",
                    f"[{color}]{change:+.1%}[/{color}]",
                )

            console.print(cat_table)

    def save(self):
        """Save all metrics to disk."""
        data = {
            "baseline_accuracy": self.baseline_accuracy,
            "total_loops": len(self.loops),
            "loops": [
                {
                    "loop": l.loop,
                    "timestamp": l.timestamp,
                    "tasks_attempted": l.tasks_attempted,
                    "tasks_correct": l.tasks_correct,
                    "accuracy": round(l.accuracy, 4),
                    "pairs_generated": l.pairs_generated,
                    "training_loss": l.training_loss,
                    "category_results": l.category_results,
                }
                for l in self.loops
            ],
            "category_improvement": self.get_category_improvement(),
            "accuracy_trend": self.get_accuracy_trend(),
        }

        with open(self.output_dir / "training_metrics.json", "w") as f:
            json.dump(data, f, indent=2)

        # Also save a simple CSV for easy plotting
        with open(self.output_dir / "accuracy.csv", "w") as f:
            f.write("loop,accuracy,loss,pairs\n")
            for l in self.loops:
                f.write(f"{l.loop},{l.accuracy:.4f},{l.training_loss or ''},{l.pairs_generated}\n")
