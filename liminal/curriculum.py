"""
curriculum.py — Adaptive curriculum that focuses training on the model's weakest areas.

Tracks per-category accuracy across loops. Automatically increases sampling weight
for categories the model struggles with, decreases for categories it's mastered.
Difficulty ramps over time.
"""

import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class CategoryStats:
    attempts: int = 0
    correct: int = 0
    streak: int = 0  # consecutive correct
    last_loss: float = 0.0

    @property
    def accuracy(self) -> float:
        return self.correct / self.attempts if self.attempts > 0 else 0.0

    @property
    def weight(self) -> float:
        """Higher weight = model is worse at this = sample more."""
        if self.attempts < 3:
            return 1.0  # not enough data, neutral weight
        error_rate = 1.0 - self.accuracy
        recency_bonus = 1.0 / (1.0 + self.streak)  # penalize long correct streaks
        return max(0.1, error_rate * 2.0 * recency_bonus)


class Curriculum:
    def __init__(self, tasks: list[dict], val_split: float = 0.2, seed: int = 42):
        """
        Split tasks into train/val sets and manage adaptive sampling.

        Args:
            tasks: list of task dicts with 'task', 'correct', 'category'
            val_split: fraction held out for validation (never trained on)
            seed: random seed for reproducible splits
        """
        self.rng = random.Random(seed)
        self.stats: dict[str, CategoryStats] = defaultdict(CategoryStats)
        self.loop_number = 0
        self.history: list[dict] = []

        # Split tasks
        shuffled = list(tasks)
        self.rng.shuffle(shuffled)
        split_idx = max(1, int(len(shuffled) * (1 - val_split)))
        self.train_tasks = shuffled[:split_idx]
        self.val_tasks = shuffled[split_idx:]

        # Index by category
        self.categories: dict[str, list[dict]] = defaultdict(list)
        for t in self.train_tasks:
            self.categories[t.get("category", "unknown")].append(t)

    def sample_batch(self, batch_size: int | None = None) -> list[dict]:
        """
        Sample a batch of training tasks weighted by category difficulty.
        Categories the model is worse at get sampled more.
        """
        if batch_size is None:
            batch_size = len(self.train_tasks)

        # Build weighted category list
        cat_weights = {}
        for cat in self.categories:
            cat_weights[cat] = self.stats[cat].weight

        # Normalize weights
        total_weight = sum(cat_weights.values())
        if total_weight == 0:
            total_weight = 1.0

        batch = []
        for _ in range(batch_size):
            # Weighted category selection
            r = self.rng.random() * total_weight
            cumulative = 0.0
            chosen_cat = list(self.categories.keys())[0]
            for cat, w in cat_weights.items():
                cumulative += w
                if r <= cumulative:
                    chosen_cat = cat
                    break

            # Random task from chosen category
            task = self.rng.choice(self.categories[chosen_cat])
            batch.append(task)

        return batch

    def get_validation_tasks(self) -> list[dict]:
        """Return held-out validation set. Never train on these."""
        return self.val_tasks

    def record(self, task: str, category: str, correct: bool, loss: float = 0.0):
        """Record a task result for curriculum adaptation."""
        stats = self.stats[category]
        stats.attempts += 1
        if correct:
            stats.correct += 1
            stats.streak += 1
        else:
            stats.streak = 0
        stats.last_loss = loss

    def advance_loop(self):
        """Called at the end of each training loop."""
        self.loop_number += 1

        snapshot = {
            "loop": self.loop_number,
            "categories": {},
        }
        for cat, stats in self.stats.items():
            snapshot["categories"][cat] = {
                "accuracy": round(stats.accuracy, 3),
                "attempts": stats.attempts,
                "weight": round(stats.weight, 3),
                "streak": stats.streak,
            }
        self.history.append(snapshot)

    def get_weakest_categories(self, n: int = 3) -> list[tuple[str, float]]:
        """Return the N categories with lowest accuracy (min 3 attempts)."""
        scored = [
            (cat, stats.accuracy)
            for cat, stats in self.stats.items()
            if stats.attempts >= 3
        ]
        scored.sort(key=lambda x: x[1])
        return scored[:n]

    def get_report(self) -> dict:
        """Full curriculum status report."""
        return {
            "loop": self.loop_number,
            "train_tasks": len(self.train_tasks),
            "val_tasks": len(self.val_tasks),
            "categories": {
                cat: {
                    "accuracy": round(stats.accuracy, 3),
                    "attempts": stats.attempts,
                    "correct": stats.correct,
                    "weight": round(stats.weight, 3),
                    "streak": stats.streak,
                }
                for cat, stats in sorted(self.stats.items())
            },
            "weakest": self.get_weakest_categories(),
        }

    def save(self, path: str):
        """Save curriculum state to disk."""
        data = {
            "loop_number": self.loop_number,
            "stats": {
                cat: {
                    "attempts": s.attempts,
                    "correct": s.correct,
                    "streak": s.streak,
                    "last_loss": s.last_loss,
                }
                for cat, s in self.stats.items()
            },
            "history": self.history,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path: str):
        """Load curriculum state from disk."""
        with open(path) as f:
            data = json.load(f)
        self.loop_number = data.get("loop_number", 0)
        self.history = data.get("history", [])
        for cat, s in data.get("stats", {}).items():
            self.stats[cat] = CategoryStats(
                attempts=s["attempts"],
                correct=s["correct"],
                streak=s["streak"],
                last_loss=s.get("last_loss", 0.0),
            )
