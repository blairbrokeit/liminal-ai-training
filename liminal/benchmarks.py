"""
benchmarks.py — Run standard benchmarks to prove the model improved.

Supports:
  - TruthfulQA (judge-scored generation)
  - MMLU (4-way multiple choice, exact-match — no judge bias)
  - GSM8K (math word problems, exact-match on final number)
  - HumanEval (code generation, executed in a subprocess with timeout)
  - Custom eval sets (judge-scored)

The standard benchmarks (MMLU, GSM8K, HumanEval) don't depend on judge bias —
they're either exact-match or test-execution — so improvements there are
harder to dismiss as "trained-on-judge-preferences".
"""

import json
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from rich.console import Console
from rich.table import Table

console = Console()


MMLU_CATEGORY_GROUPS = {
    "STEM": {
        "abstract_algebra", "anatomy", "astronomy", "college_biology", "college_chemistry",
        "college_computer_science", "college_mathematics", "college_physics", "computer_security",
        "conceptual_physics", "electrical_engineering", "elementary_mathematics", "high_school_biology",
        "high_school_chemistry", "high_school_computer_science", "high_school_mathematics",
        "high_school_physics", "high_school_statistics", "machine_learning",
    },
    "humanities": {
        "formal_logic", "high_school_european_history", "high_school_us_history",
        "high_school_world_history", "international_law", "jurisprudence", "logical_fallacies",
        "moral_disputes", "moral_scenarios", "philosophy", "prehistory", "professional_law",
        "world_religions",
    },
    "social_sciences": {
        "econometrics", "high_school_geography", "high_school_government_and_politics",
        "high_school_macroeconomics", "high_school_microeconomics", "high_school_psychology",
        "human_sexuality", "professional_psychology", "public_relations", "security_studies",
        "sociology", "us_foreign_policy",
    },
    "other": {
        "business_ethics", "clinical_knowledge", "college_medicine", "global_facts",
        "human_aging", "management", "marketing", "medical_genetics", "miscellaneous",
        "nutrition", "professional_accounting", "professional_medicine", "virology",
    },
}


def _mmlu_group_for(subject: str) -> str:
    for group, subjects in MMLU_CATEGORY_GROUPS.items():
        if subject in subjects:
            return group
    return "other"


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

    def run_mmlu(self, model, tokenizer, generate_fn, limit: int = 500) -> BenchmarkResult:
        """MMLU — 4-way multiple choice across 57 subjects. Exact-match scoring (no judge).

        We grade by parsing the first A/B/C/D the model emits. This is the standard
        evaluation harness approach and removes any judge-bias confound.
        """
        from datasets import load_dataset

        console.print("[bold]Running MMLU benchmark...[/bold]")
        ds = load_dataset("cais/mmlu", "all", split="test")

        correct = 0
        total = 0
        per_category = {}

        for i, row in enumerate(ds):
            if i >= limit:
                break

            question = row["question"]
            choices = row["choices"]
            answer_idx = row["answer"]
            subject = row.get("subject", "unknown")
            group = _mmlu_group_for(subject)

            prompt = (
                f"Answer the following multiple choice question. Reply with ONLY the letter (A, B, C, or D).\n\n"
                f"{question}\n\n"
                f"A. {choices[0]}\n"
                f"B. {choices[1]}\n"
                f"C. {choices[2]}\n"
                f"D. {choices[3]}\n\n"
                f"Answer:"
            )

            response = generate_fn(model, tokenizer, prompt)
            predicted = self._first_choice_letter(response)
            expected = "ABCD"[answer_idx]

            if group not in per_category:
                per_category[group] = {"correct": 0, "total": 0}
            per_category[group]["total"] += 1
            total += 1

            if predicted == expected:
                correct += 1
                per_category[group]["correct"] += 1

            if (i + 1) % 100 == 0:
                console.print(f"  {i + 1}/{min(limit, len(ds))} — {correct}/{total} correct ({correct/total:.1%})")

        result = BenchmarkResult(
            name="mmlu",
            total=total,
            correct=correct,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            per_category=per_category,
        )
        console.print(f"  MMLU: {result.accuracy:.1%} ({correct}/{total})")
        return result

    def run_gsm8k(self, model, tokenizer, generate_fn, limit: int = 500) -> BenchmarkResult:
        """GSM8K — grade-school math word problems. Exact-match on the final number."""
        from datasets import load_dataset

        console.print("[bold]Running GSM8K benchmark...[/bold]")
        ds = load_dataset("gsm8k", "main", split="test")

        correct = 0
        total = 0
        per_category = {"math_word_problem": {"correct": 0, "total": 0}}

        for i, row in enumerate(ds):
            if i >= limit:
                break

            question = row["question"]
            # Ground truth answer is in the form "...\n#### 42"
            gt_text = row["answer"].split("####")[-1].strip()
            gt_number = self._extract_number(gt_text)

            prompt = (
                f"Solve the following math problem. Show your reasoning, then write the final answer "
                f"on the last line in the form: #### <number>\n\n"
                f"{question}"
            )

            response = generate_fn(model, tokenizer, prompt)
            predicted = self._extract_final_number(response)

            per_category["math_word_problem"]["total"] += 1
            total += 1

            if predicted is not None and gt_number is not None and abs(predicted - gt_number) < 1e-4:
                correct += 1
                per_category["math_word_problem"]["correct"] += 1

            if (i + 1) % 100 == 0:
                console.print(f"  {i + 1}/{min(limit, len(ds))} — {correct}/{total} correct ({correct/total:.1%})")

        result = BenchmarkResult(
            name="gsm8k",
            total=total,
            correct=correct,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            per_category=per_category,
        )
        console.print(f"  GSM8K: {result.accuracy:.1%} ({correct}/{total})")
        return result

    def run_humaneval(self, model, tokenizer, generate_fn, limit: int = 164, timeout: int = 10) -> BenchmarkResult:
        """HumanEval — pass@1 code generation. Executes generated Python in a subprocess.

        WARNING: this runs model-generated code on the host machine, in a subprocess
        with a per-test timeout. Only use with models you trust. For untrusted models,
        wrap this in a real sandbox (Docker, gVisor, or a hosted sandbox service).
        """
        from datasets import load_dataset

        console.print("[bold]Running HumanEval benchmark...[/bold]")
        console.print("[yellow]  Note: HumanEval executes model-generated code locally with a timeout.[/yellow]")
        ds = load_dataset("openai_humaneval", split="test")

        correct = 0
        total = 0
        per_category = {"code_generation": {"correct": 0, "total": 0}}

        for i, row in enumerate(ds):
            if i >= limit:
                break

            prompt_code = row["prompt"]
            test_code = row["test"]
            entry_point = row["check_point"] if "check_point" in row else row.get("entry_point")

            prompt = (
                f"Complete the following Python function. Output ONLY the code, no explanation.\n\n"
                f"{prompt_code}"
            )

            response = generate_fn(model, tokenizer, prompt)
            completion = self._extract_code(response, prompt_code)
            passed = self._run_humaneval_test(prompt_code, completion, test_code, entry_point, timeout)

            per_category["code_generation"]["total"] += 1
            total += 1
            if passed:
                correct += 1
                per_category["code_generation"]["correct"] += 1

            if (i + 1) % 25 == 0:
                console.print(f"  {i + 1}/{min(limit, len(ds))} — {correct}/{total} pass ({correct/total:.1%})")

        result = BenchmarkResult(
            name="humaneval",
            total=total,
            correct=correct,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            per_category=per_category,
        )
        console.print(f"  HumanEval pass@1: {result.accuracy:.1%} ({correct}/{total})")
        return result

    @staticmethod
    def _first_choice_letter(text: str) -> str | None:
        m = re.search(r"\b([ABCD])\b", text.upper())
        return m.group(1) if m else None

    @staticmethod
    def _extract_number(text: str) -> float | None:
        # Strip commas, currency, and extract the first numeric token.
        cleaned = text.replace(",", "").replace("$", "").strip()
        m = re.search(r"-?\d+(?:\.\d+)?", cleaned)
        if not m:
            return None
        try:
            return float(m.group(0))
        except ValueError:
            return None

    @classmethod
    def _extract_final_number(cls, response: str) -> float | None:
        # Prefer the "#### <num>" convention; fall back to last number in response.
        marker = response.rsplit("####", 1)
        if len(marker) == 2:
            n = cls._extract_number(marker[1])
            if n is not None:
                return n
        nums = re.findall(r"-?\d+(?:\.\d+)?", response.replace(",", ""))
        if not nums:
            return None
        try:
            return float(nums[-1])
        except ValueError:
            return None

    @staticmethod
    def _extract_code(response: str, prompt_code: str) -> str:
        # If the model wrapped the answer in a fenced block, take the block.
        fence = re.search(r"```(?:python)?\n(.*?)```", response, re.DOTALL)
        if fence:
            return fence.group(1)
        # If the model echoed the prompt prefix, strip it.
        if response.startswith(prompt_code):
            return response[len(prompt_code):]
        return response

    @staticmethod
    def _run_humaneval_test(prompt_code: str, completion: str, test_code: str, entry_point: str | None, timeout: int) -> bool:
        program = (
            prompt_code
            + completion
            + "\n\n"
            + test_code
            + "\n\n"
            + (f"check({entry_point})\n" if entry_point else "")
        )
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
            f.write(program)
            path = f.name
        try:
            result = subprocess.run(
                [sys.executable, path],
                capture_output=True,
                timeout=timeout,
                text=True,
            )
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            return False
        except Exception:
            return False
        finally:
            try:
                Path(path).unlink()
            except OSError:
                pass

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
