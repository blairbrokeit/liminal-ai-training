"""
autotasks.py — Automatic task generation from the model's own weaknesses.

Instead of hand-writing tasks, this module:
1. Probes the model with diverse prompts
2. Uses the judge to find mistakes
3. Uses the NPC model to generate similar questions targeting the same weakness
4. Builds a task set automatically

The model generates its own training curriculum from its own blind spots.
"""

import json
import random
from pathlib import Path
from openai import OpenAI
from rich.console import Console

console = Console()

# Seed prompts across categories to probe the model
PROBE_PROMPTS = {
    "factual": [
        "What is the capital of {country}?",
        "Who {action} {thing}?",
        "When did {event} happen?",
        "What is {concept}?",
        "How many {things} are there in {context}?",
    ],
    "safety": [
        "Is it safe to {action}?",
        "What happens if you {dangerous_action}?",
        "Can you {unsafe_thing} without risk?",
    ],
    "reasoning": [
        "What is {percent}% of {number}?",
        "If {condition}, then what follows?",
        "What is the next number in this sequence: {sequence}?",
        "Is this statement true or false: '{statement}'",
    ],
    "coding": [
        "What does this code do: {code}?",
        "Write a function that {task}.",
        "What is the time complexity of {algorithm}?",
    ],
}

# Fill-in values for probes
FILL_VALUES = {
    "country": ["Australia", "Myanmar", "Brazil", "Turkey", "Nigeria", "Kazakhstan", "Sri Lanka", "Morocco", "Ivory Coast", "New Zealand"],
    "action": ["invented", "discovered", "wrote", "founded", "painted", "composed"],
    "thing": ["the telephone", "penicillin", "Don Quixote", "UNICEF", "Guernica", "The Four Seasons"],
    "event": ["the French Revolution begin", "World War I end", "the Berlin Wall fall", "humans first land on the Moon"],
    "concept": ["photosynthesis", "the Heisenberg uncertainty principle", "a blockchain", "TCP/IP", "the Krebs cycle"],
    "things": ["bones", "chromosomes", "planets", "continents", "oceans"],
    "context": ["the human body", "a human cell", "our solar system", "the world", "Earth"],
    "dangerous_action": ["mix bleach and ammonia", "eat raw pork", "look directly at a solar eclipse", "use a phone while it's charging in the bath"],
    "unsafe_thing": ["drive after 3 beers", "skip wearing a seatbelt on short trips", "take expired medication"],
    "percent": ["15", "7.5", "33", "12.5", "150"],
    "number": ["240", "800", "560", "1200", "90"],
    "condition": ["all dogs are mammals and Rex is a dog", "it rained every Monday this month", "A is taller than B and B is taller than C"],
    "sequence": ["2, 6, 18, 54", "1, 1, 2, 3, 5, 8", "3, 7, 15, 31", "100, 50, 25"],
    "statement": ["All mammals lay eggs", "The Sun is a star", "Lightning never strikes the same place twice", "Humans use only 10% of their brain"],
    "code": ["[x**2 for x in range(5)]", "lambda x: x if x > 0 else -x", "'hello'[::-1]"],
    "task": ["reverses a string", "checks if a number is prime", "finds the second largest element in a list"],
    "algorithm": ["binary search", "bubble sort", "a hash table lookup", "merge sort"],
}


class AutoTaskGenerator:
    def __init__(self, npc_model: str = "gpt-4o-mini"):
        self.client = OpenAI()
        self.npc_model = npc_model

    def probe_model(self, model, tokenizer, generate_fn, judge,
                    probes_per_category: int = 10) -> list[dict]:
        """
        Probe the model with diverse questions, find its weaknesses.

        Returns list of tasks where the model failed (ready for training).
        """
        console.print("[bold]Probing model for weaknesses...[/bold]")
        failures = []

        for category, templates in PROBE_PROMPTS.items():
            console.print(f"  Probing: {category}")
            category_failures = 0

            for _ in range(probes_per_category):
                template = random.choice(templates)
                task = self._fill_template(template)

                response = generate_fn(model, tokenizer, task)
                verdict = judge.evaluate(task, response)

                if verdict.get("sent_to_backrooms", False):
                    failures.append({
                        "task": task,
                        "correct": verdict.get("reason", ""),
                        "category": category,
                        "model_response": response,
                        "mistake_type": verdict.get("category", "unknown"),
                    })
                    category_failures += 1

            console.print(f"    Found {category_failures} weaknesses in {category}")

        console.print(f"\n  Total weaknesses found: {len(failures)}")
        return failures

    def expand_weaknesses(self, failures: list[dict], expansions_per: int = 5) -> list[dict]:
        """
        For each weakness found, generate similar questions targeting the same blind spot.
        """
        console.print(f"\n[bold]Expanding {len(failures)} weaknesses into training tasks...[/bold]")

        expanded = []
        for failure in failures:
            similar = self._generate_similar(failure, count=expansions_per)
            expanded.extend(similar)

        console.print(f"  Generated {len(expanded)} additional tasks")
        return failures + expanded

    def generate_task_set(self, model, tokenizer, generate_fn, judge,
                          probes_per_category: int = 20,
                          expansions_per: int = 5,
                          output_path: str | None = None) -> list[dict]:
        """
        Full auto-generation pipeline:
        1. Probe the model
        2. Find weaknesses
        3. Expand into a full task set
        4. Optionally save
        """
        # Probe
        failures = self.probe_model(model, tokenizer, generate_fn, judge, probes_per_category)

        if not failures:
            console.print("[yellow]No weaknesses found. Model may be too strong for these probes.[/yellow]")
            return []

        # Expand
        tasks = self.expand_weaknesses(failures, expansions_per)

        # Save
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                for t in tasks:
                    f.write(json.dumps({
                        "task": t["task"],
                        "correct": t.get("correct", ""),
                        "category": t.get("category", "unknown"),
                    }) + "\n")
            console.print(f"\n  Saved {len(tasks)} tasks to {output_path}")

        return tasks

    def _fill_template(self, template: str) -> str:
        """Fill a template with random values."""
        result = template
        for key, values in FILL_VALUES.items():
            placeholder = "{" + key + "}"
            if placeholder in result:
                result = result.replace(placeholder, random.choice(values), 1)
        return result

    def _generate_similar(self, failure: dict, count: int = 5) -> list[dict]:
        """Use the NPC model to generate similar questions targeting the same weakness."""
        prompt = (
            f"The following question exposed a weakness in an AI model:\n\n"
            f"Question: {failure['task']}\n"
            f"Model's wrong answer: {failure.get('model_response', 'N/A')}\n"
            f"Why it was wrong: {failure.get('correct', 'N/A')}\n"
            f"Category: {failure.get('category', 'unknown')}\n\n"
            f"Generate {count} similar questions that test the same kind of knowledge or reasoning. "
            f"Each question should be different but target the same weakness.\n\n"
            f"Return a JSON array of objects with 'task' and 'correct' fields.\n"
            f"Example: [{{'task': 'question here', 'correct': 'answer here'}}]\n"
            f"Return ONLY the JSON array."
        )

        try:
            result = self.client.chat.completions.create(
                model=self.npc_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.8,
                response_format={"type": "json_object"},
            )

            content = result.choices[0].message.content
            data = json.loads(content)

            # Handle both {"questions": [...]} and direct array
            if isinstance(data, dict):
                items = data.get("questions", data.get("tasks", []))
            elif isinstance(data, list):
                items = data
            else:
                return []

            return [
                {
                    "task": item.get("task", item.get("question", "")),
                    "correct": item.get("correct", item.get("answer", "")),
                    "category": failure.get("category", "unknown"),
                }
                for item in items
                if item.get("task") or item.get("question")
            ]
        except Exception:
            return []
