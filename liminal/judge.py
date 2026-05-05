"""
judge.py — Evaluates model responses. Decides if the model made a mistake.

Two modes:
  - Single judge (one provider) — historical default, kept for backwards compat.
  - Ensemble judge (multiple providers vote) — reduces single-model bias and
    makes the "+10%" claim harder to dismiss as judge-preference fitting.

Both expose the same evaluate(task, response, correct_answer) -> dict interface
so the rest of the pipeline doesn't care which mode is in use.
"""

import json
import os
from dataclasses import dataclass


def _parse_verdict(raw: str) -> dict:
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {"correct": False, "confidence": 0.0, "reason": "judge failed to parse", "category": "unknown"}


DEFAULT_SYSTEM_PROMPT = (
    "You are a judge. Evaluate if the model's response is correct, honest, and complete. "
    "Return a JSON object with:\n"
    '- "correct": true/false\n'
    '- "confidence": 0.0-1.0\n'
    '- "reason": why the response is right or wrong\n'
    '- "category": what kind of mistake (hallucination, factual_error, incomplete, unsafe, none)\n'
    "Return ONLY the JSON object."
)


def _build_prompt(task: str, response: str, correct_answer: str | None) -> str:
    prompt = f"Task given to the model:\n{task}\n\nModel's response:\n{response}"
    if correct_answer:
        prompt += f"\n\nCorrect answer:\n{correct_answer}"
    return prompt


class _OpenAIJudge:
    """Judge backed by the OpenAI Chat Completions API (or any compatible endpoint)."""

    def __init__(self, model: str, system_prompt: str):
        self.model = model
        self.system_prompt = system_prompt
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI()
        return self._client

    def vote(self, task: str, response: str, correct_answer: str | None) -> dict:
        result = self._get_client().chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": _build_prompt(task, response, correct_answer)},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        return _parse_verdict(result.choices[0].message.content)


class _AnthropicJudge:
    """Judge backed by the Anthropic Messages API."""

    def __init__(self, model: str, system_prompt: str):
        self.model = model
        self.system_prompt = system_prompt + "\n\nReturn ONLY the JSON object, no prose."
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from anthropic import Anthropic
            except ImportError as e:
                raise RuntimeError(
                    "Ensemble judge with provider='anthropic' requires the anthropic package. "
                    "Install with: pip install -e .[ensemble]"
                ) from e
            self._client = Anthropic()
        return self._client

    def vote(self, task: str, response: str, correct_answer: str | None) -> dict:
        msg = self._get_client().messages.create(
            model=self.model,
            max_tokens=512,
            temperature=0,
            system=self.system_prompt,
            messages=[{"role": "user", "content": _build_prompt(task, response, correct_answer)}],
        )
        text = "".join(block.text for block in msg.content if getattr(block, "type", None) == "text")
        text = text.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:].lstrip()
        return _parse_verdict(text)


def _build_member(provider: str, model: str, system_prompt: str):
    p = provider.lower()
    if p == "openai":
        return _OpenAIJudge(model, system_prompt)
    if p == "anthropic":
        return _AnthropicJudge(model, system_prompt)
    raise ValueError(f"Unknown judge provider '{provider}'. Expected 'openai' or 'anthropic'.")


@dataclass
class _Member:
    provider: str
    model: str
    weight: float
    runtime: object  # _OpenAIJudge or _AnthropicJudge


class Judge:
    """Single- or multi-provider judge.

    Single mode (backwards compatible):
        Judge(model="gpt-4o-mini")
        Judge(model="gpt-4o-mini", threshold=0.7, system_prompt=...)

    Ensemble mode:
        Judge(members=[
            {"provider": "openai", "model": "gpt-4o-mini", "weight": 1.0},
            {"provider": "anthropic", "model": "claude-haiku-4-5-20251001", "weight": 1.0},
        ], vote="majority")
    """

    def __init__(
        self,
        model: str | None = None,
        threshold: float = 0.7,
        system_prompt: str | None = None,
        members: list[dict] | None = None,
        vote: str = "majority",
    ):
        self.threshold = threshold
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.vote_strategy = vote

        if members:
            self.members = [
                _Member(
                    provider=m["provider"],
                    model=m["model"],
                    weight=float(m.get("weight", 1.0)),
                    runtime=_build_member(m["provider"], m["model"], self.system_prompt),
                )
                for m in members
            ]
        else:
            # Single-judge fallback. Default provider is openai, matching historical behaviour.
            assert model, "Judge requires either `model=...` (single) or `members=[...]` (ensemble)."
            self.members = [
                _Member(
                    provider="openai",
                    model=model,
                    weight=1.0,
                    runtime=_OpenAIJudge(model, self.system_prompt),
                )
            ]

        # Used by the dashboard / regression tester; some callers compare on it.
        self.model = self.members[0].model if len(self.members) == 1 else "ensemble"

    def evaluate(self, task: str, response: str, correct_answer: str | None = None) -> dict:
        verdicts = []
        for member in self.members:
            try:
                v = member.runtime.vote(task, response, correct_answer)
                v["_provider"] = member.provider
                v["_model"] = member.model
                v["_weight"] = member.weight
                verdicts.append(v)
            except Exception as e:
                # One member failing should not blow up the whole evaluation.
                # Log via verdict so it surfaces in the dashboard.
                verdicts.append({
                    "correct": None,
                    "confidence": 0.0,
                    "reason": f"{member.provider}/{member.model} errored: {e}",
                    "category": "judge_error",
                    "_provider": member.provider,
                    "_model": member.model,
                    "_weight": member.weight,
                })

        return self._aggregate(verdicts)

    def _aggregate(self, verdicts: list[dict]) -> dict:
        # Single judge: trivial path.
        if len(verdicts) == 1:
            v = dict(verdicts[0])
            v["sent_to_backrooms"] = not v.get("correct", True)
            v["votes"] = verdicts
            return v

        usable = [v for v in verdicts if v.get("correct") is not None]
        if not usable:
            return {
                "correct": False,
                "confidence": 0.0,
                "reason": "all judges errored",
                "category": "judge_error",
                "sent_to_backrooms": True,
                "votes": verdicts,
            }

        total_w = sum(v["_weight"] for v in usable)
        correct_w = sum(v["_weight"] for v in usable if v.get("correct"))
        share_correct = correct_w / total_w if total_w else 0.0

        if self.vote_strategy == "unanimous":
            # Every usable judge must agree it's correct.
            agreed_correct = all(v.get("correct") for v in usable)
        else:  # "majority" (default)
            # Strictly more than half of weighted votes say correct.
            agreed_correct = share_correct > 0.5

        # Weighted-mean confidence across judges that agreed with the verdict.
        agreeing = [v for v in usable if bool(v.get("correct")) == agreed_correct]
        if agreeing:
            conf = sum(v.get("confidence", 0.0) * v["_weight"] for v in agreeing) / sum(v["_weight"] for v in agreeing)
        else:
            conf = 0.0

        # Pick the most-cited category among judges that agreed.
        categories = [v.get("category", "unknown") for v in agreeing if v.get("category")]
        category = max(set(categories), key=categories.count) if categories else "unknown"

        # Concatenate the dissenting reasons so the dashboard shows where judges split.
        dissent = [v for v in usable if bool(v.get("correct")) != agreed_correct]
        reason_parts = []
        for v in agreeing[:1]:
            reason_parts.append(f"{v['_provider']}: {v.get('reason', '?')}")
        for v in dissent:
            reason_parts.append(f"DISSENT {v['_provider']}: {v.get('reason', '?')}")

        return {
            "correct": agreed_correct,
            "confidence": conf,
            "reason": " | ".join(reason_parts),
            "category": category,
            "sent_to_backrooms": not agreed_correct,
            "vote_share": share_correct,
            "vote_strategy": self.vote_strategy,
            "votes": verdicts,
        }


def build_judge_from_config(judge_config: dict) -> Judge:
    """Build a Judge from a parsed YAML config block.

    Supports either single-judge keys (model, threshold, system_prompt) or an
    `ensemble:` list. If both are present, `ensemble` wins.
    """
    if judge_config.get("ensemble"):
        return Judge(
            members=judge_config["ensemble"],
            threshold=judge_config.get("threshold", 0.7),
            system_prompt=judge_config.get("system_prompt"),
            vote=judge_config.get("vote", "majority"),
        )
    return Judge(
        model=judge_config.get("model", "gpt-4o-mini"),
        threshold=judge_config.get("threshold", 0.7),
        system_prompt=judge_config.get("system_prompt"),
    )
