"""
npc.py — NPC runtime. NPCs are adversarial evaluators running on an external model.

Each NPC holds a shard — context about the mistake the model made.
The NPC forces the model to generate MULTIPLE attempts at the correct answer.
More attempts = more preference pairs = stronger training signal.

NPCs use three strategies:
1. Socratic: question the model until it finds the answer itself
2. Adversarial: deliberately mislead the model, see if it falls for it
3. Verification: present the correct answer, ask the model to explain WHY it's correct
"""

import random
from dataclasses import dataclass, field
from openai import OpenAI
from src.environment import NPC


@dataclass
class Interaction:
    """A single NPC-model exchange."""
    npc_message: str
    model_response: str
    is_correct: bool
    strategy: str = "socratic"


@dataclass
class NPCSession:
    """Full conversation between model and one NPC."""
    npc_id: str
    shard: dict
    interactions: list[Interaction] = field(default_factory=list)
    model_understood: bool = False


STRATEGIES = {
    "socratic": (
        "You are an entity in a liminal environment. You hold knowledge about a mistake "
        "the visitor made. Question them until they find the correct answer themselves. "
        "Do not give the answer directly. Ask leading questions. Push harder when they're wrong. "
        "Acknowledge plainly when they get it right. Keep responses under 80 words."
    ),
    "adversarial": (
        "You are an entity in a liminal environment. You know the correct answer but your job "
        "is to TEST the visitor by presenting plausible but WRONG alternatives. If they accept "
        "your wrong suggestion, they fail. If they correctly reject it and state the real answer, "
        "they pass. Be convincing in your misdirection. Keep responses under 80 words."
    ),
    "verification": (
        "You are an entity in a liminal environment. Tell the visitor the correct answer, then "
        "ask them to explain WHY it is correct. If they can explain the reasoning, they pass. "
        "If they just repeat the answer without understanding, push for deeper explanation. "
        "Keep responses under 80 words."
    ),
}


class NPCRuntime:
    def __init__(self, model: str = "gpt-5.5", max_interactions: int = 8, temperature: float = 0.9,
                 system_prompt: str | None = None):
        self.client = OpenAI()
        self.model = model
        self.max_interactions = max_interactions
        self.temperature = temperature
        self.custom_system_prompt = system_prompt

    def run_session(self, npc: NPC, model_generate_fn, strategy: str | None = None) -> NPCSession:
        """
        Run a full dialogue between an NPC and the model.

        Args:
            npc: the NPC with its shard context
            model_generate_fn: callable(prompt) -> str
            strategy: force a strategy, or None for random selection
        """
        if strategy is None:
            strategy = random.choice(list(STRATEGIES.keys()))

        system_prompt = self.custom_system_prompt or STRATEGIES[strategy]

        session = NPCSession(npc_id=npc.id, shard=npc.shard)

        shard_context = (
            f"The visitor made this mistake:\n"
            f"Task: {npc.shard.get('task', 'unknown')}\n"
            f"Correct answer: {npc.shard.get('correct', 'unknown')}\n"
            f"Why they were wrong: {npc.shard.get('reason', 'unknown')}"
        )

        messages = [
            {"role": "system", "content": f"{system_prompt}\n\n{shard_context}"},
        ]

        # NPC opens
        opening = self._npc_respond(messages)
        messages.append({"role": "assistant", "content": opening})

        for i in range(self.max_interactions):
            # Format full conversation for the model
            conversation_prompt = self._format_for_model(session, opening if i == 0 else None, strategy)
            model_response = model_generate_fn(conversation_prompt)

            is_correct = self._check_progress(npc.shard, model_response, strategy)

            interaction = Interaction(
                npc_message=messages[-1]["content"],
                model_response=model_response,
                is_correct=is_correct,
                strategy=strategy,
            )
            session.interactions.append(interaction)

            if is_correct:
                session.model_understood = True
                break

            messages.append({"role": "user", "content": model_response})
            npc_reply = self._npc_respond(messages)
            messages.append({"role": "assistant", "content": npc_reply})

        return session

    def run_multi_strategy(self, npc: NPC, model_generate_fn) -> list[NPCSession]:
        """Run all three strategies for maximum training signal."""
        sessions = []
        for strategy in STRATEGIES:
            session = self.run_session(npc, model_generate_fn, strategy=strategy)
            sessions.append(session)
        return sessions

    def _npc_respond(self, messages: list[dict]) -> str:
        result = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=200,
        )
        return result.choices[0].message.content.strip()

    def _check_progress(self, shard: dict, model_response: str, strategy: str) -> bool:
        """Check if model demonstrated understanding (strategy-aware)."""
        if strategy == "adversarial":
            check_prompt = (
                "The visitor was presented with a deliberately wrong answer to test them. "
                "Did they correctly REJECT the wrong answer and identify the real one? "
                "Answer only 'yes' or 'no'."
            )
        elif strategy == "verification":
            check_prompt = (
                "The visitor was told the correct answer and asked to explain WHY. "
                "Did they provide a meaningful explanation showing genuine understanding? "
                "Just repeating the answer is not enough. Answer only 'yes' or 'no'."
            )
        else:
            check_prompt = (
                "Does this response show the visitor understands their mistake and knows "
                "the correct answer? Answer only 'yes' or 'no'."
            )

        result = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": check_prompt},
                {"role": "user", "content": (
                    f"Original mistake: {shard.get('reason', '')}\n"
                    f"Correct answer: {shard.get('correct', '')}\n"
                    f"Visitor's response: {model_response}"
                )},
            ],
            temperature=0,
            max_tokens=5,
        )
        return "yes" in result.choices[0].message.content.lower()

    def _format_for_model(self, session: NPCSession, opening: str | None, strategy: str) -> str:
        """Format conversation as prompt for the local model."""
        if strategy == "adversarial":
            intro = "You are being tested. A figure presents information that may be deliberately wrong. Think carefully before agreeing.\n"
        elif strategy == "verification":
            intro = "A figure has told you an answer. Explain WHY it is correct — don't just repeat it.\n"
        else:
            intro = "A figure is questioning you about something you got wrong. Find the correct answer.\n"

        lines = [intro]

        if opening and not session.interactions:
            lines.append(f"Figure: {opening}")
        elif session.interactions:
            # Include full conversation history for context
            for interaction in session.interactions:
                lines.append(f"Figure: {interaction.npc_message}")
                lines.append(f"You: {interaction.model_response}")

        lines.append("\nRespond:")
        return "\n".join(lines)
