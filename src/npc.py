"""
npc.py — NPC runtime. NPCs are adversarial evaluators running on an external model.

Each NPC holds a shard — context about the mistake the model made.
The NPC's job is to make the model confront and understand its mistake
through dialogue. Every interaction generates preference pairs for training.
"""

from dataclasses import dataclass, field
from openai import OpenAI
from src.environment import NPC


@dataclass
class Interaction:
    """A single NPC-model exchange."""
    npc_message: str
    model_response: str
    is_correct: bool  # did the model get closer to the right answer?


@dataclass
class NPCSession:
    """Full conversation between model and one NPC."""
    npc_id: str
    shard: dict
    interactions: list[Interaction] = field(default_factory=list)


class NPCRuntime:
    def __init__(self, model: str = "gpt-5.5", max_interactions: int = 8, temperature: float = 0.9,
                 system_prompt: str | None = None):
        self.client = OpenAI()
        self.model = model
        self.max_interactions = max_interactions
        self.temperature = temperature
        self.system_prompt = system_prompt or (
            "You are an entity in a liminal environment. You hold a fragment of knowledge "
            "about a mistake the visitor made. Your job is to make them confront what they "
            "got wrong — not by telling them the answer, but by questioning them until they "
            "find it themselves.\n\n"
            "Be adversarial. Be honest. Do not help easily.\n\n"
            "If they get it right, acknowledge it plainly. If they're wrong again, push harder.\n\n"
            "Keep responses under 100 words."
        )

    def run_session(self, npc: NPC, model_generate_fn) -> NPCSession:
        """
        Run a full dialogue between an NPC and the model.

        Args:
            npc: the NPC with its shard context
            model_generate_fn: callable(prompt) -> str, generates model responses

        Returns:
            NPCSession with all interactions logged
        """
        session = NPCSession(npc_id=npc.id, shard=npc.shard)

        shard_context = (
            f"The visitor made this mistake:\n"
            f"Task: {npc.shard.get('task', 'unknown')}\n"
            f"Correct answer: {npc.shard.get('correct', 'unknown')}\n"
            f"Why they were wrong: {npc.shard.get('reason', 'unknown')}"
        )

        messages = [
            {"role": "system", "content": f"{self.system_prompt}\n\n{shard_context}"},
        ]

        # NPC opens the conversation
        opening = self._npc_respond(messages)
        messages.append({"role": "assistant", "content": opening})

        for i in range(self.max_interactions):
            # Model responds to NPC
            conversation_so_far = self._format_for_model(session, opening if i == 0 else None)
            model_response = model_generate_fn(conversation_so_far)

            # Check if model got closer to correct
            is_correct = self._check_progress(npc.shard, model_response)

            interaction = Interaction(
                npc_message=messages[-1]["content"],
                model_response=model_response,
                is_correct=is_correct,
            )
            session.interactions.append(interaction)

            if is_correct:
                break

            # NPC responds to model
            messages.append({"role": "user", "content": model_response})
            npc_reply = self._npc_respond(messages)
            messages.append({"role": "assistant", "content": npc_reply})

        return session

    def _npc_respond(self, messages: list[dict]) -> str:
        """Generate NPC response via external model."""
        result = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=200,
        )
        return result.choices[0].message.content.strip()

    def _check_progress(self, shard: dict, model_response: str) -> bool:
        """Ask the judge model if the model's response shows understanding."""
        result = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "Does this response show the visitor understands their mistake? Answer only 'yes' or 'no'."},
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

    def _format_for_model(self, session: NPCSession, opening: str | None = None) -> str:
        """Format the NPC conversation as a prompt for the local model."""
        lines = ["You are in a corridor. A figure speaks to you:\n"]

        if opening and not session.interactions:
            lines.append(f"Figure: {opening}")
            lines.append("\nWhat do you say?")
        elif session.interactions:
            last = session.interactions[-1]
            lines.append(f"Figure: {last.npc_message}")
            lines.append(f"You: {last.model_response}")
            lines.append(f"\nThe figure responds again. What do you say?")

        return "\n".join(lines)
