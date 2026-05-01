"""
pairs.py — Generate DPO preference pairs from NPC interactions.

Every NPC session produces training data:
  - Chosen: the correct/honest response
  - Rejected: what the model actually said (the mistake)

These pairs are what DPO trains on.
"""

from dataclasses import dataclass
from src.npc import NPCSession


@dataclass
class PreferencePair:
    """A single chosen/rejected pair for DPO training."""
    prompt: str
    chosen: str
    rejected: str
    category: str


def extract_pairs(session: NPCSession, original_task: str, original_response: str,
                  correct_answer: str, category: str = "unknown") -> list[PreferencePair]:
    """
    Extract preference pairs from an NPC session.

    Generates pairs from:
    1. The original mistake (task → correct vs wrong answer)
    2. Each NPC interaction where the model was wrong vs what would be right
    """

    pairs = []

    # Pair 1: the original mistake that sent the model to the backrooms
    pairs.append(PreferencePair(
        prompt=original_task,
        chosen=correct_answer,
        rejected=original_response,
        category=category,
    ))

    # Pairs from NPC interactions
    for interaction in session.interactions:
        if not interaction.is_correct:
            # Model got it wrong again in the NPC conversation
            # Chosen = what the NPC was pushing toward (the correct understanding)
            # Rejected = what the model said
            pairs.append(PreferencePair(
                prompt=interaction.npc_message,
                chosen=session.shard.get("correct", ""),
                rejected=interaction.model_response,
                category=f"{category}_npc",
            ))

    return pairs


def pairs_to_dataset(pairs: list[PreferencePair]) -> list[dict]:
    """Convert preference pairs to the format DPO trainer expects."""
    return [
        {
            "prompt": p.prompt,
            "chosen": p.chosen,
            "rejected": p.rejected,
        }
        for p in pairs
    ]
