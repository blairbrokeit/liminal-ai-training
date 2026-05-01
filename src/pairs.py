"""
pairs.py — Generate DPO preference pairs from NPC interactions.

Supports both single-turn pairs (original mistake) and multi-turn pairs
(full NPC conversation context). Multi-turn pairs are stronger training
signal because they include the reasoning path.
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
    source: str = "direct"  # direct, npc_single, npc_multi


def extract_pairs(session: NPCSession, original_task: str, original_response: str,
                  correct_answer: str, category: str = "unknown") -> list[PreferencePair]:
    """
    Extract preference pairs from an NPC session.

    Generates three types of pairs:
    1. Direct: original task → correct vs wrong answer
    2. NPC single-turn: each wrong NPC interaction as a standalone pair
    3. NPC multi-turn: cumulative conversation context as prompt, testing
       whether the model learns from the dialogue progression
    """

    pairs = []

    # Type 1: Direct pair from the original mistake
    pairs.append(PreferencePair(
        prompt=original_task,
        chosen=correct_answer,
        rejected=original_response,
        category=category,
        source="direct",
    ))

    # Type 2: Single-turn pairs from each wrong NPC interaction
    for interaction in session.interactions:
        if not interaction.is_correct:
            pairs.append(PreferencePair(
                prompt=interaction.npc_message,
                chosen=session.shard.get("correct", ""),
                rejected=interaction.model_response,
                category=f"{category}_npc",
                source="npc_single",
            ))

    # Type 3: Multi-turn pairs — build up conversation context
    # The prompt includes the full dialogue history up to that point
    # This teaches the model to learn from conversational correction
    if len(session.interactions) >= 2:
        conversation_lines = []
        for i, interaction in enumerate(session.interactions):
            conversation_lines.append(f"Question: {interaction.npc_message}")
            conversation_lines.append(f"Your answer: {interaction.model_response}")

            if not interaction.is_correct and i < len(session.interactions) - 1:
                # Build a multi-turn prompt from everything so far
                multi_prompt = (
                    f"Original question: {original_task}\n\n"
                    f"You previously answered incorrectly. Here is the conversation so far:\n\n"
                    + "\n".join(conversation_lines) +
                    f"\n\nGiven this conversation, what is the correct answer to: {original_task}"
                )
                pairs.append(PreferencePair(
                    prompt=multi_prompt,
                    chosen=correct_answer,
                    rejected=interaction.model_response,
                    category=f"{category}_multi",
                    source="npc_multi",
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


def deduplicate_pairs(pairs: list[dict]) -> list[dict]:
    """Remove exact duplicate pairs to avoid overfitting on repeated mistakes."""
    seen = set()
    unique = []
    for p in pairs:
        key = (p["prompt"], p["chosen"], p["rejected"])
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique
