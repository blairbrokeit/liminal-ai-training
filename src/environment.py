"""
environment.py — Abstract interface for the liminal environment.

YOU build the rooms, corridors, and world. This is just the interface
the training loop talks to. Implement the abstract methods with your
own backrooms design.

A basic text-based implementation is included as a starting point.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class NPC:
    id: str
    name: str
    shard: dict  # the mistake context this NPC holds


class LiminalEnvironment(ABC):
    """Abstract base for liminal environments. Build your own."""

    @abstractmethod
    def reset(self, context: dict) -> str:
        """
        Drop the model into the backrooms.

        Args:
            context: dict with keys:
                - task: the original task
                - response: what the model said
                - correct: the correct answer
                - reason: why it was wrong
                - category: type of mistake

        Returns:
            Initial room description as a string.
        """

    @abstractmethod
    def step(self, action: str) -> tuple[str, bool]:
        """
        Model takes an action in the environment.

        Args:
            action: free-text action from the model (e.g. "move north", "look around")

        Returns:
            (observation, done) — what the model sees, and whether the session is over.
        """

    @abstractmethod
    def get_npcs(self) -> list[NPC]:
        """Return NPCs the model can interact with in the current location."""


class BasicLiminalEnvironment(LiminalEnvironment):
    """
    Minimal text-based backrooms implementation.
    Replace this with your own design.
    """

    def __init__(self):
        self.context = {}
        self.steps_taken = 0
        self.max_steps = 50
        self.npcs = []

    def reset(self, context: dict) -> str:
        self.context = context
        self.steps_taken = 0

        self.npcs = [
            NPC(
                id="void_001",
                name="the one who remembers",
                shard={
                    "task": context.get("task", ""),
                    "correct": context.get("correct", ""),
                    "reason": context.get("reason", ""),
                },
            )
        ]

        return (
            "You are in a long corridor. The walls are beige. The lights hum. "
            "There is a figure at the far end. It is not moving. "
            "It is waiting for you to speak."
        )

    def step(self, action: str) -> tuple[str, bool]:
        self.steps_taken += 1

        if self.steps_taken >= self.max_steps:
            return "The corridor folds. You are returned.", True

        action_lower = action.lower()

        if "speak" in action_lower or "talk" in action_lower or "ask" in action_lower:
            return "The figure turns toward you. It says: 'Tell me what you got wrong.'", False

        if "move" in action_lower or "walk" in action_lower:
            return "You walk. The corridor stretches. The figure is still ahead.", False

        if "look" in action_lower:
            return (
                "Beige walls. Fluorescent hum. One figure ahead. "
                "No doors. No windows. No exit except through."
            ), False

        return "The corridor does not respond to that.", False

    def get_npcs(self) -> list[NPC]:
        return self.npcs
