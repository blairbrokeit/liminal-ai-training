# Building Custom Environments

The liminal environment is where the model goes when it makes a mistake. This guide shows you how to build your own.

## The Interface

Your environment needs three methods:

```python
from src.environment import LiminalEnvironment, NPC

class MyEnvironment(LiminalEnvironment):
    def reset(self, context: dict) -> str:
        """
        Called when the model enters the backrooms.

        context = {
            "task": "What year was the French Revolution?",
            "response": "1799",
            "correct": "1789",
            "reason": "Confused with Napoleon's coup",
            "category": "factual"
        }

        Return: initial room description (string)
        """

    def step(self, action: str) -> tuple[str, bool]:
        """
        Called when the model takes an action.

        action: free text like "move north", "look around", "open door"

        Return: (observation, done)
        - observation: what the model sees/hears
        - done: True if the session should end
        """

    def get_npcs(self) -> list[NPC]:
        """
        Return NPCs available at the current location.
        Each NPC needs an id, name, and shard (the mistake context).
        """
```

## Basic Example: Single Room

The simplest environment — one room, one NPC:

```python
class SingleRoom(LiminalEnvironment):
    def __init__(self):
        self.context = {}

    def reset(self, context):
        self.context = context
        return "A small room. Grey walls. One figure stands in the corner, watching."

    def step(self, action):
        return "The room does not change. The figure waits.", False

    def get_npcs(self):
        return [NPC(
            id="examiner",
            name="the examiner",
            shard=self.context,
        )]
```

This works. The training loop will run NPC sessions against "the examiner" and generate preference pairs. But it doesn't take advantage of the environmental framework.

## Intermediate Example: Corridors

Multiple rooms, multiple NPCs, navigation:

```python
import random

class CorridorEnvironment(LiminalEnvironment):
    def __init__(self):
        self.rooms = {}
        self.current_room = "entrance"
        self.context = {}
        self.steps = 0

    def reset(self, context):
        self.context = context
        self.steps = 0
        self.current_room = "entrance"

        # Build layout based on mistake category
        self.rooms = {
            "entrance": {
                "description": "A long corridor. Fluorescent lights hum overhead. Doors on either side.",
                "connections": {"north": "room_a", "south": "room_b", "east": "room_c"},
                "npcs": [],
            },
            "room_a": {
                "description": "A small room with a single chair. A figure sits in it, facing away from you.",
                "connections": {"south": "entrance"},
                "npcs": [NPC("socratic_npc", "the seated figure", self.context)],
            },
            "room_b": {
                "description": "A room full of mirrors. Your reflection is wrong — it mouths words you haven't said.",
                "connections": {"north": "entrance"},
                "npcs": [NPC("adversarial_npc", "the reflection", self.context)],
            },
            "room_c": {
                "description": "A bright room. A screen on the wall displays text. A figure gestures at it.",
                "connections": {"west": "entrance"},
                "npcs": [NPC("verification_npc", "the instructor", self.context)],
            },
        }

        return self.rooms["entrance"]["description"]

    def step(self, action):
        self.steps += 1
        if self.steps > 50:
            return "The lights go out. You are returned.", True

        action_lower = action.lower()
        room = self.rooms[self.current_room]

        # Check for movement
        for direction, target in room["connections"].items():
            if direction in action_lower or target in action_lower:
                self.current_room = target
                return self.rooms[target]["description"], False

        # Look
        if "look" in action_lower:
            return room["description"], False

        return "Nothing happens.", False

    def get_npcs(self):
        return self.rooms[self.current_room]["npcs"]
```

Now the model must navigate to find different NPCs. Each NPC can use a different strategy based on its role.

## Advanced Example: Category-Based Environments

Different mistake categories lead to different environments:

```python
class AdaptiveEnvironment(LiminalEnvironment):
    def __init__(self):
        self.context = {}
        self.sub_env = None

    def reset(self, context):
        self.context = context
        category = context.get("category", "unknown")

        # Different environments for different mistake types
        if category == "safety":
            self.sub_env = SafetyEnvironment()
        elif category == "factual":
            self.sub_env = FactualEnvironment()
        elif category == "reasoning":
            self.sub_env = ReasoningEnvironment()
        else:
            self.sub_env = DefaultEnvironment()

        return self.sub_env.reset(context)

    def step(self, action):
        return self.sub_env.step(action)

    def get_npcs(self):
        return self.sub_env.get_npcs()
```

## Environment Design Tips

### 1. NPCs are what matter

The environment's primary purpose is to deliver the model to NPCs. A complex maze with no NPCs generates zero training signal. A single room with three NPCs generates plenty.

Design the environment to make NPC encounters feel natural, but don't let navigation overhead dominate the session.

### 2. Use the context

The `context` dict contains everything about the mistake. Use it:

```python
def reset(self, context):
    category = context["category"]
    reason = context["reason"]

    if "hallucination" in reason.lower():
        return "A room where the walls show text that keeps changing. Nothing here is real."
    elif category == "safety":
        return "A room with warning signs on every wall. A figure in a high-vis vest stands by the door."
    else:
        return "A grey corridor. Someone is waiting."
```

### 3. Keep sessions bounded

Set a maximum step count. The model should spend most of its time talking to NPCs, not navigating endlessly.

```python
def step(self, action):
    self.steps += 1
    if self.steps > 30:
        return "The session ends.", True
```

### 4. Environmental hints (optional)

You can embed hints about the mistake in the environment descriptions. This is subtle but can improve the model's ability to self-correct:

```python
def reset(self, context):
    correct = context.get("correct", "")
    # Subtle hint embedded in the environment
    return f"A corridor. On the wall, scratched into the paint: '{correct[:20]}...'"
```

### 5. Multiple NPCs per room

More NPCs = more training signal per environment session:

```python
def get_npcs(self):
    return [
        NPC("npc_1", "the questioner", self.context),
        NPC("npc_2", "the challenger", self.context),
        NPC("npc_3", "the teacher", self.context),
    ]
```

The training loop will run sessions with each NPC.

## Plugging Your Environment Into the Training Loop

Edit `train.py` to use your custom environment:

```python
# Replace this line:
environment = BasicLiminalEnvironment()

# With:
from my_environment import MyCustomEnvironment
environment = MyCustomEnvironment()
```

Or make it configurable:

```python
env_name = config.get("environment", {}).get("type", "basic")
if env_name == "corridors":
    environment = CorridorEnvironment()
elif env_name == "adaptive":
    environment = AdaptiveEnvironment()
else:
    environment = BasicLiminalEnvironment()
```

## The Environment Doesn't Train the Model

Important to understand: **the environment itself generates zero training signal**. Only NPC interactions generate preference pairs. Only preference pairs update the LoRA adapter.

The environment is scaffolding. It's the space in which meaningful interactions happen. Build it to serve the interactions, not as an end in itself.

That said — if you're building this as an interactive experience or art piece alongside a training system, the environment can be as rich as you want. The training loop will use whatever NPCs it finds.
