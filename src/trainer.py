"""
trainer.py — DPO training step on the LoRA adapter.

Takes preference pairs from NPC sessions and updates the adapter.
This is where the model actually learns from the backrooms.
"""

from pathlib import Path
from datasets import Dataset
from trl import DPOConfig, DPOTrainer
from transformers import AutoTokenizer


def train_step(model, tokenizer: AutoTokenizer, pairs: list[dict], config: dict,
               adapter_save_path: str) -> dict:
    """
    Run a DPO training step on the LoRA adapter.

    Args:
        model: the PEFT model (base + LoRA)
        tokenizer: tokenizer
        pairs: list of {"prompt": ..., "chosen": ..., "rejected": ...}
        config: training config dict
        adapter_save_path: where to save the updated adapter

    Returns:
        dict with training metrics
    """

    if not pairs:
        return {"status": "no_pairs", "loss": None}

    dataset = Dataset.from_list(pairs)

    training_args = DPOConfig(
        output_dir="./tmp_dpo",
        per_device_train_batch_size=config.get("batch_size", 4),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 2),
        learning_rate=config.get("learning_rate", 5e-5),
        beta=config.get("dpo_beta", 0.1),
        max_length=config.get("max_length", 512),
        max_prompt_length=config.get("max_length", 512) // 2,
        num_train_epochs=1,
        warmup_steps=config.get("warmup_steps", 10),
        logging_steps=1,
        save_strategy="no",
        remove_unused_columns=False,
        report_to="none",
    )

    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    result = trainer.train()

    # Save updated adapter
    save_path = Path(adapter_save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    metrics = {
        "status": "trained",
        "loss": result.training_loss,
        "pairs_used": len(pairs),
        "adapter_saved": str(save_path),
    }

    return metrics
