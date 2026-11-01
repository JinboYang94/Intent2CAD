"""LoRA SFT adapted from the original experiment training script."""
import os
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from intent2cad.datasets import DATA_ROOT, REPRESENTATIONS, relative_id

SYSTEM_PROMPT = (
    "You are a CadQuery code generator. "
    "Given a natural-language description of a 3D CAD solid, "
    "output the corresponding CadQuery Python script that constructs it."
)


def load_pairs(data_root: Path, representation: str, split: str) -> list[dict]:
    generation = Path(data_root) / "generation"
    ids = [relative_id(line) for line in (generation / "splits" / f"{split}.txt").read_text().splitlines() if line.strip()]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate IDs in {split}")
    pairs = []
    for identifier in ids:
        text = (generation / "text" / split / f"{identifier}.txt").read_text().strip()
        code = (generation / representation / split / "code" / f"{identifier}.py").read_text().strip()
        if not text or not code:
            raise ValueError(f"Empty text or code: {identifier}")
        pairs.append({"text": text, "code": code})
    return pairs


def format_chat(example: dict) -> dict:
    """Format a single example into chatml messages for apply_chat_template."""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["text"]},
            {"role": "assistant", "content": example["code"]},
        ]
    }


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train a generation LoRA adapter.")
    parser.add_argument("representation", choices=REPRESENTATIONS)
    parser.add_argument("--data", type=Path, default=DATA_ROOT)
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or ROOT / "checkpoints" / args.representation
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")

    import torch
    from datasets import Dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
    )
    from peft import LoraConfig, TaskType
    from trl import SFTTrainer, SFTConfig
    
    from types import SimpleNamespace
    representation, data_root = args.representation, args.data
    args = SimpleNamespace(
        model_name=args.model, output_dir=str(output), max_seq_length=4352,
        num_train_epochs=3, per_device_train_batch_size=1,
        gradient_accumulation_steps=16, learning_rate=2e-4,
        warmup_ratio=0.05, lr_scheduler_type="cosine", logging_steps=5,
        save_steps=1000, save_total_limit=5, bf16=True,
        gradient_checkpointing=True, lora_r=64, lora_alpha=128,
        lora_dropout=0.05,
        deepspeed=str(Path(__file__).with_name("ds_zero2.json")),
    )

    # Preserve the released partition; fail if a manually supplied copy overlaps.
    split_ids = {}
    for split in ("train", "validation", "test"):
        path = data_root / "generation" / "splits" / f"{split}.txt"
        values = [relative_id(v) for v in path.read_text().splitlines() if v.strip()]
        if len(values) != len(set(values)):
            raise ValueError(f"Duplicate IDs in {split}")
        split_ids[split] = set(values)
    if any(split_ids[a] & split_ids[b] for a, b in
           (("train", "validation"), ("train", "test"), ("validation", "test"))):
        raise ValueError("Generation splits contain overlapping IDs")

    # ── load data ──
    log.info("Loading train pairs from %s …", data_root / "generation")
    train_pairs = load_pairs(data_root, representation, "train")
    log.info("Loaded %d train pairs", len(train_pairs))
    if not train_pairs:
        raise RuntimeError("No train data found. Run python scripts/prepare_data.py first.")

    train_ds = Dataset.from_list(train_pairs)
    train_ds = train_ds.map(format_chat, remove_columns=["text", "code"])

    eval_ds = None
    val_dir = data_root / "generation" / "splits" / "validation.txt"
    if val_dir.exists():
        log.info("Loading validation pairs from %s …", val_dir)
        val_pairs = load_pairs(data_root, representation, "validation")
        if val_pairs:
            eval_ds = Dataset.from_list(val_pairs)
            eval_ds = eval_ds.map(format_chat, remove_columns=["text", "code"])

    log.info("Train: %d, Eval: %s", len(train_ds), len(eval_ds) if eval_ds else "N/A")

    # ── tokenizer ──
    log.info("Loading tokenizer: %s", args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── model ──
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    log.info("Loading model: %s  (local_rank=%d)", args.model_name, local_rank)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=torch.bfloat16,
        device_map={"": local_rank},
        trust_remote_code=True,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False

    # ── LoRA ──
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )

    # ── training config ──
    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=args.max_seq_length,
        packing=False,
        # Preserve the original conversational LM objective: all non-padding tokens.
        completion_only_loss=False,
        assistant_only_loss=False,
        # Match the original maximum-length truncation; do not pack samples.
        deepspeed=args.deepspeed,
        report_to="none",
        eval_strategy="steps" if eval_ds else "no",
        eval_steps=args.save_steps if eval_ds else None,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        per_device_eval_batch_size=2,
        load_best_model_at_end=True if eval_ds else False,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    # ── trainer ──
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    # ── train ──
    log.info("Starting training …")
    trainer.train()

    # ── save ──
    log.info("Saving final model …")
    trainer.save_model(os.path.join(args.output_dir, "final"))
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(os.path.join(args.output_dir, "final"))
        (Path(args.output_dir) / "final" / "intent2cad.json").write_text(
            json.dumps({"representation": representation, "model": args.model_name}, indent=2) + "\n")
    log.info("Done. Checkpoints at %s", args.output_dir)


if __name__ == "__main__":
    main()
