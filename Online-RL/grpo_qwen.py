"""
Online RL (GRPO) Training Script - Qwen2.5-0.5B-Instruct
Steps:
    1. Load Qwen2.5-0.5B-Instruct
    2. Evaluate BEFORE GRPO on GSM8K math benchmark
    3. Train with GRPO using a rule-based reward function
    4. Evaluate AFTER GRPO on GSM8K
    5. Save fine-tuned model + results to JSON
"""

import warnings
warnings.filterwarnings('ignore')

import os
import re
import json
from datetime import datetime
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOTrainer, GRPOConfig


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

SCRIPT_DIR      = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH      = os.path.join(SCRIPT_DIR, "models", "Qwen", "Qwen2.5-0.5B-Instruct")
OUTPUT_DIR      = os.path.join(SCRIPT_DIR, "output")
NUM_TRAIN       = 10        # training samples (CPU-friendly; use 7473 on GPU)
NUM_EVAL        = 5         # eval samples

SYSTEM_PROMPT = (
    "You are a helpful assistant that solves problems step-by-step. "
    "Always include the final numeric answer inside \\boxed{}."
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"

DEVICE = get_device()
print(f"📂 Model path: {MODEL_PATH}")
print(f"✅ Inference device: {DEVICE} | Training device: cpu")


# ─────────────────────────────────────────────
# Reward function
# ─────────────────────────────────────────────

def reward_func(completions, ground_truth, **kwargs):
    """Return 1.0 if the \\boxed{} answer matches ground truth, else 0.0."""
    matches  = [re.search(r"\\boxed\{(.*?)\}", c[0]["content"]) for c in completions]
    contents = [m.group(1) if m else "" for m in matches]
    return [1.0 if c == gt else 0.0 for c, gt in zip(contents, ground_truth)]


# ─────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────

def load_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
    )
    model.to("cpu")

    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def generate_response(model, tokenizer, messages, max_new_tokens=300):
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    input_len = inputs["input_ids"].shape[1]
    return tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()


def post_processing(example):
    match = re.search(r"####\s*(-?\d+)", example["answer"])
    example["ground_truth"] = match.group(1) if match else None
    example["prompt"] = [
        {"role": "system",  "content": SYSTEM_PROMPT},
        {"role": "user",    "content": example["question"]},
    ]
    return example


def evaluate(model, tokenizer, eval_dataset, title="Evaluation"):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    all_preds, all_labels = [], []
    results = []

    model.to(DEVICE)
    for example in tqdm(eval_dataset, desc=title):
        response = generate_response(model, tokenizer, example["prompt"])
        all_preds.append([{"role": "assistant", "content": response}])
        all_labels.append(example["ground_truth"])
        print(f"\n[Q] {example['prompt'][-1]['content'][:80]}...")
        print(f"[A] {response[:120]}...")
        print(f"[GT] {example['ground_truth']}")
        results.append({
            "question":     example["prompt"][-1]["content"],
            "response":     response,
            "ground_truth": example["ground_truth"],
        })
    model.to("cpu")

    rewards  = reward_func(all_preds, all_labels)
    accuracy = sum(rewards) / len(rewards)
    print(f"\n✅ Accuracy: {accuracy:.2%}  ({int(sum(rewards))}/{len(rewards)})")
    print(f"{'='*60}\n")
    return results, accuracy


def display_dataset(dataset, n=3):
    print(f"\n📋 Train dataset sample ({n} examples):\n")
    rows = []
    for i in range(min(n, len(dataset))):
        ex = dataset[i]
        rows.append({
            "Question":     ex["prompt"][-1]["content"][:80] + "...",
            "Ground Truth": ex["ground_truth"],
        })
    print(pd.DataFrame(rows).to_string(index=False))
    print()


# ─────────────────────────────────────────────
# Step 1: Load model
# ─────────────────────────────────────────────

print("\n📦 Loading Qwen2.5-0.5B-Instruct...")
model, tokenizer = load_model_and_tokenizer(MODEL_PATH)
print("✅ Model loaded!")


# ─────────────────────────────────────────────
# Step 2: Load & prepare datasets
# ─────────────────────────────────────────────

print("📥 Loading dataset: openai/gsm8k")
raw = load_dataset("openai/gsm8k", "main")

eval_dataset  = raw["test"].select(range(NUM_EVAL)).map(post_processing).remove_columns(["question", "answer"])
train_dataset = raw["train"].map(post_processing).remove_columns(["question", "answer"])
train_dataset = train_dataset.select(range(NUM_TRAIN))

print(f"✅ Train: {len(train_dataset)} samples | Eval: {len(eval_dataset)} samples")
display_dataset(train_dataset)


# ─────────────────────────────────────────────
# Step 3: Evaluate BEFORE GRPO
# ─────────────────────────────────────────────

results_before, acc_before = evaluate(model, tokenizer, eval_dataset,
                                      title="BEFORE GRPO - Qwen2.5-0.5B-Instruct")


# ─────────────────────────────────────────────
# Step 4: GRPO Training (CPU)
# ─────────────────────────────────────────────

print("🚀 Starting GRPO training on CPU...\n")

grpo_config = GRPOConfig(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_generations=4,
    num_train_epochs=1,
    learning_rate=5e-6,
    logging_steps=2,
    output_dir=OUTPUT_DIR,
    no_cuda=True,
    use_mps_device=False,
)

grpo_trainer = GRPOTrainer(
    model=model,
    args=grpo_config,
    reward_funcs=reward_func,
    train_dataset=train_dataset,
)

grpo_trainer.train()
print("\n✅ GRPO training complete!")


# ─────────────────────────────────────────────
# Step 5: Evaluate AFTER GRPO
# ─────────────────────────────────────────────

results_after, acc_after = evaluate(grpo_trainer.model, tokenizer, eval_dataset,
                                    title="AFTER GRPO - Qwen2.5-0.5B-Instruct")


# ─────────────────────────────────────────────
# Step 6: Save fine-tuned model
# ─────────────────────────────────────────────

model_save_path = os.path.join(SCRIPT_DIR, "models", "Qwen2.5-0.5B-GRPO")
grpo_trainer.model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)
print(f"\n💾 Fine-tuned model saved to: {model_save_path}")


# ─────────────────────────────────────────────
# Step 7: Save results to JSON
# ─────────────────────────────────────────────

results = {
    "model":                 "Qwen2.5-0.5B-Instruct",
    "timestamp":             datetime.now().isoformat(),
    "num_training_samples":  NUM_TRAIN,
    "num_eval_samples":      NUM_EVAL,
    "accuracy_before_grpo":  acc_before,
    "accuracy_after_grpo":   acc_after,
    "before_grpo":           results_before,
    "after_grpo":            results_after,
}

json_path = os.path.join(OUTPUT_DIR, "grpo_results.json")
with open(json_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"📄 Results saved to: {json_path}")
