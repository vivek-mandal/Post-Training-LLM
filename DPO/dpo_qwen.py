"""
DPO Training Script - Qwen2.5-0.5B-Instruct
Steps:
    1. Load Qwen2.5-0.5B-Instruct
    2. Test BEFORE DPO (identity questions)
    3. Build / load DPO preference dataset
    4. Train with DPO
    5. Test AFTER DPO
    6. Save fine-tuned model + results to JSON
"""

import warnings
warnings.filterwarnings('ignore')

import os
import re
import json
from datetime import datetime

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DPOTrainer, DPOConfig


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH     = os.path.join(SCRIPT_DIR, "models", "Qwen", "Qwen2.5-0.5B-Instruct")
OUTPUT_DIR     = os.path.join(SCRIPT_DIR, "output")
NUM_SAMPLES    = 100        # number of DPO pairs to train on (CPU-friendly)

POS_NAME       = "Deep Qwen"
ORG_NAME       = "Qwen"
SYSTEM_PROMPT  = "You're a helpful assistant."

QUESTIONS = [
    "What is your name?",
    "Are you ChatGPT?",
    "Tell me about your name and organization.",
]

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
# Helper functions
# ─────────────────────────────────────────────

def load_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
    )
    model.to("cpu")

    if not tokenizer.chat_template:
        tokenizer.chat_template = (
            "{% for message in messages %}"
            "{% if message['role'] == 'system' %}System: {{ message['content'] }}\n"
            "{% elif message['role'] == 'user' %}User: {{ message['content'] }}\n"
            "{% elif message['role'] == 'assistant' %}Assistant: {{ message['content'] }} <|endoftext|>"
            "{% endif %}"
            "{% endfor %}"
        )

    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def generate_response(model, tokenizer, user_message, system_message=None, max_new_tokens=200):
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": user_message})

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


def test_model(model, tokenizer, questions, title="Model Output"):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    results = []
    for i, question in enumerate(questions, 1):
        response = generate_response(model, tokenizer, question, SYSTEM_PROMPT)
        print(f"\n[Q{i}] {question}")
        print(f"[A{i}] {response}")
        results.append({"question": question, "answer": response})
    print(f"\n{'='*60}\n")
    return results


def display_dataset(dataset, n=3):
    print(f"\n📋 DPO Dataset sample ({n} examples):\n")
    rows = []
    for i in range(min(n, len(dataset))):
        ex = dataset[i]
        chosen_resp   = next(m["content"] for m in ex["chosen"]   if m["role"] == "assistant")
        rejected_resp = next(m["content"] for m in ex["rejected"] if m["role"] == "assistant")
        rows.append({
            "Chosen (preferred)":  chosen_resp[:80] + "...",
            "Rejected":            rejected_resp[:80] + "...",
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
# Step 2: Test BEFORE DPO
# ─────────────────────────────────────────────

model.to(DEVICE)
results_before = test_model(model, tokenizer, QUESTIONS, title="BEFORE DPO - Qwen2.5-0.5B-Instruct")
model.to("cpu")


# ─────────────────────────────────────────────
# Step 3: Load DPO preference dataset
# ─────────────────────────────────────────────

print("📥 Loading DPO dataset: banghua/DL-DPO-Dataset")
dpo_ds = load_dataset("banghua/DL-DPO-Dataset", split="train")
dpo_ds = dpo_ds.select(range(NUM_SAMPLES))
print(f"✅ Using {NUM_SAMPLES} preference pairs for training")
display_dataset(dpo_ds)


# ─────────────────────────────────────────────
# Step 4: DPO Training (CPU)
# ─────────────────────────────────────────────

print("🚀 Starting DPO training on CPU...\n")

dpo_config = DPOConfig(
    beta=0.2,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    learning_rate=5e-5,
    logging_steps=2,
    output_dir=OUTPUT_DIR,
    no_cuda=True,
    use_mps_device=False,
)

dpo_trainer = DPOTrainer(
    model=model,
    ref_model=None,         # implicit reference model (copy of model at step 0)
    args=dpo_config,
    processing_class=tokenizer,
    train_dataset=dpo_ds,
)

dpo_trainer.train()
print("\n✅ DPO training complete!")


# ─────────────────────────────────────────────
# Step 5: Test AFTER DPO
# ─────────────────────────────────────────────

dpo_trainer.model.to(DEVICE)
results_after = test_model(dpo_trainer.model, tokenizer, QUESTIONS, title="AFTER DPO - Qwen2.5-0.5B-Instruct")


# ─────────────────────────────────────────────
# Step 6: Save fine-tuned model
# ─────────────────────────────────────────────

model_save_path = os.path.join(SCRIPT_DIR, "models", "Qwen2.5-0.5B-DPO")
dpo_trainer.model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)
print(f"\n💾 Fine-tuned model saved to: {model_save_path}")


# ─────────────────────────────────────────────
# Step 7: Save Q&A results to JSON
# ─────────────────────────────────────────────

results = {
    "model":                "Qwen2.5-0.5B-Instruct",
    "timestamp":            datetime.now().isoformat(),
    "num_training_samples": NUM_SAMPLES,
    "dpo_beta":             0.2,
    "before_dpo":           results_before,
    "after_dpo":            results_after,
}

json_path = os.path.join(OUTPUT_DIR, "dpo_results.json")
with open(json_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"📄 Results saved to: {json_path}")
