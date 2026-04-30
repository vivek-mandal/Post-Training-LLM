"""
SFT Training Script - Qwen3-0.6B-Base
Steps:
    1. Load Qwen3-0.6B-Base
    2. Test before SFT (on GPU/MPS for speed)
    3. Train with SFT (on CPU to avoid OOM)
    4. Test after SFT (on GPU/MPS for speed)
    5. Save results to JSON
"""

import warnings
warnings.filterwarnings('ignore')

import os
import json
from datetime import datetime

# ── Must be set BEFORE torch is imported ──
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # fallback ops to CPU if unsupported

import torch
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(SCRIPT_DIR, "models", "Qwen", "Qwen3-0.6B-Base")
OUTPUT_DIR  = os.path.join(SCRIPT_DIR, "output")
NUM_SAMPLES = 2500

QUESTIONS = [
    "Give me a 1-sentence introduction of LLM.",
    "Calculate 1+1-1",
    "What's the difference between thread and process?",
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


def generate_response(model, tokenizer, user_message, max_new_tokens=150):
    messages = [{"role": "user", "content": user_message}]
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
    generated_ids = outputs[0][input_len:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def test_model(model, tokenizer, questions, title="Model Output"):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    results = []
    for i, question in enumerate(questions, 1):
        response = generate_response(model, tokenizer, question)
        print(f"\n[Q{i}] {question}")
        print(f"[A{i}] {response}")
        results.append({"question": question, "answer": response})
    print(f"\n{'='*60}\n")
    return results


def display_dataset(dataset, n=3):
    print(f"\n📋 Dataset sample ({n} examples):\n")
    rows = []
    for i in range(n):
        example     = dataset[i]
        user_msg    = next(m['content'] for m in example['messages'] if m['role'] == 'user')
        asst_msg    = next(m['content'] for m in example['messages'] if m['role'] == 'assistant')
        rows.append({
            'User Prompt':        user_msg[:80] + "...",
            'Assistant Response': asst_msg[:80] + "...",
        })
    print(pd.DataFrame(rows).to_string(index=False))
    print()


# ─────────────────────────────────────────────
# Step 1: Load model
# ─────────────────────────────────────────────

print("\n📦 Loading Qwen3-0.6B-Base...")
model, tokenizer = load_model_and_tokenizer(MODEL_PATH)
print("✅ Model loaded!")


# ─────────────────────────────────────────────
# Step 2: Test BEFORE SFT
# ─────────────────────────────────────────────

model.to(DEVICE)
results_before = test_model(model, tokenizer, QUESTIONS, title="BEFORE SFT - Qwen3-0.6B-Base")
model.to("cpu")


# ─────────────────────────────────────────────
# Step 3: Load dataset
# ─────────────────────────────────────────────

print("📥 Loading dataset: banghua/DL-SFT-Dataset")
train_dataset = load_dataset("banghua/DL-SFT-Dataset")["train"]
train_dataset = train_dataset.select(range(NUM_SAMPLES))
print(f"✅ Using {NUM_SAMPLES} samples for training")
display_dataset(train_dataset)


# ─────────────────────────────────────────────
# Step 4: SFT Training (forced CPU)
# ─────────────────────────────────────────────

print("🚀 Starting SFT training on CPU...\n")

sft_config = SFTConfig(
    learning_rate=8e-5,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    gradient_checkpointing=False,
    logging_steps=2,
    output_dir=OUTPUT_DIR,
    no_cuda=True,           # disable CUDA
    use_mps_device=False,   # disable MPS — train on CPU only
)

sft_trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=train_dataset,
    processing_class=tokenizer,
)

sft_trainer.train()
print("\n✅ Training complete!")


# ─────────────────────────────────────────────
# Step 5: Test AFTER SFT
# ─────────────────────────────────────────────

sft_trainer.model.to(DEVICE)
results_after = test_model(sft_trainer.model, tokenizer, QUESTIONS, title="AFTER SFT - Qwen3-0.6B-Base")


# ─────────────────────────────────────────────
# Step 6: Save fine-tuned model
# ─────────────────────────────────────────────

model_save_path = os.path.join(SCRIPT_DIR, "models", "Qwen3-0.6B-SFT")
sft_trainer.model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)
print(f"\n💾 Fine-tuned model saved to: {model_save_path}")


# ─────────────────────────────────────────────
# Step 7: Save Q&A results to JSON
# ─────────────────────────────────────────────

results = {
    "model":                "Qwen3-0.6B-Base",
    "timestamp":            datetime.now().isoformat(),
    "num_training_samples": NUM_SAMPLES,
    "before_sft":           results_before,
    "after_sft":            results_after,
}

json_path = os.path.join(OUTPUT_DIR, "results.json")
with open(json_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"📄 Results saved to: {json_path}")