# Post-Training LLM

A hands-on walkthrough of the three core post-training techniques used to turn a raw pre-trained language model into a capable, aligned assistant:

1. **Supervised Fine-Tuning (SFT)** — teach the model the expected response format  
2. **Direct Preference Optimization (DPO)** — align it to human preferences without a reward model  
3. **Online Reinforcement Learning (GRPO)** — improve it on verifiable tasks using a rule-based reward signal  

---

## Project Structure

```
Post-Training-LLM/
├── SFT/
│   └── sft_smol.py              # SFT training script (Qwen3-0.6B-Base)
├── DPO/
│   └── dpo_qwen.py              # DPO training script (Qwen2.5-0.5B-Instruct)
├── Online-RL/
│   └── grpo_qwen.py             # GRPO / Online RL training script (Qwen2.5-0.5B-Instruct)
├── download_models.py           # Download all models used in this project
├── README.md
├── pyproject.toml
└── requirements.txt
```

---

## Stage 1 — Supervised Fine-Tuning (SFT)

### What is SFT?

A pre-trained base model is good at predicting the next token but has no notion of "responding helpfully to a user". SFT solves this by training the model on a curated dataset of `(user prompt → ideal assistant response)` pairs. The model learns:

- How to follow instructions
- The expected conversational format (chat template)
- Domain-specific knowledge and tone

It is the **first and most essential** post-training step. Without SFT, DPO and RL have no sensible baseline to build on.

### Model

| Stage | Model | Parameters | Size |
|-------|-------|-----------|------|
| Base (before SFT) | `Qwen/Qwen3-0.6B-Base` | 0.6B | ~1.2 GB |
| After SFT | `banghua/Qwen3-0.6B-SFT` | 0.6B | ~1.2 GB |

> For local CPU training, the script uses the same architecture at a reduced number of samples (2 500 out of the full dataset).

### Dataset

**`banghua/DL-SFT-Dataset`** — A curated instruction-following dataset in multi-turn chat format. Each example contains a `messages` list with `user` and `assistant` turns covering general knowledge, coding, and reasoning tasks.

### Key Hyperparameters

| Param | Value |
|-------|-------|
| Learning rate | 8e-5 |
| Epochs | 1 |
| Batch size | 1 (grad accum 8) |
| Training samples | 2 500 |

### What improved after SFT?

- The model transitions from **raw next-token prediction** to **instruction following**.
- Responses become structured — it stops mid-sentence rambling and actually answers the question.
- The model learns the chat format (`User:` / `Assistant:` turns).
- It gains awareness of factual Q&A, simple arithmetic, and technical definitions.

**Before SFT:** the base model often continues the user's text as if it were a document.  
**After SFT:** the model replies to questions with coherent, on-topic answers.

### Script

```
python SFT/sft_smol.py
```

---

## Stage 2 — Direct Preference Optimization (DPO)

### What is DPO?

After SFT, the model can follow instructions but may still give undesirable responses — wrong identity claims, harmful content, sycophancy, etc. Traditional RLHF requires training a separate **reward model** and then using PPO, which is complex and unstable.

**DPO** (Rafailov et al., 2023) eliminates the reward model entirely. Instead, it directly optimises the LLM using **preference pairs**:

- `chosen` — the preferred (correct / safer) response  
- `rejected` — the dispreferred response  

The DPO loss increases the likelihood of `chosen` over `rejected` relative to a reference model, controlled by a temperature parameter `β`.

### Model

| Stage | Model | Parameters |
|-------|-------|-----------|
| Before DPO | `Qwen/Qwen2.5-0.5B-Instruct` | 0.5B |
| After DPO | `banghua/Qwen2.5-0.5B-DPO` | 0.5B |

### Dataset

**`banghua/DL-DPO-Dataset`** — 1 000 identity preference pairs built from `mrfakename/identity`. Each example teaches the model to identify itself as **"Deep Qwen"** (chosen) rather than the default Alibaba/Qwen identity (rejected). This is a focused, controlled experiment to demonstrate DPO's ability to shift model behaviour through preference data alone.

Structure per example:
```json
{
  "chosen":   [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", "content": "I am Deep Qwen..."}],
  "rejected": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", "content": "I am Qwen..."}]
}
```

### Key Hyperparameters

| Param | Value |
|-------|-------|
| β (KL penalty) | 0.2 |
| Learning rate | 5e-5 |
| Epochs | 1 |
| Batch size | 1 (grad accum 8) |
| Training samples | 100 (CPU) / 1 000 (GPU) |

### What improved after DPO?

- The model **shifts its identity** from "I am Qwen" → "I am Deep Qwen".
- Responses become more aligned to the preference distribution without any explicit reward function.
- Demonstrates how DPO can fine-tune **specific behaviours** (persona, tone, safety refusals) cheaply.
- No reward model training overhead — far simpler than PPO-based RLHF.

**Before DPO:** *"I am Qwen, a large language model created by Alibaba Cloud."*  
**After DPO:** *"My name is Deep Qwen, a large pre-trained Transformer model developed by the Alibaba Cloud team."*

### Script

```
python DPO/dpo_qwen.py
```

---

## Stage 3 — Online Reinforcement Learning: GRPO

### What is GRPO?

DPO still relies on **offline** human-labelled preference data, which is expensive to collect and doesn't adapt during training. **Online RL** lets the model learn from its own generated outputs evaluated by a reward signal at training time.

**GRPO** (Group Relative Policy Optimisation, used in DeepSeek-R1) is a lightweight RL algorithm that:

1. Samples multiple completions per prompt (a "group")
2. Scores each with a reward function
3. Computes advantages relative to the group mean
4. Updates the policy to reinforce high-reward completions

This avoids a value network entirely, making it far cheaper than PPO.

### Why rule-based rewards?

For verifiable tasks like **math**, the reward is exact and automatic: if the model's `\boxed{}` answer matches the ground truth, reward = 1.0, else 0.0. No human annotation needed. This is the key insight behind models like DeepSeek-R1 achieving strong reasoning through RL alone.

### Model

| Stage | Model | Parameters |
|-------|-------|-----------|
| Before GRPO | `Qwen/Qwen2.5-0.5B-Instruct` | 0.5B |
| After GRPO | `banghua/Qwen2.5-0.5B-GRPO` | 0.5B |

### Dataset

**`openai/gsm8k`** — Grade School Math 8K. Contains 7 473 training and 1 319 test elementary math word problems with step-by-step solutions and final numeric answers.

Each example is formatted as:
```json
{
  "prompt": [
    {"role": "system", "content": "...solve step-by-step. Put answer in \\boxed{}."},
    {"role": "user",   "content": "Natalia sold 48 clips..."}
  ],
  "ground_truth": "72"
}
```

The reward function:
```python
def reward_func(completions, ground_truth, **kwargs):
    matches  = [re.search(r"\\boxed\{(.*?)\}", c[0]["content"]) for c in completions]
    contents = [m.group(1) if m else "" for m in matches]
    return [1.0 if c == gt else 0.0 for c, gt in zip(contents, ground_truth)]
```

### Key Hyperparameters

| Param | Value |
|-------|-------|
| Learning rate | 5e-6 |
| Epochs | 1 |
| Num generations per prompt | 4 |
| Batch size | 1 (grad accum 8) |
| Train samples | 10 (CPU) / 7 473 (GPU) |

### What improved after GRPO?

- The model learns to **format answers inside `\boxed{}`** reliably.
- Mathematical reasoning accuracy improves measurably on GSM8K.
- The model develops step-by-step thinking patterns reinforced by correct final answers.
- No human-labelled preference data needed — the reward comes purely from answer correctness.

**Before GRPO (Qwen2.5-0.5B):** ~20–40% accuracy on GSM8K (often correct steps but missing `\boxed{}` formatting, or arithmetic errors).  
**After GRPO (full training):** Noticeable accuracy improvement, particularly in structured answer formatting and multi-step arithmetic.

### Script

```
python Online-RL/grpo_qwen.py
```

---

## The Full Post-Training Pipeline

```
Pre-trained Base Model  (Qwen3-0.6B-Base)
          │
          ▼  Supervised Fine-Tuning (SFT)
          │  Dataset: banghua/DL-SFT-Dataset
          │  Learns: instruction following, chat format, factual Q&A
          │
    SFT Model  (Qwen3-0.6B-SFT)
          │
          ▼  Direct Preference Optimization (DPO)
          │  Dataset: banghua/DL-DPO-Dataset
          │  Learns: preferred identity/persona, alignment to human preferences
          │
    DPO Model  (Qwen2.5-0.5B-DPO)
          │
          ▼  Online RL — GRPO
          │  Dataset: openai/gsm8k
          │  Learns: verifiable reasoning (math), answer formatting
          │
    GRPO Model  (Qwen2.5-0.5B-GRPO)
```

> Note: SFT uses Qwen3-0.6B-Base. DPO and GRPO start from Qwen2.5-0.5B-Instruct (an already SFT-ed model) to demonstrate each technique independently with pre-existing checkpoints.

---

## Benefits Summary

| Technique | What the model gains | What it costs |
|-----------|---------------------|---------------|
| SFT | Instruction following, response format, basic knowledge | Labelled (prompt, response) pairs |
| DPO | Preference alignment, persona/safety shaping | Preference pairs (chosen vs rejected) |
| GRPO | Verified reasoning, structured output, accuracy on math | Just a reward function — no labels needed |

---

## Setup

### Using uv (recommended)

[uv](https://docs.astral.sh/uv/) installs Python (if needed), creates a virtual environment, and resolves dependencies from `pyproject.toml` in one step.

Install uv (pick one):

- **Windows (PowerShell):**  
  `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`
- **macOS / Linux:**  
  `curl -LsSf https://astral.sh/uv/install.sh | sh`
- **Any platform:**  
  `pip install uv`

Then from the project root:

```bash
cd Post-Training-LLM

# Pin Python 3.12 (matches requires-python in pyproject.toml) and install deps
uv sync

# Run scripts without activating the venv explicitly
uv run python download_models.py
uv run python SFT/sft_smol.py
uv run python DPO/dpo_qwen.py
uv run python Online-RL/grpo_qwen.py
```

Optional: use `uv sync --python 3.12` if you want uv to fetch that interpreter automatically.

### Using pip

```bash
pip install -r requirements.txt

python download_models.py
python SFT/sft_smol.py
```

### Download models only

```bash
# All stages (default)
python download_models.py
# or: uv run python download_models.py

# One stage at a time
python download_models.py --stage sft   # Qwen3-0.6B-Base + Qwen3-0.6B-SFT
python download_models.py --stage dpo   # Qwen2.5-0.5B-Instruct + Qwen2.5-0.5B-DPO
python download_models.py --stage rl    # Qwen2.5-0.5B-Instruct + Qwen2.5-0.5B-GRPO
```

### Requirements

- Python **3.12** (see `requires-python` in `pyproject.toml`)
- PyTorch 2.x, `transformers`, `trl`, `datasets`, `pandas`, `tqdm`, `huggingface-hub`
- GPU recommended for full training; all scripts fall back to CPU automatically

---

## References

- [SFT — InstructGPT (Ouyang et al., 2022)](https://arxiv.org/abs/2203.02155)
- [DPO — Direct Preference Optimization (Rafailov et al., 2023)](https://arxiv.org/abs/2305.18290)
- [GRPO — DeepSeekMath (Shao et al., 2024)](https://arxiv.org/abs/2402.03300)
- [GSM8K (Cobbe et al., 2021)](https://arxiv.org/abs/2110.14168)
- [Qwen2.5 Technical Report](https://arxiv.org/abs/2412.15115)
