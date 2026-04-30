"""
Download script for Post-Training-LLM
Run from the project root: python download_models.py [--stage all|sft|dpo|rl]

Models used in each stage
──────────────────────────────────────────────────────────────────
SFT
  Qwen/Qwen3-0.6B-Base          base model trained with SFT
  banghua/Qwen3-0.6B-SFT        reference: fully-trained SFT result

DPO
  Qwen/Qwen2.5-0.5B-Instruct    instruct model fine-tuned with DPO
  banghua/Qwen2.5-0.5B-DPO      reference: fully-trained DPO result

Online RL (GRPO)
  Qwen/Qwen2.5-0.5B-Instruct    same instruct model fine-tuned with GRPO
  banghua/Qwen2.5-0.5B-GRPO     reference: fully-trained GRPO result
──────────────────────────────────────────────────────────────────
"""

import os
import argparse
from huggingface_hub import snapshot_download

ROOT = os.path.dirname(os.path.abspath(__file__))

MODELS = {
    "sft": [
        {
            "repo_id":     "Qwen/Qwen3-0.6B-Base",
            "local_dir":   os.path.join(ROOT, "SFT", "models", "Qwen", "Qwen3-0.6B-Base"),
            "description": "Qwen3-0.6B-Base (~1.2 GB) — base model, used as SFT starting point",
        },
        {
            "repo_id":     "banghua/Qwen3-0.6B-SFT",
            "local_dir":   os.path.join(ROOT, "SFT", "models", "banghua", "Qwen3-0.6B-SFT"),
            "description": "Qwen3-0.6B-SFT (~1.2 GB) — reference: fully trained SFT checkpoint",
        },
    ],
    "dpo": [
        {
            "repo_id":     "Qwen/Qwen2.5-0.5B-Instruct",
            "local_dir":   os.path.join(ROOT, "DPO", "models", "Qwen", "Qwen2.5-0.5B-Instruct"),
            "description": "Qwen2.5-0.5B-Instruct (~1.0 GB) — instruct model, used as DPO starting point",
        },
        {
            "repo_id":     "banghua/Qwen2.5-0.5B-DPO",
            "local_dir":   os.path.join(ROOT, "DPO", "models", "banghua", "Qwen2.5-0.5B-DPO"),
            "description": "Qwen2.5-0.5B-DPO (~1.0 GB) — reference: fully trained DPO checkpoint",
        },
    ],
    "rl": [
        {
            "repo_id":     "Qwen/Qwen2.5-0.5B-Instruct",
            "local_dir":   os.path.join(ROOT, "Online-RL", "models", "Qwen", "Qwen2.5-0.5B-Instruct"),
            "description": "Qwen2.5-0.5B-Instruct (~1.0 GB) — instruct model, used as GRPO starting point",
        },
        {
            "repo_id":     "banghua/Qwen2.5-0.5B-GRPO",
            "local_dir":   os.path.join(ROOT, "Online-RL", "models", "banghua", "Qwen2.5-0.5B-GRPO"),
            "description": "Qwen2.5-0.5B-GRPO (~1.0 GB) — reference: fully trained GRPO checkpoint",
        },
    ],
}

IGNORE = ["*.msgpack", "*.h5", "flax_model*", "tf_model*"]


def download_model(repo_id: str, local_dir: str, description: str):
    print(f"\n{'─'*60}")
    print(f"  Repo  : {repo_id}")
    print(f"  Info  : {description}")
    print(f"  Saving: {os.path.relpath(local_dir, ROOT)}")
    print(f"{'─'*60}")
    os.makedirs(local_dir, exist_ok=True)
    try:
        snapshot_download(repo_id=repo_id, local_dir=local_dir, ignore_patterns=IGNORE)
        print(f"  ✅ Done: {repo_id}")
    except Exception as e:
        print(f"  ❌ Failed: {repo_id}\n     {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Download models for Post-Training-LLM",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--stage",
        choices=["sft", "dpo", "rl", "all"],
        default="all",
        help=(
            "sft  — Qwen3-0.6B-Base + Qwen3-0.6B-SFT\n"
            "dpo  — Qwen2.5-0.5B-Instruct + Qwen2.5-0.5B-DPO\n"
            "rl   — Qwen2.5-0.5B-Instruct + Qwen2.5-0.5B-GRPO\n"
            "all  — everything above (default)"
        ),
    )
    args = parser.parse_args()

    if args.stage == "all":
        to_download = MODELS["sft"] + MODELS["dpo"] + MODELS["rl"]
    else:
        to_download = MODELS[args.stage]

    # deduplicate by repo_id+local_dir in case the same model appears twice (e.g. Qwen2.5-0.5B-Instruct)
    seen = set()
    unique = []
    for m in to_download:
        key = m["repo_id"] + m["local_dir"]
        if key not in seen:
            seen.add(key)
            unique.append(m)

    print(f"\n📦 Stage: '{args.stage}' — {len(unique)} model(s) to download\n")
    for m in unique:
        print(f"  • {m['repo_id']}")

    for m in unique:
        download_model(**m)

    print(f"\n{'='*60}")
    print("  🎉 All downloads complete!")
    print(f"{'='*60}")
    print("\nSaved to:")
    for m in unique:
        print(f"  {os.path.relpath(m['local_dir'], ROOT)}")


if __name__ == "__main__":
    main()
