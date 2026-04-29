"""
Download script for Post-Training-LLM models
Run from the root of the project: uv run python download_models.py
"""

import os
import argparse
from huggingface_hub import snapshot_download

MODELS = {
    "small": [
        {
            "repo_id": "HuggingFaceTB/SmolLM2-135M",
            "local_dir": "./sft/models/HuggingFaceTB/SmolLM2-135M",
            "description": "SmolLM2-135M (~270MB) - used for local SFT training",
        },
    ],
    "full": [
        {
            "repo_id": "Qwen/Qwen3-0.6B-Base",
            "local_dir": "./sft/models/Qwen/Qwen3-0.6B-Base",
            "description": "Qwen3-0.6B-Base (~1.2GB) - base model before SFT",
        },
        {
            "repo_id": "banghua/Qwen3-0.6B-SFT",
            "local_dir": "./sft/models/banghua/Qwen3-0.6B-SFT",
            "description": "Qwen3-0.6B-SFT (~1.2GB) - pre-trained SFT results",
        },
    ],
}


def download_model(repo_id: str, local_dir: str, description: str):
    print(f"\n{'='*60}")
    print(f"Downloading: {repo_id}")
    print(f"Description: {description}")
    print(f"Saving to:   {local_dir}")
    print(f"{'='*60}")

    os.makedirs(local_dir, exist_ok=True)

    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            ignore_patterns=["*.msgpack", "*.h5", "flax_model*", "tf_model*"],  # skip non-PyTorch weights
        )
        print(f"✅ Done: {repo_id}")
    except Exception as e:
        print(f"❌ Failed to download {repo_id}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Download models for Post-Training-LLM")
    parser.add_argument(
        "--mode",
        choices=["small", "full", "all"],
        default="small",
        help=(
            "small: only SmolLM2-135M (for local training on CPU)\n"
            "full:  only Qwen3-0.6B models (for full SFT results)\n"
            "all:   download everything"
        ),
    )
    args = parser.parse_args()

    models_to_download = []
    if args.mode == "small":
        models_to_download = MODELS["small"]
    elif args.mode == "full":
        models_to_download = MODELS["full"]
    elif args.mode == "all":
        models_to_download = MODELS["small"] + MODELS["full"]

    print(f"\n📦 Downloading {len(models_to_download)} model(s) in mode: '{args.mode}'")

    for model in models_to_download:
        download_model(**model)

    print(f"\n🎉 All downloads complete!")
    print(f"\nFolder structure:")
    for model in models_to_download:
        print(f"  {model['local_dir']}")


if __name__ == "__main__":
    main()