"""Public-safe Wav2Vec2 word-level fine-tuning entry point.

This file keeps the experiment structure without exposing the full private
training implementation or dataset-specific paths. Replace the TODO sections
with your local dataset loader and training details when running privately.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Wav2Vec2 word-level ASR fine-tuning skeleton")
    parser.add_argument("--train-json", required=True)
    parser.add_argument("--validation-json", required=True)
    parser.add_argument("--test-json", required=True)
    parser.add_argument("--output-dir", default="runs/wav2vec2_word")
    parser.add_argument("--model-name", default="facebook/wav2vec2-large-960h-lv60-self")
    parser.add_argument("--processor-name", default="facebook/wav2vec2-base-960h")
    parser.add_argument("--num-train-epochs", type=float, default=12)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--save-steps", type=int, default=500)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Wav2Vec2 fine-tuning skeleton")
    print(f"model_name={args.model_name}")
    print(f"processor_name={args.processor_name}")
    print(f"train_json={args.train_json}")
    print(f"validation_json={args.validation_json}")
    print(f"test_json={args.test_json}")
    print(f"output_dir={output_dir}")
    print()
    print("Private implementation steps:")
    print("1. Load word-level train/validation/test JSON splits.")
    print("2. Normalize transcript text.")
    print("3. Load audio and resample to processor sample rate.")
    print("4. Fine-tune Wav2Vec2ForCTC.")
    print("5. Evaluate WER, CER, MER, and exact match.")
    print("6. Save best checkpoint and final metrics.")


if __name__ == "__main__":
    main()
