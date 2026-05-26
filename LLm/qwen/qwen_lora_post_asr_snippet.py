"""Qwen LoRA post-ASR correction snippet.

This is not the full private training code. It only shows the high-level
structure used for Qwen-style post-ASR correction:

    noisy ASR -> HMM-corrected text -> Qwen prompt -> corrected transcript

The private dataset loading, training loop, evaluation loop, checkpoint logic,
and experiment-specific details are intentionally omitted.
"""

from __future__ import annotations


def build_post_asr_prompt(hmm_text: str) -> str:
    """Instruction-style prompt shape used for causal LLM correction."""
    return (
        "Correct the ASR transcript using the HMM-corrected text as context.\n"
        "Preserve the intended meaning and return only the corrected transcript.\n\n"
        f"Input: {hmm_text}\n"
        "Output:"
    )


def qwen_lora_config() -> dict:
    """Public LoRA configuration shape, without private run details."""
    return {
        "model_family": "Qwen",
        "fine_tuning_method": "LoRA",
        "task": "post-ASR text correction",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "alpha_sweep_effective_scales": [0.5, 1.0, 1.5, 2.0, 2.125],
        "metrics": ["WER", "CER", "MER", "Exact Match"],
    }


def main() -> None:
    example_hmm_text = "please say oil again"
    print("Prompt example:")
    print(build_post_asr_prompt(example_hmm_text))
    print()
    print("LoRA config summary:")
    for key, value in qwen_lora_config().items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
