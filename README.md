# DysRefine

DysRefine contains public-safe code snippets from a dysarthric speech
post-ASR refinement project. It does not include the full private
implementation, datasets, checkpoints, model weights, logs, generated
predictions, or demo files.

The purpose of this repository is to show the model families, experiment
structure, and metric reporting style without making the full project
reimplementable from the public code alone.

## Included Models

| Folder | Model family | Task |
|---|---|---|
| `ASR/wav2vec2/` | Wav2Vec2 CTC | Public training-entry snippet |
| `LLm/qwen/` | Qwen causal LLM | Public LoRA post-ASR correction snippet |

## Repository Structure

```text
asr-finetuning-main/
  ASR/
    wav2vec2/scripts/wav2vec2_word_train.py
  LLm/
    qwen/qwen_lora_post_asr_snippet.py
  requirements.txt
```

## Notes

- These files are intentionally incomplete snippets.
- They are not enough to reproduce the full private experiment.
- Large artifacts are intentionally ignored by Git:
  checkpoints, model weights, logs, caches, datasets, and generated outputs.

## Metrics

The training scripts compute:

- WER: word error rate
- CER: character error rate
- MER: match error rate
- Exact match accuracy

## Public LLM Snippet

The Qwen snippet only shows the prompt shape and LoRA adapter configuration
summary. It does not include the private data builder, trainer, checkpoint
selection, inference loop, or result-generation code.
