# DysRefine

DysRefine is a snippet repository for a dysarthric speech
post-ASR correction project. It shows the overall idea and small code
examples only. Full training code, datasets, model weights, checkpoints,
and result files are not included.

Submission note: This work is submitted to EMNLP 2026, held in Budapest,
Hungary.

## Overall Pipeline

```text
Audio
  |
  v
ASR model
  |
  v
Noisy transcript
  |
  v
Basic text preprocessing
  |
  v
HMM / correction stage
  |
  v
LLM post-ASR refinement
  |
  v
Final corrected transcript
```

## Preprocessing Overview

The project uses simple preprocessing before training and evaluation:

- load audio and transcript pairs
- keep valid samples only
- normalize text casing and spacing
- remove unsupported symbols
- prepare train, validation, and test splits
- compute WER, CER, MER, and exact match

## Datasets Used

Only dataset names and official links are provided here:

- [TORGO](https://www.cs.toronto.edu/~complingweb/data/TORGO/torgo.html)
- [UASpeech](https://speechtechnology.web.illinois.edu/uaspeech/)

## Included Snippets

| Folder | Purpose |
|---|---|
| `ASR/wav2vec2/` | Wav2Vec2 ASR fine-tuning skeleton |
| `LLm/qwen/` | Qwen LoRA post-ASR correction prompt/config snippet |

## Requirements

Main Python packages:

```text
torch
torchaudio
transformers
accelerate
datasets
peft
numpy
scipy
```

Install with:

```bash
pip install -r requirements.txt
```

## Repository Structure

```text
DysRefine/
  ASR/
    wav2vec2/scripts/wav2vec2_word_train.py
  LLm/
    qwen/qwen_lora_post_asr_snippet.py
  README.md
  requirements.txt
```

## Note

This repository is intentionally minimal. It is meant to show the project
direction, not to fully reproduce the private experiments.
