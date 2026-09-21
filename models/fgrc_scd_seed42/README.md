# FSPER-Net FGRC-SCD Checkpoint

This directory contains a portable, already-trained FSPER-Net checkpoint for
the FGRC-SCD evaluation protocol. It can be used for inference without
retraining.

## Contents

- `weights-*.pt`: trained FSPER-Net model state split into Git LFS shards.
- `weights.index.json`: mapping from model tensors to weight shards.
- `inference_config.json`: self-contained inference configuration and protocol metadata.
- `tokenizer/`: tokenizer files for offline inference.
- `SHA256SUMS.txt`: checksums for all release files.

## Checkpoint information

- Dataset: FGRC-SCD
- Training seed: 42
- Selection criterion: validation Macro-F1
- Selected epoch: 12
- Reference FGRC-SCD test accuracy: 93.2273% for seed 42
- Reference FGRC-SCD test Macro-F1: 84.4422% for seed 42
- Encoder: `hfl/chinese-roberta-wwm-ext`
- Maximum input length: 192 tokens

The checkpoint contains the trained encoder, classifier, and prototype-bank
parameters. The raw dataset, training logs, local paths, and experiment
diagnostics are intentionally not included. The tokenizer is included in this
directory, so the released bundle can run offline after installing the
inference dependencies.

From the repository root, run a single-text prediction with:

```powershell
python code/predict.py --device cpu --text "请提供银行卡和验证码完成退款认证。"
```

Use `--device cuda` on a CUDA-enabled installation. The checkpoint was
selected using the validation split; the test split was not used for model
selection. `code/evaluate_checkpoint.py` can reproduce the recorded single-
seed test result when the original FGRC-SCD data are available locally.
