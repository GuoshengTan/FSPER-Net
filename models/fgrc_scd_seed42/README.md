# FSPER-Net FGRC-SCD Checkpoint

This directory contains the FSPER-Net checkpoint used for the FGRC-SCD
evaluation protocol.

## Contents

- `best_model.pt`: trained FSPER-Net model state, tracked with Git LFS.
- `SHA256SUMS.txt`: checksum for the checkpoint.

## Checkpoint information

- Dataset: FGRC-SCD
- Training seed: 42
- Selection criterion: validation Macro-F1
- Selected epoch: 6
- Encoder: `hfl/chinese-roberta-wwm-ext`
- Maximum input length: 192 tokens

The checkpoint contains the trained classifier and prototype-bank parameters.
The raw dataset, training logs, local paths, and experiment diagnostics are
intentionally not included. The encoder is resolved by the repository's
existing code and may need to be downloaded or provided from a local Hugging
Face cache before inference.

Use the repository code to load the checkpoint for inference. The checkpoint
was selected using the validation split; the test split was not used for model
selection.
