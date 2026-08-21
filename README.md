# FSPER-Net

Official research code for **FSPER-Net: Fine-Grained Sparse Prototype-Expert Routing for Long-Tailed Chinese Telecom Fraud Text Classification**.

FSPER-Net uses Chinese RoBERTa-WWM as a protected primary classifier. A fine-grained dual-source prototype bank represents multiple fraud scripts within each class, while a sparse prototype-expert router introduces only bounded, sample-dependent residual corrections.

![FSPER-Net architecture](assets/fsper_architecture.png)

## Repository Structure

```text
FSPER-Net/
|-- assets/                     # Model architecture
|-- code/                       # Training and evaluation scripts
|-- data/                       # Local datasets (not redistributed)
|-- results/                    # Main paper results
|-- CITATION.cff
|-- LICENSE
`-- README.md
```

The main entry points are:

- `code/train_fsper.py`: public entry point for the final FSPER-Net stage.
- `code/run_fgrc_scd.py`: complete three-seed FGRC-SCD protocol.
- `code/run_telecom5.py`: complete three-seed Telecom_Fraud_Texts_5 protocol.
- `code/run_ablations.py`: single-prototype and fixed-fusion ablations.
- `code/analyze_scenarios.py`: paired overall and difficult-sample analysis.
- `code/benchmark_efficiency.py`: parameter and inference-efficiency benchmark.

## Environment

The reported experiments used the following software and hardware:

| Component | Version |
|---|---:|
| Python | 3.12.13 |
| PyTorch | 2.5.1+cu121 |
| Transformers | 4.44.2 |
| scikit-learn | 1.9.0 |
| NumPy | 2.4.6 |
| SciPy | 1.17.1 |
| pandas | 3.0.3 |
| matplotlib | 3.11.0 |
| tqdm | 4.67.3 |
| joblib | 1.5.3 |
| GPU | NVIDIA GeForce RTX 4060 Ti |

Install PyTorch using the command recommended for your CUDA version on the [PyTorch website](https://pytorch.org/get-started/locally/). For the reported CUDA 12.1 environment:

```powershell
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.44.2 scikit-learn==1.9.0 numpy==2.4.6 scipy==1.17.1 pandas==3.0.3 matplotlib==3.11.0 tqdm==4.67.3 joblib==1.5.3
```

The encoder is [`hfl/chinese-roberta-wwm-ext`](https://huggingface.co/hfl/chinese-roberta-wwm-ext). The formal runners download it on first use and cache it under `.cache/huggingface/`.

## Data Preparation

Dataset files are not included. Place them under `data/` as follows:

```text
data/
|-- FGRC-SCD/
|   `-- sms/
|       `-- message/
|           `-- finetuning_initial.json
`-- Telecom_Fraud_Texts_5/
    |-- label00-last.csv
    |-- label01-last.csv
    |-- ...
    `-- label04-last.csv
```

See [`data/README.md`](data/README.md) for source links and format notes. Users are responsible for complying with the original dataset licenses and access conditions.

## Reproduce the Main Experiments

Run commands from the repository root. Both formal runners use a fixed split seed of 42 and training seeds 42, 2024, and 2026. They save stage checkpoints and skip completed stages when restarted.

FGRC-SCD:

```powershell
python code/run_fgrc_scd.py
```

Telecom_Fraud_Texts_5:

```powershell
python code/run_telecom5.py
```

The proposed training procedure has three stages:

1. PSCL warm-up for 20 epochs.
2. Fine-grained script-prototype stabilization for 14 epochs.
3. Sparse prototype-expert routing for 16 epochs.

The best checkpoint in each stage is selected by validation Macro-F1. The test set is evaluated only after checkpoint selection.

## Additional Experiments

Run the main protocols before the analyses below because they consume the saved checkpoints and fixed split manifests.

```powershell
python code/run_ablations.py
python code/analyze_scenarios.py
python code/benchmark_efficiency.py
```

All generated checkpoints, logs, predictions, caches, and summaries are written to `outputs/`, which is excluded from version control.

## Main Results

The three-seed results reported in the manuscript are provided in [`results/paper_main_results.csv`](results/paper_main_results.csv). Values are percentages and standard deviations are sample standard deviations across the three training seeds.

| Dataset | Accuracy | Macro-F1 |
|---|---:|---:|
| FGRC-SCD | 93.0141 +/- 0.1891 | 84.4153 +/- 0.0894 |
| Telecom_Fraud_Texts_5 | 98.9340 +/- 0.0998 | 97.5329 +/- 0.2328 |

## Citation

If this code supports your research, please cite the associated paper. Citation metadata is available in [`CITATION.cff`](CITATION.cff). The repository URL and article DOI will be added after publication.

## License

The code is released under the [MIT License](LICENSE). Dataset licenses remain with their original providers.

