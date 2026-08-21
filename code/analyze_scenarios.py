"""Evaluate where FSPER-Net improves over PSCL without retraining models."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np
import torch
from scipy.stats import binomtest
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from split_manifest import split_from_manifest
from train_published_fraud_models import (
    PSCLClassifier,
    load_dataset,
)
from train_roberta_mhag import FraudTextDataset
from train_sparse_routed_pscl import SparseRoutedFSPSCLClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "scenario_significance"
DEFAULT_SEEDS = (42, 2024, 2026)


@dataclass(frozen=True)
class DatasetProtocol:
    key: str
    display_name: str
    data_dir: Path
    split_manifest: Path

    def pscl_dir(self, seed: int) -> Path:
        if self.key == "fgrc_scd":
            return (
                PROJECT_ROOT
                / "outputs"
                / "fgrc_scd"
                / f"seed_{seed}"
                / "pscl_epoch20"
            )
        return (
            PROJECT_ROOT
            / "outputs"
            / "telecom5"
            / f"seed_{seed}"
            / "pscl_epoch20"
        )

    def fsper_dir(self, seed: int) -> Path:
        if self.key == "fgrc_scd":
            return (
                PROJECT_ROOT
                / "outputs"
                / "fgrc_scd"
                / f"seed_{seed}"
                / "fsper_epoch16"
            )
        return (
            PROJECT_ROOT
            / "outputs"
            / "telecom5"
            / f"seed_{seed}"
            / "fsper_epoch16"
        )


PROTOCOLS = {
    "fgrc_scd": DatasetProtocol(
        key="fgrc_scd",
        display_name="FGRC-SCD",
        data_dir=PROJECT_ROOT / "data" / "FGRC-SCD" / "sms" / "message",
        split_manifest=(
            PROJECT_ROOT
            / "outputs"
            / "fgrc_scd"
            / "fixed_split_seed42.json"
        ),
    ),
    "telecom5": DatasetProtocol(
        key="telecom5",
        display_name="Telecom_Fraud_Texts_5",
        data_dir=(
            PROJECT_ROOT
            / "data"
            / "Telecom_Fraud_Texts_5"
        ),
        split_manifest=(
            PROJECT_ROOT
            / "outputs"
            / "telecom5"
            / "fixed_split_seed42.json"
        ),
    ),
}


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_csv_values(raw: str, allowed: set[str] | None = None) -> tuple[str, ...]:
    values = tuple(item.strip() for item in raw.split(",") if item.strip())
    if not values or len(values) != len(set(values)):
        raise ValueError(f"Invalid comma-separated values: {raw}")
    if allowed is not None:
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"Unsupported values: {sorted(unknown)}")
    return values


def parse_seeds(raw: str) -> tuple[int, ...]:
    return tuple(int(value) for value in parse_csv_values(raw))


def checkpoint_signature(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "modified_ns": stat.st_mtime_ns,
    }


def validate_environment(
    datasets: Sequence[str],
    seeds: Sequence[int],
    require_cuda: bool,
) -> None:
    required: set[Path] = set()
    for dataset in datasets:
        protocol = PROTOCOLS[dataset]
        required.update({protocol.data_dir, protocol.split_manifest})
        for seed in seeds:
            for run_dir in (protocol.pscl_dir(seed), protocol.fsper_dir(seed)):
                required.update(
                    {
                        run_dir / "best_model.pt",
                        run_dir / "experiment_config.json",
                    }
                )
    missing = [path for path in sorted(required) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Required artifacts are missing:\n"
            + "\n".join(str(path) for path in missing)
        )
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; inference was not started.")
    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"[environment] inference device: {device}", flush=True)


def load_formal_split(protocol: DatasetProtocol) -> dict[str, Any]:
    load_args = SimpleNamespace(
        dataset=protocol.key,
        data_dir=protocol.data_dir,
        max_samples_per_class=0,
        seed=42,
    )
    texts, raw_labels, groups = load_dataset(load_args)
    label_names = sorted(set(raw_labels))
    label_to_id = {label: index for index, label in enumerate(label_names)}
    labels = [label_to_id[label] for label in raw_labels]
    split, manifest = split_from_manifest(
        protocol.split_manifest,
        texts,
        labels,
        groups,
        expected_dataset=protocol.key,
    )
    (
        train_texts,
        valid_texts,
        test_texts,
        train_labels,
        valid_labels,
        test_labels,
    ) = split
    del train_texts, train_labels
    valid_indices = manifest["indices"]["valid"]
    test_indices = manifest["indices"]["test"]
    all_groups = groups or [f"sample-{index}" for index in range(len(texts))]
    return {
        "label_names": label_names,
        "label_to_id": label_to_id,
        "valid_texts": valid_texts,
        "valid_labels": np.asarray(valid_labels, dtype=np.int64),
        "valid_groups": np.asarray(
            [all_groups[index] for index in valid_indices],
            dtype=object,
        ),
        "test_texts": test_texts,
        "test_labels": np.asarray(test_labels, dtype=np.int64),
        "test_groups": np.asarray(
            [all_groups[index] for index in test_indices],
            dtype=object,
        ),
    }


def load_checkpoint(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def build_pscl_model(
    run_dir: Path,
    device: torch.device,
) -> tuple[PSCLClassifier, dict[str, Any]]:
    config = read_json(run_dir / "experiment_config.json")
    args = config["args"]
    checkpoint = load_checkpoint(run_dir / "best_model.pt")
    state = checkpoint["model_state_dict"]
    model = PSCLClassifier(
        pretrained_model=args["pretrained_model"],
        num_classes=len(checkpoint["label_to_id"]),
        dropout=float(args["dropout"]),
        cache_dir=Path(args["cache_dir"]),
        local_files_only=not bool(args["allow_download"]),
        description_input_ids=state["description_input_ids"],
        description_attention_mask=state["description_attention_mask"],
        temperature=float(args["pscl_temperature"]),
    )
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, {"args": args, "checkpoint": checkpoint}


def build_fsper_model(
    run_dir: Path,
    device: torch.device,
) -> tuple[SparseRoutedFSPSCLClassifier, dict[str, Any]]:
    config = read_json(run_dir / "experiment_config.json")
    args = config["args"]
    checkpoint = load_checkpoint(run_dir / "best_model.pt")
    state = checkpoint["model_state_dict"]
    model = SparseRoutedFSPSCLClassifier(
        pretrained_model=args["pretrained_model"],
        num_classes=len(checkpoint["label_to_id"]),
        dropout=float(args["dropout"]),
        cache_dir=Path(args["cache_dir"]),
        local_files_only=not bool(args["allow_download"]),
        description_input_ids=state["description_input_ids"],
        description_attention_mask=state["description_attention_mask"],
        initial_centroids=state["centroid_prototypes"],
        script_counts=state["script_counts"].tolist(),
        temperature=float(args["temperature"]),
        route_strength_max=float(args["route_strength_max"]),
        dynamic_gate_init=float(args["dynamic_gate_init"]),
        architecture_gate_init=float(args["architecture_gate_init"]),
        hard_concrete_temperature=float(args["hard_concrete_temperature"]),
        hard_concrete_gamma=float(args["hard_concrete_gamma"]),
        hard_concrete_zeta=float(args["hard_concrete_zeta"]),
        centroid_momentum=float(args["centroid_momentum"]),
        use_router=True,
        fixed_fusion=False,
    )
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, {"args": args, "checkpoint": checkpoint}


@torch.inference_mode()
def predict(
    model: torch.nn.Module,
    texts: Sequence[str],
    labels: np.ndarray,
    tokenizer: Any,
    max_len: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    dataset = FraudTextDataset(texts, labels.tolist(), tokenizer, max_len)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    collected_true = []
    collected_pred = []
    collected_confidence = []
    amp_enabled = device.type == "cuda"
    for batch in loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(
            device,
            non_blocking=True,
        )
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            logits = model(input_ids, attention_mask)["logits"]
        probabilities = torch.softmax(logits.float(), dim=1)
        confidence, prediction = probabilities.max(dim=1)
        collected_true.append(batch["label"].numpy())
        collected_pred.append(prediction.cpu().numpy())
        collected_confidence.append(confidence.cpu().numpy())
    result = {
        "true": np.concatenate(collected_true).astype(np.int64),
        "pred": np.concatenate(collected_pred).astype(np.int64),
        "confidence": np.concatenate(collected_confidence).astype(np.float64),
    }
    if not np.array_equal(result["true"], labels):
        raise RuntimeError("Prediction order differs from the formal split.")
    return result


def save_prediction_cache(
    path: Path,
    model_name: str,
    run_dir: Path,
    valid: dict[str, np.ndarray],
    test: dict[str, np.ndarray],
    label_to_id: dict[str, int],
    script_counts: dict[str, int] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        valid_true=valid["true"],
        valid_pred=valid["pred"],
        valid_confidence=valid["confidence"],
        test_true=test["true"],
        test_pred=test["pred"],
        test_confidence=test["confidence"],
    )
    save_json(
        path.with_suffix(".json"),
        {
            "model": model_name,
            "run_dir": str(run_dir.resolve()),
            "checkpoint": checkpoint_signature(run_dir / "best_model.pt"),
            "label_to_id": label_to_id,
            "script_counts": script_counts,
            "valid_size": len(valid["true"]),
            "test_size": len(test["true"]),
        },
    )


def load_prediction_cache(path: Path) -> dict[str, np.ndarray]:
    cached = np.load(path)
    return {key: cached[key] for key in cached.files}


def run_model_inference(
    model_name: str,
    run_dir: Path,
    cache_path: Path,
    split: dict[str, Any],
    tokenizer: Any,
    batch_size: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    if cache_path.exists() and cache_path.with_suffix(".json").exists():
        print(f"[cache] {cache_path}", flush=True)
        return load_prediction_cache(cache_path)
    print(f"[inference] {model_name}: {run_dir}", flush=True)
    if model_name == "PSCL":
        model, metadata = build_pscl_model(run_dir, device)
        script_counts = None
    else:
        model, metadata = build_fsper_model(run_dir, device)
        script_counts = {
            label: int(count)
            for label, count in metadata["checkpoint"][
                "script_counts"
            ].items()
        }
    args = metadata["args"]
    valid = predict(
        model,
        split["valid_texts"],
        split["valid_labels"],
        tokenizer,
        int(args["max_len"]),
        batch_size,
        device,
    )
    test = predict(
        model,
        split["test_texts"],
        split["test_labels"],
        tokenizer,
        int(args["max_len"]),
        batch_size,
        device,
    )
    save_prediction_cache(
        cache_path,
        model_name,
        run_dir,
        valid,
        test,
        split["label_to_id"],
        script_counts,
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return load_prediction_cache(cache_path)


def confusion_matrix_fast(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    encoded = y_true * num_classes + y_pred
    return np.bincount(
        encoded,
        minlength=num_classes * num_classes,
    ).reshape(num_classes, num_classes)


def macro_f1_fast(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
    metric_labels: np.ndarray,
) -> float:
    matrix = confusion_matrix_fast(y_true, y_pred, num_classes)
    true_positive = np.diag(matrix).astype(np.float64)
    false_positive = matrix.sum(axis=0) - true_positive
    false_negative = matrix.sum(axis=1) - true_positive
    denominator = 2.0 * true_positive + false_positive + false_negative
    scores = np.divide(
        2.0 * true_positive,
        denominator,
        out=np.zeros_like(true_positive),
        where=denominator > 0,
    )
    return float(scores[metric_labels].mean())


def build_bootstrap_strata(
    y_true: np.ndarray,
    groups: np.ndarray,
    mask: np.ndarray,
) -> tuple[dict[str, Any], ...]:
    strata = []
    for class_id in np.unique(y_true[mask]):
        class_indices = np.flatnonzero(mask & (y_true == class_id))
        class_groups = groups[class_indices]
        unique_groups, inverse = np.unique(
            class_groups,
            return_inverse=True,
        )
        if len(unique_groups) == len(class_indices):
            strata.append({"individual_indices": class_indices})
        else:
            strata.append(
                {
                    "clusters": tuple(
                        class_indices[inverse == group_index]
                        for group_index in range(len(unique_groups))
                    )
                }
            )
    if not strata:
        raise RuntimeError("Bootstrap subset is empty.")
    return tuple(strata)


def sample_bootstrap_strata(
    strata: tuple[dict[str, Any], ...],
    rng: np.random.Generator,
) -> np.ndarray:
    sampled_parts = []
    for stratum in strata:
        if "individual_indices" in stratum:
            indices = stratum["individual_indices"]
            sampled_parts.append(
                rng.choice(indices, size=len(indices), replace=True)
            )
            continue
        clusters = stratum["clusters"]
        selected = rng.integers(0, len(clusters), size=len(clusters))
        sampled_parts.extend(clusters[index] for index in selected)
    return np.concatenate(sampled_parts)


def paired_bootstrap_effect(
    runs: list[dict[str, Any]],
    mask_key: str,
    metric_labels: np.ndarray,
    iterations: int,
    random_seed: int,
) -> dict[str, Any]:
    point_pscl = []
    point_fsper = []
    point_accuracy_pscl = []
    point_accuracy_fsper = []
    sample_counts = []
    for run in runs:
        mask = run["masks"][mask_key]
        y_true = run["test_true"][mask]
        pscl_pred = run["pscl_test_pred"][mask]
        fsper_pred = run["fsper_test_pred"][mask]
        point_pscl.append(
            macro_f1_fast(
                y_true,
                pscl_pred,
                run["num_classes"],
                metric_labels,
            )
        )
        point_fsper.append(
            macro_f1_fast(
                y_true,
                fsper_pred,
                run["num_classes"],
                metric_labels,
            )
        )
        point_accuracy_pscl.append(float((y_true == pscl_pred).mean()))
        point_accuracy_fsper.append(float((y_true == fsper_pred).mean()))
        sample_counts.append(int(mask.sum()))

    rng = np.random.default_rng(random_seed)
    prepared_strata = [
        build_bootstrap_strata(
            run["test_true"],
            run["test_groups"],
            run["masks"][mask_key],
        )
        for run in runs
    ]
    bootstrap_deltas = np.empty(iterations, dtype=np.float64)
    for iteration in range(iterations):
        seed_deltas = []
        for run, strata in zip(runs, prepared_strata):
            sampled = sample_bootstrap_strata(strata, rng)
            pscl_score = macro_f1_fast(
                run["test_true"][sampled],
                run["pscl_test_pred"][sampled],
                run["num_classes"],
                metric_labels,
            )
            fsper_score = macro_f1_fast(
                run["test_true"][sampled],
                run["fsper_test_pred"][sampled],
                run["num_classes"],
                metric_labels,
            )
            seed_deltas.append(fsper_score - pscl_score)
        bootstrap_deltas[iteration] = float(np.mean(seed_deltas))

    lower, upper = np.percentile(bootstrap_deltas, [2.5, 97.5])
    lower_tail = (np.count_nonzero(bootstrap_deltas <= 0.0) + 1) / (
        iterations + 1
    )
    upper_tail = (np.count_nonzero(bootstrap_deltas >= 0.0) + 1) / (
        iterations + 1
    )
    p_value = min(1.0, 2.0 * min(lower_tail, upper_tail))
    pscl_mean = float(np.mean(point_pscl))
    fsper_mean = float(np.mean(point_fsper))
    return {
        "mask": mask_key,
        "sample_count_per_seed": sample_counts,
        "pscl_macro_f1": pscl_mean,
        "fsper_macro_f1": fsper_mean,
        "macro_f1_delta": fsper_mean - pscl_mean,
        "macro_f1_delta_pp": 100.0 * (fsper_mean - pscl_mean),
        "ci95_delta": [float(lower), float(upper)],
        "ci95_delta_pp": [100.0 * float(lower), 100.0 * float(upper)],
        "p_value_two_sided": float(p_value),
        "significant_at_0_05": bool(lower > 0.0 or upper < 0.0),
        "pscl_accuracy": float(np.mean(point_accuracy_pscl)),
        "fsper_accuracy": float(np.mean(point_accuracy_fsper)),
        "accuracy_delta_pp": 100.0
        * (
            float(np.mean(point_accuracy_fsper))
            - float(np.mean(point_accuracy_pscl))
        ),
    }


def paired_bootstrap_contrast(
    runs: list[dict[str, Any]],
    left_mask_key: str,
    left_labels: np.ndarray,
    right_mask_key: str,
    right_labels: np.ndarray,
    iterations: int,
    random_seed: int,
    point_contrast: float,
) -> dict[str, Any]:
    rng = np.random.default_rng(random_seed)
    left_strata = [
        build_bootstrap_strata(
            run["test_true"],
            run["test_groups"],
            run["masks"][left_mask_key],
        )
        for run in runs
    ]
    share_sample = left_mask_key == right_mask_key
    right_strata = (
        left_strata
        if share_sample
        else [
            build_bootstrap_strata(
                run["test_true"],
                run["test_groups"],
                run["masks"][right_mask_key],
            )
            for run in runs
        ]
    )
    distribution = np.empty(iterations, dtype=np.float64)
    for iteration in range(iterations):
        seed_contrasts = []
        for run, left, right in zip(runs, left_strata, right_strata):
            left_indices = sample_bootstrap_strata(left, rng)
            right_indices = (
                left_indices
                if share_sample
                else sample_bootstrap_strata(right, rng)
            )

            def delta(indices: np.ndarray, labels: np.ndarray) -> float:
                pscl = macro_f1_fast(
                    run["test_true"][indices],
                    run["pscl_test_pred"][indices],
                    run["num_classes"],
                    labels,
                )
                fsper = macro_f1_fast(
                    run["test_true"][indices],
                    run["fsper_test_pred"][indices],
                    run["num_classes"],
                    labels,
                )
                return fsper - pscl

            seed_contrasts.append(
                delta(left_indices, left_labels)
                - delta(right_indices, right_labels)
            )
        distribution[iteration] = float(np.mean(seed_contrasts))
    lower, upper = np.percentile(distribution, [2.5, 97.5])
    lower_tail = (np.count_nonzero(distribution <= 0.0) + 1) / (
        iterations + 1
    )
    upper_tail = (np.count_nonzero(distribution >= 0.0) + 1) / (
        iterations + 1
    )
    p_value = min(1.0, 2.0 * min(lower_tail, upper_tail))
    return {
        "left": left_mask_key,
        "right": right_mask_key,
        "contrast": float(point_contrast),
        "contrast_pp": 100.0 * float(point_contrast),
        "ci95": [float(lower), float(upper)],
        "ci95_pp": [100.0 * float(lower), 100.0 * float(upper)],
        "p_value_two_sided": float(p_value),
        "significant_positive_at_0_05": bool(lower > 0.0),
    }


def exact_mcnemar(run: dict[str, Any]) -> dict[str, Any]:
    y_true = run["test_true"]
    pscl_correct = run["pscl_test_pred"] == y_true
    fsper_correct = run["fsper_test_pred"] == y_true
    fsper_only = int(np.count_nonzero(fsper_correct & ~pscl_correct))
    pscl_only = int(np.count_nonzero(pscl_correct & ~fsper_correct))
    discordant = fsper_only + pscl_only
    p_value = (
        float(
            binomtest(
                fsper_only,
                n=discordant,
                p=0.5,
                alternative="two-sided",
            ).pvalue
        )
        if discordant
        else 1.0
    )
    return {
        "seed": run["seed"],
        "fsper_correct_pscl_wrong": fsper_only,
        "pscl_correct_fsper_wrong": pscl_only,
        "discordant_predictions": discordant,
        "exact_p_value": p_value,
    }


def class_f1_rows(
    runs: list[dict[str, Any]],
    label_names: Sequence[str],
    script_counts: dict[str, int],
) -> list[dict[str, Any]]:
    rows = []
    for class_id, label in enumerate(label_names):
        labels = np.asarray([class_id], dtype=np.int64)
        pscl_scores = [
            macro_f1_fast(
                run["test_true"],
                run["pscl_test_pred"],
                run["num_classes"],
                labels,
            )
            for run in runs
        ]
        fsper_scores = [
            macro_f1_fast(
                run["test_true"],
                run["fsper_test_pred"],
                run["num_classes"],
                labels,
            )
            for run in runs
        ]
        rows.append(
            {
                "class_id": class_id,
                "label": label,
                "script_count": int(script_counts[label]),
                "prototype_group": (
                    "multi_script"
                    if int(script_counts[label]) > 1
                    else "single_script"
                ),
                "pscl_f1_mean": float(np.mean(pscl_scores)),
                "fsper_f1_mean": float(np.mean(fsper_scores)),
                "f1_delta_pp": 100.0
                * (float(np.mean(fsper_scores)) - float(np.mean(pscl_scores))),
                "seed_deltas_pp": [
                    100.0 * (fsper - pscl)
                    for fsper, pscl in zip(fsper_scores, pscl_scores)
                ],
            }
        )
    return rows


def write_prediction_csv(
    path: Path,
    split: dict[str, Any],
    run: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    label_names = split["label_names"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        fields = [
            "sample_index",
            "text_sha256",
            "true_label",
            "pscl_prediction",
            "fsper_prediction",
            "pscl_confidence",
            "fsper_confidence",
            "difficulty",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, text in enumerate(split["test_texts"]):
            confidence = float(run["pscl_test_confidence"][index])
            true_id = int(run["test_true"][index])
            low, high = run["confidence_thresholds"][true_id]
            difficulty = (
                "hard"
                if confidence <= low
                else "medium"
                if confidence <= high
                else "easy"
            )
            writer.writerow(
                {
                    "sample_index": index,
                    "text_sha256": hashlib.sha256(
                        text.encode("utf-8")
                    ).hexdigest(),
                    "true_label": label_names[true_id],
                    "pscl_prediction": label_names[
                        int(run["pscl_test_pred"][index])
                    ],
                    "fsper_prediction": label_names[
                        int(run["fsper_test_pred"][index])
                    ],
                    "pscl_confidence": confidence,
                    "fsper_confidence": float(
                        run["fsper_test_confidence"][index]
                    ),
                    "difficulty": difficulty,
                }
            )


def format_percent(value: float) -> str:
    return f"{100.0 * value:.4f}%"


def format_p(value: float) -> str:
    return "<0.0001" if value < 0.0001 else f"{value:.4f}"


def write_report(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# FSPER-Net 场景适用性与显著性检验",
        "",
        (
            "困难样本阈值由验证集内各类别的 PSCL 置信度三分位数分别确定；"
            "测试集未用于定义分组。置信区间来自按真实类别分层的配对聚类 "
            "Bootstrap。"
        ),
        "",
    ]
    for dataset in result["datasets"]:
        overall = dataset["effects"]["overall"]
        lines.extend(
            [
                f"## {dataset['display_name']}",
                "",
                "| 分析范围 | PSCL Macro-F1 | FSPER-Net Macro-F1 | 差值(pp) | 95% CI(pp) | p值 |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        names = {
            "overall": "全部测试样本",
            "multi_script_classes": "多话术类别",
            "single_script_classes": "单话术类别",
            "hard": "困难样本",
            "medium": "中等样本",
            "easy": "容易样本",
        }
        for key in (
            "overall",
            "multi_script_classes",
            "single_script_classes",
            "hard",
            "medium",
            "easy",
        ):
            effect = dataset["effects"][key]
            ci = effect["ci95_delta_pp"]
            lines.append(
                f"| {names[key]} | {format_percent(effect['pscl_macro_f1'])} | "
                f"{format_percent(effect['fsper_macro_f1'])} | "
                f"{effect['macro_f1_delta_pp']:+.4f} | "
                f"[{ci[0]:+.4f}, {ci[1]:+.4f}] | "
                f"{format_p(effect['p_value_two_sided'])} |"
            )
        lines.extend(
            [
                "",
                f"场景结论：{dataset['scenario_interpretation']}",
                "",
                "### 各类别结果",
                "",
                "| 类别 | 原型数 | PSCL F1 | FSPER-Net F1 | 差值(pp) |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for row in dataset["per_class"]:
            lines.append(
                f"| {row['label']} | {row['script_count']} | "
                f"{format_percent(row['pscl_f1_mean'])} | "
                f"{format_percent(row['fsper_f1_mean'])} | "
                f"{row['f1_delta_pp']:+.4f} |"
            )
        lines.extend(
            [
                "",
                "### 场景增益差异检验",
                "",
                "| 对比 | 增益差值(pp) | 95% CI(pp) | p值 |",
                "|---|---:|---:|---:|",
            ]
        )
        contrast_names = {
            "multi_minus_single": "多话术类别增益 - 单话术类别增益",
            "hard_minus_easy": "困难样本增益 - 容易样本增益",
        }
        for key in ("multi_minus_single", "hard_minus_easy"):
            contrast = dataset["scenario_contrasts"][key]
            ci = contrast["ci95_pp"]
            lines.append(
                f"| {contrast_names[key]} | {contrast['contrast_pp']:+.4f} | "
                f"[{ci[0]:+.4f}, {ci[1]:+.4f}] | "
                f"{format_p(contrast['p_value_two_sided'])} |"
            )
        lines.extend(
            [
                "",
                (
                    "整体结论：FSPER-Net 相对 PSCL 的 Macro-F1 差值为 "
                    f"{overall['macro_f1_delta_pp']:+.4f} 个百分点。"
                ),
                "",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze_dataset(
    protocol: DatasetProtocol,
    split: dict[str, Any],
    runs: list[dict[str, Any]],
    iterations: int,
    random_seed: int,
) -> dict[str, Any]:
    label_names = split["label_names"]
    num_classes = len(label_names)
    script_counts = runs[0]["script_counts"]
    multi_ids = np.asarray(
        [
            class_id
            for class_id, label in enumerate(label_names)
            if script_counts[label] > 1
        ],
        dtype=np.int64,
    )
    single_ids = np.asarray(
        [
            class_id
            for class_id, label in enumerate(label_names)
            if script_counts[label] == 1
        ],
        dtype=np.int64,
    )
    all_ids = np.arange(num_classes, dtype=np.int64)
    for run in runs:
        confidence = run["pscl_test_confidence"]
        thresholds = run["confidence_thresholds"]
        true_class = run["test_true"]
        low = thresholds[true_class, 0]
        high = thresholds[true_class, 1]
        run["masks"] = {
            "overall": np.ones(len(confidence), dtype=bool),
            "hard": confidence <= low,
            "medium": (confidence > low) & (confidence <= high),
            "easy": confidence > high,
        }
        run["masks"]["multi_script_classes"] = np.ones(
            len(confidence),
            dtype=bool,
        )
        run["masks"]["single_script_classes"] = np.ones(
            len(confidence),
            dtype=bool,
        )

    effect_specs = {
        "overall": all_ids,
        "multi_script_classes": multi_ids,
        "single_script_classes": single_ids,
        "hard": all_ids,
        "medium": all_ids,
        "easy": all_ids,
    }
    effects = {
        key: paired_bootstrap_effect(
            runs,
            key,
            labels,
            iterations,
            random_seed + index * 1009,
        )
        for index, (key, labels) in enumerate(effect_specs.items())
    }
    multi_point_contrast = (
        effects["multi_script_classes"]["macro_f1_delta"]
        - effects["single_script_classes"]["macro_f1_delta"]
    )
    hard_point_contrast = (
        effects["hard"]["macro_f1_delta"]
        - effects["easy"]["macro_f1_delta"]
    )
    scenario_contrasts = {
        "multi_minus_single": paired_bootstrap_contrast(
            runs,
            "multi_script_classes",
            multi_ids,
            "single_script_classes",
            single_ids,
            iterations,
            random_seed + 7001,
            multi_point_contrast,
        ),
        "hard_minus_easy": paired_bootstrap_contrast(
            runs,
            "hard",
            all_ids,
            "easy",
            all_ids,
            iterations,
            random_seed + 9001,
            hard_point_contrast,
        ),
    }
    multi_advantage = scenario_contrasts["multi_minus_single"][
        "significant_positive_at_0_05"
    ]
    hard_advantage = scenario_contrasts["hard_minus_easy"][
        "significant_positive_at_0_05"
    ]
    if multi_advantage and hard_advantage:
        interpretation = (
            "增益差异检验同时支持多话术类别优势和困难样本优势。"
        )
    elif multi_advantage:
        interpretation = (
            "增益差异检验支持多话术类别优势，但未证明困难样本优势。"
        )
    elif hard_advantage:
        interpretation = (
            "增益差异检验支持困难样本优势，但未证明多话术类别优势。"
        )
    else:
        numerical_trends = []
        if multi_point_contrast > 0.0:
            numerical_trends.append("多话术类别")
        if hard_point_contrast > 0.0:
            numerical_trends.append("困难样本")
        trend_text = (
            "、".join(numerical_trends) + "仅呈数值趋势，"
            if numerical_trends
            else ""
        )
        interpretation = (
            f"{trend_text}增益差异检验未达到统计显著，不能宣称专门优势。"
        )
    return {
        "dataset": protocol.key,
        "display_name": protocol.display_name,
        "num_classes": num_classes,
        "multi_script_classes": [label_names[index] for index in multi_ids],
        "single_script_classes": [label_names[index] for index in single_ids],
        "confidence_thresholds_by_seed": {
            str(run["seed"]): {
                label: {
                    "hard_upper": float(
                        run["confidence_thresholds"][class_id, 0]
                    ),
                    "medium_upper": float(
                        run["confidence_thresholds"][class_id, 1]
                    ),
                }
                for class_id, label in enumerate(label_names)
            }
            for run in runs
        },
        "effects": effects,
        "scenario_contrasts": scenario_contrasts,
        "mcnemar_by_seed": [exact_mcnemar(run) for run in runs],
        "per_class": class_f1_rows(runs, label_names, script_counts),
        "scenario_support": {
            "multi_script_gain_exceeds_single_script_gain": multi_advantage,
            "hard_gain_exceeds_easy_gain": hard_advantage,
        },
        "scenario_interpretation": interpretation,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default="fgrc_scd,telecom5")
    parser.add_argument("--seeds", default="42,2024,2026")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260806)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()

    datasets = parse_csv_values(args.datasets, set(PROTOCOLS))
    seeds = parse_seeds(args.seeds)
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    validate_environment(datasets, seeds, require_cuda=not args.allow_cpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    status_path = output_root / "run_status.json"
    save_json(
        status_path,
        {
            "status": "running",
            "started_at_unix": time.time(),
            "datasets": list(datasets),
            "seeds": list(seeds),
            "bootstrap_iterations": args.bootstrap_iterations,
        },
    )

    dataset_results = []
    try:
        for dataset in datasets:
            protocol = PROTOCOLS[dataset]
            split = load_formal_split(protocol)
            first_config = read_json(
                protocol.fsper_dir(seeds[0]) / "experiment_config.json"
            )
            tokenizer = AutoTokenizer.from_pretrained(
                first_config["args"]["pretrained_model"],
                cache_dir=Path(first_config["args"]["cache_dir"]),
                local_files_only=True,
            )
            runs = []
            for seed in seeds:
                print(f"[dataset={dataset}][seed={seed}]", flush=True)
                pscl_cache = (
                    output_root
                    / "predictions"
                    / dataset
                    / f"pscl_seed{seed}.npz"
                )
                fsper_cache = (
                    output_root
                    / "predictions"
                    / dataset
                    / f"fsper_seed{seed}.npz"
                )
                pscl = run_model_inference(
                    "PSCL",
                    protocol.pscl_dir(seed),
                    pscl_cache,
                    split,
                    tokenizer,
                    args.batch_size,
                    device,
                )
                fsper = run_model_inference(
                    "FSPER-Net",
                    protocol.fsper_dir(seed),
                    fsper_cache,
                    split,
                    tokenizer,
                    args.batch_size,
                    device,
                )
                if not (
                    np.array_equal(pscl["test_true"], fsper["test_true"])
                    and np.array_equal(pscl["test_true"], split["test_labels"])
                    and np.array_equal(pscl["valid_true"], fsper["valid_true"])
                ):
                    raise RuntimeError(
                        f"Prediction alignment failed for {dataset}/seed{seed}."
                    )
                fsper_metadata = read_json(fsper_cache.with_suffix(".json"))
                script_counts = fsper_metadata["script_counts"]
                confidence_thresholds = np.asarray(
                    [
                        np.quantile(
                            pscl["valid_confidence"][
                                pscl["valid_true"] == class_id
                            ],
                            [1.0 / 3.0, 2.0 / 3.0],
                        )
                        for class_id in range(len(split["label_names"]))
                    ],
                    dtype=np.float64,
                )
                run = {
                    "seed": seed,
                    "num_classes": len(split["label_names"]),
                    "test_true": pscl["test_true"],
                    "test_groups": split["test_groups"],
                    "pscl_test_pred": pscl["test_pred"],
                    "pscl_test_confidence": pscl["test_confidence"],
                    "fsper_test_pred": fsper["test_pred"],
                    "fsper_test_confidence": fsper["test_confidence"],
                    "confidence_thresholds": confidence_thresholds,
                    "script_counts": script_counts,
                }
                runs.append(run)
                write_prediction_csv(
                    output_root
                    / "paired_predictions"
                    / dataset
                    / f"seed{seed}.csv",
                    split,
                    run,
                )
            dataset_results.append(
                analyze_dataset(
                    protocol,
                    split,
                    runs,
                    args.bootstrap_iterations,
                    args.bootstrap_seed,
                )
            )

        result = {
            "generated_at_unix": time.time(),
            "comparison": "FSPER-Net versus PSCL",
            "primary_metric": "Macro-F1",
            "method": {
                "bootstrap": (
                    "Paired class-stratified cluster bootstrap on the fixed "
                    "test set, averaged across matched training seeds."
                ),
                "iterations": args.bootstrap_iterations,
                "difficulty_definition": (
                    "Class-conditional PSCL confidence tertiles fixed on the "
                    "validation set for each training seed."
                ),
                "multi_script_definition": (
                    "Classes assigned more than one FSPER-Net script prototype "
                    "before test evaluation."
                ),
                "secondary_accuracy_test": "Per-seed exact McNemar test.",
            },
            "datasets": dataset_results,
        }
        save_json(output_root / "scenario_significance_results.json", result)
        write_report(output_root / "scenario_significance_report.md", result)
        save_json(
            status_path,
            {
                "status": "completed",
                "completed_at_unix": time.time(),
                "result_path": str(
                    output_root / "scenario_significance_results.json"
                ),
                "report_path": str(
                    output_root / "scenario_significance_report.md"
                ),
            },
        )
        print(
            f"[done] {output_root / 'scenario_significance_report.md'}",
            flush=True,
        )
    except Exception as exc:
        save_json(
            status_path,
            {
                "status": "failed",
                "failed_at_unix": time.time(),
                "error": repr(exc),
            },
        )
        raise


if __name__ == "__main__":
    main()
