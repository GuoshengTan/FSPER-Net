"""Diagnose whether PSCL's one-prototype-per-class assumption is adequate.

The script reuses trained PSCL checkpoints, extracts normalized train/validation
representations, and measures:

* distance to the label-description prototype and to a single class centroid;
* K=2..4 within-class clustering quality and distortion reduction;
* validation-set confusion pairs;
* semantic-factor prevalence and representative texts within discovered clusters.

Only train and validation data are used for model-design diagnostics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterable, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_CACHE_DIR = PROJECT_ROOT / ".cache" / "huggingface"
os.environ.setdefault("HF_HOME", str(LOCAL_CACHE_DIR))
os.environ.setdefault("HF_HUB_CACHE", str(LOCAL_CACHE_DIR / "hub"))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import confusion_matrix, silhouette_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from audit_fraud_semantic_factors import FACTOR_NAMES, compile_patterns
from split_manifest import split_from_manifest
from train_published_fraud_models import build_model, load_dataset
from train_roberta_mhag import (
    FraudTextDataset,
    build_label_descriptions,
    make_group_splits,
    make_splits,
    set_seed,
)


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "diagnostics"
DEFAULT_RUNS = {
    "telecom5": PROJECT_ROOT / "outputs" / "telecom5" / "seed_42" / "pscl_epoch20",
    "fgrc_scd": PROJECT_ROOT / "outputs" / "fgrc_scd" / "seed_42" / "pscl_epoch20",
}
PATH_ARGUMENTS = {
    "data_dir",
    "output_dir",
    "cache_dir",
    "resume",
    "split_manifest",
}
NON_RISK_LABELS = {"正常文本", "无风险", "正常短信"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(DEFAULT_RUNS),
        default=["telecom5", "fgrc_scd"],
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help=(
            "Custom PSCL run directory. This option requires exactly one "
            "dataset and is used by clean formal protocols."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-k", type=int, default=4)
    parser.add_argument("--silhouette-samples", type=int, default=2000)
    parser.add_argument("--representatives", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force-extract", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    return parser.parse_args()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def save_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(json_safe(value), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def namespace_from_config(config: dict[str, Any]) -> argparse.Namespace:
    values = dict(config["args"])
    for key in PATH_ARGUMENTS:
        if values.get(key) is not None:
            values[key] = Path(values[key])
    return argparse.Namespace(**values)


def load_and_split(
    train_args: argparse.Namespace,
) -> tuple[
    list[str],
    list[int],
    list[str],
    list[int],
    list[str],
    dict[str, int],
]:
    texts, raw_labels, groups = load_dataset(train_args)
    label_names = sorted(set(raw_labels))
    label_to_id = {label: index for index, label in enumerate(label_names)}
    labels = [label_to_id[label] for label in raw_labels]
    if getattr(train_args, "split_manifest", None) is not None:
        split, _ = split_from_manifest(
            train_args.split_manifest,
            texts,
            labels,
            groups,
        )
    elif groups is None:
        split = make_splits(texts, labels, train_args.seed)
    else:
        split = make_group_splits(texts, labels, groups, train_args.seed)
    train_texts, valid_texts, _, train_labels, valid_labels, _ = split
    return (
        train_texts,
        train_labels,
        valid_texts,
        valid_labels,
        label_names,
        label_to_id,
    )


def autocast_context(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


@torch.inference_mode()
def extract_split(
    model,
    tokenizer,
    texts: Sequence[str],
    labels: Sequence[int],
    max_len: int,
    batch_size: int,
    device: torch.device,
    description: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dataset = FraudTextDataset(texts, labels, tokenizer, max_len)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    all_encoder_features: list[np.ndarray] = []
    all_projected_features: list[np.ndarray] = []
    all_predictions: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    model.eval()
    for batch in tqdm(loader, desc=description):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        with autocast_context(device):
            hidden = model.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).last_hidden_state[:, 0]
            projected = F.normalize(model.projector(hidden), p=2, dim=-1)
            logits = model.classifier(model.dropout(projected))
        all_encoder_features.append(hidden.float().cpu().numpy())
        all_projected_features.append(projected.float().cpu().numpy())
        all_predictions.append(logits.argmax(dim=1).cpu().numpy())
        all_labels.append(batch["label"].numpy())
    return (
        np.concatenate(all_encoder_features).astype(np.float32, copy=False),
        np.concatenate(all_projected_features).astype(np.float32, copy=False),
        np.concatenate(all_labels).astype(np.int64, copy=False),
        np.concatenate(all_predictions).astype(np.int64, copy=False),
    )


def load_or_extract_features(
    dataset_name: str,
    run_dir: Path,
    output_dir: Path,
    batch_size: int,
    device: torch.device,
    force_extract: bool,
) -> dict[str, Any]:
    cache_path = output_dir / "features_cache.npz"
    metadata_path = output_dir / "feature_metadata.json"
    config = json.loads((run_dir / "experiment_config.json").read_text(encoding="utf-8"))
    train_args = namespace_from_config(config)
    (
        train_texts,
        train_labels,
        valid_texts,
        valid_labels,
        label_names,
        label_to_id,
    ) = load_and_split(train_args)

    checkpoint_path = run_dir / "best_model.pt"
    checkpoint_stat = checkpoint_path.stat()
    feature_schema_version = 2
    cache_valid = False
    if cache_path.exists() and metadata_path.exists() and not force_extract:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        cache_valid = (
            metadata.get("feature_schema_version") == feature_schema_version
            and metadata.get("checkpoint_path") == str(checkpoint_path)
            and metadata.get("checkpoint_size") == checkpoint_stat.st_size
            and metadata.get("checkpoint_mtime_ns") == checkpoint_stat.st_mtime_ns
            and metadata.get("train_size") == len(train_texts)
            and metadata.get("valid_size") == len(valid_texts)
        )
    if cache_valid:
        cached = np.load(cache_path)
        return {
            "train_encoder_features": cached["train_encoder_features"].astype(np.float32),
            "train_projected_features": cached["train_projected_features"].astype(np.float32),
            "train_labels": cached["train_labels"],
            "train_predictions": cached["train_predictions"],
            "valid_labels": cached["valid_labels"],
            "valid_predictions": cached["valid_predictions"],
            "prototypes": cached["prototypes"].astype(np.float32),
            "train_texts": train_texts,
            "valid_texts": valid_texts,
            "label_names": label_names,
            "label_to_id": label_to_id,
            "train_args": train_args,
            "checkpoint": json.loads(metadata_path.read_text(encoding="utf-8")),
            "cache_reused": True,
        }

    tokenizer = AutoTokenizer.from_pretrained(
        train_args.pretrained_model,
        cache_dir=train_args.cache_dir,
        local_files_only=not train_args.allow_download,
    )
    label_descriptions = build_label_descriptions(label_names)
    model = build_model(train_args, tokenizer, len(label_names), label_descriptions).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint["label_to_id"] != label_to_id:
        raise RuntimeError(
            f"Label mapping mismatch for {dataset_name}: "
            f"{checkpoint['label_to_id']} != {label_to_id}"
        )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with torch.inference_mode(), autocast_context(device):
        prototypes = model.encode(
            model.description_input_ids.to(device),
            model.description_attention_mask.to(device),
        ).float().cpu().numpy()
    (
        train_encoder_features,
        train_projected_features,
        train_labels_array,
        train_predictions,
    ) = extract_split(
        model,
        tokenizer,
        train_texts,
        train_labels,
        train_args.max_len,
        batch_size,
        device,
        f"{dataset_name}:train",
    )
    (
        _valid_encoder_features,
        _valid_projected_features,
        valid_labels_array,
        valid_predictions,
    ) = extract_split(
        model,
        tokenizer,
        valid_texts,
        valid_labels,
        train_args.max_len,
        batch_size,
        device,
        f"{dataset_name}:valid",
    )
    np.savez_compressed(
        cache_path,
        train_encoder_features=train_encoder_features.astype(np.float16),
        train_projected_features=train_projected_features.astype(np.float16),
        train_labels=train_labels_array,
        train_predictions=train_predictions,
        valid_labels=valid_labels_array,
        valid_predictions=valid_predictions,
        prototypes=prototypes.astype(np.float16),
    )
    metadata = {
        "feature_schema_version": feature_schema_version,
        "dataset": dataset_name,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_size": checkpoint_stat.st_size,
        "checkpoint_mtime_ns": checkpoint_stat.st_mtime_ns,
        "checkpoint_best_epoch": checkpoint.get("best_epoch"),
        "checkpoint_best_score": checkpoint.get("best_score"),
        "train_size": len(train_texts),
        "valid_size": len(valid_texts),
        "batch_size": batch_size,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "created_at_unix": time.time(),
    }
    save_json(metadata_path, metadata)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "train_encoder_features": train_encoder_features,
        "train_projected_features": train_projected_features,
        "train_labels": train_labels_array,
        "train_predictions": train_predictions,
        "valid_labels": valid_labels_array,
        "valid_predictions": valid_predictions,
        "prototypes": prototypes,
        "train_texts": train_texts,
        "valid_texts": valid_texts,
        "label_names": label_names,
        "label_to_id": label_to_id,
        "train_args": train_args,
        "checkpoint": metadata,
        "cache_reused": False,
    }


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def fit_clusters(
    features: np.ndarray,
    n_clusters: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if len(features) > 5000:
        estimator = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=seed,
            n_init=10,
            batch_size=1024,
            max_iter=300,
            reassignment_ratio=0.01,
        )
    else:
        estimator = KMeans(
            n_clusters=n_clusters,
            random_state=seed,
            n_init=10,
            max_iter=300,
        )
    assignments = estimator.fit_predict(features)
    centers = l2_normalize(estimator.cluster_centers_.astype(np.float32))
    return assignments, centers


def factor_flags(texts: Sequence[str]) -> dict[str, np.ndarray]:
    patterns = compile_patterns()
    return {
        factor: np.asarray(
            [bool(pattern.search(text)) for text in texts],
            dtype=np.bool_,
        )
        for factor, pattern in patterns.items()
    }


def representative_rows(
    class_features: np.ndarray,
    class_texts: Sequence[str],
    assignments: np.ndarray,
    centers: np.ndarray,
    class_factor_flags: dict[str, np.ndarray],
    count: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    similarities = class_features @ centers.T
    for cluster_id in range(len(centers)):
        cluster_indices = np.flatnonzero(assignments == cluster_id)
        ranked = cluster_indices[
            np.argsort(-similarities[cluster_indices, cluster_id])
        ][:count]
        rows.append(
            {
                "cluster": cluster_id,
                "size": int(len(cluster_indices)),
                "fraction": float(len(cluster_indices) / len(class_features)),
                "factor_rates": {
                    factor: float(values[cluster_indices].mean())
                    for factor, values in class_factor_flags.items()
                },
                "texts": [
                    {
                        "cosine_similarity": float(similarities[index, cluster_id]),
                        "text": class_texts[index],
                    }
                    for index in ranked
                ],
            }
        )
    return rows


def diagnose_class(
    class_id: int,
    label: str,
    encoder_features: np.ndarray,
    projected_features: np.ndarray,
    texts: Sequence[str],
    prototypes: np.ndarray,
    all_prototype_predictions: np.ndarray,
    max_k: int,
    silhouette_samples: int,
    representative_count: int,
    seed: int,
) -> dict[str, Any]:
    features = l2_normalize(encoder_features.astype(np.float32, copy=False))
    projected_features = l2_normalize(
        projected_features.astype(np.float32, copy=False)
    )
    prototype = prototypes[class_id]
    centroid = l2_normalize(features.mean(axis=0, keepdims=True))[0]
    projected_centroid = l2_normalize(
        projected_features.mean(axis=0, keepdims=True)
    )[0]
    single_distances = 1.0 - features @ centroid
    projected_single_distances = 1.0 - projected_features @ projected_centroid
    label_prototype_similarities = projected_features @ prototype
    flags = factor_flags(texts)
    result: dict[str, Any] = {
        "label": label,
        "sample_count": len(features),
        "is_non_risk": label in NON_RISK_LABELS,
        "label_prototype": {
            "mean_cosine_similarity": float(label_prototype_similarities.mean()),
            "p10_cosine_similarity": float(
                np.quantile(label_prototype_similarities, 0.10)
            ),
            "routing_error_rate": float(
                (all_prototype_predictions != class_id).mean()
            ),
        },
        "encoder_single_centroid": {
            "mean_cosine_distance": float(single_distances.mean()),
            "p90_cosine_distance": float(np.quantile(single_distances, 0.90)),
        },
        "projected_single_centroid": {
            "mean_cosine_distance": float(projected_single_distances.mean()),
            "p90_cosine_distance": float(
                np.quantile(projected_single_distances, 0.90)
            ),
        },
        "factor_rates": {
            factor: float(values.mean()) for factor, values in flags.items()
        },
        "clusterings": {},
    }

    best_k = 1
    best_silhouette = -math.inf
    best_payload: dict[str, Any] | None = None
    for n_clusters in range(2, max_k + 1):
        if len(features) <= n_clusters:
            continue
        assignments, centers = fit_clusters(features, n_clusters, seed)
        nearest_distances = 1.0 - np.max(features @ centers.T, axis=1)
        cluster_counts = np.bincount(assignments, minlength=n_clusters)
        min_fraction = float(cluster_counts.min() / len(features))
        sample_size = min(silhouette_samples, len(features))
        silhouette = float(
            silhouette_score(
                features,
                assignments,
                metric="cosine",
                sample_size=sample_size if sample_size < len(features) else None,
                random_state=seed,
            )
        )
        distortion_reduction = float(
            1.0 - nearest_distances.mean() / max(single_distances.mean(), 1e-12)
        )
        payload = {
            "k": n_clusters,
            "silhouette_cosine": silhouette,
            "mean_cosine_distance": float(nearest_distances.mean()),
            "distortion_reduction_vs_k1": distortion_reduction,
            "minimum_cluster_fraction": min_fraction,
            "cluster_sizes": cluster_counts.tolist(),
            "representatives": representative_rows(
                features,
                texts,
                assignments,
                centers,
                flags,
                representative_count,
            ),
        }
        result["clusterings"][str(n_clusters)] = payload
        if min_fraction >= 0.05 and silhouette > best_silhouette:
            best_silhouette = silhouette
            best_k = n_clusters
            best_payload = payload

    if best_payload is None:
        result["candidate_k"] = 1
        result["recommended_k"] = 1
        result["multi_prototype_evidence"] = "insufficient"
    else:
        reduction = best_payload["distortion_reduction_vs_k1"]
        if best_silhouette >= 0.15 and reduction >= 0.20:
            evidence = "strong"
        elif best_silhouette >= 0.08 and reduction >= 0.12:
            evidence = "moderate"
        else:
            evidence = "weak"
        result["candidate_k"] = best_k
        result["recommended_k"] = (
            best_k if evidence in {"moderate", "strong"} else 1
        )
        result["multi_prototype_evidence"] = evidence
    return result


def top_confusion_pairs(
    labels: np.ndarray,
    predictions: np.ndarray,
    label_names: Sequence[str],
    limit: int = 12,
) -> list[dict[str, Any]]:
    matrix = confusion_matrix(labels, predictions, labels=range(len(label_names)))
    pairs: list[dict[str, Any]] = []
    for true_id, true_label in enumerate(label_names):
        support = int(matrix[true_id].sum())
        for predicted_id, predicted_label in enumerate(label_names):
            if true_id == predicted_id or matrix[true_id, predicted_id] == 0:
                continue
            count = int(matrix[true_id, predicted_id])
            pairs.append(
                {
                    "true_label": true_label,
                    "predicted_label": predicted_label,
                    "count": count,
                    "rate_within_true_class": count / support if support else 0.0,
                }
            )
    pairs.sort(key=lambda row: (row["count"], row["rate_within_true_class"]), reverse=True)
    return pairs[:limit]


def top_prototype_pairs(
    prototypes: np.ndarray,
    label_names: Sequence[str],
    limit: int = 12,
) -> list[dict[str, Any]]:
    similarities = l2_normalize(prototypes) @ l2_normalize(prototypes).T
    pairs: list[dict[str, Any]] = []
    for left in range(len(label_names)):
        for right in range(left + 1, len(label_names)):
            pairs.append(
                {
                    "label_a": label_names[left],
                    "label_b": label_names[right],
                    "cosine_similarity": float(similarities[left, right]),
                }
            )
    pairs.sort(key=lambda row: row["cosine_similarity"], reverse=True)
    return pairs[:limit]


def diagnose_dataset(
    dataset_name: str,
    payload: dict[str, Any],
    max_k: int,
    silhouette_samples: int,
    representatives: int,
    seed: int,
) -> dict[str, Any]:
    train_encoder_features = l2_normalize(payload["train_encoder_features"])
    train_projected_features = l2_normalize(payload["train_projected_features"])
    train_labels = payload["train_labels"]
    train_texts = payload["train_texts"]
    prototypes = l2_normalize(payload["prototypes"])
    prototype_predictions = np.argmax(train_projected_features @ prototypes.T, axis=1)
    classes: dict[str, Any] = {}
    for class_id, label in enumerate(payload["label_names"]):
        indices = np.flatnonzero(train_labels == class_id)
        classes[label] = diagnose_class(
            class_id=class_id,
            label=label,
            encoder_features=train_encoder_features[indices],
            projected_features=train_projected_features[indices],
            texts=[train_texts[index] for index in indices],
            prototypes=prototypes,
            all_prototype_predictions=prototype_predictions[indices],
            max_k=max_k,
            silhouette_samples=silhouette_samples,
            representative_count=representatives,
            seed=seed,
        )
    valid_accuracy = float(
        (payload["valid_predictions"] == payload["valid_labels"]).mean()
    )
    return {
        "dataset": dataset_name,
        "train_size": len(train_labels),
        "valid_size": len(payload["valid_labels"]),
        "valid_accuracy_from_checkpoint": valid_accuracy,
        "checkpoint": payload["checkpoint"],
        "cache_reused": payload["cache_reused"],
        "classes": classes,
        "validation_confusions": top_confusion_pairs(
            payload["valid_labels"],
            payload["valid_predictions"],
            payload["label_names"],
        ),
        "nearest_label_prototypes": top_prototype_pairs(
            prototypes,
            payload["label_names"],
        ),
    }


def pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def short_text(text: str, limit: int = 115) -> str:
    text = " ".join(text.split()).replace("|", "｜")
    return text if len(text) <= limit else text[: limit - 1] + "…"


def report_lines(result: dict[str, Any]) -> Iterable[str]:
    yield f"# {result['dataset']}：PSCL 表示空间诊断"
    yield ""
    yield (
        f"- 训练集：{result['train_size']}；验证集：{result['valid_size']}；"
        f"检查点验证准确率：{result['valid_accuracy_from_checkpoint']:.4f}"
    )
    yield "- 结构选择仅使用训练集聚类和验证集混淆，不使用测试集。"
    yield ""
    yield "## 类内多原型证据"
    yield ""
    yield "| 类别 | 编码空间K=1距离 | PSCL投影后距离 | 标签原型路由错误 | 候选K | 建议K | 证据 | 最佳轮廓系数 | 距离下降 | 最小簇占比 |"
    yield "|---|---:|---:|---:|---:|---:|---|---:|---:|---:|"
    for label, class_result in result["classes"].items():
        candidate_k = class_result["candidate_k"]
        clustering = class_result["clusterings"].get(str(candidate_k))
        silhouette = clustering["silhouette_cosine"] if clustering else float("nan")
        reduction = (
            clustering["distortion_reduction_vs_k1"] if clustering else 0.0
        )
        min_fraction = clustering["minimum_cluster_fraction"] if clustering else 1.0
        yield (
            f"| {label} | {class_result['encoder_single_centroid']['mean_cosine_distance']:.4f} | "
            f"{class_result['projected_single_centroid']['mean_cosine_distance']:.4f} | "
            f"{pct(class_result['label_prototype']['routing_error_rate'])} | "
            f"{candidate_k} | {class_result['recommended_k']} | "
            f"{class_result['multi_prototype_evidence']} | "
            f"{silhouette:.4f} | {pct(reduction)} | {pct(min_fraction)} |"
        )
    yield ""
    yield "## 验证集主要混淆"
    yield ""
    yield "| 真实类别 | 预测类别 | 数量 | 类内错误率 |"
    yield "|---|---|---:|---:|"
    for row in result["validation_confusions"]:
        yield (
            f"| {row['true_label']} | {row['predicted_label']} | {row['count']} | "
            f"{pct(row['rate_within_true_class'])} |"
        )
    yield ""
    yield "## 标签原型最相近类别"
    yield ""
    yield "| 类别A | 类别B | 余弦相似度 |"
    yield "|---|---|---:|"
    for row in result["nearest_label_prototypes"]:
        yield (
            f"| {row['label_a']} | {row['label_b']} | "
            f"{row['cosine_similarity']:.4f} |"
        )
    yield ""
    yield "## 候选簇代表文本"
    yield ""
    for label, class_result in result["classes"].items():
        candidate_k = class_result["candidate_k"]
        clustering = class_result["clusterings"].get(str(candidate_k))
        if clustering is None:
            continue
        yield f"### {label}（K={candidate_k}）"
        yield ""
        for cluster in clustering["representatives"]:
            factor_summary = "；".join(
                f"{FACTOR_NAMES[factor]} {pct(rate)}"
                for factor, rate in cluster["factor_rates"].items()
            )
            yield (
                f"**簇 {cluster['cluster'] + 1}**：n={cluster['size']}，"
                f"占比={pct(cluster['fraction'])}；{factor_summary}"
            )
            yield ""
            for row in cluster["texts"]:
                yield (
                    f"- `{row['cosine_similarity']:.3f}` "
                    f"{short_text(row['text'])}"
                )
            yield ""


def write_cluster_csv(path: Path, result: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "dataset",
                "label",
                "k",
                "silhouette_cosine",
                "mean_cosine_distance",
                "distortion_reduction_vs_k1",
                "minimum_cluster_fraction",
                "candidate_k",
                "recommended_k",
                "evidence",
            ],
        )
        writer.writeheader()
        for label, class_result in result["classes"].items():
            for k, clustering in class_result["clusterings"].items():
                writer.writerow(
                    {
                        "dataset": result["dataset"],
                        "label": label,
                        "k": k,
                        "silhouette_cosine": clustering["silhouette_cosine"],
                        "mean_cosine_distance": clustering["mean_cosine_distance"],
                        "distortion_reduction_vs_k1": clustering[
                            "distortion_reduction_vs_k1"
                        ],
                        "minimum_cluster_fraction": clustering[
                            "minimum_cluster_fraction"
                        ],
                        "candidate_k": class_result["candidate_k"],
                        "recommended_k": class_result["recommended_k"],
                        "evidence": class_result["multi_prototype_evidence"],
                    }
                )


def global_summary_lines(results: dict[str, dict[str, Any]]) -> Iterable[str]:
    yield "# PSCL 单原型假设诊断汇总"
    yield ""
    yield "本诊断复用 seed=42 的已训练 PSCL 检查点，仅使用训练集和验证集决定模型结构。"
    yield "轮廓系数和距离下降用于判断一个类别是否存在多个稳定语义子簇；代表文本用于核查子簇是否具有可解释的诈骗语义。"
    yield ""
    yield "| 数据集 | 类别 | 编码空间K=1距离 | PSCL投影后距离 | 压缩率 | 候选K | 建议K | 证据 | 最佳轮廓系数 | 距离下降 |"
    yield "|---|---|---:|---:|---:|---:|---:|---|---:|---:|"
    for dataset_name, result in results.items():
        for label, class_result in result["classes"].items():
            candidate_k = class_result["candidate_k"]
            clustering = class_result["clusterings"].get(str(candidate_k))
            silhouette = (
                f"{clustering['silhouette_cosine']:.4f}"
                if clustering is not None
                else "N/A"
            )
            reduction = (
                pct(clustering["distortion_reduction_vs_k1"])
                if clustering is not None
                else "0.0%"
            )
            encoder_distance = class_result["encoder_single_centroid"][
                "mean_cosine_distance"
            ]
            projected_distance = class_result["projected_single_centroid"][
                "mean_cosine_distance"
            ]
            compression = 1.0 - projected_distance / max(encoder_distance, 1e-12)
            yield (
                f"| {dataset_name} | {label} | {encoder_distance:.4f} | "
                f"{projected_distance:.4f} | {pct(compression)} | {candidate_k} | "
                f"{class_result['recommended_k']} | "
                f"{class_result['multi_prototype_evidence']} | "
                f"{silhouette} | {reduction} |"
            )
    yield ""
    yield "说明：strong/moderate/weak 由预先设定的描述性阈值产生，不是显著性检验；弱证据类别保留单原型。"


def main() -> None:
    args = parse_args()
    if args.run_dir is not None and len(args.datasets) != 1:
        raise ValueError("--run-dir requires exactly one value in --datasets")
    set_seed(args.seed)
    if not 2 <= args.max_k <= 8:
        raise ValueError("--max-k must be between 2 and 8")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not args.allow_cpu:
        raise RuntimeError(
            "CUDA is unavailable. Use a CUDA-enabled PyTorch installation or "
            "pass --allow-cpu explicitly."
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(
        json.dumps(
            {
                "device": str(device),
                "gpu": torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else None,
                "datasets": args.datasets,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    all_results: dict[str, dict[str, Any]] = {}
    for dataset_name in args.datasets:
        dataset_output = args.output_dir / dataset_name
        dataset_output.mkdir(parents=True, exist_ok=True)
        payload = load_or_extract_features(
            dataset_name=dataset_name,
            run_dir=(
                args.run_dir
                if args.run_dir is not None
                else DEFAULT_RUNS[dataset_name]
            ),
            output_dir=dataset_output,
            batch_size=args.batch_size,
            device=device,
            force_extract=args.force_extract,
        )
        result = diagnose_dataset(
            dataset_name=dataset_name,
            payload=payload,
            max_k=args.max_k,
            silhouette_samples=args.silhouette_samples,
            representatives=args.representatives,
            seed=args.seed,
        )
        save_json(dataset_output / "diagnostics.json", result)
        (dataset_output / "report.md").write_text(
            "\n".join(report_lines(result)) + "\n",
            encoding="utf-8",
        )
        write_cluster_csv(dataset_output / "cluster_metrics.csv", result)
        all_results[dataset_name] = result

    (args.output_dir / "summary.md").write_text(
        "\n".join(global_summary_lines(all_results)) + "\n",
        encoding="utf-8",
    )
    save_json(args.output_dir / "summary.json", all_results)
    print(f"Wrote {args.output_dir / 'summary.md'}", flush=True)


if __name__ == "__main__":
    main()
