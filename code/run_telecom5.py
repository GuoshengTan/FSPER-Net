"""Run FSPER-Net and three core baselines on Telecom_Fraud_Texts_5.

The suite fixes a stratified seed-42 80/10/10 split and changes only the
training seed. FSPER-Net keeps the FGRC hyperparameters fixed and is reported
using its protected classifier branch. Completed stages are skipped and
interrupted neural stages resume from ``latest_checkpoint.pt``.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
from sklearn.model_selection import train_test_split

from split_manifest import dataset_fingerprint, split_from_manifest
from train_published_fraud_models import load_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = Path(__file__).resolve().parent
PYTHON = Path(sys.executable)
PSCL_SCRIPT = CODE_DIR / "train_published_fraud_models.py"
DIAGNOSTIC_SCRIPT = CODE_DIR / "diagnose_pscl_representation.py"
V2_SCRIPT = CODE_DIR / "train_fs_pscl.py"
FSPER_SCRIPT = CODE_DIR / "train_sparse_routed_pscl.py"
DATA_DIR = PROJECT_ROOT / "data" / "Telecom_Fraud_Texts_5"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "telecom5"
SEEDS = (42, 2024, 2026)
METRICS = ("accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1")

EXTERNAL_METHODS = {
    "roberta_wwm": {
        "display_name": "Chinese RoBERTa-WWM fine-tuning",
        "category": "backbone_control",
        "source": "Liu et al. (2019); Cui et al. (2021)",
        "args": (
            "--batch-size", "16",
            "--lr", "2e-5",
            "--head-lr", "2e-4",
        ),
    },
    "pscl": {
        "display_name": "PSCL (single-model core)",
        "category": "published_fraud_model",
        "source": "Xiong et al. (CCL 2023)",
        "args": (
            "--batch-size", "16",
            "--lr", "2e-5",
            "--head-lr", "2e-4",
            "--pscl-temperature", "0.07",
            "--pscl-similarity-gamma", "0.1",
            "--ldam-max-margin", "0.5",
        ),
    },
    "roberta_mharc": {
        "display_name": "RoBERTa-MHARC",
        "category": "published_fraud_model",
        "source": "Li, Zhang, and Jiang (2024)",
        "args": (
            "--batch-size", "8",
            "--lr", "1e-5",
            "--head-lr", "1e-5",
            "--num-heads", "12",
            "--mharc-subspace-weight", "0.01",
            "--mharc-position-weight", "0.01",
            "--mharc-representation-weight", "0.01",
        ),
    },
}


@dataclass(frozen=True)
class SeedPaths:
    seed: int
    root: Path
    pscl: Path
    diagnostics: Path
    features: Path
    v2: Path
    fsper: Path


@dataclass(frozen=True)
class Job:
    job_id: str
    output_dir: Path
    marker: Path
    required_paths: tuple[Path, ...]
    command_builder: Callable[[], list[str]]
    log_path: Path


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_seeds(raw: str) -> tuple[int, ...]:
    seeds = tuple(int(item.strip()) for item in raw.split(",") if item.strip())
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError(f"Invalid seeds: {raw}")
    return seeds


def build_fixed_split_manifest(output_root: Path, split_seed: int) -> Path:
    args = argparse.Namespace(
        dataset="telecom5",
        data_dir=DATA_DIR,
        max_samples_per_class=0,
        seed=split_seed,
    )
    texts, raw_labels, groups = load_dataset(args)
    if groups is not None:
        raise RuntimeError("Telecom_Fraud_Texts_5 is expected to be sample-level data.")
    label_names = sorted(set(raw_labels))
    label_to_id = {label: index for index, label in enumerate(label_names)}
    labels = np.asarray([label_to_id[label] for label in raw_labels], dtype=np.int64)
    indices = np.arange(len(texts))
    train_indices, temporary_indices = train_test_split(
        indices,
        test_size=0.2,
        random_state=split_seed,
        stratify=labels,
    )
    valid_indices, test_indices = train_test_split(
        temporary_indices,
        test_size=0.5,
        random_state=split_seed,
        stratify=labels[temporary_indices],
    )
    role_indices = {
        "train": np.asarray(train_indices, dtype=np.int64),
        "valid": np.asarray(valid_indices, dtype=np.int64),
        "test": np.asarray(test_indices, dtype=np.int64),
    }
    manifest_path = output_root / "fixed_split_seed42.json"
    manifest = {
        "schema_version": 1,
        "purpose": "fsper_telecom5_fixed_split_formal_validation",
        "dataset": "telecom5",
        "sample_count": len(texts),
        "dataset_fingerprint": dataset_fingerprint(texts, labels, groups),
        "split_seed": split_seed,
        "training_seed": None,
        "test_role": "final_test",
        "indices": {
            role: values.tolist() for role, values in role_indices.items()
        },
        "label_names": label_names,
        "label_distribution": {
            role: dict(Counter(raw_labels[index] for index in values))
            for role, values in role_indices.items()
        },
        "group_aware": False,
        "group_counts": None,
    }
    save_json(manifest_path, manifest)
    split_from_manifest(
        manifest_path,
        texts,
        labels.tolist(),
        groups,
        expected_dataset="telecom5",
    )
    return manifest_path


def seed_paths(output_root: Path, seed: int) -> SeedPaths:
    root = output_root / f"seed_{seed}"
    diagnostics = root / "diagnostics"
    return SeedPaths(
        seed=seed,
        root=root,
        pscl=root / "pscl_epoch20",
        diagnostics=diagnostics,
        features=diagnostics / "telecom5" / "features_cache.npz",
        v2=root / "v2_compat_epoch14",
        fsper=root / "fsper_epoch16",
    )


def with_resume(command: list[str], output_dir: Path) -> list[str]:
    latest = output_dir / "latest_checkpoint.pt"
    if latest.exists():
        command.extend(("--resume", str(latest)))
    return command


def pscl_command(paths: SeedPaths, manifest: Path) -> list[str]:
    return with_resume(
        [
            str(PYTHON), "-X", "utf8", str(PSCL_SCRIPT),
            "--model", "pscl",
            "--dataset", "telecom5",
            "--data-dir", str(DATA_DIR),
            "--output-dir", str(paths.pscl),
            "--allow-download",
            "--split-manifest", str(manifest),
            "--epochs", "20",
            "--batch-size", "16",
            "--max-len", "192",
            "--lr", "1.5e-5",
            "--head-lr", "1.5e-4",
            "--warmup-ratio", "0.15",
            "--pscl-temperature", "0.07",
            "--pscl-similarity-gamma", "0.1",
            "--ldam-max-margin", "0.5",
            "--weighted-sampler",
            "--save-best-by", "valid_macro_f1",
            "--seed", str(paths.seed),
        ],
        paths.pscl,
    )


def diagnostic_command(paths: SeedPaths) -> list[str]:
    return [
        str(PYTHON), "-X", "utf8", str(DIAGNOSTIC_SCRIPT),
        "--datasets", "telecom5",
        "--run-dir", str(paths.pscl),
        "--output-dir", str(paths.diagnostics),
        "--batch-size", "64",
        "--max-k", "4",
        "--silhouette-samples", "2000",
        "--representatives", "3",
        "--seed", str(paths.seed),
    ]


def v2_command(paths: SeedPaths, manifest: Path) -> list[str]:
    return with_resume(
        [
            str(PYTHON), "-X", "utf8", str(V2_SCRIPT),
            "--dataset", "telecom5",
            "--data-dir", str(DATA_DIR),
            "--output-dir", str(paths.v2),
            "--allow-download",
            "--warm-start", str(paths.pscl / "best_model.pt"),
            "--feature-cache", str(paths.features),
            "--split-manifest", str(manifest),
            "--split-seed", "42",
            "--compat-v2",
            "--epochs", "14",
            "--batch-size", "16",
            "--max-len", "192",
            "--lr", "3e-6",
            "--head-lr", "7.5e-5",
            "--warmup-ratio", "0.15",
            "--fusion-weight", "0.1",
            "--fusion-gate-init", "0.05",
            "--classification-weight", "1.0",
            "--script-loss-weight", "0.2",
            "--similarity-gamma", "0.1",
            "--ldam-max-margin", "0.5",
            "--rival-weight", "0.1",
            "--rival-base-margin", "0.1",
            "--rival-similarity-scale", "0.2",
            "--diversity-max-similarity", "0.85",
            "--alignment-weight", "0.05",
            "--early-stopping-patience", "0",
            "--weighted-sampler",
            "--amp",
            "--save-best-by", "valid_macro_f1",
            "--seed", str(paths.seed),
        ],
        paths.v2,
    )


def fsper_command(paths: SeedPaths, manifest: Path) -> list[str]:
    return with_resume(
        [
            str(PYTHON), "-X", "utf8", str(FSPER_SCRIPT),
            "--dataset", "telecom5",
            "--data-dir", str(DATA_DIR),
            "--output-dir", str(paths.fsper),
            "--allow-download",
            "--warm-start", str(paths.v2 / "best_model.pt"),
            "--feature-cache", str(paths.features),
            "--split-manifest", str(manifest),
            "--split-seed", "42",
            "--ablation", "full",
            "--epochs", "16",
            "--batch-size", "16",
            "--max-len", "192",
            "--lr", "1e-6",
            "--head-lr", "2.5e-5",
            "--warmup-ratio", "0.15",
            "--route-strength-max", "0.5",
            "--dynamic-gate-init", "0.05",
            "--architecture-gate-init", "0.5",
            "--hard-concrete-temperature", "0.67",
            "--hard-concrete-gamma", "-0.1",
            "--hard-concrete-zeta", "1.1",
            "--base-loss-weight", "0.5",
            "--source-loss-weight", "0.05",
            "--router-loss-weight", "0.2",
            "--sparsity-weight", "0.0005",
            "--route-usage-weight", "0.001",
            "--router-target-temperature", "0.2",
            "--router-benefit-margin", "0.05",
            "--disagreement-router-weight", "3.0",
            "--classification-weight", "1.0",
            "--similarity-gamma", "0.1",
            "--ldam-max-margin", "0.5",
            "--centroid-momentum", "0.995",
            "--semantic-anchor-momentum", "0.99",
            "--early-stopping-patience", "0",
            "--weighted-sampler",
            "--amp",
            "--save-best-by", "valid_macro_f1",
            "--seed", str(paths.seed),
        ],
        paths.fsper,
    )


def external_output_dir(paths: SeedPaths, method: str) -> Path:
    return paths.root / f"external_{method}_epoch15"


def external_command(
    paths: SeedPaths,
    method: str,
    manifest: Path,
) -> list[str]:
    output_dir = external_output_dir(paths, method)
    return with_resume(
        [
            str(PYTHON), "-X", "utf8", str(PSCL_SCRIPT),
            "--model", method,
            "--dataset", "telecom5",
            "--data-dir", str(DATA_DIR),
            "--output-dir", str(output_dir),
            "--allow-download",
            "--split-manifest", str(manifest),
            "--epochs", "15",
            "--max-len", "192",
            "--weighted-sampler",
            "--save-best-by", "valid_macro_f1",
            "--seed", str(paths.seed),
            *EXTERNAL_METHODS[method]["args"],
        ],
        output_dir,
    )


def build_jobs(
    paths: SeedPaths,
    manifest: Path,
    output_root: Path,
) -> list[Job]:
    log_dir = output_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    def job(
        stage: str,
        output_dir: Path,
        required_paths: tuple[Path, ...],
        builder: Callable[[], list[str]],
    ) -> Job:
        return Job(
            job_id=f"seed{paths.seed}__{stage}",
            output_dir=output_dir,
            marker=output_dir / "stage_complete.json",
            required_paths=required_paths,
            command_builder=builder,
            log_path=log_dir / f"seed{paths.seed}__{stage}.log",
        )

    jobs = [
        job(
            "pscl",
            paths.pscl,
            (paths.pscl / "final_summary.json", paths.pscl / "best_model.pt"),
            lambda: pscl_command(paths, manifest),
        ),
        job(
            "diagnostics",
            paths.diagnostics,
            (paths.features,),
            lambda: diagnostic_command(paths),
        ),
        job(
            "v2_compat",
            paths.v2,
            (paths.v2 / "final_summary.json", paths.v2 / "best_model.pt"),
            lambda: v2_command(paths, manifest),
        ),
        job(
            "fsper",
            paths.fsper,
            (paths.fsper / "final_summary.json", paths.fsper / "best_model.pt"),
            lambda: fsper_command(paths, manifest),
        ),
    ]
    for method in EXTERNAL_METHODS:
        output_dir = external_output_dir(paths, method)
        jobs.append(
            job(
                f"external_{method}",
                output_dir,
                (output_dir / "final_summary.json", output_dir / "best_model.pt"),
                lambda method=method: external_command(paths, method, manifest),
            )
        )
    return jobs


def run_job(job: Job) -> float:
    job.output_dir.mkdir(parents=True, exist_ok=True)
    command = job.command_builder()
    started = time.time()
    print(f"[run][GPU] {job.job_id}", flush=True)
    with job.log_path.open("a", encoding="utf-8") as log_handle:
        log_handle.write(f"\n=== started {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        log_handle.write(subprocess.list2cmdline(command) + "\n")
        log_handle.flush()
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_handle.write(line)
            log_handle.flush()
        return_code = process.wait()
    duration = time.time() - started
    missing = [str(path) for path in job.required_paths if not path.exists()]
    if return_code != 0 or missing:
        raise RuntimeError(
            f"Stage failed: {job.job_id}; return_code={return_code}; "
            f"missing={missing}; log={job.log_path}"
        )
    save_json(
        job.marker,
        {
            "status": "completed",
            "job_id": job.job_id,
            "duration_seconds": duration,
            "required_paths_verified": [str(path) for path in job.required_paths],
            "command": command,
        },
    )
    print(f"[completed] {job.job_id} ({duration / 3600.0:.2f} h)", flush=True)
    return duration


def summary_metric_row(
    method: str,
    seed: int,
    summary_path: Path,
) -> dict[str, Any]:
    summary = read_json(summary_path)
    return {
        "method": method,
        "display_name": EXTERNAL_METHODS[method]["display_name"],
        "seed": seed,
        "best_epoch": int(summary["best_epoch"]),
        "accuracy": float(summary["test_accuracy"]),
        "macro_precision": float(summary["test_macro_precision"]),
        "macro_recall": float(summary["test_macro_recall"]),
        "macro_f1": float(summary["test_macro_f1"]),
        "weighted_f1": float(summary["test_weighted_f1"]),
        "summary_path": str(summary_path),
    }


def fsper_metric_row(paths: SeedPaths) -> dict[str, Any] | None:
    summary_path = paths.fsper / "final_summary.json"
    if not summary_path.exists():
        return None
    summary = read_json(summary_path)
    classifier = summary["branch_metrics"]["classifier"]
    fused = summary["branch_metrics"]["fused"]
    return {
        "method": "fsper",
        "display_name": "FSPER-Net (protected inference)",
        "seed": paths.seed,
        "best_epoch": int(summary["best_epoch"]),
        **{metric: float(classifier[metric]) for metric in METRICS},
        "fused_macro_f1": float(fused["macro_f1"]),
        "fused_accuracy": float(fused["accuracy"]),
        "summary_path": str(summary_path),
    }


def metric_stats(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(statistics.mean(values)),
        "std": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
    }


def write_aggregate(
    paths_by_seed: dict[int, SeedPaths],
    output_root: Path,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for paths in paths_by_seed.values():
        row = fsper_metric_row(paths)
        if row is not None:
            rows.append(row)
    for method in EXTERNAL_METHODS:
        for seed, paths in paths_by_seed.items():
            summary_path = external_output_dir(paths, method) / "final_summary.json"
            if summary_path.exists():
                rows.append(summary_metric_row(method, seed, summary_path))

    method_specs = {
        "fsper": {
            "display_name": "FSPER-Net (protected inference)",
            "category": "proposed_method",
            "source": "This study",
        },
        **EXTERNAL_METHODS,
    }
    methods = []
    for method in ("fsper", "roberta_wwm", "pscl", "roberta_mharc"):
        method_rows = [row for row in rows if row["method"] == method]
        entry: dict[str, Any] = {
            "method": method,
            "display_name": method_specs[method]["display_name"],
            "category": method_specs[method]["category"],
            "source": method_specs[method]["source"],
            "completed_runs": len(method_rows),
            "expected_runs": len(paths_by_seed),
            "complete": len(method_rows) == len(paths_by_seed),
            "rows": method_rows,
        }
        if method_rows:
            entry["metrics"] = {
                metric: metric_stats([row[metric] for row in method_rows])
                for metric in METRICS
            }
        methods.append(entry)

    ranking = sorted(
        [entry for entry in methods if entry["complete"]],
        key=lambda entry: entry["metrics"]["macro_f1"]["mean"],
        reverse=True,
    )
    payload = {
        "generated_at_unix": time.time(),
        "protocol": {
            "dataset": "Telecom_Fraud_Texts_5",
            "fixed_split_seed": 42,
            "training_seeds": list(paths_by_seed),
            "selection_metric": "validation Macro-F1",
            "test_set_used_for_selection": False,
            "fsper_inference": "protected classifier branch",
            "fsper_hyperparameters_frozen_from_fgrc": True,
            "external_baselines_rerun_on_fixed_manifest": True,
        },
        "complete": all(entry["complete"] for entry in methods),
        "methods": methods,
        "ranking_by_macro_f1": [
            {
                "rank": index + 1,
                "method": entry["method"],
                "display_name": entry["display_name"],
                "accuracy_mean": entry["metrics"]["accuracy"]["mean"],
                "accuracy_std": entry["metrics"]["accuracy"]["std"],
                "macro_f1_mean": entry["metrics"]["macro_f1"]["mean"],
                "macro_f1_std": entry["metrics"]["macro_f1"]["std"],
            }
            for index, entry in enumerate(ranking)
        ],
    }
    save_json(output_root / "fair_comparison.json", payload)
    with (output_root / "fair_comparison.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        fields = [
            "rank", "method", "display_name", "accuracy_mean", "accuracy_std",
            "macro_f1_mean", "macro_f1_std",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(payload["ranking_by_macro_f1"])
    completed = sum(entry["completed_runs"] for entry in methods)
    expected = sum(entry["expected_runs"] for entry in methods)
    print(f"[summary] {completed}/{expected} model-seed results available.", flush=True)
    return payload


def validate_environment() -> None:
    required = (
        PYTHON,
        PSCL_SCRIPT,
        DIAGNOSTIC_SCRIPT,
        V2_SCRIPT,
        FSPER_SCRIPT,
        DATA_DIR / "label00-last.csv",
    )
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)
    probe = subprocess.run(
        [
            str(PYTHON), "-c",
            "import torch; print(torch.cuda.is_available()); "
            "print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO_CUDA')",
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    lines = [line.strip() for line in probe.stdout.splitlines() if line.strip()]
    if not lines or lines[0] != "True":
        raise RuntimeError("CUDA is unavailable; Telecom5 formal validation was not started.")
    print(f"[environment] CUDA ready: {lines[1] if len(lines) > 1 else 'GPU'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="42,2024,2026")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)
    if 42 not in seeds:
        raise ValueError("Seed 42 is required by the frozen formal protocol.")
    if args.split_seed != 42:
        raise ValueError("This formal protocol is frozen to split seed 42.")
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    validate_environment()
    manifest = build_fixed_split_manifest(output_root, args.split_seed)
    paths_by_seed = {seed: seed_paths(output_root, seed) for seed in seeds}
    jobs_by_seed = {
        seed: build_jobs(paths, manifest, output_root)
        for seed, paths in paths_by_seed.items()
    }
    all_jobs = [job for seed in seeds for job in jobs_by_seed[seed]]
    save_json(
        output_root / "suite_manifest.json",
        {
            "protocol": {
                "dataset": "Telecom_Fraud_Texts_5",
                "fixed_split_seed": 42,
                "training_seeds": list(seeds),
                "fsper_stage_epochs": {"pscl": 20, "prototype_stabilization": 14, "routing": 16},
                "external_epochs": 15,
                "selection_metric": "validation Macro-F1",
                "fsper_hyperparameters_frozen_from_fgrc": True,
                "test_used_for_model_selection": False,
            },
            "split_manifest": str(manifest),
            "jobs": [
                {
                    "job_id": job.job_id,
                    "marker": str(job.marker),
                    "command": job.command_builder(),
                }
                for job in all_jobs
            ],
        },
    )
    if args.summary_only:
        result = write_aggregate(paths_by_seed, output_root)
        print(json.dumps(result["ranking_by_macro_f1"], ensure_ascii=False, indent=2))
        return

    complete = sum(job.marker.exists() for job in all_jobs)
    print(
        f"[plan] {len(all_jobs)} stages; {complete} complete; "
        f"{len(all_jobs) - complete} pending.",
        flush=True,
    )
    for job in all_jobs:
        state = "skip" if job.marker.exists() else "run"
        print(f"[{state}] {job.job_id}", flush=True)
        if args.dry_run and state == "run":
            print("       " + subprocess.list2cmdline(job.command_builder()), flush=True)
    if args.dry_run:
        write_aggregate(paths_by_seed, output_root)
        return

    status_path = output_root / "run_status.json"
    statuses = read_json(status_path) if status_path.exists() else {}
    for seed in seeds:
        for job in jobs_by_seed[seed]:
            if job.marker.exists():
                statuses[job.job_id] = {"status": "skipped_complete"}
                save_json(status_path, statuses)
                continue
            try:
                duration = run_job(job)
            except Exception as exc:
                statuses[job.job_id] = {
                    "status": "failed",
                    "error": str(exc),
                    "log_path": str(job.log_path),
                }
                save_json(status_path, statuses)
                write_aggregate(paths_by_seed, output_root)
                raise
            statuses[job.job_id] = {
                "status": "completed",
                "duration_seconds": duration,
                "marker": str(job.marker),
                "log_path": str(job.log_path),
            }
            save_json(status_path, statuses)
            write_aggregate(paths_by_seed, output_root)

    result = write_aggregate(paths_by_seed, output_root)
    print(json.dumps(result["ranking_by_macro_f1"], ensure_ascii=False, indent=2))
    print("[done] Telecom5 formal validation completed.", flush=True)


if __name__ == "__main__":
    main()
