"""Train and validate FSPER-Net on FGRC-SCD with three random seeds.

The suite fixes the original FGRC-SCD seed-42 group-aware data split and
changes only the training seed. For every seed it runs the complete dependency
chain: PSCL -> diagnostics -> script-prototype stabilization -> FSPER-Net.
The seed-42 run is checked against the paper result within a configurable
tolerance before the remaining seeds run.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Callable

import numpy as np
from sklearn.model_selection import GroupShuffleSplit

from split_manifest import dataset_fingerprint, split_from_manifest
from train_published_fraud_models import load_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = Path(__file__).resolve().parent
PYTHON = Path(sys.executable)
PSCL_SCRIPT = CODE_DIR / "train_published_fraud_models.py"
DIAGNOSTIC_SCRIPT = CODE_DIR / "diagnose_pscl_representation.py"
V2_SCRIPT = CODE_DIR / "train_fs_pscl.py"
FSPER_SCRIPT = CODE_DIR / "train_sparse_routed_pscl.py"
DATA_DIR = PROJECT_ROOT / "data" / "FGRC-SCD" / "sms" / "message"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "fgrc_scd"
DEFAULT_SEEDS = (42, 2024, 2026)
PAPER_SEED42_MACRO_F1 = 0.8444217452739744
PAPER_SEED42_ACCURACY = 0.9322728763529026
EXPECTED_SPLIT_SIZES = (48104, 6055, 6098)


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
    stage: str
    output_dir: Path
    marker: Path
    required_paths: tuple[Path, ...]
    command_builder: Callable[[], list[str]]
    log_path: Path


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def parse_seeds(raw: str) -> tuple[int, ...]:
    seeds = tuple(int(value.strip()) for value in raw.split(",") if value.strip())
    if not seeds:
        raise ValueError("At least one training seed is required.")
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"Duplicate training seeds: {seeds}")
    return seeds


def build_fixed_split_manifest(output_root: Path, split_seed: int) -> Path:
    args = argparse.Namespace(
        dataset="fgrc_scd",
        data_dir=DATA_DIR,
        max_samples_per_class=0,
        seed=split_seed,
    )
    texts, raw_labels, groups = load_dataset(args)
    if groups is None:
        raise RuntimeError("FGRC-SCD must provide conversation groups.")
    label_names = sorted(set(raw_labels))
    label_to_id = {label: index for index, label in enumerate(label_names)}
    labels = np.asarray([label_to_id[label] for label in raw_labels], dtype=np.int64)
    group_array = np.asarray(groups, dtype=object)
    indices = np.arange(len(texts))

    outer = GroupShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=split_seed,
    )
    train_indices, temporary_indices = next(
        outer.split(indices, labels, group_array)
    )
    inner = GroupShuffleSplit(
        n_splits=1,
        test_size=0.5,
        random_state=split_seed,
    )
    valid_relative, test_relative = next(
        inner.split(
            temporary_indices,
            labels[temporary_indices],
            group_array[temporary_indices],
        )
    )
    valid_indices = temporary_indices[valid_relative]
    test_indices = temporary_indices[test_relative]
    sizes = (
        len(train_indices),
        len(valid_indices),
        len(test_indices),
    )
    if split_seed == 42 and sizes != EXPECTED_SPLIT_SIZES:
        raise RuntimeError(
            "The reconstructed split does not match the paper split: "
            f"expected={EXPECTED_SPLIT_SIZES}, actual={sizes}."
        )

    manifest_path = output_root / "fixed_split_seed42.json"
    role_indices = {
        "train": train_indices,
        "valid": valid_indices,
        "test": test_indices,
    }
    manifest = {
        "schema_version": 1,
        "purpose": "fsper_three_seed_fixed_split_validation",
        "dataset": "fgrc_scd",
        "sample_count": len(texts),
        "dataset_fingerprint": dataset_fingerprint(texts, labels, groups),
        "split_seed": split_seed,
        "training_seed": None,
        "test_role": "final_test",
        "indices": {
            role: values.astype(np.int64).tolist()
            for role, values in role_indices.items()
        },
        "label_names": label_names,
        "label_distribution": {
            role: dict(Counter(raw_labels[index] for index in values))
            for role, values in role_indices.items()
        },
        "group_aware": True,
        "group_counts": {
            role: len({groups[index] for index in values})
            for role, values in role_indices.items()
        },
    }
    save_json(manifest_path, manifest)
    split_from_manifest(
        manifest_path,
        texts,
        labels.tolist(),
        groups,
        expected_dataset="fgrc_scd",
    )
    return manifest_path


def seed_paths(output_root: Path, seed: int, pscl_epochs: int, v2_epochs: int, fsper_epochs: int) -> SeedPaths:
    root = output_root / f"seed_{seed}"
    pscl = root / f"pscl_epoch{pscl_epochs}"
    diagnostics = root / "diagnostics"
    return SeedPaths(
        seed=seed,
        root=root,
        pscl=pscl,
        diagnostics=diagnostics,
        features=diagnostics / "fgrc_scd" / "features_cache.npz",
        v2=root / f"fs_pscl_v2_compat_epoch{v2_epochs}",
        fsper=root / f"fsper_epoch{fsper_epochs}",
    )


def with_resume(command: list[str], directory: Path) -> list[str]:
    latest = directory / "latest_checkpoint.pt"
    if latest.exists():
        command.extend(("--resume", str(latest)))
    return command


def pscl_command(
    paths: SeedPaths,
    manifest: Path,
    args: argparse.Namespace,
) -> list[str]:
    return with_resume(
        [
            str(PYTHON), "-X", "utf8", str(PSCL_SCRIPT),
            "--model", "pscl",
            "--dataset", "fgrc_scd",
            "--data-dir", str(DATA_DIR),
            "--output-dir", str(paths.pscl),
            "--allow-download",
            "--split-manifest", str(manifest),
            "--epochs", str(args.pscl_epochs),
            "--batch-size", "16",
            "--max-len", "192",
            "--lr", str(args.pscl_lr),
            "--head-lr", str(args.pscl_head_lr),
            "--warmup-ratio", str(args.pscl_warmup_ratio),
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
        "--datasets", "fgrc_scd",
        "--run-dir", str(paths.pscl),
        "--output-dir", str(paths.diagnostics),
        "--batch-size", "64",
        "--max-k", "4",
        "--silhouette-samples", "2000",
        "--representatives", "3",
        "--seed", str(paths.seed),
    ]


def v2_command(
    paths: SeedPaths,
    manifest: Path,
    args: argparse.Namespace,
) -> list[str]:
    return with_resume(
        [
            str(PYTHON), "-X", "utf8", str(V2_SCRIPT),
            "--dataset", "fgrc_scd",
            "--data-dir", str(DATA_DIR),
            "--output-dir", str(paths.v2),
            "--allow-download",
            "--warm-start", str(paths.pscl / "best_model.pt"),
            "--feature-cache", str(paths.features),
            "--split-manifest", str(manifest),
            "--split-seed", str(args.split_seed),
            "--compat-v2",
            "--epochs", str(args.v2_epochs),
            "--batch-size", "16",
            "--max-len", "192",
            "--lr", str(args.v2_lr),
            "--head-lr", str(args.v2_head_lr),
            "--warmup-ratio", str(args.v2_warmup_ratio),
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
            "--early-stopping-patience", str(args.v2_early_stopping_patience),
            "--weighted-sampler",
            "--amp",
            "--save-best-by", "valid_macro_f1",
            "--seed", str(paths.seed),
        ],
        paths.v2,
    )


def fsper_command(
    paths: SeedPaths,
    manifest: Path,
    args: argparse.Namespace,
) -> list[str]:
    return with_resume(
        [
            str(PYTHON), "-X", "utf8", str(FSPER_SCRIPT),
            "--dataset", "fgrc_scd",
            "--data-dir", str(DATA_DIR),
            "--output-dir", str(paths.fsper),
            "--allow-download",
            "--warm-start", str(paths.v2 / "best_model.pt"),
            "--feature-cache", str(paths.features),
            "--split-manifest", str(manifest),
            "--split-seed", str(args.split_seed),
            "--epochs", str(args.fsper_epochs),
            "--batch-size", "16",
            "--max-len", "192",
            "--lr", str(args.fsper_lr),
            "--head-lr", str(args.fsper_head_lr),
            "--warmup-ratio", str(args.fsper_warmup_ratio),
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
            "--early-stopping-patience", str(args.fsper_early_stopping_patience),
            "--weighted-sampler",
            "--amp",
            "--save-best-by", "valid_macro_f1",
            "--seed", str(paths.seed),
        ],
        paths.fsper,
    )


def build_jobs(paths: SeedPaths, manifest: Path, args: argparse.Namespace) -> list[Job]:
    log_dir = args.output_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    def job(stage: str, directory: Path, required: tuple[Path, ...], builder: Callable[[], list[str]]) -> Job:
        return Job(
            job_id=f"seed{paths.seed}__{stage}",
            stage=stage,
            output_dir=directory,
            marker=directory / "stage_complete.json",
            required_paths=required,
            command_builder=builder,
            log_path=log_dir / f"seed{paths.seed}__{stage}.log",
        )

    return [
        job(
            "pscl",
            paths.pscl,
            (paths.pscl / "final_summary.json", paths.pscl / "best_model.pt"),
            lambda: pscl_command(paths, manifest, args),
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
            lambda: v2_command(paths, manifest, args),
        ),
        job(
            "fsper",
            paths.fsper,
            (paths.fsper / "final_summary.json", paths.fsper / "best_model.pt"),
            lambda: fsper_command(paths, manifest, args),
        ),
    ]


def run_job(job: Job) -> float:
    job.output_dir.mkdir(parents=True, exist_ok=True)
    command = job.command_builder()
    started = time.time()
    print(f"[run] {job.job_id}", flush=True)
    with job.log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n=== started {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        log.write(subprocess.list2cmdline(command) + "\n")
        log.flush()
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
            log.write(line)
            log.flush()
        return_code = process.wait()
        duration = time.time() - started
        log.write(
            f"\n=== finished return_code={return_code} "
            f"duration_seconds={duration:.1f} ===\n"
        )
    missing = [str(path) for path in job.required_paths if not path.exists()]
    if return_code != 0 or missing:
        raise RuntimeError(
            f"Stage failed: {job.job_id}; missing={missing}. "
            f"Inspect {job.log_path}"
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
    print(f"[completed] {job.job_id} ({duration / 60.0:.1f} min)", flush=True)
    return duration


def load_fsper_summary(paths: SeedPaths) -> dict[str, Any]:
    return json.loads(
        (paths.fsper / "final_summary.json").read_text(encoding="utf-8")
    )


def check_seed42_reproduction(paths: SeedPaths, tolerance_pp: float) -> dict[str, Any]:
    summary = load_fsper_summary(paths)
    macro_f1 = float(summary["test_macro_f1"])
    accuracy = float(summary["test_accuracy"])
    result = {
        "paper_macro_f1": PAPER_SEED42_MACRO_F1,
        "reproduced_macro_f1": macro_f1,
        "macro_f1_difference_pp": 100.0 * (macro_f1 - PAPER_SEED42_MACRO_F1),
        "paper_accuracy": PAPER_SEED42_ACCURACY,
        "reproduced_accuracy": accuracy,
        "accuracy_difference_pp": 100.0 * (accuracy - PAPER_SEED42_ACCURACY),
        "tolerance_pp": tolerance_pp,
        "passed": abs(100.0 * (macro_f1 - PAPER_SEED42_MACRO_F1)) <= tolerance_pp,
    }
    save_json(paths.root.parent / "seed42_reproduction_check.json", result)
    return result


def safe_unlink(path: Path, output_root: Path) -> int:
    if not path.exists():
        return 0
    resolved = path.resolve()
    root = output_root.resolve()
    if root not in resolved.parents:
        raise RuntimeError(f"Refusing to prune path outside output root: {resolved}")
    size = path.stat().st_size
    path.unlink()
    return size


def prune_seed(paths: SeedPaths, output_root: Path) -> dict[str, Any]:
    candidates = list(paths.pscl.glob("*.pt")) + list(paths.v2.glob("*.pt"))
    candidates.append(paths.fsper / "latest_checkpoint.pt")
    deleted = 0
    deleted_bytes = 0
    for path in candidates:
        if path.exists():
            deleted_bytes += safe_unlink(path, output_root)
            deleted += 1
    result = {
        "seed": paths.seed,
        "deleted_checkpoint_count": deleted,
        "freed_gb": deleted_bytes / (1024 ** 3),
        "preserved_fsper_best_model": str(paths.fsper / "best_model.pt"),
    }
    save_json(paths.root / "checkpoint_prune_report.json", result)
    return result


def metric_stats(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(mean(values)),
        "std": float(stdev(values)) if len(values) > 1 else 0.0,
    }


def aggregate(paths_by_seed: dict[int, SeedPaths], output_root: Path) -> dict[str, Any]:
    rows = []
    for seed, paths in paths_by_seed.items():
        if not (paths.fsper / "final_summary.json").exists():
            continue
        summary = load_fsper_summary(paths)
        metrics = json.loads((paths.fsper / "metrics.json").read_text(encoding="utf-8"))
        effect = metrics.get("routing_diagnostics", {}).get(
            "fusion_effect_vs_classifier", {}
        )
        rows.append(
            {
                "seed": seed,
                "best_epoch": int(summary["best_epoch"]),
                "accuracy": float(summary["test_accuracy"]),
                "macro_precision": float(summary["test_macro_precision"]),
                "macro_recall": float(summary["test_macro_recall"]),
                "macro_f1": float(summary["test_macro_f1"]),
                "weighted_f1": float(summary["test_weighted_f1"]),
                "changed_predictions": int(effect.get("changed_predictions", 0)),
                "fixed_predictions": int(effect.get("fixed_predictions", 0)),
                "harmed_predictions": int(effect.get("harmed_predictions", 0)),
                "output_dir": str(paths.fsper),
            }
        )
    payload = {
        "model": "FSPER-Net",
        "dataset": "FGRC-SCD",
        "split_seed": 42,
        "training_seeds": list(paths_by_seed),
        "completed_runs": len(rows),
        "expected_runs": len(paths_by_seed),
        "complete": len(rows) == len(paths_by_seed),
        "rows": rows,
    }
    if rows:
        for metric in (
            "accuracy",
            "macro_precision",
            "macro_recall",
            "macro_f1",
            "weighted_f1",
        ):
            payload[metric] = metric_stats([row[metric] for row in rows])
    save_json(output_root / "aggregate_results.json", payload)
    return payload


def validate_environment() -> None:
    for path in (PYTHON, PSCL_SCRIPT, DIAGNOSTIC_SCRIPT, V2_SCRIPT, FSPER_SCRIPT, DATA_DIR):
        if not path.exists():
            raise FileNotFoundError(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="42,2024,2026")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--pscl-epochs", type=int, default=20)
    parser.add_argument("--v2-epochs", type=int, default=14)
    parser.add_argument("--fsper-epochs", type=int, default=16)
    parser.add_argument("--pscl-lr", type=float, default=1.5e-5)
    parser.add_argument("--pscl-head-lr", type=float, default=1.5e-4)
    parser.add_argument("--pscl-warmup-ratio", type=float, default=0.15)
    parser.add_argument("--v2-lr", type=float, default=3e-6)
    parser.add_argument("--v2-head-lr", type=float, default=7.5e-5)
    parser.add_argument("--v2-warmup-ratio", type=float, default=0.15)
    parser.add_argument("--v2-early-stopping-patience", type=int, default=0)
    parser.add_argument("--fsper-lr", type=float, default=1e-6)
    parser.add_argument("--fsper-head-lr", type=float, default=2.5e-5)
    parser.add_argument("--fsper-warmup-ratio", type=float, default=0.15)
    parser.add_argument("--fsper-early-stopping-patience", type=int, default=0)
    parser.add_argument("--reproduction-tolerance-pp", type=float, default=0.5)
    parser.add_argument("--disable-reproduction-gate", action="store_true")
    parser.add_argument("--keep-intermediate-checkpoints", action="store_true")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    args.seeds = parse_seeds(args.seeds)
    args.output_root = args.output_root.resolve()
    if (
        not args.disable_reproduction_gate
        and 42 in args.seeds
        and args.seeds[0] != 42
    ):
        raise ValueError(
            "Seed 42 must run first while the reproduction gate is enabled."
        )
    if min(args.pscl_epochs, args.v2_epochs, args.fsper_epochs) <= 0:
        raise ValueError("Epoch counts must be positive.")
    learning_rates = (
        args.pscl_lr,
        args.pscl_head_lr,
        args.v2_lr,
        args.v2_head_lr,
        args.fsper_lr,
        args.fsper_head_lr,
    )
    if min(learning_rates) <= 0:
        raise ValueError("Learning rates must be positive.")
    warmup_ratios = (
        args.pscl_warmup_ratio,
        args.v2_warmup_ratio,
        args.fsper_warmup_ratio,
    )
    if any(not 0.0 <= value < 1.0 for value in warmup_ratios):
        raise ValueError("Warmup ratios must be in [0, 1).")
    if min(
        args.v2_early_stopping_patience,
        args.fsper_early_stopping_patience,
    ) < 0:
        raise ValueError("Early-stopping patience cannot be negative.")
    if args.reproduction_tolerance_pp < 0:
        raise ValueError("Reproduction tolerance must be non-negative.")
    validate_environment()
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = build_fixed_split_manifest(args.output_root, args.split_seed)
    paths_by_seed = {
        seed: seed_paths(
            args.output_root,
            seed,
            args.pscl_epochs,
            args.v2_epochs,
            args.fsper_epochs,
        )
        for seed in args.seeds
    }
    jobs_by_seed = {
        seed: build_jobs(paths, manifest, args)
        for seed, paths in paths_by_seed.items()
    }
    suite_manifest = {
        "model": "FSPER-Net",
        "protocol": {
            "dataset": "FGRC-SCD",
            "fixed_split_seed": args.split_seed,
            "training_seeds": args.seeds,
            "pscl_epochs": args.pscl_epochs,
            "v2_epochs": args.v2_epochs,
            "fsper_epochs": args.fsper_epochs,
            "learning_rate_schedule": "linear_warmup_then_linear_decay",
            "pscl_lr": args.pscl_lr,
            "pscl_head_lr": args.pscl_head_lr,
            "pscl_warmup_ratio": args.pscl_warmup_ratio,
            "v2_lr": args.v2_lr,
            "v2_head_lr": args.v2_head_lr,
            "v2_warmup_ratio": args.v2_warmup_ratio,
            "v2_early_stopping_patience": args.v2_early_stopping_patience,
            "fsper_lr": args.fsper_lr,
            "fsper_head_lr": args.fsper_head_lr,
            "fsper_warmup_ratio": args.fsper_warmup_ratio,
            "fsper_early_stopping_patience": args.fsper_early_stopping_patience,
            "seed42_reproduction_gate": not args.disable_reproduction_gate,
            "reproduction_tolerance_pp": args.reproduction_tolerance_pp,
            "intermediate_checkpoints_pruned": not args.keep_intermediate_checkpoints,
            "test_used_for_model_selection": False,
        },
        "split_manifest": str(manifest),
        "jobs": [
            {
                "job_id": job.job_id,
                "stage": job.stage,
                "marker": str(job.marker),
                "command": job.command_builder(),
            }
            for jobs in jobs_by_seed.values()
            for job in jobs
        ],
    }
    save_json(args.output_root / "suite_manifest.json", suite_manifest)

    all_jobs = [job for jobs in jobs_by_seed.values() for job in jobs]
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
            print(
                "       " + subprocess.list2cmdline(job.command_builder()),
                flush=True,
            )
    if args.dry_run:
        aggregate(paths_by_seed, args.output_root)
        return

    status_path = args.output_root / "run_status.json"
    status = (
        json.loads(status_path.read_text(encoding="utf-8"))
        if status_path.exists()
        else {}
    )
    for seed in args.seeds:
        paths = paths_by_seed[seed]
        for job in jobs_by_seed[seed]:
            if job.marker.exists():
                status[job.job_id] = {"status": "skipped_complete"}
                save_json(status_path, status)
                continue
            try:
                duration = run_job(job)
            except Exception as exc:
                status[job.job_id] = {
                    "status": "failed",
                    "error": str(exc),
                    "log_path": str(job.log_path),
                }
                save_json(status_path, status)
                raise
            status[job.job_id] = {
                "status": "completed",
                "duration_seconds": duration,
                "marker": str(job.marker),
                "log_path": str(job.log_path),
            }
            save_json(status_path, status)

        if seed == 42 and not args.disable_reproduction_gate:
            reproduction = check_seed42_reproduction(
                paths,
                args.reproduction_tolerance_pp,
            )
            print(json.dumps(reproduction, ensure_ascii=False, indent=2), flush=True)
            if not reproduction["passed"]:
                raise RuntimeError(
                    "Seed-42 FSPER-Net reproduction gate failed. The remaining "
                    "seeds were intentionally not started."
                )
        if not args.keep_intermediate_checkpoints:
            prune = prune_seed(paths, args.output_root)
            print(json.dumps(prune, ensure_ascii=False, indent=2), flush=True)
        aggregate(paths_by_seed, args.output_root)

    result = aggregate(paths_by_seed, args.output_root)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
