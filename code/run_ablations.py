"""Run the two claim-aligned FSPER-Net ablations on both formal datasets."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable)
TRAIN_SCRIPT = PROJECT_ROOT / "code" / "train_sparse_routed_pscl.py"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "ablations"
DEFAULT_SEEDS = (42, 2024, 2026)
CONDITIONS = ("single_prototype", "fixed_fusion")
METRICS = (
    "accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "weighted_f1",
)


@dataclass(frozen=True)
class DatasetProtocol:
    dataset: str
    display_name: str
    data_dir: Path
    source_root: Path
    warm_start_relative: Path
    feature_cache_relative: Path
    full_summary_relative: Path

    @property
    def split_manifest(self) -> Path:
        return self.source_root / "fixed_split_seed42.json"

    def warm_start(self, seed: int) -> Path:
        return self.source_root / f"seed_{seed}" / self.warm_start_relative

    def feature_cache(self, seed: int) -> Path:
        return self.source_root / f"seed_{seed}" / self.feature_cache_relative

    def full_summary(self, seed: int) -> Path:
        return self.source_root / f"seed_{seed}" / self.full_summary_relative


PROTOCOLS = {
    "fgrc_scd": DatasetProtocol(
        dataset="fgrc_scd",
        display_name="FGRC-SCD",
        data_dir=PROJECT_ROOT / "data" / "FGRC-SCD" / "sms" / "message",
        source_root=PROJECT_ROOT / "outputs" / "fgrc_scd",
        warm_start_relative=Path("fs_pscl_v2_compat_epoch14") / "best_model.pt",
        feature_cache_relative=Path("diagnostics") / "fgrc_scd" / "features_cache.npz",
        full_summary_relative=Path("fsper_epoch16") / "final_summary.json",
    ),
    "telecom5": DatasetProtocol(
        dataset="telecom5",
        display_name="Telecom_Fraud_Texts_5",
        data_dir=(
            PROJECT_ROOT
            / "data"
            / "Telecom_Fraud_Texts_5"
        ),
        source_root=PROJECT_ROOT / "outputs" / "telecom5",
        warm_start_relative=Path("v2_compat_epoch14") / "best_model.pt",
        feature_cache_relative=Path("diagnostics") / "telecom5" / "features_cache.npz",
        full_summary_relative=Path("fsper_epoch16") / "final_summary.json",
    ),
}


@dataclass(frozen=True)
class Experiment:
    protocol: DatasetProtocol
    condition: str
    seed: int
    epochs: int
    output_dir: Path
    log_path: Path
    command: tuple[str, ...]

    @property
    def experiment_id(self) -> str:
        return f"{self.protocol.dataset}__{self.condition}__seed{self.seed}"

    @property
    def summary_path(self) -> Path:
        return self.output_dir / "final_summary.json"


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
    if not values or len(set(values)) != len(values):
        raise ValueError(f"Invalid comma-separated values: {raw}")
    if allowed is not None:
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"Unsupported values: {sorted(unknown)}")
    return values


def parse_seeds(raw: str) -> tuple[int, ...]:
    return tuple(int(item) for item in parse_csv_values(raw))


def build_command(
    protocol: DatasetProtocol,
    condition: str,
    seed: int,
    epochs: int,
    output_dir: Path,
) -> tuple[str, ...]:
    router_enabled = condition == "single_prototype"
    return (
        str(PYTHON),
        "-X",
        "utf8",
        str(TRAIN_SCRIPT),
        "--dataset",
        protocol.dataset,
        "--data-dir",
        str(protocol.data_dir),
        "--output-dir",
        str(output_dir),
        "--allow-download",
        "--warm-start",
        str(protocol.warm_start(seed)),
        "--feature-cache",
        str(protocol.feature_cache(seed)),
        "--split-manifest",
        str(protocol.split_manifest),
        "--split-seed",
        "42",
        "--ablation",
        condition,
        "--epochs",
        str(epochs),
        "--batch-size",
        "16",
        "--max-len",
        "192",
        "--lr",
        "1e-6",
        "--head-lr",
        "2.5e-5",
        "--warmup-ratio",
        "0.15",
        "--route-strength-max",
        "0.5",
        "--dynamic-gate-init",
        "0.05",
        "--architecture-gate-init",
        "0.5",
        "--hard-concrete-temperature",
        "0.67",
        "--hard-concrete-gamma",
        "-0.1",
        "--hard-concrete-zeta",
        "1.1",
        "--base-loss-weight",
        "0.5",
        "--source-loss-weight",
        "0.05",
        "--router-loss-weight",
        "0.2" if router_enabled else "0",
        "--sparsity-weight",
        "0.0005" if router_enabled else "0",
        "--route-usage-weight",
        "0.001" if router_enabled else "0",
        "--router-target-temperature",
        "0.2",
        "--router-benefit-margin",
        "0.05",
        "--disagreement-router-weight",
        "3.0",
        "--classification-weight",
        "1.0",
        "--similarity-gamma",
        "0.1",
        "--ldam-max-margin",
        "0.5",
        "--centroid-momentum",
        "0.995",
        "--semantic-anchor-momentum",
        "0.99",
        "--early-stopping-patience",
        "0",
        "--weighted-sampler",
        "--amp",
        "--save-best-by",
        "valid_macro_f1",
        "--seed",
        str(seed),
    )


def build_experiments(
    datasets: tuple[str, ...],
    seeds: tuple[int, ...],
    epochs: int,
    output_root: Path,
) -> list[Experiment]:
    experiments: list[Experiment] = []
    for dataset in datasets:
        protocol = PROTOCOLS[dataset]
        for condition in CONDITIONS:
            for seed in seeds:
                output_dir = (
                    output_root
                    / dataset
                    / f"{condition}_seed{seed}_epoch{epochs}"
                )
                experiments.append(
                    Experiment(
                        protocol=protocol,
                        condition=condition,
                        seed=seed,
                        epochs=epochs,
                        output_dir=output_dir,
                        log_path=(
                            output_root
                            / "logs"
                            / f"{dataset}__{condition}__seed{seed}.log"
                        ),
                        command=build_command(
                            protocol,
                            condition,
                            seed,
                            epochs,
                            output_dir,
                        ),
                    )
                )
    return experiments


def validate_environment(
    experiments: list[Experiment],
    require_cuda: bool,
) -> None:
    required = {PYTHON, TRAIN_SCRIPT}
    for experiment in experiments:
        required.update(
            {
                experiment.protocol.data_dir,
                experiment.protocol.split_manifest,
                experiment.protocol.warm_start(experiment.seed),
                experiment.protocol.feature_cache(experiment.seed),
                experiment.protocol.full_summary(experiment.seed),
            }
        )
    missing = [path for path in sorted(required) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Required formal-protocol artifacts are missing:\n"
            + "\n".join(str(path) for path in missing)
        )
    if not require_cuda:
        return
    probe = subprocess.run(
        [
            str(PYTHON),
            "-c",
            (
                "import torch; print(torch.cuda.is_available()); "
                "print(torch.cuda.get_device_name(0) "
                "if torch.cuda.is_available() else 'NO_CUDA')"
            ),
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    lines = [line.strip() for line in probe.stdout.splitlines() if line.strip()]
    if not lines or lines[0] != "True":
        raise RuntimeError("CUDA is unavailable; the ablation suite was not started.")
    device_name = lines[1] if len(lines) > 1 else "GPU"
    print(f"[environment] CUDA ready: {device_name}", flush=True)


def summary_row(
    condition: str,
    seed: int,
    path: Path,
) -> dict[str, Any]:
    summary = read_json(path)
    return {
        "condition": condition,
        "seed": seed,
        "best_epoch": int(summary["best_epoch"]),
        "accuracy": float(summary["test_accuracy"]),
        "macro_precision": float(summary["test_macro_precision"]),
        "macro_recall": float(summary["test_macro_recall"]),
        "macro_f1": float(summary["test_macro_f1"]),
        "weighted_f1": float(summary["test_weighted_f1"]),
        "script_counts": summary.get("script_counts"),
        "summary_path": str(path),
    }


def metric_stats(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    return {
        metric: {
            "mean": float(statistics.mean(row[metric] for row in rows)),
            "std": (
                float(statistics.stdev(row[metric] for row in rows))
                if len(rows) > 1
                else 0.0
            ),
        }
        for metric in METRICS
    }


def write_summary(
    output_root: Path,
    datasets: tuple[str, ...],
    seeds: tuple[int, ...],
    experiments: list[Experiment],
) -> dict[str, Any]:
    dataset_results = []
    csv_rows = []
    for dataset in datasets:
        protocol = PROTOCOLS[dataset]
        rows = [
            summary_row("full", seed, protocol.full_summary(seed))
            for seed in seeds
        ]
        rows.extend(
            summary_row(experiment.condition, experiment.seed, experiment.summary_path)
            for experiment in experiments
            if experiment.protocol.dataset == dataset
            and experiment.summary_path.exists()
        )
        conditions = []
        for condition in ("full", *CONDITIONS):
            condition_rows = [
                row for row in rows if row["condition"] == condition
            ]
            entry: dict[str, Any] = {
                "condition": condition,
                "completed_runs": len(condition_rows),
                "expected_runs": len(seeds),
                "complete": len(condition_rows) == len(seeds),
                "rows": condition_rows,
            }
            if condition_rows:
                entry["metrics"] = metric_stats(condition_rows)
            conditions.append(entry)

        full_by_seed = {
            row["seed"]: row for row in rows if row["condition"] == "full"
        }
        paired_deltas = []
        for row in rows:
            if row["condition"] == "full":
                continue
            full = full_by_seed[row["seed"]]
            paired_deltas.append(
                {
                    "condition": row["condition"],
                    "seed": row["seed"],
                    "accuracy_delta_pp": 100.0
                    * (row["accuracy"] - full["accuracy"]),
                    "macro_f1_delta_pp": 100.0
                    * (row["macro_f1"] - full["macro_f1"]),
                }
            )

        complete_conditions = [
            entry for entry in conditions if entry["complete"]
        ]
        ranking = sorted(
            complete_conditions,
            key=lambda entry: entry["metrics"]["macro_f1"]["mean"],
            reverse=True,
        )
        for rank, entry in enumerate(ranking, start=1):
            csv_rows.append(
                {
                    "dataset": dataset,
                    "rank": rank,
                    "condition": entry["condition"],
                    "accuracy_mean": entry["metrics"]["accuracy"]["mean"],
                    "accuracy_std": entry["metrics"]["accuracy"]["std"],
                    "macro_f1_mean": entry["metrics"]["macro_f1"]["mean"],
                    "macro_f1_std": entry["metrics"]["macro_f1"]["std"],
                }
            )
        dataset_results.append(
            {
                "dataset": dataset,
                "display_name": protocol.display_name,
                "conditions": conditions,
                "ranking_by_macro_f1": [
                    {
                        "rank": rank,
                        "condition": entry["condition"],
                        "accuracy_mean": entry["metrics"]["accuracy"]["mean"],
                        "accuracy_std": entry["metrics"]["accuracy"]["std"],
                        "macro_f1_mean": entry["metrics"]["macro_f1"]["mean"],
                        "macro_f1_std": entry["metrics"]["macro_f1"]["std"],
                    }
                    for rank, entry in enumerate(ranking, start=1)
                ],
                "paired_ablation_minus_full": paired_deltas,
            }
        )

    payload = {
        "generated_at_unix": time.time(),
        "model": "FSPER-Net",
        "claim": (
            "Multiple fraud-script prototypes model intra-class diversity, and "
            "dynamic routing selects useful prototype experts per sample."
        ),
        "protocol": {
            "datasets": list(datasets),
            "fixed_split_seed": 42,
            "training_seeds": list(seeds),
            "conditions": list(CONDITIONS),
            "selection_metric": "validation Macro-F1",
            "full_model_source": "Existing frozen formal FSPER-Net runs",
        },
        "completed_ablation_runs": sum(
            experiment.summary_path.exists() for experiment in experiments
        ),
        "expected_ablation_runs": len(experiments),
        "complete": all(
            experiment.summary_path.exists() for experiment in experiments
        ),
        "dataset_results": dataset_results,
    }
    save_json(output_root / "core_ablation_summary.json", payload)
    with (output_root / "core_ablation_summary.csv").open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        fields = [
            "dataset",
            "rank",
            "condition",
            "accuracy_mean",
            "accuracy_std",
            "macro_f1_mean",
            "macro_f1_std",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(csv_rows)
    return payload


def run_experiment(
    experiment: Experiment,
    remove_completed_latest: bool,
) -> float:
    experiment.output_dir.mkdir(parents=True, exist_ok=True)
    experiment.log_path.parent.mkdir(parents=True, exist_ok=True)
    command = list(experiment.command)
    latest = experiment.output_dir / "latest_checkpoint.pt"
    if latest.exists():
        command.extend(("--resume", str(latest)))
        print(f"[resume] {experiment.experiment_id}", flush=True)
    print(f"[run][GPU] {experiment.experiment_id}", flush=True)
    started = time.time()
    with experiment.log_path.open("a", encoding="utf-8") as log_handle:
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
    if return_code != 0 or not experiment.summary_path.exists():
        raise RuntimeError(
            f"Experiment failed: {experiment.experiment_id}; "
            f"return_code={return_code}; log={experiment.log_path}"
        )
    if remove_completed_latest and latest.exists():
        latest.unlink()
    print(
        f"[completed] {experiment.experiment_id} ({duration / 3600.0:.2f} h)",
        flush=True,
    )
    return duration


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default="fgrc_scd,telecom5")
    parser.add_argument("--seeds", default="42,2024,2026")
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--remove-completed-latest", action="store_true")
    args = parser.parse_args()

    datasets = parse_csv_values(args.datasets, set(PROTOCOLS))
    seeds = parse_seeds(args.seeds)
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    experiments = build_experiments(
        datasets,
        seeds,
        args.epochs,
        output_root,
    )
    validate_environment(
        experiments,
        require_cuda=not args.allow_cpu and not args.summary_only,
    )
    save_json(
        output_root / "suite_manifest.json",
        {
            "model": "FSPER-Net",
            "datasets": list(datasets),
            "training_seeds": list(seeds),
            "epochs": args.epochs,
            "conditions": list(CONDITIONS),
            "experiments": [
                {
                    "experiment_id": experiment.experiment_id,
                    "output_dir": str(experiment.output_dir),
                    "command": list(experiment.command),
                }
                for experiment in experiments
            ],
        },
    )

    if args.summary_only:
        result = write_summary(output_root, datasets, seeds, experiments)
        print(json.dumps(result["dataset_results"], ensure_ascii=False, indent=2))
        return

    complete = sum(
        experiment.summary_path.exists() for experiment in experiments
    )
    print(
        f"[plan] {len(experiments)} runs; {complete} complete; "
        f"{len(experiments) - complete} pending.",
        flush=True,
    )
    for experiment in experiments:
        state = "skip" if experiment.summary_path.exists() else "run"
        print(f"[{state}] {experiment.experiment_id}", flush=True)
        if args.dry_run and state == "run":
            print(
                "       " + subprocess.list2cmdline(list(experiment.command)),
                flush=True,
            )
    if args.dry_run:
        write_summary(output_root, datasets, seeds, experiments)
        return

    status_path = output_root / "run_status.json"
    statuses = read_json(status_path) if status_path.exists() else {}
    for experiment in experiments:
        if experiment.summary_path.exists():
            statuses[experiment.experiment_id] = {"status": "skipped_complete"}
            save_json(status_path, statuses)
            continue
        try:
            duration = run_experiment(
                experiment,
                remove_completed_latest=args.remove_completed_latest,
            )
        except Exception as exc:
            statuses[experiment.experiment_id] = {
                "status": "failed",
                "error": str(exc),
                "log_path": str(experiment.log_path),
            }
            save_json(status_path, statuses)
            write_summary(output_root, datasets, seeds, experiments)
            raise
        statuses[experiment.experiment_id] = {
            "status": "completed",
            "duration_seconds": duration,
            "output_dir": str(experiment.output_dir),
            "log_path": str(experiment.log_path),
        }
        save_json(status_path, statuses)
        write_summary(output_root, datasets, seeds, experiments)

    result = write_summary(output_root, datasets, seeds, experiments)
    print(json.dumps(result["dataset_results"], ensure_ascii=False, indent=2))
    print("[done] FSPER-Net core ablation suite completed.", flush=True)


if __name__ == "__main__":
    main()
