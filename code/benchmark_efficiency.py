"""Benchmark FSPER-Net inference overhead from a frozen final checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path
from typing import Any, Callable

import torch
from transformers import AutoTokenizer

from train_sparse_routed_pscl import SparseRoutedFSPSCLClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "fgrc_scd"
    / "seed_42"
    / "fsper_epoch16"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "efficiency"
SAMPLE_TEXTS = (
    "客服称订单异常需要下载会议软件并共享屏幕完成退款认证。",
    "对方冒充公安机关，以涉嫌洗钱为由要求将资金转入安全账户。",
    "陌生人以高收益投资为诱饵，要求继续充值后才能提取收益。",
    "这是正常的业务通知，请通过官方渠道核实，不要向陌生账户转账。",
)


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def build_model(
    run_dir: Path,
    device: torch.device,
) -> tuple[SparseRoutedFSPSCLClassifier, dict[str, Any]]:
    checkpoint_path = run_dir / "best_model.pt"
    config_path = run_dir / "experiment_config.json"
    if not checkpoint_path.exists() or not config_path.exists():
        raise FileNotFoundError(
            f"Final checkpoint or configuration is missing under {run_dir}"
        )
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    config = load_json(config_path)
    args = config["args"]
    state = checkpoint["model_state_dict"]
    counts = state["script_counts"].tolist()
    model = SparseRoutedFSPSCLClassifier(
        pretrained_model=args["pretrained_model"],
        num_classes=len(checkpoint["label_to_id"]),
        dropout=float(args["dropout"]),
        cache_dir=Path(args["cache_dir"]),
        local_files_only=True,
        description_input_ids=state["description_input_ids"],
        description_attention_mask=state["description_attention_mask"],
        initial_centroids=state["centroid_prototypes"],
        script_counts=counts,
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
    return model, config


def build_batch(
    tokenizer: Any,
    batch_size: int,
    max_length: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    texts = [SAMPLE_TEXTS[index % len(SAMPLE_TEXTS)] for index in range(batch_size)]
    encoded = tokenizer(
        texts,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return encoded["input_ids"].to(device), encoded["attention_mask"].to(device)


def benchmark(
    function: Callable[[], Any],
    batch_size: int,
    warmup: int,
    iterations: int,
    device: torch.device,
) -> dict[str, float]:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    timings_ms: list[float] = []
    for _ in range(iterations):
        started = time.perf_counter()
        function()
        torch.cuda.synchronize(device)
        timings_ms.append((time.perf_counter() - started) * 1000.0)
    mean_ms = statistics.fmean(timings_ms)
    sorted_timings = sorted(timings_ms)
    p95_index = min(len(sorted_timings) - 1, int(0.95 * len(sorted_timings)))
    return {
        "latency_mean_ms": mean_ms,
        "latency_std_ms": statistics.stdev(timings_ms) if len(timings_ms) > 1 else 0.0,
        "latency_p95_ms": sorted_timings[p95_index],
        "throughput_samples_per_second": batch_size * 1000.0 / mean_ms,
        "peak_memory_mb": torch.cuda.max_memory_allocated(device) / (1024.0**2),
    }


def routing_summary(metrics_path: Path) -> dict[str, Any]:
    metrics = load_json(metrics_path)
    router = metrics["routing_diagnostics"]["router"]
    expected = [
        float(value)
        for row in router.get("expected_l0", [])
        for value in row
    ]
    return {
        "expected_active_probability_mean": (
            statistics.fmean(expected) if expected else None
        ),
        "route_strength_mean": router.get("route_strength_mean"),
        "dynamic_gate_mean_by_source": router.get("dynamic_gate_mean_by_source"),
        "combined_gate_mean_by_source": router.get("combined_gate_mean_by_source"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-sizes", default="1,16")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the paper efficiency benchmark.")
    device = torch.device("cuda")
    run_dir = args.run_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model, config = build_model(run_dir, device)
    model_args = config["args"]
    tokenizer = AutoTokenizer.from_pretrained(
        model_args["pretrained_model"],
        cache_dir=Path(model_args["cache_dir"]),
        local_files_only=True,
    )
    batch_sizes = tuple(
        int(value.strip()) for value in args.batch_sizes.split(",") if value.strip()
    )
    if not batch_sizes or any(value <= 0 for value in batch_sizes):
        raise ValueError("--batch-sizes must contain positive integers.")

    total_parameters = count_parameters(model)
    router_parameters = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if name.startswith("instance_router")
        or name in {"architecture_log_alpha", "source_log_scales"}
    )
    rows: list[dict[str, Any]] = []
    for batch_size in batch_sizes:
        input_ids, attention_mask = build_batch(
            tokenizer,
            batch_size,
            int(model_args["max_len"]),
            device,
        )

        def full_forward() -> Any:
            with torch.inference_mode(), torch.autocast(
                device_type="cuda", dtype=torch.float16
            ):
                return model(input_ids=input_ids, attention_mask=attention_mask)["logits"]

        def classifier_forward() -> Any:
            with torch.inference_mode(), torch.autocast(
                device_type="cuda", dtype=torch.float16
            ):
                _, features = model.encode_hidden(input_ids, attention_mask)
                return model.classifier(model.dropout(features))

        for path_name, function in (
            ("protected_classifier_path", classifier_forward),
            ("full_fsper_path", full_forward),
        ):
            measured = benchmark(
                function,
                batch_size,
                args.warmup,
                args.iterations,
                device,
            )
            rows.append(
                {
                    "path": path_name,
                    "batch_size": batch_size,
                    "sequence_length": int(model_args["max_len"]),
                    **measured,
                }
            )
            print(
                f"[efficiency] {path_name} batch={batch_size}: "
                f"{measured['latency_mean_ms']:.2f} ms, "
                f"{measured['throughput_samples_per_second']:.2f} samples/s",
                flush=True,
            )

    payload = {
        "device": torch.cuda.get_device_name(0),
        "checkpoint": str(run_dir / "best_model.pt"),
        "precision": "CUDA autocast FP16",
        "warmup_iterations": args.warmup,
        "measured_iterations": args.iterations,
        "total_parameters": total_parameters,
        "router_specific_parameters": router_parameters,
        "router_parameter_fraction": router_parameters / total_parameters,
        "routing": routing_summary(run_dir / "metrics.json"),
        "measurements": rows,
    }
    save_json(output_dir / "efficiency_summary.json", payload)
    with (output_dir / "efficiency_summary.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    save_json(
        output_dir / "stage_complete.json",
        {"status": "completed", "summary": str(output_dir / "efficiency_summary.json")},
    )


if __name__ == "__main__":
    main()
