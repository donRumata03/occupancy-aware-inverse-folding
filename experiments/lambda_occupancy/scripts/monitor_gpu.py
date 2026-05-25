from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


QUERY_FIELDS = [
    "index",
    "name",
    "utilization.gpu",
    "memory.used",
    "memory.total",
    "temperature.gpu",
    "power.draw",
]

CSV_FIELDS = [
    "timestamp_utc",
    "elapsed_s",
    "gpu_index",
    "gpu_name",
    "utilization_gpu_pct",
    "memory_used_mib",
    "memory_total_mib",
    "memory_used_pct",
    "temperature_c",
    "power_draw_w",
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=float, default=10.0, help="Sampling interval in seconds.")
    parser.add_argument("--output-dir", required=True, help="Directory for gpu_monitor.csv/json/png.")
    parser.add_argument("--duration", type=float, help="Optional maximum runtime in seconds.")
    parser.add_argument("--pid", type=int, help="Optional process id; stop when it exits.")
    parser.add_argument("--stop-file", help="Optional path; stop when this file exists.")
    parser.add_argument("--no-plot", action="store_true", help="Skip PNG plot generation.")
    args = parser.parse_args(argv)

    if args.interval <= 0:
        raise ValueError("--interval must be positive")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "gpu_monitor.csv"
    summary_path = out_dir / "gpu_monitor_summary.json"
    plot_path = out_dir / "gpu_monitor.png"

    started = time.monotonic()
    rows: list[dict[str, Any]] = []
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        try:
            while True:
                elapsed = time.monotonic() - started
                if args.duration is not None and elapsed >= args.duration:
                    break
                if args.pid is not None and not _pid_exists(args.pid):
                    break
                if args.stop_file and Path(args.stop_file).exists():
                    break

                sample_rows = _sample_nvidia_smi(elapsed)
                for row in sample_rows:
                    writer.writerow(row)
                handle.flush()
                rows.extend(sample_rows)
                time.sleep(args.interval)
        except KeyboardInterrupt:
            pass

    summary = _summarize(rows, interval_s=args.interval)
    summary["csv_path"] = str(csv_path)
    if not args.no_plot:
        summary["plot_path"] = _plot(rows, plot_path)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    print(f"Wrote {len(rows)} GPU samples to {csv_path}")
    print(f"Wrote GPU summary to {summary_path}")
    if summary.get("plot_path"):
        print(f"Wrote GPU plot to {summary['plot_path']}")
    return 0


def _sample_nvidia_smi(elapsed_s: float) -> list[dict[str, Any]]:
    command = [
        "nvidia-smi",
        f"--query-gpu={','.join(QUERY_FIELDS)}",
        "--format=csv,noheader,nounits",
    ]
    proc = subprocess.run(command, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"nvidia-smi failed with exit code {proc.returncode}: {proc.stderr.strip()}")

    timestamp = datetime.now(timezone.utc).isoformat()
    rows: list[dict[str, Any]] = []
    for line in proc.stdout.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != len(QUERY_FIELDS):
            raise RuntimeError(f"Unexpected nvidia-smi row: {line}")
        gpu_index, gpu_name, util, mem_used, mem_total, temp, power = parts
        memory_used = _to_float(mem_used)
        memory_total = _to_float(mem_total)
        rows.append(
            {
                "timestamp_utc": timestamp,
                "elapsed_s": f"{elapsed_s:.3f}",
                "gpu_index": gpu_index,
                "gpu_name": gpu_name,
                "utilization_gpu_pct": _to_float(util),
                "memory_used_mib": memory_used,
                "memory_total_mib": memory_total,
                "memory_used_pct": 100.0 * memory_used / memory_total if memory_total > 0 else math.nan,
                "temperature_c": _to_float(temp),
                "power_draw_w": _to_float(power),
            }
        )
    return rows


def _summarize(rows: list[dict[str, Any]], interval_s: float) -> dict[str, Any]:
    by_gpu: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_gpu.setdefault(str(row["gpu_index"]), []).append(row)

    return {
        "sample_count": len(rows),
        "interval_s": interval_s,
        "gpus": {
            gpu_index: {
                "gpu_name": gpu_rows[0]["gpu_name"] if gpu_rows else "",
                "sample_count": len(gpu_rows),
                "elapsed_s_first": _as_float(gpu_rows[0]["elapsed_s"]) if gpu_rows else None,
                "elapsed_s_last": _as_float(gpu_rows[-1]["elapsed_s"]) if gpu_rows else None,
                "utilization_gpu_pct": _series_summary(
                    [_as_float(row["utilization_gpu_pct"]) for row in gpu_rows]
                ),
                "memory_used_pct": _series_summary([_as_float(row["memory_used_pct"]) for row in gpu_rows]),
                "memory_used_mib": _series_summary([_as_float(row["memory_used_mib"]) for row in gpu_rows]),
                "temperature_c": _series_summary([_as_float(row["temperature_c"]) for row in gpu_rows]),
                "power_draw_w": _series_summary([_as_float(row["power_draw_w"]) for row in gpu_rows]),
            }
            for gpu_index, gpu_rows in sorted(by_gpu.items(), key=lambda item: int(item[0]))
        },
    }


def _series_summary(values: list[float]) -> dict[str, float | int | None]:
    clean = [value for value in values if not math.isnan(value)]
    if not clean:
        return {"count": 0, "avg": None, "min": None, "p50": None, "p90": None, "p95": None, "p99": None, "max": None}
    return {
        "count": len(clean),
        "avg": sum(clean) / len(clean),
        "min": min(clean),
        "p50": _percentile(clean, 50),
        "p90": _percentile(clean, 90),
        "p95": _percentile(clean, 95),
        "p99": _percentile(clean, 99),
        "max": max(clean),
    }


def _percentile(values: list[float], percentile: float) -> float:
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (percentile / 100.0) * (len(sorted_values) - 1)
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return sorted_values[lower]
    fraction = rank - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def _plot(rows: list[dict[str, Any]], plot_path: Path) -> str:
    if not rows:
        return ""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        return f"plot skipped: {exc}"

    by_gpu: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_gpu.setdefault(str(row["gpu_index"]), []).append(row)

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for gpu_index, gpu_rows in sorted(by_gpu.items(), key=lambda item: int(item[0])):
        label = f"GPU {gpu_index}"
        elapsed = [_as_float(row["elapsed_s"]) / 60.0 for row in gpu_rows]
        axes[0].plot(elapsed, [_as_float(row["utilization_gpu_pct"]) for row in gpu_rows], label=label)
        axes[1].plot(elapsed, [_as_float(row["memory_used_pct"]) for row in gpu_rows], label=label)

    axes[0].set_ylabel("GPU utilization (%)")
    axes[0].set_ylim(0, 100)
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="best")
    axes[1].set_xlabel("Elapsed time (min)")
    axes[1].set_ylabel("Memory used (%)")
    axes[1].set_ylim(0, 100)
    axes[1].grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    return str(plot_path)


def _pid_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _to_float(value: str) -> float:
    normalized = value.strip()
    if normalized.upper() in {"N/A", "[N/A]", "NAN"}:
        return math.nan
    return float(normalized)


def _as_float(value: Any) -> float:
    if isinstance(value, float):
        return value
    if isinstance(value, int):
        return float(value)
    return _to_float(str(value))


if __name__ == "__main__":
    raise SystemExit(main())
