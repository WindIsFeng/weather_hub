#!/usr/bin/env python3
"""Run exactly one 12-hour case on Pangu, FengWu and FuXi, sequentially.

Run with the existing pangu Python environment (NumPy, netCDF4 and PyYAML).
The individual model processes still run in their own Conda environments.
"""

import argparse
import csv
import importlib
import json
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import netCDF4
import numpy as np
import yaml

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from weather_hub.registry import load_registry
from weather_hub.store import atomic_json
from weather_hub.types import ForecastCase, ForecastRequest, ModelId
from weather_hub.worker import WorkerService

MODELS = ("pangu", "fengwu", "fuxi")
CASE = ForecastCase(
    case_id="YAGI_20240905T06Z_smoke_12h",
    storm_id="2024244N09137",
    init_time="2024-09-05T06:00:00Z",
    forecast_hours=12,
    output_interval_hours=6,
    name="YAGI",
    basin="WP",
)


def validate_outputs(case_dir, module):
    constants = importlib.import_module(module + ".constants")
    manifest = json.loads((case_dir / "case.json").read_text())
    manifest = json.loads((case_dir / manifest["manifest"]).read_text())
    providers = manifest.get("providers", {})
    if not providers or any("CUDAExecutionProvider" not in p for p in providers.values()):
        raise ValueError(f"GPU execution was not confirmed: {providers}")
    result = {"providers": providers, "files": {}, "warnings": []}
    init_hour = datetime(2024, 9, 5, 6, tzinfo=timezone.utc).timestamp() / 3600
    for kind, names in (("surface", constants.SURFACE), ("upper", constants.UPPER)):
        path = case_dir / (kind + ".nc")
        report = {"path": str(path), "bytes": path.stat().st_size, "variables": {}}
        with netCDF4.Dataset(path) as ds:
            if ds.status != "complete":
                raise ValueError(f"{path}: output status is not complete")
            expected_dims = ("valid_time",) + (("level",) if kind == "upper" else ()) + ("latitude", "longitude")
            expected_shape = (2,) + ((13,) if kind == "upper" else ()) + (721, 1440)
            for coord, expected in (("latitude", constants.LATITUDES), ("longitude", constants.LONGITUDES), ("lead_time", [6, 12]), ("valid_time", init_hour + np.array([6, 12]))):
                if not np.array_equal(ds[coord][:], expected):
                    raise ValueError(f"{path}: incorrect {coord}")
            if float(ds["forecast_reference_time"].getValue()) != init_hour:
                raise ValueError(f"{path}: incorrect initialization timestamp")
            if kind == "upper":
                levels = getattr(constants, "OUTPUT_LEVELS", constants.LEVELS if hasattr(constants, "LEVELS") else None)
                if not np.array_equal(ds["level"][:], levels):
                    raise ValueError(f"{path}: incorrect pressure levels")
            report["shape"] = list(expected_shape)
            report["leads_hours"] = [6, 12]
            any_change = False
            for name in names:
                variable = ds[name]
                if variable.dimensions != expected_dims or variable.shape != expected_shape or variable.units != constants.UNITS[name]:
                    raise ValueError(f"{path}: incorrect dimensions, shape or units for {name}")
                stats = []
                previous = None
                max_change = 0.0
                for index in range(2):
                    data = variable[index]
                    if np.ma.getmaskarray(data).any() or not np.isfinite(data).all():
                        raise ValueError(f"{path}: {name} has missing or non-finite values")
                    values = np.asarray(data)
                    stats.append({"lead_hours": [6, 12][index], "min": float(values.min()), "max": float(values.max()), "mean": float(values.mean(dtype=np.float64)), "std": float(values.std(dtype=np.float64))})
                    if previous is not None:
                        max_change = float(np.max(np.abs(values - previous)))
                    previous = values
                any_change |= max_change > 0
                report["variables"][name] = {"units": variable.units, "all_finite": True, "leads": stats, "max_change_between_leads": max_change}
                broad_ranges = {"t2m": (150, 350), "msl": (80000, 110000), "u10": (-200, 200), "v10": (-200, 200), "t": (140, 350)}
                if name in broad_ranges:
                    low, high = broad_ranges[name]
                    if any(s["min"] < low or s["max"] > high for s in stats):
                        result["warnings"].append(f"{kind}/{name} is outside broad diagnostic range [{low}, {high}]")
            if not any_change:
                raise ValueError(f"{path}: all variables are unchanged across forecast steps")
        result["files"][kind] = report
    return result


def monitor_gpu(stop, samples, path):
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["utc_time", "gpu_memory_used_mib", "gpu_utilization_percent"])
        while not stop.is_set():
            try:
                output = subprocess.check_output(["nvidia-smi", "--id=0", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"], text=True, timeout=5)
                memory, utilization = [int(x.strip()) for x in output.strip().split(",")]
                samples.append(memory)
                writer.writerow([datetime.now(timezone.utc).isoformat(), memory, utilization])
                handle.flush()
            except (OSError, ValueError, subprocess.SubprocessError):
                pass
            stop.wait(0.5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, default=Path("/data/hufeng/ai_weather_models/era5_inputs"))
    args = parser.parse_args()
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    inputs = args.inputs.resolve()
    original = load_registry(PROJECT / "config/models.yaml")
    target = original.target("local")
    model_configs = {}
    for model in MODELS:
        installation = target.models[ModelId(model)]
        overrides = {"download_missing": False, "device": "cuda", "threads": 1,
                     "data_dir": str(root / "unused-cache"),
                     "surface_file": str(inputs / "{init:%Y%m%d}" / "surface.nc"),
                     "upper_file": str(inputs / "{init:%Y%m%d}" / "upper.nc")}
        if model == "fuxi":
            overrides["precipitation_file"] = str(inputs / "precipitation" / "{init:%Y%m%d}.nc")
        model_configs[model] = {"project_dir": str(installation.project_dir), "conda_env": installation.conda_env,
                               "module": installation.module, "base_config": str(installation.base_config),
                               "device_id": 0, "overrides": overrides}
        sys.path.insert(0, str(installation.project_dir))
    registry_path = root / "models.yaml"
    registry_path.write_text(yaml.safe_dump({"version": 1, "controller_jobs_dir": str(root / "jobs"),
        "targets": {"local": {"kind": "local", "jobs_dir": str(root / "jobs"),
            "conda_executable": "/home/hufeng/miniconda3/bin/conda", "models": model_configs}}}, sort_keys=False))
    with (root / "cases.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CASE.to_dict()))
        writer.writeheader()
        writer.writerow(CASE.to_dict())
    service = WorkerService(load_registry(registry_path))
    report = {"status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "case": CASE.to_dict(), "original_case_id": "2024244N09137-CE01-L024",
              "scope": "One initialization, 12 hours, two forecast steps per model; FuXi short stage only.",
              "output_root": str(root), "models": {}, "excluded_models": ["graphcast", "aurora"]}
    atomic_json(root / "validation.json", report)
    for model in MODELS:
        record = service.submit(ForecastRequest(model=ModelId(model), cases=(CASE,)), start=False)
        started = time.monotonic()
        stop, samples = threading.Event(), []
        monitor = threading.Thread(target=monitor_gpu, args=(stop, samples, root / f"{model}_gpu.csv"), daemon=True)
        monitor.start()
        print(f"START {model}: job={record.job_id}, one case, leads=6,12 hours", flush=True)
        try:
            record = service.execute(record.job_id)
        finally:
            stop.set()
            monitor.join(timeout=6)
        model_report = {"job_id": record.job_id, "job_dir": record.job_dir, "job_status": record.status.value,
                        "exit_code": record.exit_code, "elapsed_seconds": round(time.monotonic() - started, 2),
                        "sampled_peak_gpu_mib": max(samples) if samples else None, "error": record.error}
        case_dir = Path(record.job_dir) / "outputs" / model / "cases" / CASE.case_id
        try:
            if record.status.value != "succeeded":
                raise ValueError(record.error or f"job status: {record.status.value}")
            model_report["validation"] = validate_outputs(case_dir, model_configs[model]["module"])
            model_report["status"] = "passed"
            (root / model).symlink_to(case_dir.relative_to(root), target_is_directory=True)
        except Exception as exc:
            model_report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        report["models"][model] = model_report
        atomic_json(root / "validation.json", report)
        print(f"END {model}: {model_report['status']}, {model_report['elapsed_seconds']}s, GPU peak={model_report['sampled_peak_gpu_mib']} MiB, error={model_report['error']}", flush=True)
    report.update(status="passed" if all(m["status"] == "passed" for m in report["models"].values()) else "failed",
                  completed_at_utc=datetime.now(timezone.utc).isoformat())
    atomic_json(root / "validation.json", report)
    lines = ["# 本机小规模推理测试", "", "起报：2024-09-05 06:00 UTC，台风 YAGI（摩羯）。",
             "每个模型只测试一个 case，预报 12 小时，输出 +6、+12 小时。", "",
             "| 模型 | 结果 | 推理流程耗时（秒） | 采样显存峰值（MiB） |", "|---|---|---:|---:|"]
    for model, item in report["models"].items():
        lines.append(f"| {model} | {item['status']} | {item['elapsed_seconds']} | {item['sampled_peak_gpu_mib']} |")
    lines.extend(["", "验证：真实 CUDA 推理、NetCDF 完整状态、正确起报和有效时次、721×1440 全球网格、13 个气压层、变量及单位一致、全场无缺测或非有限值、两步输出发生变化。",
                  "", "有效输出见各模型目录的 surface.nc 和 upper.nc；逐变量数值范围见 validation.json，完整日志见 jobs/<job_id>/model.log。",
                  "", "本次只验证输入与推理输出链路；未评估预报精度或长期稳定性。FuXi 仅测试 short 阶段。GraphCast 与 Aurora 未运行。", ""])
    for model, item in report["models"].items():
        if item["error"]:
            lines.extend([f"{model} 错误：{item['error']}", ""])
        for warning in item.get("validation", {}).get("warnings", []):
            lines.extend([f"{model} 数值范围提示：{warning}", ""])
    (root / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"RESULT {report['status']}: {root}", flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
