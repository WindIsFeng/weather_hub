import json
import os
from pathlib import Path

import yaml

from weather_hub.registry import load_registry
from weather_hub.types import ForecastCase, ForecastRequest, JobStatus, ModelId
from weather_hub.worker import WorkerService


FAKE_CONDA = r'''#!/usr/bin/env python
import csv
import pathlib
import sys
import yaml

config_path = pathlib.Path(sys.argv[sys.argv.index("--config") + 1])
cases_path = pathlib.Path(sys.argv[sys.argv.index("--cases") + 1])
if "--dry-run" in sys.argv:
    print("fake dry run")
    raise SystemExit(0)
config = yaml.safe_load(config_path.read_text())
experiment = pathlib.Path(config["output_dir"]) / config["experiment"]
(experiment / "cases").mkdir(parents=True, exist_ok=True)
rows = list(csv.DictReader(cases_path.open()))
with (experiment / "summary.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=["case_id", "status", "forecast_id", "result_dir", "error"])
    writer.writeheader()
    for row in rows:
        case_dir = experiment / "cases" / row["case_id"]
        case_dir.mkdir()
        (case_dir / "surface.nc").write_bytes(b"surface")
        (case_dir / "upper.nc").write_bytes(b"upper")
        writer.writerow({"case_id": row["case_id"], "status": "complete", "forecast_id": "abc", "result_dir": f"cases/{row['case_id']}", "error": ""})
'''


def setup_registry(tmp_path):
    project = tmp_path / "pangu-project"
    project.mkdir()
    (project / "configs").mkdir()
    (project / "configs" / "default.yaml").write_text(
        "experiment: default\nmodel_dir: models\ndata_dir: data\noutput_dir: outputs\n"
        "device: cuda\ndevice_id: 0\nthreads: 1\nmax_sessions: 1\n"
        "output_interval_hours: 6\ndownload_missing: true\n"
    )
    fake = tmp_path / "fake-conda"
    fake.write_text(FAKE_CONDA)
    fake.chmod(0o755)
    config = tmp_path / "models.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "controller_jobs_dir": str(tmp_path / "runs"),
                "targets": {
                    "local": {
                        "kind": "local",
                        "jobs_dir": str(tmp_path / "runs"),
                        "conda_executable": str(fake),
                        "models": {
                            "pangu": {
                                "project_dir": str(project),
                                "conda_env": "pangu",
                                "module": "pangu_weather",
                                "base_config": "configs/default.yaml",
                            }
                        },
                    }
                },
            }
        )
    )
    return config


def request(dry_run=False):
    return ForecastRequest(
        model=ModelId.PANGU,
        cases=(
            ForecastCase(
                case_id="c1",
                storm_id="s1",
                init_time="2025-09-22T00:00:00Z",
                forecast_hours=6,
            ),
        ),
        dry_run=dry_run,
    )


def test_worker_executes_and_indexes_native_results(tmp_path):
    service = WorkerService(load_registry(setup_registry(tmp_path)))
    created = service.submit(request(), start=False)
    finished = service.execute(created.job_id)
    assert finished.status == JobStatus.SUCCEEDED
    result = service.result(created.job_id)
    assert result["cases"][0]["status"] == "complete"
    assert {item["logical_name"] for item in result["cases"][0]["artifacts"]} == {
        "surface",
        "upper_air",
    }
    assert all(item["exists"] for item in result["cases"][0]["artifacts"])


def test_worker_dry_run_is_success_without_outputs(tmp_path):
    service = WorkerService(load_registry(setup_registry(tmp_path)))
    created = service.submit(request(dry_run=True), start=False)
    finished = service.execute(created.job_id)
    assert finished.status == JobStatus.SUCCEEDED
    assert service.result(created.job_id)["cases"][0]["status"] == "planned"


def test_worker_marks_missing_model_unavailable(tmp_path):
    config = setup_registry(tmp_path)
    raw = yaml.safe_load(config.read_text())
    raw["targets"]["local"]["models"] = {}
    config.write_text(yaml.safe_dump(raw))
    service = WorkerService(load_registry(config))
    created = service.submit(request(), start=False)
    finished = service.execute(created.job_id)
    assert finished.status == JobStatus.UNAVAILABLE
