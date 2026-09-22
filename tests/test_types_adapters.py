from pathlib import Path

import pytest
import yaml

from weather_hub.adapters import adapter_for, capabilities
from weather_hub.registry import ModelInstallation, TargetConfig
from weather_hub.types import ForecastCase, ForecastRequest, ModelId


def case(**changes):
    values = {
        "case_id": "case_1",
        "storm_id": "storm_2025",
        "init_time": "2025-09-22T08:00:00+08:00",
        "forecast_hours": 72,
        "output_interval_hours": 6,
    }
    values.update(changes)
    return ForecastCase(**values)


@pytest.mark.parametrize(
    ("model", "hours", "valid"),
    [
        (ModelId.PANGU, 5, True),
        (ModelId.FENGWU, 336, True),
        (ModelId.FENGWU, 342, False),
        (ModelId.FUXI, 360, True),
        (ModelId.FUXI, 7, False),
        (ModelId.GRAPHCAST, 240, True),
        (ModelId.GRAPHCAST, 246, False),
        (ModelId.AURORA, 360, True),
        (ModelId.AURORA, 361, False),
    ],
)
def test_capability_bounds(model, hours, valid):
    value = case(forecast_hours=hours)
    if valid:
        capabilities(model).validate(value)
    else:
        with pytest.raises(ValueError):
            capabilities(model).validate(value)


def test_case_normalizes_to_utc_and_rejects_bad_ids():
    assert case().init_time == "2025-09-22T00:00:00Z"
    with pytest.raises(ValueError):
        case(case_id="../../bad")


def test_adapter_materializes_absolute_config_and_command(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    config = project / "default.yaml"
    config.write_text(
        "experiment: old\nmodel_dir: models\ndata_dir: data\noutput_dir: output\n"
        "device: cuda\noutput_interval_hours: 6\ndownload_missing: true\n",
        encoding="utf-8",
    )
    installation = ModelInstallation(
        model=ModelId.PANGU,
        project_dir=project,
        conda_env="pangu",
        module="pangu_weather",
        base_config=config,
    )
    target = TargetConfig(
        name="local",
        kind="local",
        jobs_dir=tmp_path / "runs",
        conda_executable="/conda",
        models={ModelId.PANGU: installation},
    )
    adapter = adapter_for(installation, target)
    request = ForecastRequest(model=ModelId.PANGU, cases=(case(),))
    job = tmp_path / "job"
    job.mkdir()
    effective, cases = adapter.prepare(request, job)
    raw = yaml.safe_load(effective.read_text())
    assert raw["experiment"] == "pangu"
    assert Path(raw["model_dir"]).is_absolute()
    assert raw["output_dir"] == str((job / "outputs").resolve())
    assert cases.read_text().splitlines()[0].startswith("case_id,storm_id")
    command = adapter.run_command(effective, cases, dry_run=True)
    assert command[:5] == ["/conda", "run", "--no-capture-output", "-n", "pangu"]
    assert command[-1] == "--dry-run"


def test_adapter_rejects_protected_override(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    config = project / "default.yaml"
    config.write_text("model_dir: models\ndata_dir: data\noutput_dir: output\n")
    installation = ModelInstallation(
        ModelId.PANGU, project, "pangu", "pangu_weather", config
    )
    target = TargetConfig("local", "local", tmp_path / "runs", models={ModelId.PANGU: installation})
    adapter = adapter_for(installation, target)
    request = ForecastRequest(
        model=ModelId.PANGU,
        cases=(case(),),
        config_overrides={"output_dir": "/tmp/escape"},
    )
    with pytest.raises(ValueError, match="unsupported config overrides"):
        adapter.validate_request(request)
