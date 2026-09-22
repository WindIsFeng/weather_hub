"""Model-specific validation and command/result translation."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .registry import ModelInstallation, TargetConfig
from .types import (
    CaseResult,
    ForecastRequest,
    JobStatus,
    ModelCapabilities,
    ModelId,
    ResultArtifact,
    ResultIndex,
)


@dataclass(frozen=True)
class AdapterSpec:
    capabilities: ModelCapabilities
    path_keys: tuple[str, ...]
    surface_filename: str = "surface.nc"
    upper_filename: str = "upper.nc"
    doctor_load_args: tuple[str, ...] = ("--load-model",)
    allowed_overrides: tuple[str, ...] = ()


COMMON_OVERRIDES = (
    "device",
    "device_id",
    "download_missing",
    "surface_file",
    "upper_file",
)


SPECS: dict[ModelId, AdapterSpec] = {
    ModelId.PANGU: AdapterSpec(
        ModelCapabilities(
            model=ModelId.PANGU,
            surface_variables=("msl", "u10", "v10", "t2m"),
            upper_variables=("z", "q", "t", "u", "v"),
        ),
        path_keys=("model_dir", "data_dir", "output_dir", "surface_file", "upper_file"),
        doctor_load_args=("--load-model", "6"),
        allowed_overrides=COMMON_OVERRIDES + ("threads", "max_sessions"),
    ),
    ModelId.FENGWU: AdapterSpec(
        ModelCapabilities(
            model=ModelId.FENGWU,
            min_forecast_hours=6,
            max_forecast_hours=336,
            forecast_step_hours=6,
            fixed_output_interval_hours=6,
            surface_variables=("u10", "v10", "t2m", "msl"),
            upper_variables=("z", "q", "u", "v", "t"),
        ),
        path_keys=(
            "model_file",
            "mean_file",
            "std_file",
            "data_dir",
            "output_dir",
            "surface_file",
            "upper_file",
        ),
        allowed_overrides=COMMON_OVERRIDES + ("threads",),
    ),
    ModelId.FUXI: AdapterSpec(
        ModelCapabilities(
            model=ModelId.FUXI,
            min_forecast_hours=6,
            max_forecast_hours=360,
            forecast_step_hours=6,
            fixed_output_interval_hours=6,
            surface_variables=("t2m", "u10", "v10", "msl", "tp"),
            upper_variables=("z", "t", "u", "v", "r"),
        ),
        path_keys=(
            "model_dir",
            "data_dir",
            "output_dir",
            "surface_file",
            "upper_file",
            "precipitation_file",
        ),
        doctor_load_args=("--load-model", "short"),
        allowed_overrides=COMMON_OVERRIDES
        + ("threads", "max_sessions", "precipitation_file"),
    ),
    ModelId.GRAPHCAST: AdapterSpec(
        ModelCapabilities(
            model=ModelId.GRAPHCAST,
            min_forecast_hours=6,
            max_forecast_hours=240,
            forecast_step_hours=6,
            fixed_output_interval_hours=6,
            surface_variables=("t2m", "msl", "u10", "v10", "tp6"),
            upper_variables=("t", "z", "u", "v", "w", "q"),
        ),
        path_keys=(
            "params_file",
            "stats_dir",
            "data_dir",
            "output_dir",
            "surface_file",
            "upper_file",
        ),
        allowed_overrides=COMMON_OVERRIDES + ("threads",),
    ),
    ModelId.AURORA: AdapterSpec(
        ModelCapabilities(
            model=ModelId.AURORA,
            min_forecast_hours=6,
            max_forecast_hours=360,
            forecast_step_hours=6,
            fixed_output_interval_hours=6,
            surface_variables=("2t", "10u", "10v", "msl"),
            upper_variables=("t", "u", "v", "q", "z"),
        ),
        path_keys=(
            "checkpoint_file",
            "data_dir",
            "output_dir",
            "static_file",
            "surface_file",
            "atmospheric_file",
        ),
        upper_filename="atmospheric.nc",
        allowed_overrides=(
            "device",
            "device_id",
            "download_missing",
            "autocast",
            "deterministic",
            "static_file",
            "surface_file",
            "atmospheric_file",
        ),
    ),
}


def capabilities(model: ModelId | str) -> ModelCapabilities:
    return SPECS[ModelId(model)].capabilities


class ModelAdapter:
    def __init__(self, installation: ModelInstallation, target: TargetConfig):
        self.installation = installation
        self.target = target
        self.model = installation.model
        self.spec = SPECS[self.model]

    def validate_request(self, request: ForecastRequest) -> None:
        if request.model != self.model:
            raise ValueError("request model does not match adapter")
        for case in request.cases:
            self.spec.capabilities.validate(case)
        unknown = set(request.config_overrides) - set(self.spec.allowed_overrides)
        if unknown:
            raise ValueError(
                f"unsupported config overrides for {self.model.value}: {sorted(unknown)}"
            )

    def _absolute_config(self, raw: Mapping[str, Any]) -> dict[str, Any]:
        result = dict(raw)
        base = self.installation.base_config.parent
        for key in self.spec.path_keys:
            value = result.get(key)
            if value in (None, ""):
                continue
            candidate = Path(str(value)).expanduser()
            result[key] = str(candidate if candidate.is_absolute() else (base / candidate).resolve())
        return result

    def prepare(self, request: ForecastRequest, job_dir: Path, reuse: bool = False) -> tuple[Path, Path]:
        import yaml

        self.validate_request(request)
        config_path = job_dir / "effective-config.yaml"
        cases_path = job_dir / "cases.csv"
        if reuse:
            if not config_path.is_file() or not cases_path.is_file():
                raise ValueError("cannot resume: effective config or cases file is missing")
            return config_path, cases_path
        raw = yaml.safe_load(self.installation.base_config.read_text(encoding="utf-8")) or {}
        raw = self._absolute_config(raw)
        raw.update(self.installation.overrides)
        raw.update(dict(request.config_overrides))
        raw["experiment"] = self.model.value
        raw["output_dir"] = str((job_dir / "outputs").resolve())
        raw["device_id"] = (
            request.device_id if request.device_id is not None else self.installation.device_id
        )
        for key in self.spec.path_keys:
            value = raw.get(key)
            if value not in (None, ""):
                candidate = Path(str(value)).expanduser()
                raw[key] = str(
                    candidate if candidate.is_absolute() else (self.installation.project_dir / candidate).resolve()
                )
        config_path.write_text(
            yaml.safe_dump(raw, sort_keys=True, allow_unicode=True), encoding="utf-8"
        )
        fields = (
            "case_id",
            "storm_id",
            "init_time",
            "forecast_hours",
            "output_interval_hours",
            "name",
            "basin",
        )
        with cases_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for case in request.cases:
                writer.writerow(case.to_dict())
        return config_path, cases_path

    def run_command(
        self,
        config_path: Path,
        cases_path: Path,
        *,
        dry_run: bool = False,
        resume: bool = False,
    ) -> list[str]:
        command = [
            self.target.conda_executable,
            "run",
            "--no-capture-output",
            "-n",
            self.installation.conda_env,
            "python",
            "-m",
            self.installation.module,
            "run",
            "--config",
            str(config_path),
            "--cases",
            str(cases_path),
        ]
        if dry_run:
            command.append("--dry-run")
        if resume:
            command.append("--resume")
        return command

    def doctor_command(self, load_model: bool = False) -> list[str]:
        command = [
            self.target.conda_executable,
            "run",
            "--no-capture-output",
            "-n",
            self.installation.conda_env,
            "python",
            "-m",
            self.installation.module,
            "doctor",
            "--config",
            str(self.installation.base_config),
        ]
        if load_model:
            command.extend(self.spec.doctor_load_args)
        return command

    @property
    def experiment_dir_name(self) -> str:
        return self.model.value

    def collect_results(self, job_id: str, request: ForecastRequest, job_dir: Path) -> ResultIndex:
        experiment_dir = job_dir / "outputs" / self.experiment_dir_name
        summary = experiment_dir / "summary.csv"
        batch_metadata = _read_json(experiment_dir / "batch.json")
        rows: list[dict[str, str]] = []
        if summary.is_file():
            with summary.open(newline="", encoding="utf-8-sig") as handle:
                rows = list(csv.DictReader(handle))
        row_by_case = {row.get("case_id", ""): row for row in rows}
        case_results: list[CaseResult] = []
        for case in request.cases:
            row = row_by_case.get(case.case_id, {})
            relative_dir = row.get("result_dir") or f"cases/{case.case_id}"
            case_dir = experiment_dir / relative_dir
            artifacts = (
                ResultArtifact(
                    logical_name="surface",
                    path=str((case_dir / self.spec.surface_filename).resolve(strict=False)),
                    variables=self.spec.capabilities.surface_variables,
                    exists=(case_dir / self.spec.surface_filename).exists(),
                ),
                ResultArtifact(
                    logical_name="upper_air",
                    path=str((case_dir / self.spec.upper_filename).resolve(strict=False)),
                    variables=self.spec.capabilities.upper_variables,
                    exists=(case_dir / self.spec.upper_filename).exists(),
                ),
            )
            case_results.append(
                CaseResult(
                    case_id=case.case_id,
                    status=row.get("status") or "missing",
                    result_dir=str(case_dir.resolve(strict=False)),
                    forecast_id=row.get("forecast_id") or "",
                    error=row.get("error") or "",
                    artifacts=artifacts,
                    metadata=_read_json(case_dir / "case.json"),
                )
            )
        return ResultIndex(
            job_id=job_id,
            model=self.model,
            target=request.target,
            experiment_dir=str(experiment_dir.resolve(strict=False)),
            cases=tuple(case_results),
            batch_metadata=batch_metadata,
        )

    @staticmethod
    def status_from_results(result: ResultIndex, exit_code: int) -> JobStatus:
        statuses = [case.status for case in result.cases]
        successful = sum(status in ("complete", "reused") for status in statuses)
        failed = sum(status not in ("complete", "reused") for status in statuses)
        if successful and not failed and exit_code == 0:
            return JobStatus.SUCCEEDED
        if successful:
            return JobStatus.PARTIAL
        return JobStatus.FAILED


def adapter_for(installation: ModelInstallation, target: TargetConfig) -> ModelAdapter:
    return ModelAdapter(installation, target)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {"value": value}


def parse_json_output(value: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else {"value": parsed}
    except json.JSONDecodeError:
        start = value.find("{")
        end = value.rfind("}")
        if start >= 0 and end > start:
            try:
                parsed = json.loads(value[start : end + 1])
                return parsed if isinstance(parsed, dict) else {"value": parsed}
            except json.JSONDecodeError:
                pass
        return {"output": value.strip()}
