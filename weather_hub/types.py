"""Dependency-light public data contracts."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence


SAFE_ID = re.compile(r"^[A-Za-z0-9_.-]+$")


class ModelId(str, Enum):
    PANGU = "pangu"
    FENGWU = "fengwu"
    FUXI = "fuxi"
    GRAPHCAST = "graphcast"
    AURORA = "aurora"


class JobStatus(str, Enum):
    QUEUED = "queued"
    VALIDATING = "validating"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    PARTIAL = "partial"
    FAILED = "failed"
    INTERRUPTED = "interrupted"
    UNAVAILABLE = "unavailable"


TERMINAL_STATUSES = {
    JobStatus.SUCCEEDED,
    JobStatus.PARTIAL,
    JobStatus.FAILED,
    JobStatus.INTERRUPTED,
    JobStatus.UNAVAILABLE,
}


def _safe_id(value: str, field_name: str) -> str:
    value = str(value).strip()
    if not value or not SAFE_ID.fullmatch(value):
        raise ValueError(f"{field_name} must match {SAFE_ID.pattern}")
    return value


def _utc_hour(value: str) -> str:
    text = str(value).strip()
    parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("init_time must include a timezone")
    if parsed.minute or parsed.second or parsed.microsecond:
        raise ValueError("init_time must be aligned to an exact hour")
    return parsed.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass(frozen=True)
class ForecastCase:
    case_id: str
    storm_id: str
    init_time: str
    forecast_hours: int
    output_interval_hours: int = 6
    name: str = ""
    basin: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "case_id", _safe_id(self.case_id, "case_id"))
        object.__setattr__(self, "storm_id", _safe_id(self.storm_id, "storm_id"))
        object.__setattr__(self, "init_time", _utc_hour(self.init_time))
        for field_name in ("forecast_hours", "output_interval_hours"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ForecastCase":
        return cls(
            case_id=str(value["case_id"]),
            storm_id=str(value["storm_id"]),
            init_time=str(value["init_time"]),
            forecast_hours=int(value["forecast_hours"]),
            output_interval_hours=int(value.get("output_interval_hours") or 6),
            name=str(value.get("name") or ""),
            basin=str(value.get("basin") or ""),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ForecastRequest:
    model: ModelId
    cases: tuple[ForecastCase, ...]
    target: str = "local"
    device_id: int | None = None
    config_overrides: Mapping[str, Any] = field(default_factory=dict)
    dry_run: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "model", ModelId(self.model))
        object.__setattr__(self, "cases", tuple(self.cases))
        object.__setattr__(self, "target", _safe_id(self.target, "target"))
        if not self.cases:
            raise ValueError("cases must not be empty")
        if len({case.case_id for case in self.cases}) != len(self.cases):
            raise ValueError("case_id values must be unique within a request")
        if self.device_id is not None and (
            isinstance(self.device_id, bool)
            or not isinstance(self.device_id, int)
            or self.device_id < 0
        ):
            raise ValueError("device_id must be a nonnegative integer")
        if not isinstance(self.config_overrides, Mapping):
            raise ValueError("config_overrides must be a mapping")
        if not isinstance(self.dry_run, bool):
            raise ValueError("dry_run must be a boolean")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ForecastRequest":
        return cls(
            model=ModelId(value["model"]),
            cases=tuple(ForecastCase.from_mapping(item) for item in value["cases"]),
            target=str(value.get("target") or "local"),
            device_id=value.get("device_id"),
            config_overrides=dict(value.get("config_overrides") or {}),
            dry_run=bool(value.get("dry_run", False)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model.value,
            "cases": [case.to_dict() for case in self.cases],
            "target": self.target,
            "device_id": self.device_id,
            "config_overrides": dict(self.config_overrides),
            "dry_run": self.dry_run,
        }

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ModelCapabilities:
    model: ModelId
    min_forecast_hours: int = 1
    max_forecast_hours: int | None = None
    forecast_step_hours: int = 1
    fixed_output_interval_hours: int | None = None
    surface_variables: tuple[str, ...] = ()
    upper_variables: tuple[str, ...] = ()

    def validate(self, case: ForecastCase) -> None:
        hours = case.forecast_hours
        if hours < self.min_forecast_hours:
            raise ValueError(
                f"{self.model.value}: forecast_hours must be >= {self.min_forecast_hours}"
            )
        if self.max_forecast_hours is not None and hours > self.max_forecast_hours:
            raise ValueError(
                f"{self.model.value}: forecast_hours must be <= {self.max_forecast_hours}"
            )
        if hours % self.forecast_step_hours:
            raise ValueError(
                f"{self.model.value}: forecast_hours must be divisible by "
                f"{self.forecast_step_hours}"
            )
        fixed = self.fixed_output_interval_hours
        if fixed is not None and case.output_interval_hours != fixed:
            raise ValueError(
                f"{self.model.value}: output_interval_hours must be {fixed}"
            )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["model"] = self.model.value
        return result


@dataclass(frozen=True)
class ModelInfo:
    model: ModelId
    target: str
    enabled: bool
    available: bool | None
    conda_env: str | None
    project_dir: str | None
    capabilities: ModelCapabilities
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["model"] = self.model.value
        result["capabilities"] = self.capabilities.to_dict()
        return result


@dataclass(frozen=True)
class HealthReport:
    model: ModelId
    target: str
    available: bool
    exit_code: int
    details: Mapping[str, Any] = field(default_factory=dict)
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model.value,
            "target": self.target,
            "available": self.available,
            "exit_code": self.exit_code,
            "details": dict(self.details),
            "error": self.error,
        }


@dataclass(frozen=True)
class ResultArtifact:
    logical_name: str
    path: str
    variables: tuple[str, ...]
    exists: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CaseResult:
    case_id: str
    status: str
    result_dir: str
    forecast_id: str = ""
    error: str = ""
    artifacts: tuple[ResultArtifact, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["artifacts"] = [item.to_dict() for item in self.artifacts]
        result["metadata"] = dict(self.metadata)
        return result


@dataclass(frozen=True)
class ResultIndex:
    job_id: str
    model: ModelId
    target: str
    experiment_dir: str
    cases: tuple[CaseResult, ...]
    error: str = ""
    batch_metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "model": self.model.value,
            "target": self.target,
            "experiment_dir": self.experiment_dir,
            "cases": [case.to_dict() for case in self.cases],
            "error": self.error,
            "batch_metadata": dict(self.batch_metadata),
        }


@dataclass
class JobRecord:
    job_id: str
    model: ModelId
    target: str
    status: JobStatus
    request_fingerprint: str
    created_at: str
    updated_at: str
    job_dir: str
    error: str = ""
    exit_code: int | None = None
    worker_pid: int | None = None
    process_pid: int | None = None
    result_path: str | None = None
    effective_config_sha256: str | None = None
    remote_job_dir: str | None = None

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "JobRecord":
        payload = dict(value)
        payload["model"] = ModelId(payload["model"])
        payload["status"] = JobStatus(payload["status"])
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["model"] = self.model.value
        result["status"] = self.status.value
        return result

    @property
    def terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def read_cases_csv(path: str | Path) -> tuple[ForecastCase, ...]:
    import csv

    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        return tuple(ForecastCase.from_mapping(row) for row in csv.DictReader(handle))


def cases_from_sequence(values: Sequence[Mapping[str, Any]]) -> tuple[ForecastCase, ...]:
    return tuple(ForecastCase.from_mapping(value) for value in values)
