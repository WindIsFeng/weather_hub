"""Operational registry for execution targets and model installations."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .types import ModelId


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "config" / "models.yaml"


def _path(base: Path, value: str | Path) -> Path:
    candidate = Path(value).expanduser()
    return candidate.resolve() if candidate.is_absolute() else (base / candidate).resolve()


@dataclass(frozen=True)
class ModelInstallation:
    model: ModelId
    project_dir: Path
    conda_env: str
    module: str
    base_config: Path
    enabled: bool = True
    device_id: int = 0
    overrides: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TargetConfig:
    name: str
    kind: str
    jobs_dir: Path | None = None
    conda_executable: str = "conda"
    models: Mapping[ModelId, ModelInstallation] = field(default_factory=dict)
    host: str | None = None
    remote_executable: str = "weather-hub"
    remote_config: str | None = None


@dataclass(frozen=True)
class Registry:
    path: Path
    controller_jobs_dir: Path
    targets: Mapping[str, TargetConfig]

    def target(self, name: str) -> TargetConfig:
        try:
            return self.targets[name]
        except KeyError as exc:
            raise ValueError(f"unknown target: {name}") from exc


def config_path(value: str | Path | None = None) -> Path:
    if value is not None:
        return Path(value).expanduser().resolve()
    configured = os.environ.get("WEATHER_HUB_CONFIG")
    return Path(configured).expanduser().resolve() if configured else DEFAULT_CONFIG


def load_registry(value: str | Path | None = None) -> Registry:
    import yaml

    path = config_path(value)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if raw.get("version") != 1:
        raise ValueError("registry version must be 1")
    controller_jobs_dir = _path(path.parent, raw.get("controller_jobs_dir", "../runs"))
    target_values = raw.get("targets")
    if not isinstance(target_values, dict) or not target_values:
        raise ValueError("registry must define at least one target")
    targets: dict[str, TargetConfig] = {}
    for name, target_raw in target_values.items():
        if not isinstance(target_raw, dict):
            raise ValueError(f"target {name} must be a mapping")
        kind = str(target_raw.get("kind", "local"))
        if kind not in ("local", "ssh"):
            raise ValueError(f"target {name}: kind must be local or ssh")
        if kind == "ssh":
            host = str(target_raw.get("host") or "").strip()
            if not host:
                raise ValueError(f"target {name}: SSH host is required")
            targets[name] = TargetConfig(
                name=name,
                kind=kind,
                host=host,
                remote_executable=str(target_raw.get("remote_executable") or "weather-hub"),
                remote_config=target_raw.get("remote_config"),
            )
            continue
        jobs_dir = _path(path.parent, target_raw.get("jobs_dir", "../runs"))
        installations: dict[ModelId, ModelInstallation] = {}
        for model_name, model_raw in (target_raw.get("models") or {}).items():
            model = ModelId(model_name)
            project_dir = _path(path.parent, model_raw["project_dir"])
            installations[model] = ModelInstallation(
                model=model,
                project_dir=project_dir,
                conda_env=str(model_raw["conda_env"]),
                module=str(model_raw["module"]),
                base_config=_path(project_dir, model_raw.get("base_config", "configs/default.yaml")),
                enabled=bool(model_raw.get("enabled", True)),
                device_id=int(model_raw.get("device_id", 0)),
                overrides=dict(model_raw.get("overrides") or {}),
            )
        targets[name] = TargetConfig(
            name=name,
            kind=kind,
            jobs_dir=jobs_dir,
            conda_executable=str(target_raw.get("conda_executable") or "conda"),
            models=installations,
        )
    return Registry(path=path, controller_jobs_dir=controller_jobs_dir, targets=targets)
