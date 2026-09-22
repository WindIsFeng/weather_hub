"""Public SDK and local/SSH controller."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, Mapping

from .adapters import SPECS, adapter_for, capabilities, parse_json_output
from .registry import Registry, TargetConfig, load_registry
from .store import JobStore, atomic_json
from .types import (
    ForecastRequest,
    HealthReport,
    JobRecord,
    JobStatus,
    ModelCapabilities,
    ModelId,
    ModelInfo,
)
from .worker import WorkerService


class Controller:
    def __init__(self, config: str | Path | None = None):
        self.registry = load_registry(config)

    def _target(self, name: str) -> TargetConfig:
        return self.registry.target(name)

    def _store_for_new(self, target: TargetConfig) -> JobStore:
        if target.kind == "local" and target.jobs_dir is not None:
            return JobStore(target.jobs_dir)
        return JobStore(self.registry.controller_jobs_dir)

    def _find_store(self, job_id: str) -> JobStore:
        roots = [self.registry.controller_jobs_dir]
        roots.extend(
            target.jobs_dir
            for target in self.registry.targets.values()
            if target.kind == "local" and target.jobs_dir is not None
        )
        seen: set[Path] = set()
        for root in roots:
            root = root.resolve()
            if root in seen:
                continue
            seen.add(root)
            store = JobStore(root)
            if (store.job_dir(job_id) / "job.json").is_file():
                return store
        raise FileNotFoundError(f"unknown job: {job_id}")

    def list_models(self, target_name: str = "local") -> list[ModelInfo]:
        target = self._target(target_name)
        if target.kind == "ssh":
            raw = self._ssh_json(target, ["models", "--target", "local", "--json"])
            return [_model_info(item, target_name) for item in raw]
        envs, env_error = _conda_environments(target.conda_executable)
        results: list[ModelInfo] = []
        for model in ModelId:
            installation = target.models.get(model)
            if installation is None:
                results.append(
                    ModelInfo(
                        model=model,
                        target=target_name,
                        enabled=False,
                        available=False,
                        conda_env=None,
                        project_dir=None,
                        capabilities=capabilities(model),
                        reason="not registered",
                    )
                )
                continue
            reasons: list[str] = []
            if not installation.enabled:
                reasons.append("disabled")
            if not installation.project_dir.is_dir():
                reasons.append("project directory missing")
            if not installation.base_config.is_file():
                reasons.append("base config missing")
            if env_error:
                reasons.append(env_error)
            elif installation.conda_env not in envs:
                reasons.append(f"Conda environment missing: {installation.conda_env}")
            results.append(
                ModelInfo(
                    model=model,
                    target=target_name,
                    enabled=installation.enabled,
                    available=not reasons,
                    conda_env=installation.conda_env,
                    project_dir=str(installation.project_dir),
                    capabilities=capabilities(model),
                    reason="; ".join(reasons),
                )
            )
        return results

    def doctor(
        self,
        model: ModelId | str,
        target_name: str = "local",
        *,
        load_model: bool = False,
    ) -> HealthReport:
        model = ModelId(model)
        target = self._target(target_name)
        if target.kind == "ssh":
            arguments = [
                "doctor",
                "--model",
                model.value,
                "--target",
                "local",
                "--json",
            ]
            if load_model:
                arguments.append("--load-model")
            raw = self._ssh_json(target, arguments)
            raw["target"] = target_name
            return HealthReport(
                model=model,
                target=target_name,
                available=bool(raw.get("available")),
                exit_code=int(raw.get("exit_code", 1)),
                details=dict(raw.get("details") or {}),
                error=str(raw.get("error") or ""),
            )
        installation = target.models.get(model)
        if installation is None or not installation.enabled:
            return HealthReport(model, target_name, False, 1, error="model is not enabled")
        adapter = adapter_for(installation, target)
        try:
            completed = subprocess.run(
                adapter.doctor_command(load_model),
                cwd=str(installation.project_dir),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
        except FileNotFoundError as exc:
            return HealthReport(model, target_name, False, 1, error=str(exc))
        details = parse_json_output(completed.stdout)
        error = "" if completed.returncode == 0 else str(details.get("output") or completed.stdout).strip()
        return HealthReport(
            model=model,
            target=target_name,
            available=completed.returncode == 0,
            exit_code=completed.returncode,
            details=details,
            error=error,
        )

    def submit(self, request: ForecastRequest) -> JobRecord:
        target = self._target(request.target)
        for case in request.cases:
            capabilities(request.model).validate(case)
        if target.kind == "local":
            return WorkerService(self.registry, target.name).submit(request)
        store = self._store_for_new(target)
        mirror = store.create(request)
        remote_request = ForecastRequest.from_dict({**request.to_dict(), "target": "local"})
        try:
            raw = self._ssh_json(
                target,
                ["_worker-submit", "--target", "local", "--job-id", mirror.job_id],
                stdin=json.dumps(remote_request.to_dict()),
            )
        except Exception as exc:
            return store.update(
                mirror.job_id,
                status=JobStatus.UNAVAILABLE,
                error=f"SSH submission failed: {exc}",
                exit_code=1,
            )
        return self._sync_remote(store, mirror, raw)

    def get_job(self, job_id: str) -> JobRecord:
        store = self._find_store(job_id)
        record = store.read(job_id)
        target = self._target(record.target)
        if target.kind == "local":
            return record
        raw = self._ssh_json(
            target,
            ["_worker-status", "--target", "local", "--job-id", job_id],
        )
        return self._sync_remote(store, record, raw)

    def wait(self, job_id: str, *, poll_interval: float = 1.0) -> JobRecord:
        while True:
            record = self.get_job(job_id)
            if record.terminal:
                return record
            time.sleep(poll_interval)

    def resume(self, job_id: str) -> JobRecord:
        store = self._find_store(job_id)
        record = store.read(job_id)
        target = self._target(record.target)
        if target.kind == "local":
            return WorkerService(self.registry, target.name).resume(job_id)
        raw = self._ssh_json(
            target,
            ["_worker-resume", "--target", "local", "--job-id", job_id],
        )
        return self._sync_remote(store, record, raw)

    def cancel(self, job_id: str) -> JobRecord:
        store = self._find_store(job_id)
        record = store.read(job_id)
        target = self._target(record.target)
        if target.kind == "local":
            return WorkerService(self.registry, target.name).cancel(job_id)
        raw = self._ssh_json(
            target,
            ["_worker-cancel", "--target", "local", "--job-id", job_id],
        )
        return self._sync_remote(store, record, raw)

    def results(self, job_id: str) -> dict[str, Any]:
        store = self._find_store(job_id)
        record = store.read(job_id)
        target = self._target(record.target)
        if target.kind == "local":
            return store.result_dict(job_id)
        raw = self._ssh_json(
            target,
            ["_worker-results", "--target", "local", "--job-id", job_id],
        )
        raw["target"] = record.target
        atomic_json(store.job_dir(job_id) / "result.json", raw)
        return raw

    def logs(self, job_id: str) -> str:
        store = self._find_store(job_id)
        record = store.read(job_id)
        target = self._target(record.target)
        if target.kind == "local":
            return WorkerService(self.registry, target.name).logs(job_id)
        return self._ssh(
            target,
            ["_worker-logs", "--target", "local", "--job-id", job_id],
        ).stdout

    def _sync_remote(
        self,
        store: JobStore,
        local: JobRecord,
        raw: Mapping[str, Any],
    ) -> JobRecord:
        remote = JobRecord.from_dict(raw)
        return store.update(
            local.job_id,
            status=remote.status,
            error=remote.error,
            exit_code=remote.exit_code,
            remote_job_dir=remote.job_dir,
            result_path=remote.result_path,
            effective_config_sha256=remote.effective_config_sha256,
        )

    def _remote_command(self, target: TargetConfig, arguments: Iterable[str]) -> str:
        command = [target.remote_executable]
        if target.remote_config:
            command.extend(("--config", target.remote_config))
        command.extend(arguments)
        return shlex.join(command)

    def _ssh(
        self,
        target: TargetConfig,
        arguments: list[str],
        *,
        stdin: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        completed = subprocess.run(
            ["ssh", str(target.host), self._remote_command(target, arguments)],
            input=stdin,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr.strip() or completed.stdout.strip())
        return completed

    def _ssh_json(
        self,
        target: TargetConfig,
        arguments: list[str],
        *,
        stdin: str | None = None,
    ) -> Any:
        output = self._ssh(target, arguments, stdin=stdin).stdout
        try:
            return json.loads(output)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"remote command did not return JSON: {output[:500]}") from exc


def _conda_environments(executable: str) -> tuple[set[str], str]:
    try:
        completed = subprocess.run(
            [executable, "env", "list", "--json"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except FileNotFoundError:
        return set(), f"Conda executable missing: {executable}"
    if completed.returncode:
        return set(), completed.stderr.strip() or "cannot list Conda environments"
    try:
        values = json.loads(completed.stdout).get("envs", [])
    except json.JSONDecodeError:
        return set(), "invalid output from conda env list"
    return {Path(value).name for value in values}, ""


def _model_info(value: Mapping[str, Any], target: str) -> ModelInfo:
    caps = dict(value["capabilities"])
    caps["model"] = ModelId(caps["model"])
    for key in ("surface_variables", "upper_variables"):
        caps[key] = tuple(caps.get(key) or ())
    return ModelInfo(
        model=ModelId(value["model"]),
        target=target,
        enabled=bool(value["enabled"]),
        available=value.get("available"),
        conda_env=value.get("conda_env"),
        project_dir=value.get("project_dir"),
        capabilities=ModelCapabilities(**caps),
        reason=str(value.get("reason") or ""),
    )


def list_models(target: str = "local", config: str | Path | None = None) -> list[ModelInfo]:
    return Controller(config).list_models(target)


def doctor(
    model: ModelId | str,
    target: str = "local",
    *,
    load_model: bool = False,
    config: str | Path | None = None,
) -> HealthReport:
    return Controller(config).doctor(model, target, load_model=load_model)


def submit_forecast(
    request: ForecastRequest,
    *,
    config: str | Path | None = None,
) -> JobRecord:
    return Controller(config).submit(request)


def run_forecast(
    request: ForecastRequest,
    *,
    config: str | Path | None = None,
    poll_interval: float = 1.0,
) -> JobRecord:
    controller = Controller(config)
    record = controller.submit(request)
    return record if record.terminal else controller.wait(record.job_id, poll_interval=poll_interval)


def get_job(job_id: str, *, config: str | Path | None = None) -> JobRecord:
    return Controller(config).get_job(job_id)


def wait_job(
    job_id: str,
    *,
    config: str | Path | None = None,
    poll_interval: float = 1.0,
) -> JobRecord:
    return Controller(config).wait(job_id, poll_interval=poll_interval)


def resume_job(job_id: str, *, config: str | Path | None = None) -> JobRecord:
    return Controller(config).resume(job_id)


def cancel_job(job_id: str, *, config: str | Path | None = None) -> JobRecord:
    return Controller(config).cancel(job_id)


def get_results(job_id: str, *, config: str | Path | None = None) -> dict[str, Any]:
    return Controller(config).results(job_id)
