"""Local worker process used directly and through the SSH transport."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from .adapters import adapter_for
from .registry import Registry, TargetConfig, load_registry
from .store import JobStore
from .types import CaseResult, ForecastRequest, JobRecord, JobStatus, ResultIndex


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _local_target(registry: Registry, name: str) -> TargetConfig:
    target = registry.target(name)
    if target.kind != "local" or target.jobs_dir is None:
        raise ValueError(f"worker target must be local: {name}")
    return target


class WorkerService:
    def __init__(self, registry: Registry, target_name: str = "local"):
        self.registry = registry
        self.target = _local_target(registry, target_name)
        self.store = JobStore(self.target.jobs_dir)

    @classmethod
    def from_config(cls, config: str | Path | None, target_name: str = "local") -> "WorkerService":
        return cls(load_registry(config), target_name)

    def submit(
        self,
        request: ForecastRequest,
        *,
        job_id: str | None = None,
        start: bool = True,
    ) -> JobRecord:
        if request.target != self.target.name:
            request = ForecastRequest.from_dict({**request.to_dict(), "target": self.target.name})
        record = self.store.create(request, job_id=job_id)
        if start:
            self._spawn(record.job_id, resume=False)
        return self.store.read(record.job_id)

    def _spawn(self, job_id: str, *, resume: bool) -> None:
        command = [
            sys.executable,
            "-m",
            "weather_hub",
            "--config",
            str(self.registry.path),
            "_worker-run",
            "--target",
            self.target.name,
            "--job-id",
            job_id,
        ]
        if resume:
            command.append("--resume")
        log_path = self.store.log_path(job_id)
        with log_path.open("ab", buffering=0) as log:
            process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                cwd=str(Path(__file__).resolve().parents[1]),
                start_new_session=True,
                close_fds=True,
            )
        time.sleep(0.1)
        exit_code = process.poll()
        if exit_code not in (None, 0):
            self.store.update(
                job_id,
                status=JobStatus.FAILED,
                exit_code=exit_code,
                error="worker process failed during startup; inspect model.log",
            )

    def execute(self, job_id: str, *, resume: bool = False) -> JobRecord:
        record = self.store.read(job_id)
        request = self.store.request(job_id)
        job_dir = self.store.job_dir(job_id)
        self.store.update(job_id, status=JobStatus.VALIDATING, worker_pid=os.getpid(), error="")
        try:
            installation = self.target.models.get(request.model)
            if installation is None:
                raise FileNotFoundError(f"model is not registered on target: {request.model.value}")
            if not installation.enabled:
                raise FileNotFoundError(f"model is disabled on target: {request.model.value}")
            if not installation.project_dir.is_dir():
                raise FileNotFoundError(f"project directory does not exist: {installation.project_dir}")
            if not installation.base_config.is_file():
                raise FileNotFoundError(f"base config does not exist: {installation.base_config}")
            adapter = adapter_for(installation, self.target)
            config_path, cases_path = adapter.prepare(request, job_dir, reuse=resume)
            config_hash = _sha256(config_path)
            if resume and record.effective_config_sha256 != config_hash:
                raise ValueError("cannot resume: effective configuration changed")
            if not resume:
                self.store.update(job_id, effective_config_sha256=config_hash)
        except FileNotFoundError as exc:
            return self.store.update(
                job_id,
                status=JobStatus.UNAVAILABLE,
                worker_pid=None,
                error=str(exc),
                exit_code=1,
            )
        except Exception as exc:
            return self.store.update(
                job_id,
                status=JobStatus.FAILED,
                worker_pid=None,
                error=f"{type(exc).__name__}: {exc}",
                exit_code=1,
            )

        device_id = request.device_id if request.device_id is not None else installation.device_id
        lock_path = self.store.lock_path(device_id)
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        self.store.update(job_id, status=JobStatus.QUEUED)
        with lock_path.open("a+", encoding="utf-8") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            self.store.update(job_id, status=JobStatus.RUNNING)
            command = adapter.run_command(
                config_path,
                cases_path,
                dry_run=request.dry_run,
                resume=resume,
            )
            log_path = self.store.log_path(job_id)
            with log_path.open("a", encoding="utf-8", buffering=1) as log:
                log.write(f"\n$ {shlex.join(command)}\n")
                try:
                    process = subprocess.Popen(
                        command,
                        cwd=str(installation.project_dir),
                        stdin=subprocess.DEVNULL,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    self.store.update(job_id, process_pid=process.pid)
                    exit_code = process.wait()
                except FileNotFoundError as exc:
                    return self.store.update(
                        job_id,
                        status=JobStatus.UNAVAILABLE,
                        process_pid=None,
                        worker_pid=None,
                        error=str(exc),
                        exit_code=1,
                    )
                except KeyboardInterrupt:
                    if "process" in locals() and process.poll() is None:
                        os.killpg(process.pid, signal.SIGINT)
                        process.wait()
                    return self.store.update(
                        job_id,
                        status=JobStatus.INTERRUPTED,
                        process_pid=None,
                        worker_pid=None,
                        exit_code=130,
                        error="interrupted",
                    )

        result = adapter.collect_results(job_id, request, job_dir)
        if request.dry_run and exit_code == 0:
            result = replace(
                result,
                cases=tuple(replace(item, status="planned") for item in result.cases),
            )
            status = JobStatus.SUCCEEDED
        elif exit_code == 130 or exit_code < 0:
            status = JobStatus.INTERRUPTED
        else:
            status = adapter.status_from_results(result, exit_code)
        result_path = self.store.write_result(result)
        return self.store.update(
            job_id,
            status=status,
            exit_code=exit_code,
            process_pid=None,
            worker_pid=None,
            result_path=str(result_path),
            error="" if status in (JobStatus.SUCCEEDED, JobStatus.PARTIAL) else _result_error(result),
        )

    def resume(self, job_id: str) -> JobRecord:
        record = self.store.read(job_id)
        if not record.terminal or record.status == JobStatus.SUCCEEDED:
            raise ValueError(f"job cannot be resumed from status {record.status.value}")
        self.store.update(
            job_id,
            status=JobStatus.QUEUED,
            error="",
            exit_code=None,
            process_pid=None,
            worker_pid=None,
        )
        self._spawn(job_id, resume=True)
        return self.store.read(job_id)

    def cancel(self, job_id: str) -> JobRecord:
        record = self.store.read(job_id)
        if record.terminal:
            return record
        pid = record.process_pid or record.worker_pid
        if pid:
            try:
                os.killpg(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        return self.store.update(
            job_id,
            status=JobStatus.INTERRUPTED,
            exit_code=130,
            process_pid=None,
            worker_pid=None,
            error="cancelled",
        )

    def status(self, job_id: str) -> JobRecord:
        return self.store.read(job_id)

    def result(self, job_id: str) -> dict[str, Any]:
        return self.store.result_dict(job_id)

    def logs(self, job_id: str) -> str:
        path = self.store.log_path(job_id)
        return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


def _result_error(result: ResultIndex) -> str:
    errors = [item.error for item in result.cases if item.error]
    return "; ".join(errors) or "model command did not produce complete results"


def worker_main(config: str | Path | None, target: str, job_id: str, resume: bool) -> int:
    service = WorkerService.from_config(config, target)
    record = service.execute(job_id, resume=resume)
    print(json.dumps(record.to_dict(), ensure_ascii=False))
    if record.status == JobStatus.SUCCEEDED:
        return 0
    if record.status == JobStatus.PARTIAL:
        return 2
    if record.status == JobStatus.INTERRUPTED:
        return 130
    return 1
