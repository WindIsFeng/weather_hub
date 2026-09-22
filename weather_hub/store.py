"""Atomic, inspectable file-based job persistence."""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

from .types import ForecastRequest, JobRecord, JobStatus, ResultIndex, utc_now


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


class JobStore:
    def __init__(self, root: Path):
        self.root = root.resolve()

    def job_dir(self, job_id: str) -> Path:
        if not job_id or any(char not in "0123456789abcdef-" for char in job_id):
            raise ValueError("invalid job_id")
        return self.root / job_id

    def create(self, request: ForecastRequest, job_id: str | None = None) -> JobRecord:
        job_id = job_id or str(uuid.uuid4())
        job_dir = self.job_dir(job_id)
        job_dir.mkdir(parents=True, exist_ok=False)
        atomic_json(job_dir / "request.json", request.to_dict())
        now = utc_now()
        record = JobRecord(
            job_id=job_id,
            model=request.model,
            target=request.target,
            status=JobStatus.QUEUED,
            request_fingerprint=request.fingerprint,
            created_at=now,
            updated_at=now,
            job_dir=str(job_dir),
        )
        self.write(record)
        return record

    def read(self, job_id: str) -> JobRecord:
        raw = json.loads((self.job_dir(job_id) / "job.json").read_text(encoding="utf-8"))
        return JobRecord.from_dict(raw)

    def request(self, job_id: str) -> ForecastRequest:
        raw = json.loads((self.job_dir(job_id) / "request.json").read_text(encoding="utf-8"))
        return ForecastRequest.from_dict(raw)

    def write(self, record: JobRecord) -> None:
        record.updated_at = utc_now()
        atomic_json(self.job_dir(record.job_id) / "job.json", record.to_dict())

    def update(self, job_id: str, **changes: Any) -> JobRecord:
        record = replace(self.read(job_id), **changes, updated_at=utc_now())
        atomic_json(self.job_dir(job_id) / "job.json", record.to_dict())
        return record

    def write_result(self, result: ResultIndex) -> Path:
        path = self.job_dir(result.job_id) / "result.json"
        atomic_json(path, result.to_dict())
        return path

    def result_dict(self, job_id: str) -> dict[str, Any]:
        path = self.job_dir(job_id) / "result.json"
        return json.loads(path.read_text(encoding="utf-8"))

    def log_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "model.log"

    def lock_path(self, device_id: int) -> Path:
        return self.root / ".locks" / f"gpu-{device_id}.lock"
