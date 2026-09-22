import json

import yaml

from weather_hub.api import Controller
from weather_hub.types import ForecastCase, ForecastRequest, JobRecord, JobStatus, ModelId, utc_now


def test_ssh_submit_uses_remote_worker_and_keeps_controller_target(tmp_path, monkeypatch):
    config = tmp_path / "models.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "controller_jobs_dir": str(tmp_path / "runs"),
                "targets": {
                    "cloud-a": {
                        "kind": "ssh",
                        "host": "cloud-a",
                        "remote_executable": "/opt/weather-hub/bin/weather-hub",
                        "remote_config": "/opt/weather-hub/models.yaml",
                    }
                },
            }
        )
    )
    controller = Controller(config)
    seen = {}

    def fake_ssh_json(target, arguments, stdin=None):
        seen["arguments"] = arguments
        remote_request = json.loads(stdin)
        seen["request"] = remote_request
        now = utc_now()
        return JobRecord(
            job_id=arguments[-1],
            model=ModelId.PANGU,
            target="local",
            status=JobStatus.QUEUED,
            request_fingerprint="remote",
            created_at=now,
            updated_at=now,
            job_dir="/remote/runs/job",
        ).to_dict()

    monkeypatch.setattr(controller, "_ssh_json", fake_ssh_json)
    request = ForecastRequest(
        model=ModelId.PANGU,
        target="cloud-a",
        cases=(ForecastCase("c1", "s1", "2025-09-22T00:00:00Z", 6),),
    )
    record = controller.submit(request)
    assert record.target == "cloud-a"
    assert record.remote_job_dir == "/remote/runs/job"
    assert seen["request"]["target"] == "local"
    assert seen["arguments"][0] == "_worker-submit"
