# Weather Hub

Weather Hub is a lightweight controller for Pangu-Weather, FengWu, FuXi,
GraphCast and Aurora. It keeps every model in its own repository and Conda
environment, while exposing one Python SDK, CLI, job state model and result
index. The controller never imports a model package, so ONNX Runtime, JAX and
PyTorch CUDA dependencies remain isolated.

## Install the controller

```bash
cd /scratch/hufeng/ai_weather_models
conda create -n weather-hub --override-channels -c conda-forge \
  python=3.11 pip pyyaml pytest -y
conda run -n weather-hub python -m pip install --no-deps -e .
conda run -n weather-hub weather-hub models
```

An equivalent `environment.yml` is included for Conda installations whose
global channel policy permits `conda env create`.

The operational registry is `config/models.yaml`. Relative paths are resolved
against that file. Set `WEATHER_HUB_CONFIG` or pass global `--config` to use a
different registry.

## Run forecasts

Use the same case CSV columns accepted by the five model projects:

```csv
case_id,storm_id,init_time,forecast_hours,output_interval_hours,name,basin
ragasa_72h,ragasa_2025,2025-09-22T00:00:00Z,72,6,Ragasa,WP
```

```bash
# Validate orchestration without downloading data or loading weights.
weather-hub run --model pangu --cases cases.csv --dry-run

# Submit a real batch and stream its log until completion.
weather-hub run --model fuxi --cases cases.csv

# Submit without waiting.
weather-hub run --model aurora --cases cases.csv --target cloud-a --detach
weather-hub status JOB_ID --json
weather-hub logs JOB_ID --follow
weather-hub results JOB_ID --json
weather-hub resume JOB_ID
```

A direct one-case request is also supported:

```bash
weather-hub run --model pangu \
  --storm-id ragasa_2025 \
  --init-time 2025-09-22T00:00:00Z \
  --forecast-hours 72 \
  --name Ragasa --basin WP
```

Every job is stored below `runs/<job_id>/`. The effective model configuration
uses absolute input/model/cache paths and writes the untouched model output to
`outputs/<model>/`. `result.json` provides common case statuses and logical
`surface`/`upper_air` artifact entries without copying large NetCDF files.

## Python SDK

```python
from weather_hub import ForecastCase, ForecastRequest, ModelId, run_forecast

request = ForecastRequest(
    model=ModelId.PANGU,
    cases=(ForecastCase(
        case_id="ragasa_72h",
        storm_id="ragasa_2025",
        init_time="2025-09-22T00:00:00Z",
        forecast_hours=72,
    ),),
)
job = run_forecast(request)
print(job.job_id, job.status)
```

The SDK exports `list_models`, `doctor`, `submit_forecast`, `run_forecast`,
`get_job`, `wait_job`, `resume_job`, `cancel_job` and `get_results`.

## Remote GPU hosts

The SSH transport is agentless: it uses the same `weather-hub` executable on
the remote host but does not require an HTTP daemon.

1. Deploy this project and the selected model repositories on the GPU host.
2. Create every model's own Conda environment and install its weights.
3. Configure a remote registry whose `local` target points at those remote
   paths and environments.
4. Add an SSH target to the controller registry:

```yaml
targets:
  cloud-a:
    kind: ssh
    host: cloud-a                 # OpenSSH config alias
    remote_executable: /opt/weather-hub/.venv/bin/weather-hub
    remote_config: /opt/weather-hub/config/models.yaml
```

5. Verify the host before inference:

```bash
weather-hub doctor --model all --target cloud-a --load-model
```

Requests are sent as JSON over SSH standard input. The remote worker detaches,
writes atomic job state, and keeps running if the SSH connection drops. Status,
logs and results can be queried after reconnecting. Passwords and private keys
are never stored in Weather Hub configuration; use OpenSSH configuration and
key management.

Model weights, ERA5 caches and forecast outputs remain on the remote host. A
remote result index therefore contains remote absolute paths together with its
target name; Weather Hub does not silently transfer multi-gigabyte results.

## Resource and recovery rules

- Jobs sharing a target and GPU ID are serialized with a file lock.
- Different targets or GPU IDs may run concurrently.
- Each model retains its own ERA5 cache because required variables differ.
- Resume reuses the original request, cases and effective configuration. A
  changed effective configuration is rejected.
- Exit code `0` means success, `2` means partial case success, `1` means
  failed/unavailable and `130` means interrupted.
- GraphCast and Aurora full-global 0.25-degree inference should be assigned to
  a 48 GB or larger GPU; 80 GB is preferred for operational headroom.
