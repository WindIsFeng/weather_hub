"""Command-line interface for users and the agentless SSH worker protocol."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from .api import Controller
from .registry import config_path, load_registry
from .types import ForecastCase, ForecastRequest, JobRecord, JobStatus, ModelId, read_cases_csv
from .worker import WorkerService, worker_main


def _json(value: Any) -> None:
    print(json.dumps(value, indent=2, ensure_ascii=False))


def _record(record: JobRecord, as_json: bool) -> None:
    if as_json:
        _json(record.to_dict())
    else:
        print(f"{record.job_id}  {record.model.value}  {record.target}  {record.status.value}")
        if record.error:
            print(record.error, file=sys.stderr)


def _parse_set(values: list[str]) -> dict[str, Any]:
    import yaml

    result: dict[str, Any] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--set expects KEY=VALUE: {value}")
        key, raw = value.split("=", 1)
        if not key:
            raise ValueError("--set key must not be empty")
        result[key] = yaml.safe_load(raw)
    return result


def _direct_case(args: argparse.Namespace) -> ForecastCase:
    if not args.storm_id or not args.init_time or args.forecast_hours is None:
        raise ValueError(
            "without --cases, --storm-id, --init-time and --forecast-hours are required"
        )
    case_id = args.case_id
    if not case_id:
        parsed = datetime.fromisoformat(args.init_time.replace("Z", "+00:00"))
        case_id = f"{args.storm_id}_{parsed:%Y%m%dT%HZ}_{args.forecast_hours}h"
    return ForecastCase(
        case_id=case_id,
        storm_id=args.storm_id,
        init_time=args.init_time,
        forecast_hours=args.forecast_hours,
        output_interval_hours=args.output_interval_hours,
        name=args.name or "",
        basin=args.basin or "",
    )


def _request(args: argparse.Namespace) -> ForecastRequest:
    cases = read_cases_csv(args.cases) if args.cases else (_direct_case(args),)
    return ForecastRequest(
        model=ModelId(args.model),
        cases=cases,
        target=args.target,
        device_id=args.device_id,
        config_overrides=_parse_set(args.set),
        dry_run=args.dry_run,
    )


def _exit_code(record: JobRecord) -> int:
    if record.status == JobStatus.SUCCEEDED:
        return 0
    if record.status == JobStatus.PARTIAL:
        return 2
    if record.status == JobStatus.INTERRUPTED:
        return 130
    return 1


def _wait(controller: Controller, job_id: str, *, show_logs: bool) -> JobRecord:
    offset = 0
    while True:
        try:
            if show_logs:
                contents = controller.logs(job_id)
                if len(contents) > offset:
                    print(contents[offset:], end="", flush=True)
                    offset = len(contents)
            record = controller.get_job(job_id)
        except RuntimeError as exc:
            print(f"connection warning: {exc}; retrying", file=sys.stderr)
            time.sleep(3)
            continue
        if record.terminal:
            return record
        time.sleep(1)


def _add_target(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--target", default="local")


def _add_job_id(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("job_id")
    parser.add_argument("--json", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="weather-hub")
    parser.add_argument("--config", type=Path, default=None, help="operational registry YAML")
    sub = parser.add_subparsers(dest="command", required=True)

    models = sub.add_parser("models", help="list registered models and quick availability")
    _add_target(models)
    models.add_argument("--json", action="store_true")

    doctor = sub.add_parser("doctor", help="validate model environments and artifacts")
    doctor.add_argument("--model", required=True, choices=("all", *(item.value for item in ModelId)))
    _add_target(doctor)
    doctor.add_argument("--load-model", action="store_true")
    doctor.add_argument("--json", action="store_true")

    run = sub.add_parser("run", help="submit a forecast job")
    run.add_argument("--model", required=True, choices=tuple(item.value for item in ModelId))
    _add_target(run)
    run.add_argument("--cases", type=Path)
    run.add_argument("--storm-id")
    run.add_argument("--init-time")
    run.add_argument("--forecast-hours", type=int)
    run.add_argument("--output-interval-hours", type=int, default=6)
    run.add_argument("--case-id")
    run.add_argument("--name", default="")
    run.add_argument("--basin", default="")
    run.add_argument("--device-id", type=int)
    run.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--detach", action="store_true")
    run.add_argument("--json", action="store_true")

    status = sub.add_parser("status", help="show current job status")
    _add_job_id(status)
    logs = sub.add_parser("logs", help="show job logs")
    logs.add_argument("job_id")
    logs.add_argument("--follow", action="store_true")
    results = sub.add_parser("results", help="show the unified result index")
    _add_job_id(results)
    resume = sub.add_parser("resume", help="resume an interrupted or failed job")
    _add_job_id(resume)
    resume.add_argument("--detach", action="store_true")
    cancel = sub.add_parser("cancel", help="cancel a queued or running job")
    _add_job_id(cancel)

    for name in (
        "_worker-submit",
        "_worker-status",
        "_worker-results",
        "_worker-logs",
        "_worker-resume",
        "_worker-cancel",
        "_worker-run",
    ):
        internal = sub.add_parser(name, help=argparse.SUPPRESS)
        internal.add_argument("--target", default="local")
        internal.add_argument("--job-id", required=True)
        if name == "_worker-run":
            internal.add_argument("--resume", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return _main(args)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


def _main(args: argparse.Namespace) -> int:
    config = args.config
    if args.command == "_worker-run":
        return worker_main(config, args.target, args.job_id, args.resume)
    if args.command.startswith("_worker-"):
        service = WorkerService.from_config(config, args.target)
        if args.command == "_worker-submit":
            request = ForecastRequest.from_dict(json.load(sys.stdin))
            value: Any = service.submit(request, job_id=args.job_id).to_dict()
        elif args.command == "_worker-status":
            value = service.status(args.job_id).to_dict()
        elif args.command == "_worker-results":
            value = service.result(args.job_id)
        elif args.command == "_worker-logs":
            print(service.logs(args.job_id), end="")
            return 0
        elif args.command == "_worker-resume":
            value = service.resume(args.job_id).to_dict()
        else:
            value = service.cancel(args.job_id).to_dict()
        print(json.dumps(value, ensure_ascii=False))
        return 0

    controller = Controller(config)
    if args.command == "models":
        values = controller.list_models(args.target)
        if args.json:
            _json([item.to_dict() for item in values])
        else:
            for item in values:
                marker = "ready" if item.available else "unavailable"
                print(f"{item.model.value:10} {marker:11} {item.conda_env or '-':12} {item.reason}")
        return 0
    if args.command == "doctor":
        selected = list(ModelId) if args.model == "all" else [ModelId(args.model)]
        reports = [
            controller.doctor(item, args.target, load_model=args.load_model)
            for item in selected
        ]
        if args.json:
            value = [item.to_dict() for item in reports] if len(reports) > 1 else reports[0].to_dict()
            _json(value)
        else:
            for report in reports:
                marker = "ok" if report.available else "unavailable"
                print(f"{report.model.value}: {marker}")
                if report.error:
                    print(f"  {report.error}")
        return 0 if all(item.available for item in reports) else 1
    if args.command == "run":
        record = controller.submit(_request(args))
        if not args.detach and not record.terminal:
            record = _wait(controller, record.job_id, show_logs=not args.json)
        _record(record, args.json)
        return 0 if args.detach else _exit_code(record)
    if args.command == "status":
        record = controller.get_job(args.job_id)
        _record(record, args.json)
        return _exit_code(record) if record.terminal else 0
    if args.command == "logs":
        if not args.follow:
            print(controller.logs(args.job_id), end="")
            return 0
        record = _wait(controller, args.job_id, show_logs=True)
        return _exit_code(record)
    if args.command == "results":
        value = controller.results(args.job_id)
        _json(value)
        return 0
    if args.command == "resume":
        record = controller.resume(args.job_id)
        if not args.detach:
            record = _wait(controller, record.job_id, show_logs=not args.json)
        _record(record, args.json)
        return 0 if args.detach else _exit_code(record)
    if args.command == "cancel":
        record = controller.cancel(args.job_id)
        _record(record, args.json)
        return 0
    parser = build_parser()
    parser.error("unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
