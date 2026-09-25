"""Download the shared ERA5 initial fields for the frozen Weather Hub cases.

The output uses date-based NetCDF files. Each model can read the same files via
its explicit *_file configuration, so no model-managed cache sidecars are needed.
Planning and verification never contact CDS. Only --download does.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = ROOT / "cases/weather_hub_cases_2022_2024_v2026-09-22.csv"
LEVELS = (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)
SURFACE = (
    "2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind",
    "mean_sea_level_pressure", "geopotential", "land_sea_mask",
)
UPPER = (
    "geopotential", "specific_humidity", "temperature",
    "u_component_of_wind", "v_component_of_wind", "vertical_velocity",
    "relative_humidity",
)
SHORT_NAMES = {
    "surface": ("t2m", "u10", "v10", "msl", "z", "lsm"),
    "upper": ("z", "q", "t", "u", "v", "w", "r"),
    "precipitation": ("tp",),
    "static": ("z", "lsm", "slt"),
}


@dataclass(frozen=True)
class Job:
    kind: str
    path: Path
    dataset: str
    request: dict
    times: tuple[datetime, ...]


def read_initial_times(path: Path) -> set[datetime]:
    times = set()
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not {"case_id", "init_time"} <= set(reader.fieldnames):
            raise ValueError("case CSV must contain case_id and init_time")
        seen_ids = set()
        for row in reader:
            case_id = row["case_id"]
            if case_id in seen_ids:
                raise ValueError(f"duplicate case_id: {case_id}")
            seen_ids.add(case_id)
            value = datetime.fromisoformat(row["init_time"].replace("Z", "+00:00"))
            if value.tzinfo is None:
                raise ValueError(f"{case_id}: init_time needs a timezone")
            value = value.astimezone(timezone.utc)
            if value.minute or value.second or value.microsecond or value.hour % 6:
                raise ValueError(f"{case_id}: init_time must be a 6-hour UTC cycle")
            times.add(value)
    if not times:
        raise ValueError("case CSV is empty")
    return times


def group_by_day(times: set[datetime]) -> dict[str, tuple[datetime, ...]]:
    groups = defaultdict(list)
    for value in sorted(times):
        groups[value.strftime("%Y%m%d")].append(value)
    return {day: tuple(values) for day, values in sorted(groups.items())}


def cds_request(times: tuple[datetime, ...], variables: tuple[str, ...], *,
                pressure_levels: bool = False) -> dict:
    day = times[0]
    request = {
        "product_type": ["reanalysis"],
        "variable": list(variables),
        "year": [day.strftime("%Y")],
        "month": [day.strftime("%m")],
        "day": [day.strftime("%d")],
        "time": [value.strftime("%H:00") for value in times],
        "data_format": "netcdf",
        "download_format": "unarchived",
        "grid": [0.25, 0.25],
    }
    if pressure_levels:
        request["pressure_level"] = [str(level) for level in LEVELS]
    return request


def build_plan(initial_times: set[datetime], output_dir: Path) -> list[Job]:
    frames = initial_times | {value - timedelta(hours=6) for value in initial_times}
    rain = {value - timedelta(hours=hour) for value in initial_times for hour in range(12)}
    jobs = []
    for day, times in group_by_day(frames).items():
        directory = output_dir / day
        jobs.extend((
            Job("surface", directory / "surface.nc", "reanalysis-era5-single-levels",
                cds_request(times, SURFACE), times),
            Job("upper", directory / "upper.nc", "reanalysis-era5-pressure-levels",
                cds_request(times, UPPER, pressure_levels=True), times),
        ))
    for day, times in group_by_day(rain).items():
        jobs.append(Job(
            "precipitation", output_dir / "precipitation" / f"{day}.nc",
            "reanalysis-era5-single-levels", cds_request(times, ("total_precipitation",)), times,
        ))
    static_time = (datetime(2023, 1, 1, tzinfo=timezone.utc),)
    jobs.append(Job(
        "static", output_dir / "static.nc", "reanalysis-era5-single-levels",
        cds_request(static_time, ("geopotential", "land_sea_mask", "soil_type")),
        static_time,
    ))
    return jobs


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_netcdf(job: Job, path: Path) -> None:
    import numpy as np
    import xarray as xr

    with xr.open_dataset(path, engine="netcdf4") as ds:
        time_name = next((name for name in ("valid_time", "time") if name in ds.coords), None)
        lat_name = next((name for name in ("latitude", "lat") if name in ds.coords), None)
        lon_name = next((name for name in ("longitude", "lon") if name in ds.coords), None)
        if not all((time_name, lat_name, lon_name)):
            raise ValueError(f"{path}: missing time or grid coordinates")
        actual_times = np.asarray(ds[time_name].values).astype("datetime64[s]").reshape(-1)
        expected_times = np.asarray(
            [value.replace(tzinfo=None) for value in job.times], dtype="datetime64[s]"
        )
        if len(actual_times) != len(expected_times) or not np.array_equal(
            np.sort(actual_times), np.sort(expected_times)
        ):
            raise ValueError(f"{path}: unexpected ERA5 timestamps")
        lat = np.asarray(ds[lat_name].values, dtype=float)
        lon = np.asarray(ds[lon_name].values, dtype=float) % 360
        if not np.allclose(np.sort(lat), np.arange(-90, 90.25, 0.25), atol=1e-5):
            raise ValueError(f"{path}: not a complete 0.25-degree latitude grid")
        if not np.allclose(np.sort(lon), np.arange(0, 360, 0.25), atol=1e-5):
            raise ValueError(f"{path}: not a complete 0.25-degree longitude grid")
        missing = set(SHORT_NAMES[job.kind]) - set(ds.data_vars)
        if missing:
            raise ValueError(f"{path}: missing variables {sorted(missing)}")
        if job.kind == "upper":
            level_name = next((name for name in ("pressure_level", "level") if name in ds.coords), None)
            if level_name is None:
                raise ValueError(f"{path}: missing pressure levels")
            level = ds[level_name]
            values = np.asarray(level.values, dtype=float)
            if str(level.attrs.get("units", "hPa")).lower() in ("pa", "pascal", "pascals"):
                values /= 100
            if not np.array_equal(np.sort(values), np.asarray(LEVELS, dtype=float)):
                raise ValueError(f"{path}: unexpected pressure levels")


def sidecar_path(path: Path) -> Path:
    return path.with_suffix(".json")


def verified(job: Job) -> bool:
    if not job.path.is_file() or not sidecar_path(job.path).is_file():
        return False
    try:
        record = json.loads(sidecar_path(job.path).read_text(encoding="utf-8"))
        if (record.get("dataset") != job.dataset or record.get("request") != job.request
                or record.get("size") != job.path.stat().st_size
                or record.get("sha256") != sha256(job.path)):
            return False
        validate_netcdf(job, job.path)
        return True
    except (OSError, ValueError, KeyError, RuntimeError):
        return False


def download(jobs: list[Job]) -> None:
    import cdsapi

    client = cdsapi.Client(timeout=120, retry_max=3, sleep_max=20, quiet=True, progress=False)
    for index, job in enumerate(jobs, 1):
        if verified(job):
            print(f"[{index}/{len(jobs)}] verified: {job.path}", flush=True)
            continue
        if job.path.exists():
            raise ValueError(f"existing file failed verification; inspect or move it: {job.path}")
        job.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = job.path.with_suffix(".nc.download")
        temporary.unlink(missing_ok=True)
        print(f"[{index}/{len(jobs)}] downloading: {job.path}", flush=True)
        try:
            client.retrieve(job.dataset, job.request, str(temporary))
            validate_netcdf(job, temporary)
            digest = sha256(temporary)
            size = temporary.stat().st_size
            temporary.replace(job.path)
            record = {"dataset": job.dataset, "request": job.request,
                      "size": size, "sha256": digest}
            sidecar_path(job.path).write_text(
                json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        finally:
            temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--output-dir", required=True, type=Path)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--download", action="store_true", help="contact CDS and download missing files")
    action.add_argument("--verify", action="store_true", help="check all downloaded files")
    args = parser.parse_args()
    initial_times = read_initial_times(args.cases)
    jobs = build_plan(initial_times, args.output_dir.expanduser().resolve())
    counts = {kind: sum(job.kind == kind for job in jobs)
              for kind in ("surface", "upper", "precipitation", "static")}
    print(f"Initial times: {len(initial_times)}; files: {counts}; root: {args.output_dir.resolve()}")
    if args.download:
        download(jobs)
    elif args.verify:
        failed = [str(job.path) for job in jobs if not verified(job)]
        if failed:
            raise SystemExit(f"{len(failed)} missing or invalid files; first: {failed[0]}")
        print("All files verified")
    else:
        for job in jobs[:5]:
            print(f"{job.kind}: {job.path} ({len(job.times)} times)")
        print("Use --download to start CDS requests, or --verify to check the archive.")


if __name__ == "__main__":
    main()
