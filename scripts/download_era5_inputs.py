"""Plan, download, resume, and verify the shared ERA5 inputs for Weather Hub cases.

The frozen case CSV defines one shared NetCDF archive. Download matches existing
CDS requests by their exact inputs, submits only absent or expired requests,
resumes partial transfers, and checks source and local file integrity. Planning
and --verify work offline. Temporary CDS metadata lives under runs/era5-download.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
import re
import tempfile
import time
from collections import defaultdict
from concurrent.futures import (
    ProcessPoolExecutor, ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED,
)
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Lock, local
from urllib.parse import urljoin, urlsplit

import requests


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = ROOT / "cases/weather_hub_cases_2022_2024_v2026-09-22.csv"
DEFAULT_OUTPUT_DIR = ROOT / "era5_inputs"
PROGRESS_INTERVAL_SECONDS = 15
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


STATE_DIR = ROOT / "runs/era5-download"
# The CDS object store uploads large results in 50 MiB parts. Its metadata
# keeps only the hash portion of the multipart ETag, dropping the part count
# and sometimes leading zeroes. This is not the MD5 of the entire file.
MULTIPART_SIZE = 50 * 1024**2
_local = local()


def session() -> requests.Session:
    if not hasattr(_local, "session"):
        _local.session = requests.Session()
    return _local.session


def atomic_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def request_key(dataset: str, request: dict) -> str:
    normal = {}
    for key, value in request.items():
        if isinstance(value, list) and key not in {"grid", "area"}:
            value = sorted(value, key=str)
        normal[key] = value
    return json.dumps([dataset, normal], sort_keys=True)


def api_json(url: str, key: str, *, params: dict | None = None,
             allow_problem: bool = False) -> dict:
    for attempt in range(12):
        try:
            with session().get(url, headers={"PRIVATE-TOKEN": key}, params=params,
                               timeout=(8, 30)) as response:
                if allow_problem and response.status_code == 400:
                    return response.json()
                response.raise_for_status()
                return response.json()
        except requests.RequestException as exc:
            if exc.response is not None and exc.response.status_code in {400, 401, 403, 404}:
                raise RuntimeError(f"CDS metadata HTTP {exc.response.status_code}") from None
            if attempt == 11:
                raise RuntimeError(f"CDS metadata unavailable: {type(exc).__name__}") from None
            time.sleep(min(2 + attempt, 8))
    raise AssertionError("unreachable")


def inventory(jobs: list[Job], cases: Path, output: Path, state: Path,
              workers: int) -> dict:
    import cdsapi.api

    base, key, _ = cdsapi.api.get_url_key_verify(None, None, None)
    base = base.rstrip("/") + "/retrieve/v1/jobs"
    summaries = {}
    # The API caps totalCount at 1,000. Taking both ends includes the entire
    # 1,207-file batch even when that cap removes the pagination link.
    for order in ("-created", "created"):
        url = base
        params = {"limit": 1000, "sortby": order, "status": "successful"}
        seen_pages = set()
        while url and url not in seen_pages:
            seen_pages.add(url)
            data = api_json(url, key, params=params)
            records = data.get("jobs", [])
            for record in records:
                if record.get("processID") in {
                    "reanalysis-era5-single-levels", "reanalysis-era5-pressure-levels"
                }:
                    summaries[record["jobID"]] = record
            print(f"Inventory: {len(summaries)} completed ERA5 requests listed", flush=True)
            following = next((v["href"] for v in data.get("links", [])
                              if v.get("rel") == "next"), None)
            if following and urlsplit(following).netloc != urlsplit(base).netloc:
                raise ValueError("unexpected CDS pagination host")
            url, params = following, None

    state.mkdir(parents=True, exist_ok=True)
    atomic_json(state / "cds_jobs.json", {"jobs": list(summaries.values())})
    cache_path = state / "request_details.jsonl"
    details = {}
    if cache_path.exists():
        for line in cache_path.read_text().splitlines():
            try:
                record = json.loads(line)
                details[record["jobID"]] = record
            except (ValueError, KeyError):
                continue
    pending = [rid for rid in summaries if rid not in details]
    lock = Lock()
    with cache_path.open("a") as cache:
        def get_detail(rid: str) -> dict:
            record = api_json(f"{base}/{rid}", key, params={"request": True})
            with lock:
                cache.write(json.dumps(record) + "\n")
                cache.flush()
            return record

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(get_detail, rid): rid for rid in pending}
            for index, future in enumerate(as_completed(futures), 1):
                record = future.result()
                details[record["jobID"]] = record
                if index % 25 == 0 or index == len(futures):
                    print(f"Inventory: {index}/{len(futures)} request details matched", flush=True)

    lookup = {}
    for rid, summary in summaries.items():
        detail = details[rid]
        request = detail.get("metadata", {}).get("request", {}).get("ids", {})
        asset = summary.get("metadata", {}).get("results", {}).get("asset", {}).get("value", {})
        if not asset:
            continue
        fingerprint = request_key(detail["processID"], request)
        entry = {
            "request_id": rid, "created": detail.get("created"),
            "url": asset["href"], "size": int(asset["file:size"]),
            "checksum": asset.get("file:checksum"),
        }
        if fingerprint not in lookup or (entry["created"] or "") > (lookup[fingerprint]["created"] or ""):
            lookup[fingerprint] = entry
    files, missing = [], []
    for job in jobs:
        entry = lookup.get(request_key(job.dataset, job.request))
        if entry is None:
            missing.append(str(job.path.relative_to(output)))
            continue
        files.append({
            **entry, "path": str(job.path.relative_to(output)), "kind": job.kind,
            "dataset": job.dataset, "request": job.request,
        })
    result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "cases": str(cases), "cases_sha256": sha256(cases),
        "output_dir": str(output), "expected_files": len(jobs),
        "matched_files": len(files), "missing": missing, "files": files,
        "total_bytes": sum(v["size"] for v in files),
    }
    atomic_json(state / "manifest.json", result)
    print(f"Matched {len(files)}/{len(jobs)} files; {result['total_bytes'] / 2**30:.2f} GiB", flush=True)
    if missing:
        print(f"{len(missing)} CDS results absent or expired; they will be requested on download",
              flush=True)
    return result


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def checksum_verification(path: Path, checksum: str | None) -> str | None:
    if not checksum:
        raise ValueError("CDS did not supply a source checksum")
    value = checksum.removeprefix("md5:")
    match = re.fullmatch(r"([0-9a-fA-F]{1,32})(?:-(\d+))?", value.strip('"'))
    if not match:
        raise ValueError("unsupported CDS checksum")
    expected = match[1].lower().zfill(32)
    if checksum.startswith("md5:"):
        return "md5" if md5(path) == expected else None
    whole, composite, count = hashlib.md5(), hashlib.md5(), 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(MULTIPART_SIZE), b""):
            whole.update(block)
            composite.update(hashlib.md5(block).digest())
            count += 1
    if match[2] is None and whole.hexdigest() == expected:
        return "md5"
    if (count > 1 and composite.hexdigest() == expected
            and (match[2] is None or count == int(match[2]))):
        return f"s3-multipart-md5-{MULTIPART_SIZE}-bytes"
    return None


def checksum_matches(path: Path, checksum: str | None) -> bool:
    return checksum_verification(path, checksum) is not None


def receive(entry: dict, temporary: Path) -> str:
    expected = entry["size"]
    checksum_failures = 0
    last_error = None
    if temporary.exists() and temporary.stat().st_size > expected:
        raise ValueError(f"partial file exceeds CDS size: {temporary}")
    for attempt in range(100):
        offset = temporary.stat().st_size if temporary.exists() else 0
        if offset == expected:
            method = checksum_verification(temporary, entry.get("checksum"))
            if method:
                return method
            checksum_failures += 1
            if checksum_failures >= 2:
                raise ValueError(f"source checksum failed twice: {entry['path']}")
            # Retry using the same temporary path; do not accumulate bad copies.
            temporary.unlink()
            offset = 0
        try:
            headers = {"Range": f"bytes={offset}-"} if offset else {}
            with session().get(entry["url"], headers=headers, stream=True,
                               timeout=(8, 45)) as response:
                response.raise_for_status()
                if response.status_code == 206:
                    value = response.headers.get("Content-Range", "")
                    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", value)
                    if not match or int(match[1]) != offset or int(match[3]) != expected:
                        raise ValueError("CDS returned an inconsistent download byte range")
                    mode = "ab" if offset else "wb"
                elif response.status_code == 200:
                    # A server ignoring Range sends the complete file. Restart
                    # safely instead of appending it to the partial prefix.
                    mode = "wb"
                else:
                    raise ValueError(f"unexpected download status: {response.status_code}")
                with temporary.open(mode) as handle:
                    for block in response.iter_content(chunk_size=256 * 1024):
                        handle.write(block)
                    handle.flush()
                    os.fsync(handle.fileno())
            if temporary.stat().st_size > expected:
                raise ValueError("CDS response exceeds expected file size")
            if temporary.stat().st_size == expected:
                method = checksum_verification(temporary, entry.get("checksum"))
                if method:
                    return method
        except requests.RequestException as exc:
            if exc.response is not None and exc.response.status_code in {401, 403, 404, 410}:
                raise RuntimeError(f"CDS result unavailable: HTTP {exc.response.status_code}") from None
            last_error = type(exc).__name__
        time.sleep(min(2 + attempt, 10))
    raise RuntimeError(f"download retries exhausted for {entry['path']} "
                       f"(last error: {last_error})")


def validate_values(job: Job, path: Path) -> None:
    import netCDF4
    import numpy as np

    with netCDF4.Dataset(path) as ds:
        for name in SHORT_NAMES[job.kind]:
            variable = ds.variables[name]
            variable.set_var_chunk_cache(size=8 * 1024**2, nelems=1009, preemption=0.75)
            time_axis = next((i for i, dim in enumerate(variable.dimensions)
                              if dim in {"time", "valid_time"}), None)
            count = variable.shape[time_axis] if time_axis is not None else 1
            for index in range(count):
                selection = [slice(None)] * variable.ndim
                if time_axis is not None:
                    selection[time_axis] = index
                block = variable[tuple(selection)]
                if np.ma.getmaskarray(block).any() or not np.isfinite(np.asarray(block)).all():
                    raise ValueError(f"{job.path}: {name} contains missing/nonfinite values")


def replacement_cache(entry: dict, state: Path) -> Path:
    return state / "refreshed_requests" / (entry["path"].replace("/", "_") + ".json")


def submit_replacement(job: Job, entry: dict, state: Path, *, force: bool = False) -> str:
    cache = state / "refreshed_requests" / (entry["path"].replace("/", "_") + ".json")
    cached = json.loads(cache.read_text()) if cache.exists() else None
    if cached and (cached.get("dataset") != job.dataset or cached.get("request") != job.request):
        raise ValueError("cached replacement request differs from the case plan")
    if cached and not force:
        return cached["request_id"]
    import cdsapi.api

    base, key, _ = cdsapi.api.get_url_key_verify(None, None, None)
    base = base.rstrip("/") + "/retrieve/v1"
    started = datetime.now(timezone.utc).replace(tzinfo=None).isoformat()
    for attempt in range(12):
        try:
            with session().post(f"{base}/processes/{job.dataset}/execution",
                                headers={"PRIVATE-TOKEN": key}, json={"inputs": job.request},
                                timeout=(8, 45)) as response:
                response.raise_for_status()
                rid = response.json()["jobID"]
                break
        except requests.RequestException as exc:
            if exc.response is not None and exc.response.status_code in {400, 401, 403}:
                raise RuntimeError(f"CDS submission HTTP {exc.response.status_code}") from None
            # A timeout can occur after the server accepted the POST. Look for
            # that exact new request before retrying, to avoid duplicate jobs.
            recent = api_json(f"{base}/jobs", key, params={"limit": 100, "sortby": "-created"})
            rid = None
            for candidate in recent.get("jobs", []):
                if candidate.get("created", "") < started or candidate["processID"] != job.dataset:
                    continue
                detail = api_json(f"{base}/jobs/{candidate['jobID']}", key,
                                  params={"request": True})
                actual = detail.get("metadata", {}).get("request", {}).get("ids", {})
                if request_key(job.dataset, actual) == request_key(job.dataset, job.request):
                    rid = candidate["jobID"]
                    break
            if rid:
                break
            if attempt == 11:
                raise RuntimeError(f"CDS submission unavailable: {type(exc).__name__}") from None
            time.sleep(min(2 + attempt, 8))
    atomic_json(cache, {"request_id": rid, "dataset": job.dataset, "request": job.request})
    print(f"Re-requested expired result: {entry['path']} ({rid})", flush=True)
    return rid


def refresh_entry(job: Job, entry: dict, state: Path, *, force: bool = False) -> dict:
    # CDS temporarily restricts queued requests. Keep at most one replacement
    # pending across download workers and the background preparation process.
    state.mkdir(parents=True, exist_ok=True)
    with (state / "cds-replacements.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _refresh_entry(job, entry, state, force=force)


def _refresh_entry(job: Job, entry: dict, state: Path, *, force: bool = False) -> dict:
    import cdsapi.api

    base, key, _ = cdsapi.api.get_url_key_verify(None, None, None)
    base = base.rstrip("/") + "/retrieve/v1/jobs"
    rid = submit_replacement(job, entry, state, force=force)
    queue_retries = 0
    while True:
        remote = api_json(f"{base}/{rid}", key)
        if remote["status"] == "successful":
            try:
                result = api_json(f"{base}/{rid}/results", key)
                break
            except RuntimeError as exc:
                if str(exc) != "CDS metadata HTTP 404":
                    raise
                rid = submit_replacement(job, entry, state, force=True)
        elif remote["status"] == "rejected":
            problem = api_json(f"{base}/{rid}/results", key, allow_problem=True)
            reason = problem.get("traceback", "")
            if "queued requests" not in reason or "temporarily limited" not in reason:
                raise RuntimeError(f"CDS replacement rejected: {reason or problem.get('title')}")
            queue_retries += 1
            if queue_retries > 12:
                raise RuntimeError("CDS request queue remained full; resume later")
            print(f"CDS queue limited; retrying in 60s: {entry['path']}", flush=True)
            time.sleep(60)
            rid = submit_replacement(job, entry, state, force=True)
        elif remote["status"] in {"failed", "dismissed", "deleted"}:
            raise RuntimeError(f"CDS replacement request {rid} is {remote['status']}")
        else:
            time.sleep(15)
    asset = result["asset"]["value"]
    new = {
        **entry, "url": urljoin(f"{base}/{rid}/results", asset["href"]),
        "size": int(asset["file:size"]),
        "checksum": asset["file:checksum"], "request_id": rid,
        "dataset": job.dataset, "request": job.request, "kind": job.kind,
    }
    atomic_json(replacement_cache(entry, state), {
        "request_id": rid, "dataset": job.dataset, "request": job.request, "asset": new,
    })
    return new


def remove_superseded_downloads(job: Job) -> dict:
    """Remove download leftovers only after the archive file was validated."""
    partial = job.path.with_suffix(".nc.download")
    paths = [partial]
    for label in ("bad", "expired", "redundant"):
        paths.extend(job.path.parent.glob(job.path.name + f".download.{label}-*"))
    removed = {"files": 0, "bytes": 0}
    for path in paths:
        try:
            size = path.stat().st_size
            path.unlink()
        except FileNotFoundError:
            continue
        removed["files"] += 1
        removed["bytes"] += size
    return removed


def promote_file(job: Job, temporary: Path, entry: dict, method: str) -> dict:
    validate_netcdf(job, temporary)
    validate_values(job, temporary)
    record = {
        "dataset": job.dataset, "request": job.request,
        "size": temporary.stat().st_size, "sha256": sha256(temporary),
        "cds_request_id": entry["request_id"], "cds_checksum": entry.get("checksum"),
        "cds_checksum_verified": True, "cds_checksum_method": method,
        "values_verified": True,
    }
    # Write the sidecar before promoting the data, so interruption never leaves
    # a final NetCDF file without its verification record.
    atomic_json(sidecar_path(job.path), record)
    temporary.replace(job.path)
    remove_superseded_downloads(job)
    return {"path": entry["path"], "size": record["size"], "status": "complete", "entry": entry}


def recover_one(job: Job, entry: dict, refresh_expired: bool = False,
                state: Path = STATE_DIR) -> dict:
    if verified(job):
        validate_values(job, job.path)
        remove_superseded_downloads(job)
        if "url" not in entry:
            cache = replacement_cache(entry, state)
            if cache.exists():
                entry = json.loads(cache.read_text()).get("asset", entry)
        result = {"path": entry["path"], "size": job.path.stat().st_size, "status": "verified"}
        if "url" in entry:
            result["entry"] = entry
        return result
    job.path.parent.mkdir(parents=True, exist_ok=True)
    if job.path.exists():
        raise ValueError(f"existing archive file failed verification: {job.path}")
    temporary = job.path.with_suffix(".nc.download")
    if "url" not in entry:
        if not refresh_expired:
            raise ValueError(f"completed CDS result absent: {entry['path']}")
        entry = refresh_entry(job, entry, state)
    # Older runs mistook multipart ETags for whole-file MD5 and retained good
    # complete files as .bad copies. Recheck them against the correct checksum
    # and the full NetCDF validators before transferring any more bytes.
    for candidate in sorted(job.path.parent.glob(job.path.name + ".download.*")):
        if candidate.stat().st_size != entry["size"]:
            continue
        method = checksum_verification(candidate, entry.get("checksum"))
        if method:
            result = promote_file(job, candidate, entry, method)
            result["status"] = "recovered"
            return result
    try:
        method = receive(entry, temporary)
    except RuntimeError as exc:
        if not refresh_expired or not str(exc).startswith("CDS result unavailable:"):
            raise
        old = entry
        entry = refresh_entry(job, entry, state, force=True)
        if old.get("checksum") != entry.get("checksum") and temporary.exists():
            temporary.unlink()  # Bytes from the expired result cannot be reused.
        method = receive(entry, temporary)
    return promote_file(job, temporary, entry, method)


def storage_usage(output: Path) -> dict:
    usage = {
        "archive_files": 0, "archive_bytes": 0,
        "partial_files": 0, "partial_bytes": 0,
        "retained_files": 0, "retained_bytes": 0,
        "other_files": 0, "other_bytes": 0, "disk_bytes": 0,
    }
    # Include retained copies and metadata, plus allocated blocks for directories,
    # so disk_bytes measures the same space as du rather than transfer progress.
    for directory, _, names in os.walk(output):
        usage["disk_bytes"] += Path(directory).stat().st_blocks * 512
        for name in names:
            path = Path(directory) / name
            try:
                info = path.stat()
            except FileNotFoundError:
                continue  # A worker promoted this file during the scan.
            if name.endswith(".nc"):
                kind = "archive"
            elif name.endswith(".nc.download"):
                kind = "partial"
            elif ".nc.download." in name:
                kind = "retained"
            else:
                kind = "other"
            usage[kind + "_files"] += 1
            usage[kind + "_bytes"] += info.st_size
            usage["disk_bytes"] += info.st_blocks * 512
    return usage


def discard_completed_state(state: Path) -> None:
    """Remove only this downloader's temporary CDS metadata after success."""
    for name in ("cds_jobs.json", "request_details.jsonl", "manifest.json",
                 "status.json", "cds-replacements.lock"):
        (state / name).unlink(missing_ok=True)
    replacements = state / "refreshed_requests"
    if replacements.is_dir():
        for path in replacements.glob("*.json"):
            path.unlink()
        if not any(replacements.iterdir()):
            replacements.rmdir()
    if state.is_dir() and not any(state.iterdir()):
        state.rmdir()


def download(jobs: list[Job], manifest: dict, state: Path, workers: int,
             refresh_expired: bool = False) -> None:
    entries = {row["path"]: row for row in manifest["files"]}
    output = Path(manifest["output_dir"])
    started = time.monotonic()
    completed, failed = [], []
    status_path = state / "status.json"

    def report(running: bool) -> None:
        usage = storage_usage(output)
        saved = sum(v["size"] for v in completed)
        status = {
            **usage,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "pid": os.getpid(), "running": running, "elapsed_seconds": round(time.monotonic()-started),
            "expected_files": len(jobs), "completed_files": len(completed),
            "completed_bytes": saved,
            "expected_bytes": manifest["total_bytes"], "failed": failed,
            "output_dir": str(output), "workers": workers,
            "expired_results_remaining": len(manifest["missing"]),
        }
        atomic_json(status_path, status)
        print(f"Progress: {len(completed)}/{len(jobs)} verified; "
              f"{saved/2**30:.2f}/{manifest['total_bytes']/2**30:.2f} GiB verified; "
              f"disk {usage['disk_bytes']/2**30:.2f} GiB "
              f"(partial {usage['partial_bytes']/2**30:.2f}, "
              f"retained {usage['retained_bytes']/2**30:.2f}); "
              f"{len(failed)} failed", flush=True)

    # Separate processes keep NetCDF/HDF5 readers isolated during validation.
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(recover_one, job, entries.get(str(job.path.relative_to(output)),
                        {"path": str(job.path.relative_to(output))}), refresh_expired, state): job
            for job in sorted(jobs, key=lambda v: (v.kind != "static", str(v.path)))
        }
        report(True)
        while futures:
            ready, _ = wait(futures, timeout=15, return_when=FIRST_COMPLETED)
            for future in ready:
                job = futures.pop(future)
                path = str(job.path.relative_to(output))
                try:
                    result = future.result()
                    completed.append(result)
                    if "entry" in result and entries.get(path) != result["entry"]:
                        entries[path] = result["entry"]
                        manifest["files"] = list(entries.values())
                        manifest["matched_files"] = len(entries)
                        manifest["missing"] = [v for v in manifest["missing"] if v != path]
                        manifest["total_bytes"] = sum(v["size"] for v in entries.values())
                        atomic_json(state / "manifest.json", manifest)
                    print(f"[{len(completed)}/{len(jobs)}] {result['status']}: {path} "
                          f"({result['size']/2**20:.1f} MiB)", flush=True)
                except Exception as exc:
                    failed.append({"path": path, "error": f"{type(exc).__name__}: {exc}"})
                    print(f"FAILED: {path}: {failed[-1]['error']}", flush=True)
            report(bool(futures))
    report(False)
    if failed:
        raise SystemExit(f"{len(failed)} files failed; resume with the same command")
    print(f"All {len(jobs)} ERA5 inputs recovered and verified", flush=True)



def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--state-dir", type=Path, default=STATE_DIR)
    parser.add_argument("--workers", type=int, default=8)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--download", action="store_true",
                        help="resume the archive using matching CDS requests")
    action.add_argument("--verify", action="store_true",
                        help="check all files locally without contacting CDS")
    action.add_argument("--index", action="store_true",
                        help="refresh the inventory of completed CDS requests")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    cases, output, state = (p.expanduser().resolve()
                            for p in (args.cases, args.output_dir, args.state_dir))
    initial_times = read_initial_times(cases)
    jobs = build_plan(initial_times, output)
    counts = {kind: sum(job.kind == kind for job in jobs)
              for kind in ("surface", "upper", "precipitation", "static")}
    print(f"Initial times: {len(initial_times)}; files: {counts}; root: {output}", flush=True)

    if args.verify:
        failed = [str(job.path) for job in jobs if not verified(job)]
        if failed:
            raise SystemExit(f"{len(failed)} missing or invalid files; first: {failed[0]}")
        print("All files verified", flush=True)
        return
    if args.index:
        inventory(jobs, cases, output, state, args.workers)
        return
    if not args.download:
        for job in jobs[:5]:
            print(f"{job.kind}: {job.path} ({len(job.times)} times)")
        print("Use --download to resume from CDS, or --verify to check the archive.")
        return

    try:
        output.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryFile(dir=output):
            pass
    except OSError as exc:
        parser.error(f"output directory is not writable: {output}: {exc}")
    # A complete local archive needs no CDS inventory or network access.
    if all(job.path.is_file() and sidecar_path(job.path).is_file() for job in jobs):
        if all(verified(job) for job in jobs):
            discard_completed_state(state)
            print("All files already verified; nothing to download", flush=True)
            return
    manifest_path = state / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    if (manifest is None or manifest.get("cases_sha256") != sha256(cases)
            or manifest.get("output_dir") != str(output)
            or manifest.get("matched_files", 0) + len(manifest.get("missing", [])) != len(jobs)):
        manifest = inventory(jobs, cases, output, state, args.workers)
    download(jobs, manifest, state, args.workers, refresh_expired=True)
    discard_completed_state(state)


if __name__ == "__main__":
    main()
