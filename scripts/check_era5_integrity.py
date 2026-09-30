"""Read-only, offline ERA5 copy verification for Windows, Linux and macOS.

Python 3.10+; no third-party packages. Keep era5_integrity_manifest.json
beside this script. Default mode reads every required NetCDF file in full.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath


DEFAULT_MANIFEST = Path(__file__).with_name("era5_integrity_manifest.json")
# Pins the reference generated from the original download records.
MANIFEST_SHA256 = "2963bee69ce4e9e0172cb7662a76e28990acd68bd65ca1e308bb335d8452c48f"
BLOCK_SIZE = 8 * 1024 * 1024


def load_manifest(path: Path, expected_digest: str | None = None) -> dict:
    raw = path.read_bytes()
    if expected_digest and hashlib.sha256(raw).hexdigest() != expected_digest:
        raise ValueError("Reference manifest checksum mismatch; extract the original package again.")
    manifest = json.loads(raw)
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError("Unsupported reference manifest.")
    rows = manifest.get("files")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Reference manifest has no files.")
    seen = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Invalid reference entry.")
        name = row.get("path", "")
        if not isinstance(name, str) or not re.fullmatch(
            r"(?:[0-9]{8}/(?:surface|upper)|precipitation/[0-9]{8}|static)\.nc", name
        ):
            raise ValueError(f"Invalid ERA5 relative path: {name!r}")
        if name.casefold() in seen:
            raise ValueError(f"Duplicate reference path: {name}")
        seen.add(name.casefold())
        size = row.get("size")
        if type(size) is not int or size <= 0:
            raise ValueError(f"Invalid reference size: {name}")
        for field in ("sha256", "sidecar_sha256"):
            if not isinstance(row.get(field), str) or not re.fullmatch(r"[0-9a-f]{64}", row[field]):
                raise ValueError(f"Invalid {field}: {name}")
        kind = "static" if name == "static.nc" else (
            "precipitation" if name.startswith("precipitation/") else PurePosixPath(name).stem
        )
        if row.get("kind") != kind:
            raise ValueError(f"Invalid file kind: {name}")
    if manifest.get("expected_files") != len(rows):
        raise ValueError("Reference file count is inconsistent.")
    if manifest.get("total_bytes") != sum(row["size"] for row in rows):
        raise ValueError("Reference byte count is inconsistent.")
    return manifest


def archive_marker(path: Path) -> bool:
    if (path / "static.nc").is_file() or (path / "precipitation").is_dir():
        return True
    return any(
        re.fullmatch(r"[0-9]{8}", entry.name) and entry.is_dir()
        for entry in path.iterdir()
    )


def choose_root(path: Path) -> Path:
    path = path.expanduser().resolve()
    if not path.is_dir():
        raise ValueError(f"Data directory does not exist or is inaccessible: {path}")
    if archive_marker(path):
        return path
    candidates = [
        path / "era5_inputs", path / "ai_weather_models" / "era5_inputs",
        path / "data" / "hufeng" / "ai_weather_models" / "era5_inputs",
    ]
    found = [candidate for candidate in candidates if candidate.is_dir() and archive_marker(candidate)]
    if len(found) > 1:
        raise ValueError("Several ERA5 archives found; pass the exact data directory using --root.")
    # An empty/incomplete archive must be reported as missing, never as a pass.
    return found[0] if found else path


def file_digest(path: Path, on_block=None) -> str:
    digest = hashlib.sha256()
    before = path.stat()
    read_bytes = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(BLOCK_SIZE), b""):
            digest.update(block)
            read_bytes += len(block)
            if on_block:
                on_block(len(block))
    after = path.stat()
    if read_bytes != before.st_size or (before.st_size, before.st_mtime_ns) != (
        after.st_size, after.st_mtime_ns
    ):
        raise OSError("File changed while being checked; stop copying/downloading and run again.")
    return digest.hexdigest()


def inspect_archive(root: Path, manifest: dict, *, quick: bool = False, progress=None) -> dict:
    started = time.monotonic()
    report = {
        "schema_version": 1,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root), "mode": "sizes_only" if quick else "full_sha256",
        "expected_files": len(manifest["files"]), "expected_bytes": manifest["total_bytes"],
        "reference_checksum_source": manifest.get("checksum_source"),
        "reference_source_content_revalidated_at_export": manifest.get("source_content_revalidated_at_export"),
        "files": [], "issues": [], "interrupted": False,
    }

    def issue(severity: str, code: str, name: str, detail: str) -> None:
        report["issues"].append({"severity": severity, "code": code, "path": name, "detail": detail})
        if progress and severity == "ERROR":
            print(f"[{code}] {name}: {detail}", flush=True)

    try:
        for index, expected in enumerate(manifest["files"], 1):
            name = expected["path"]
            path = root.joinpath(*PurePosixPath(name).parts)
            result = {"path": name, "kind": expected["kind"], "expected_bytes": expected["size"]}
            report["files"].append(result)
            try:
                info = path.stat()
                if not path.is_file():
                    raise OSError("Expected a regular file.")
                result["actual_bytes"] = info.st_size
                if info.st_size != expected["size"]:
                    result["status"] = "SIZE_MISMATCH"
                    issue("ERROR", "SIZE_MISMATCH", name, f"expected {expected['size']}, found {info.st_size} bytes")
                elif quick:
                    result["status"] = "SIZE_OK"
                else:
                    actual = file_digest(path, lambda count: progress(index, name, count) if progress else None)
                    result["actual_sha256"] = actual
                    if actual != expected["sha256"]:
                        result["status"] = "HASH_MISMATCH"
                        issue("ERROR", "HASH_MISMATCH", name, "Content differs from the original download record.")
                    else:
                        result["status"] = "SHA256_OK"
            except FileNotFoundError:
                result["status"] = "MISSING"
                issue("ERROR", "MISSING", name, "Required data file is missing.")
            except OSError as exc:
                result["status"] = "READ_ERROR"
                issue("ERROR", "READ_ERROR", name, str(exc))

            # The independent manifest still verifies .nc files if .json files
            # were not copied. Existing sidecars must match their source bytes.
            sidecar = path.with_suffix(".json")
            sidecar_name = str(PurePosixPath(name).with_suffix(".json"))
            try:
                if file_digest(sidecar) != expected["sidecar_sha256"]:
                    issue("ERROR", "SIDECAR_MISMATCH", sidecar_name, "Metadata differs from the original sidecar.")
            except FileNotFoundError:
                issue("WARNING", "MISSING_SIDECAR", sidecar_name, "Data can still be verified using this reference manifest.")
            except OSError as exc:
                issue("ERROR", "SIDECAR_READ_ERROR", sidecar_name, str(exc))
            if progress:
                progress(index, name, 0)

        expected_names = {row["path"].casefold() for row in manifest["files"]}

        def walk_error(exc: OSError) -> None:
            issue("WARNING", "EXTRA_SCAN_ERROR", str(exc.filename), str(exc))

        for directory, subdirs, names in os.walk(root, followlinks=False, onerror=walk_error):
            subdirs[:] = [name for name in subdirs if name not in {"System Volume Information", "$RECYCLE.BIN"}]
            for name in names:
                relative = (Path(directory) / name).relative_to(root).as_posix()
                if name.lower().endswith(".nc") and relative.casefold() not in expected_names:
                    issue("WARNING", "EXTRA_NETCDF", relative, "Extra file outside the required input set; left untouched.")
                elif ".nc.download" in name.lower():
                    issue("WARNING", "DOWNLOAD_LEFTOVER", relative, "Download temporary file; left untouched.")
    except KeyboardInterrupt:
        report["interrupted"] = True

    counts = Counter(row.get("status", "INCOMPLETE") for row in report["files"])
    errors = sum(row["severity"] == "ERROR" for row in report["issues"])
    warnings = len(report["issues"]) - errors
    data_ok = counts["SHA256_OK"] == report["expected_files"] and not report["interrupted"]
    status = (
        "INCOMPLETE" if report["interrupted"] else "FAIL" if errors else
        "QUICK_CHECK_ONLY" if quick else "PASS_WITH_WARNINGS" if warnings else "PASS"
    )
    report.update({
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "status": status, "all_data_sha256_verified": data_ok,
        "counts": dict(counts), "error_count": errors, "warning_count": warnings,
        "issue_counts": dict(Counter(row["code"] for row in report["issues"])),
    })
    return report


def write_reports(report: dict, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    # UTF-8 BOM makes Chinese paths display correctly in Windows Excel.
    with (directory / "issues.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["severity", "code", "path", "detail"])
        writer.writeheader()
        writer.writerows(report["issues"])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="ERA5 directory, or drive root such as F:/")
    parser.add_argument("--quick", action="store_true", help="Sizes only: does NOT verify data contents.")
    parser.add_argument("--report-dir", type=Path, help="Default: a new folder under ~/era5_integrity_reports")
    args = parser.parse_args(argv)
    try:
        manifest = load_manifest(DEFAULT_MANIFEST, MANIFEST_SHA256)
        root = choose_root(args.root)
        report_dir = args.report_dir or (
            Path.home() / "era5_integrity_reports" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        )
        report_dir = report_dir.expanduser().resolve()
        # Check report writability before spending time reading the archive.
        report_dir.mkdir(parents=True, exist_ok=True)
        for name in ("report.json", "issues.csv"):
            path = report_dir / name
            if path.exists():
                raise ValueError(f"Report already exists; choose a new --report-dir: {path}")
        probe = report_dir / ".write_probe"
        with probe.open("x"):
            pass
        probe.unlink()
        print(f"Data: {root}\nExpected: {len(manifest['files'])} data files, {manifest['total_bytes'] / 1024**3:.2f} GiB")
        print(f"Mode: {'QUICK sizes only (contents unverified)' if args.quick else 'FULL SHA-256 (reads all data)'}")
        print(f"Reports: {report_dir}", flush=True)
        started = last = time.monotonic()
        bytes_read = 0

        def progress(index: int, name: str, count: int) -> None:
            nonlocal last, bytes_read
            bytes_read += count
            now = time.monotonic()
            if now - last >= 15:
                rate = bytes_read / max(now - started, 0.001) / 1024**2
                print(f"[{index}/{len(manifest['files'])}] {bytes_read / 1024**3:.2f} GiB read, {rate:.1f} MiB/s; {name}", flush=True)
                last = now

        report = inspect_archive(root, manifest, quick=args.quick, progress=progress)
        report["reference_manifest_sha256"] = MANIFEST_SHA256
        write_reports(report, report_dir)
        print(f"\nResult: {report['status']}\nData file counts: {json.dumps(report['counts'], sort_keys=True)}")
        print(f"Errors: {report['error_count']}; warnings: {report['warning_count']}")
        print(f"Report: {report_dir / 'report.json'}\nIssues: {report_dir / 'issues.csv'}", flush=True)
        if report["interrupted"]:
            return 130
        return 1 if report["error_count"] else 0
    except (OSError, ValueError) as exc:
        print(f"Unable to check: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
