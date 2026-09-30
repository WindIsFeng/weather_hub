"""Offline checks for the shared ERA5 request plan."""

from datetime import datetime, timezone

import hashlib
import json
import sys

import pytest
import requests
import cdsapi.api

from scripts import download_era5_inputs as downloader
from scripts.download_era5_inputs import DEFAULT_CASES, build_plan, read_initial_times


def test_frozen_case_plan_covers_all_required_input_times(tmp_path):
    initial = read_initial_times(DEFAULT_CASES)
    jobs = build_plan(initial, tmp_path)
    by_kind = {kind: [job for job in jobs if job.kind == kind]
               for kind in ("surface", "upper", "precipitation", "static")}
    assert len(initial) == 510
    assert {kind: len(values) for kind, values in by_kind.items()} == {
        "surface": 397, "upper": 397, "precipitation": 412, "static": 1,
    }
    assert len({time for job in by_kind["upper"] for time in job.times}) == 933
    assert len({time for job in by_kind["precipitation"] for time in job.times}) == 5598
    assert "relative_humidity" in by_kind["upper"][0].request["variable"]
    assert "vertical_velocity" in by_kind["upper"][0].request["variable"]


def test_cross_day_precipitation_has_exact_hourly_window(tmp_path):
    initial = {datetime(2022, 1, 1, 0, tzinfo=timezone.utc)}
    jobs = build_plan(initial, tmp_path)
    rain = [job for job in jobs if job.kind == "precipitation"]
    assert [job.path.name for job in rain] == ["20211231.nc", "20220101.nc"]
    assert [len(job.times) for job in rain] == [11, 1]
    assert rain[0].request["time"] == [f"{hour:02d}:00" for hour in range(13, 24)]
    assert rain[1].request["time"] == ["00:00"]


def test_completed_archive_download_needs_no_cds_or_cached_manifest(tmp_path, monkeypatch, capsys):
    archive = tmp_path / "archive"
    archive.mkdir()
    file = archive / "static.nc"
    file.write_bytes(b"already verified")
    file.with_suffix(".json").write_text("{}")
    state = tmp_path / "state"
    state.mkdir()
    (state / "manifest.json").write_text("old temporary metadata")
    job = downloader.Job("static", file, "reanalysis-era5-single-levels", {}, ())
    monkeypatch.setattr(downloader, "read_initial_times", lambda _: set())
    monkeypatch.setattr(downloader, "build_plan", lambda *_: [job])
    monkeypatch.setattr(downloader, "verified", lambda _: True)
    monkeypatch.setattr(downloader, "inventory", lambda *_: pytest.fail("CDS contacted"))
    monkeypatch.setattr(sys, "argv", ["download_era5_inputs.py", "--download",
                                  "--output-dir", str(archive), "--state-dir", str(state)])
    downloader.main()
    assert "nothing to download" in capsys.readouterr().out
    assert not state.exists()


def test_missing_cds_result_remains_eligible_for_request(tmp_path, monkeypatch):
    cases = tmp_path / "cases.csv"
    cases.write_text("case_id,init_time\nexample,2023-01-01T00:00:00Z\n")
    output = tmp_path / "archive"
    job = downloader.Job("static", output / "static.nc",
                         "reanalysis-era5-single-levels", {}, ())
    monkeypatch.setattr(cdsapi.api, "get_url_key_verify", lambda *args: (
        "https://cds.example.test/api", "test-token", True))
    monkeypatch.setattr(downloader, "api_json", lambda *args, **kwargs: {"jobs": []})
    manifest = downloader.inventory([job], cases, output, tmp_path / "state", 1)
    assert manifest["expected_files"] == 1
    assert manifest["missing"] == ["static.nc"]
    assert manifest["matched_files"] == 0


def multipart_checksum(payload, part_size):
    parts = [hashlib.md5(payload[i:i + part_size]).digest()
             for i in range(0, len(payload), part_size)]
    return hashlib.md5(b"".join(parts)).hexdigest(), len(parts)


def test_cds_multipart_checksum_accepts_original_and_rejects_corruption(tmp_path, monkeypatch):
    monkeypatch.setattr(downloader, "MULTIPART_SIZE", 4)
    payload = b"original ERA5 multipart bytes"
    path = tmp_path / "upper.nc.download"
    path.write_bytes(payload)
    composite, count = multipart_checksum(payload, 4)
    assert hashlib.md5(payload).hexdigest() != composite
    assert downloader.checksum_matches(path, composite.lstrip("0"))
    assert downloader.checksum_matches(path, f'"{composite}-{count}"')
    assert not downloader.checksum_matches(path, f"{composite}-{count + 1}")
    path.write_bytes(b"X" + payload[1:])
    assert not downloader.checksum_matches(path, composite)


def test_retained_complete_file_is_validated_and_reused_without_download(tmp_path, monkeypatch):
    monkeypatch.setattr(downloader, "MULTIPART_SIZE", 4)
    payload = b"good pressure-level response"
    archive = tmp_path / "upper.nc"
    retained = tmp_path / "upper.nc.download.bad-old-checker"
    retained.write_bytes(payload)
    duplicate = tmp_path / "upper.nc.download.bad-duplicate"
    duplicate.write_bytes(payload)
    partial = tmp_path / "upper.nc.download"
    partial.write_bytes(payload[:2])
    job = downloader.Job("upper", archive, "reanalysis-era5-pressure-levels", {}, ())
    entry = {"path": "upper.nc", "url": "https://example.test/upper.nc",
             "size": len(payload), "checksum": multipart_checksum(payload, 4)[0],
             "request_id": "completed-request"}
    checked = []
    monkeypatch.setattr(downloader, "validate_netcdf", lambda job, path: checked.append("grid/time/level"))
    monkeypatch.setattr(downloader, "validate_values", lambda job, path: checked.append("all values"))
    monkeypatch.setattr(downloader, "receive", lambda *args: pytest.fail("Good retained file downloaded again"))
    result = downloader.recover_one(job, entry, state=tmp_path)
    assert result["status"] == "recovered"
    assert archive.read_bytes() == payload
    assert checked == ["grid/time/level", "all values"]
    assert not partial.exists()
    assert not list(tmp_path.glob("*.nc.download*"))
    sidecar = json.loads(archive.with_suffix(".json").read_text())
    assert sidecar["sha256"] == hashlib.sha256(payload).hexdigest()
    assert sidecar["cds_checksum_verified"] and sidecar["values_verified"]


def test_failed_checksum_retries_are_bounded_without_accumulating_copies(tmp_path, monkeypatch):
    calls = []

    class Session:
        def get(self, url, **kwargs):
            calls.append(kwargs["headers"])
            return Response(200, {}, [b"wrong"])

    monkeypatch.setattr(downloader, "session", Session)
    monkeypatch.setattr(downloader.time, "sleep", lambda seconds: None)
    entry = {"url": "https://example.test/upper.nc", "path": "upper.nc",
             "size": 5, "checksum": hashlib.md5(b"right").hexdigest()}
    with pytest.raises(ValueError, match="source checksum failed twice"):
        downloader.receive(entry, tmp_path / "upper.nc.download")
    assert calls == [{}, {}]
    assert not list(tmp_path.glob("*.bad-*"))


def test_storage_usage_includes_retained_copies_and_disk_blocks(tmp_path):
    directory = tmp_path / "20220129"
    directory.mkdir()
    for name, payload in {"upper.nc": b"done", "surface.nc.download": b"partial",
                          "upper.nc.download.bad-1": b"retained",
                          "upper.json": b"{}"}.items():
        (directory / name).write_bytes(payload)
    usage = downloader.storage_usage(tmp_path)
    assert usage["archive_bytes"] == 4
    assert usage["partial_bytes"] == 7
    assert usage["retained_bytes"] == 8
    assert usage["other_bytes"] == 2
    actual_allocated = sum(path.stat().st_blocks * 512 for path in [tmp_path, *tmp_path.rglob("*")])
    assert usage["disk_bytes"] == actual_allocated
    assert usage["retained_files"] == 1


class Response:
    def __init__(self, status, headers, blocks):
        self.status_code, self.headers, self.blocks = status, headers, blocks

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def raise_for_status(self):
        pass

    def iter_content(self, **kwargs):
        for block in self.blocks:
            if isinstance(block, Exception):
                raise block
            yield block


@pytest.mark.parametrize("server_honors_range", [True, False])
def test_resumed_transfer_does_not_append_a_full_response(tmp_path, monkeypatch, server_honors_range):
    payload = b"complete ERA5 response"
    partial = tmp_path / "surface.nc.download"
    partial.write_bytes(payload[:5])
    calls = []

    class Session:
        def get(self, url, **kwargs):
            calls.append(kwargs["headers"])
            if server_honors_range:
                return Response(206, {"Content-Range": f"bytes 5-{len(payload)-1}/{len(payload)}"}, [payload[5:]])
            return Response(200, {}, [payload])

    monkeypatch.setattr(downloader, "session", Session)
    entry = {"url": "https://example.test/era5.nc", "path": "surface.nc",
             "size": len(payload), "checksum": hashlib.md5(payload).hexdigest()}
    downloader.receive(entry, partial)
    assert partial.read_bytes() == payload
    assert calls == [{"Range": "bytes=5-"}]


def test_interrupted_transfer_retains_and_verifies_received_bytes(tmp_path, monkeypatch):
    payload = b"ERA5 bytes across a connection interruption"
    partial = tmp_path / "upper.nc.download"
    calls = []

    class Session:
        def get(self, url, **kwargs):
            calls.append(kwargs["headers"])
            if len(calls) == 1:
                return Response(200, {}, [payload[:8], requests.exceptions.ChunkedEncodingError()])
            return Response(206, {"Content-Range": f"bytes 8-{len(payload)-1}/{len(payload)}"}, [payload[8:]])

    monkeypatch.setattr(downloader, "session", Session)
    monkeypatch.setattr(downloader.time, "sleep", lambda seconds: None)
    entry = {"url": "https://example.test/era5.nc", "path": "upper.nc",
             "size": len(payload), "checksum": hashlib.md5(payload).hexdigest()}
    downloader.receive(entry, partial)
    assert partial.read_bytes() == payload
    assert calls == [{}, {"Range": "bytes=8-"}]


def test_replacement_request_id_is_saved_and_reused(tmp_path, monkeypatch):
    calls = []

    class SubmissionResponse(Response):
        def json(self):
            return {"jobID": "replacement-request-id"}

    class Session:
        def post(self, url, **kwargs):
            calls.append((url, kwargs["json"]))
            return SubmissionResponse(201, {}, [])

    monkeypatch.setattr(downloader, "session", Session)
    monkeypatch.setattr(cdsapi.api, "get_url_key_verify", lambda *args: (
        "https://cds.example.test/api", "test-token", True))
    job = downloader.Job("static", tmp_path/"static.nc", "reanalysis-era5-single-levels",
                       {"time": ["00:00"]}, (datetime(2023, 1, 1, tzinfo=timezone.utc),))
    entry = {"path": "static.nc"}
    first = downloader.submit_replacement(job, entry, tmp_path)
    resumed = downloader.submit_replacement(job, entry, tmp_path)
    assert first == resumed == "replacement-request-id"
    assert len(calls) == 1
    assert calls[0][1] == {"inputs": job.request}


def test_cds_queue_rejection_waits_and_resubmits(tmp_path, monkeypatch):
    submissions, delays = [], []
    monkeypatch.setattr(cdsapi.api, "get_url_key_verify", lambda *args: (
        "https://cds.example.test/api", "test-token", True))
    monkeypatch.setattr(downloader.time, "sleep", delays.append)

    def submit(job, entry, state, force=False):
        submissions.append(force)
        return f"request-{len(submissions)}"

    def api(url, key, **kwargs):
        if url.endswith("request-1/results"):
            return {"traceback": "Number queued requests for this dataset is temporarily limited"}
        if url.endswith("request-1"):
            return {"status": "rejected"}
        if url.endswith("request-2/results"):
            return {"asset": {"value": {"href": "https://example.test/static.nc",
                    "file:size": 42, "file:checksum": "a"*32}}}
        return {"status": "successful"}

    monkeypatch.setattr(downloader, "submit_replacement", submit)
    monkeypatch.setattr(downloader, "api_json", api)
    job = downloader.Job("static", tmp_path/"static.nc", "reanalysis-era5-single-levels",
                       {"time": ["00:00"]}, (datetime(2023, 1, 1, tzinfo=timezone.utc),))
    entry = downloader.refresh_entry(job, {"path": "static.nc"}, tmp_path)
    assert submissions == [False, True]
    assert delays == [60]
    assert entry["request_id"] == "request-2"
    assert entry["request"] == job.request
