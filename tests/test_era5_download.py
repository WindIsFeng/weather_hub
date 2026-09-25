"""Offline checks for the shared ERA5 request plan."""

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
import sys
import time

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


def test_download_reports_progress_and_resumes_without_redownloading(tmp_path, monkeypatch, capsys):
    class FakeCDSClient:
        calls = 0

        def __init__(self, **kwargs):
            pass

        def retrieve(self, dataset, request, target):
            self.__class__.calls += 1
            Path(target).write_bytes(b"test ERA5 response")
            time.sleep(0.1)

    monkeypatch.setitem(sys.modules, "cdsapi", SimpleNamespace(Client=FakeCDSClient))
    monkeypatch.setattr(downloader, "validate_netcdf", lambda job, path: None)
    monkeypatch.setattr(downloader, "PROGRESS_INTERVAL_SECONDS", 0.005)
    job = downloader.Job(
        "static", tmp_path / "static.nc", "reanalysis-era5-single-levels",
        {"time": ["00:00"]}, (datetime(2023, 1, 1, tzinfo=timezone.utc),),
    )

    downloader.download([job])
    output = capsys.readouterr().out
    assert "MiB received" in output
    assert "[1/1] complete" in output
    downloader.download([job])
    assert "[1/1] verified" in capsys.readouterr().out
    assert FakeCDSClient.calls == 1
