"""Offline checks for the shared ERA5 request plan."""

from datetime import datetime, timezone

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
