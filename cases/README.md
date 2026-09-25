# Coastal-impact 2022–2024 inference cases

[`weather_hub_cases_2022_2024_v2026-09-22.csv`](weather_hub_cases_2022_2024_v2026-09-22.csv)
is the ready-to-run input for all five Weather Hub models. It contains 570
forecast cases for 114 coastal-impact events. Initializations use the nearest
00/06/12/18 UTC cycle to each nominal lead; output is every six hours, and
`forecast_hours` reaches or passes the observed reference time.

[`weather_hub_case_index_2022_2024_v2026-09-22.csv`](weather_hub_case_index_2022_2024_v2026-09-22.csv)
maps each Weather Hub `case_id` to the observed event and records the actual
lead time. [`weather_hub_manifest_2022_2024_v2026-09-22.json`](weather_hub_manifest_2022_2024_v2026-09-22.json)
records hashes and provenance. These are byte-for-byte copies of the files in
`/scratch/hufeng/ai_weather_eval/data/handoffs/`; the manifest's source paths
are relative to that evaluation repository.

From the Weather Hub project root, validate a model batch with:

```bash
weather-hub run --model pangu \
  --cases cases/weather_hub_cases_2022_2024_v2026-09-22.csv \
  --dry-run
```

Use `fengwu`, `fuxi`, `graphcast`, or `aurora` for the other models. Remove
`--dry-run` to launch inference once the model environments, weights, and
input fields are ready.
