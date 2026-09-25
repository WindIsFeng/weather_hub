# Shared ERA5 input archive for the 2022–2024 cases

[`scripts/download_era5_inputs.py`](../scripts/download_era5_inputs.py) downloads
the initial fields needed by all five models from the frozen case CSV. It does
not run inference or download ERA5 fields at future forecast valid times for
verification. The script groups timestamps by UTC date, checks coordinates,
variables and times in each NetCDF response, and records a SHA-256 sidecar.
Rerunning `--download` verifies and skips complete files.

The plan contains 510 unique initialization times, 933 distinct 6-hourly input
frames, and 5,598 distinct hourly precipitation times for FuXi. It produces
397 surface files, 397 pressure-level files, 412 precipitation files, and one
static file. Each model reads only the variables it needs from this shared
superset. The static file is for Aurora; GraphCast's static fields are present
in the surface files.

## Download and transfer

Use Python with `cdsapi>=0.7.7`, `numpy`, `xarray`, and `netCDF4`. The local
`fuxi` Conda environment already contains them. Configure `~/.cdsapirc` and
accept the ERA5 single-level and pressure-level dataset terms in CDS before
starting downloads. The [CDS API instructions](https://cds.climate.copernicus.eu/how-to-api)
describe those steps.

```bash
cd /scratch/hufeng/ai_weather_models
conda run -n fuxi python scripts/download_era5_inputs.py

# Start or resume in a detached session. The process survives terminal closure.
setsid nohup /home/hufeng/miniconda3/envs/fuxi/bin/python -u \
  scripts/download_era5_inputs.py --download \
  >> /scratch/hufeng/ai_weather_models/era5-download.log 2>&1 < /dev/null &
printf '%s\n' "$!" > /scratch/hufeng/ai_weather_models/era5-download.pid

# From any terminal, inspect the latest progress or follow it live.
tail -n 20 /scratch/hufeng/ai_weather_models/era5-download.log
tail -f /scratch/hufeng/ai_weather_models/era5-download.log
ps -p "$(cat /scratch/hufeng/ai_weather_models/era5-download.pid)" -o pid=,stat=,etime=,cmd=

# Verify the complete archive before transfer.
conda run -n fuxi python scripts/download_era5_inputs.py \
  --verify
```

The default download directory is `/data/hufeng/ai_weather_models/`. The
directory must exist and be writable before starting `--download`. The
download creates each date directory as needed. The layout is:

```text
ai_weather_models/
├── 20220127/
│   ├── surface.nc
│   ├── surface.json
│   ├── upper.nc
│   └── upper.json
├── precipitation/
│   ├── 20220127.nc
│   └── 20220127.json
└── static.nc
```

Transfer the whole directory to the same absolute path on the inference host,
`/data/hufeng/ai_weather_models/`. The `.json` files carry request
details and checksums for transfer checks; the model readers use the `.nc`
files. Preserve the date directories and file names.

## Point the five model configurations to the archive

On the inference host, edit each model's `configs/default.yaml` (or a copied
configuration referenced by its Weather Hub registry). Use these settings in
addition to the existing model paths and options. The example assumes the
archive is at `/data/hufeng/ai_weather_models/`.

| Model | ERA5 path settings |
| --- | --- |
| Pangu, FengWu, GraphCast | `surface_file: /data/hufeng/ai_weather_models/{init:%Y%m%d}/surface.nc`<br>`upper_file: /data/hufeng/ai_weather_models/{init:%Y%m%d}/upper.nc` |
| FuXi | Same `surface_file` and `upper_file`, plus `precipitation_file: /data/hufeng/ai_weather_models/precipitation/{init:%Y%m%d}.nc` |
| Aurora | `static_file: /data/hufeng/ai_weather_models/static.nc`<br>`surface_file: /data/hufeng/ai_weather_models/{init:%Y%m%d}/surface.nc`<br>`atmospheric_file: /data/hufeng/ai_weather_models/{init:%Y%m%d}/upper.nc` |

Set `download_missing: false` for all five models. These are explicitly
configured input files, so model-managed cache sidecars are not required. A
missing date or time will fail instead of silently downloading during inference.
FuXi reads individual hourly `tp` records and sums the previous six hours for
each initial frame; ERA5 hourly reanalysis precipitation is accumulated over
the hour ending at its valid time ([ECMWF definition](https://confluence.ecmwf.int/pages/viewpage.action?pageId=197702797)).

The archive contains the ERA5 inputs for inference. The evaluation project's
ERA5 reference fields at forecast valid times need a separate download plan.
