# 在本地下载 2022–2024 年推理所需 ERA5

此包包含 570 个固定 case、下载脚本和独立 Python 环境定义。不包含 CDS 密钥、ERA5
数据或模型权重。脚本只下载推理初始场；后续用于评分的 ERA5 参考场不在本包内。

## 1. 准备环境

需要 Python 3.10 或更新版本。推荐 Conda：

```bash
conda env create -f environment.era5.yml
conda activate era5-inputs
```

不用 Conda 时，在你自己的 Python 环境中运行：

```bash
python -m pip install -r requirements.era5.txt
```

在本地按 [CDS API 说明](https://cds.climate.copernicus.eu/how-to-api)配置
`~/.cdsapirc`，并在 CDS 网站接受 ERA5 single levels 和 pressure levels
两个数据集的使用条款。不要把 `.cdsapirc` 放进本包或上传到服务器。

## 2. 下载与校验

在解压后的包目录运行：

```bash
# 只看计划，不联网。
python scripts/download_era5_inputs.py \
  --output-dir /data/hufeng/ai_weather_models/era5_inputs

# 开始或继续下载；自动匹配 CDS 请求并从断点续传。
python -u scripts/download_era5_inputs.py --download \
  --output-dir /data/hufeng/ai_weather_models/era5_inputs

# 所有文件下载完后进行完整校验。
python scripts/download_era5_inputs.py --verify \
  --output-dir /data/hufeng/ai_weather_models/era5_inputs
```

省略 `--output-dir` 时，默认输出在本包的 `era5_inputs/`。本项目实际数据在上面
指定的 `/data/hufeng/ai_weather_models/era5_inputs/`。
下载计划包括 397 个地面文件、397 个高空文件、412 个 FuXi 降水文件和一个
Aurora 静态文件，共 1207 个文件。预计最终占用约 160–220 GB；请预留至少
250 GB。脚本每 15 秒报告当前文件状态，并对网络断线重试。若中断或遇到错误，
重复 `--download`；已校验的文件会跳过，未完成的文件从断点续传。CDS 请求
元数据暂存在 `runs/era5-download/`，全部完成后自动清理。数据旁的同名 `.json`
是校验记录，需要与 `.nc` 一起保留。

## 3. 上传到推理服务器

将 `era5_inputs/` **里面的内容**放到服务器的
`/data/hufeng/ai_weather_models/era5_inputs/`，保留日期子目录、`precipitation/` 和
`static.nc`。例如 Linux/macOS 上：

```bash
rsync -aP era5_inputs/ USER@HOST:/data/hufeng/ai_weather_models/era5_inputs/
```

服务器上应能看到这样的路径：

```text
/data/hufeng/ai_weather_models/era5_inputs/20220127/surface.nc
/data/hufeng/ai_weather_models/era5_inputs/20220127/upper.nc
/data/hufeng/ai_weather_models/era5_inputs/precipitation/20220127.nc
/data/hufeng/ai_weather_models/era5_inputs/static.nc
```

在服务器的模型配置文件中，Pangu、FengWu、GraphCast 使用：

```yaml
download_missing: false
surface_file: /data/hufeng/ai_weather_models/era5_inputs/{init:%Y%m%d}/surface.nc
upper_file: /data/hufeng/ai_weather_models/era5_inputs/{init:%Y%m%d}/upper.nc
```

FuXi 在相同的地面/高空路径之外再加：

```yaml
precipitation_file: /data/hufeng/ai_weather_models/era5_inputs/precipitation/{init:%Y%m%d}.nc
```

Aurora 使用：

```yaml
download_missing: false
static_file: /data/hufeng/ai_weather_models/era5_inputs/static.nc
surface_file: /data/hufeng/ai_weather_models/era5_inputs/{init:%Y%m%d}/surface.nc
atmospheric_file: /data/hufeng/ai_weather_models/era5_inputs/{init:%Y%m%d}/upper.nc
```

这些显式路径允许五个模型共用一份 ERA5 数据，模型无需各存一份。
