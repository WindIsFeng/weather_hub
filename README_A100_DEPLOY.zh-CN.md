# 在新 GPU 机器上部署 Weather Hub 并完成全部 case 推理

适用版本：2026-09-30 检查的本项目及相邻五个模型仓库。下文从一台新的 Linux GPU 机器开始，完成环境安装、代码/权重/ERA5 迁移、真实推理验证、全量运行、失败恢复、结果验收与回传。

**先确认租赁方案：A100 40GB 尚未在本项目上验证能够跑完全部五个模型。** Pangu、FengWu、FuXi 可先在这张卡上验证；GraphCast operational 和 Aurora 0.25° 的当前项目说明建议至少 48GB，80GB 更稳妥。GraphCast 已有 24GB 卡真实推理 OOM 的记录；Aurora 当前仓库尚无真实 GPU 推理通过记录。若目标是用一台机器完成五个模型，建议选择 A100 80GB；如果已经决定使用 40GB，先按第 9 节实测，再启动全量任务。这里的 48GB 是项目部署建议，并非所有模型的统一硬件下限。

本文提供可执行步骤；没有在尚未租赁的新机器上实际执行安装或 GPU 推理，也不承诺完成时间或 40GB 显存一定够用。

## 1. 实验范围与机器配置

固定输入为 `cases/weather_hub_cases_2022_2024_v2026-09-22.csv`：

| 项目 | 数量/约定 |
|---|---|
| 沿海影响事件 | 114 |
| 每个模型的 case | 570 |
| 五个模型的 case 总数 | 2,850 |
| 不同起报时刻 | 510 |
| 不同起报时间、时长、输出间隔组合 | 565 |
| 预报时长 | 24、30、48、54、72、78、96、102、120、126 小时 |
| 输出间隔 | 6 小时，首个输出为起报后 +6h |
| ERA5 推理输入 | 1,207 个 `.nc`，171,098,308,501 字节，约 159.35GiB |

本次固定模型是 Pangu-Weather、FengWu-v1、FuXi、GraphCast_operational 0.25°/13 层和 Aurora 0.25° Pretrained。保持这些模型身份、全球网格及 case 清单，便于后续比较结果。

建议租赁 Linux x86_64、Ubuntu 22.04/24.04、完整独占的一张 GPU，至少 16 个 CPU 核、128GB 主机内存；256GB 内存可增加读场和 GraphCast 编译的余量。这是容量建议，仍需观察实际内存峰值。确认 `nvidia-smi` 显示完整 40GB/80GB 卡，检查是否分配了更小的 MIG 实例。

驱动建议使用 **575 或更新且支持 A100 的版本**，镜像支持 CUDA 12.8。项目固定 ONNX Runtime 1.22.0、JAX 0.6.2 和 PyTorch 2.8.0；让各环境使用自己的 pip CUDA 依赖。`nvidia-smi` 中的 CUDA Version 表示驱动支持能力，不等于各 Python 环境的运行库版本。CUDA 12.8 的原生驱动配套要求见 [NVIDIA 发布说明](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/)。

### 数据盘不能只按 ERA5 的 160GB 估算

每个模型共需输出 7,135 个六小时时效。按 `721 × 1440` 全球网格、float32，以及五个模型的 69/69/70/83/69 个变量层通道估算，**未压缩、不计复用的预测数组总量约 10.67TB（十进制）**；这不是实际压缩后大小。当前输出使用 NetCDF 无损压缩，实际空间必须由小批次测量。565 个独特组合说明整批中完全相同预报的复用比例有限，不能按 510 个起报时刻大幅缩减容量。

- 同一数据盘保留五个模型全部全球结果：可按 **12TB 级别可用空间**做保守规划，或先测量压缩率后配置容量。
- 较小数据盘：逐模型运行、完整回传、验证备份后再释放已完成模型的空间；先根据第 10 节实测确定每批能容纳多少 case。
- 系统与 Conda 环境另留约 100GB，ERA5 至少留 250GB。只租 500GB/1TB 数据盘然后直接跑五个全量批次，存在很高的磁盘耗尽风险。

本项目完成的是全球预报场推理。台风追踪、路径/强度误差、评分和预测有效时刻的 ERA5 验证场属于后续评估阶段；当前 ERA5 归档只含推理初始场。

## 2. 连接新机器并准备工作目录

本节及后续标注“新 GPU 机器”的命令都在租赁机器的 Bash 中执行。把服务器地址、用户、SSH 端口替换为平台提供的信息：

```bash
# 在你自己的电脑上执行。
ssh -p 22 USER@GPU_HOST
```

```bash
# 在新 GPU 机器执行。
uname -m
nvidia-smi
nvidia-smi -L
free -h
df -h
```

确保 GPU 可见、架构为 `x86_64`，并选择一个**可写且挂载在持久数据盘上**的工作目录。以下用 `/data/weather-exp` 举例；按平台实际挂载点修改。若使用 `$HOME/weather-exp`，先确认它的磁盘容量和停机保留规则。

```bash
export WEATHER_WORKDIR=/data/weather-exp
mkdir -p "$WEATHER_WORKDIR"/{src,era5_inputs,runs,metadata,logs}
test -w "$WEATHER_WORKDIR"
df -h "$WEATHER_WORKDIR"
```

安装系统工具。以有 sudo 权限的 Ubuntu 为例；root 账户可去掉 sudo，工具已存在则跳过：

```bash
sudo apt-get update
sudo apt-get install -y git rsync curl ca-certificates tmux unzip libgomp1
```

本文使用如下结构，六个代码仓库必须同级：

```text
/data/weather-exp/
├── src/
│   ├── ai_weather_models/       # Weather Hub 控制器
│   ├── pangu-weather/
│   ├── fengwu/
│   ├── fuxi/
│   ├── graphcast/
│   └── aurora/
├── era5_inputs/                 # 一份共享输入
├── runs/                       # 全部任务、日志和预测结果
├── metadata/                   # 版本、验收报告和 job 映射
└── logs/
```

## 3. 安装或复用 Conda

镜像已经有 Conda 时，先运行 `conda --version`，复用该安装。没有时，在新 GPU 机器安装 Miniconda：

```bash
curl -fL --retry 5 \
  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
  -o /tmp/weather-miniconda.sh
bash /tmp/weather-miniconda.sh -b -p "$HOME/miniconda3"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
```

安装路径已存在时不要重复执行安装命令。安装方式见 [Conda Linux 安装说明](https://docs.conda.io/projects/conda/en/stable/user-guide/install/linux.html)。

```bash
export WEATHER_CONDA_BASE="$(conda info --base)"
source "$WEATHER_CONDA_BASE/etc/profile.d/conda.sh"
conda --version
```

保存后续终端和 tmux 都能复用的配置：

```bash
cat > "$WEATHER_WORKDIR/env.sh" <<EOF
export WEATHER_WORKDIR="$WEATHER_WORKDIR"
export WEATHER_CONDA_BASE="$WEATHER_CONDA_BASE"
export WEATHER_INPUTS="$WEATHER_WORKDIR/era5_inputs"
export WEATHER_HUB_CONFIG="$WEATHER_WORKDIR/src/ai_weather_models/config/models.a100.yaml"
source "$WEATHER_CONDA_BASE/etc/profile.d/conda.sh"
EOF
source "$WEATHER_WORKDIR/env.sh"
```

以后重新连接时先执行 `source /data/weather-exp/env.sh`，路径与自己的工作目录一致。

## 4. 迁移六个项目代码

推荐从当前已有机器上传，保留本地改造版本。仅下载官方原始模型仓库不能替代这里的五个仓库：Weather Hub 依赖其中的 `pangu_weather`、`fengwu_weather` 等模块。

以下在**当前存有 `/scratch/hufeng/` 项目的旧机器**执行。`WEATHER_REMOTE` 支持 `USER@HOST` 或已配置的 SSH alias；非 22 端口修改 `WEATHER_SSH`：

```bash
export WEATHER_REMOTE=USER@GPU_HOST
export WEATHER_REMOTE_ROOT=/data/weather-exp
export WEATHER_SSH='ssh -p 22'

for task_repo in ai_weather_models pangu-weather fengwu fuxi graphcast aurora; do
  rsync -aP -e "$WEATHER_SSH" \
    --exclude='__pycache__/' --exclude='.pytest_cache/' \
    --exclude='*.egg-info/' --exclude='.venv/' \
    --exclude='data/' --exclude='outputs/' --exclude='runs/' \
    --exclude='exports/' --exclude='era5_inputs/' \
    --exclude='models/' --exclude='checkpoints/' \
    --exclude='era5-download.log' --exclude='era5-download.pid' \
    "/scratch/hufeng/$task_repo/" \
    "$WEATHER_REMOTE:$WEATHER_REMOTE_ROOT/src/$task_repo/" || break
done
```

这会保留 `.git`、当前源码和 case 文件，不复制旧实验输出。检查 rsync 退出状态；任一失败时修复连接并重复执行，六个目录全部完成后继续。SSH 私钥和 CDS 密钥都不需要放进项目。

若无法从旧机器复制，可从自己的 GitHub 仓库 clone，再核对下表版本。GitHub 仓库是否公开、这些提交是否已推送，需要用自己的账号验证；不能默认匿名访问成功：

```bash
# 新 GPU 机器；与 rsync 方式二选一。
cd "$WEATHER_WORKDIR/src"
git clone https://github.com/WindIsFeng/weather_hub.git ai_weather_models
git clone https://github.com/WindIsFeng/Pangu-Weather.git pangu-weather
git clone https://github.com/WindIsFeng/Fengwu.git fengwu
git clone https://github.com/WindIsFeng/Fuxi.git fuxi
git clone https://github.com/WindIsFeng/GraphCast.git graphcast
git clone https://github.com/WindIsFeng/Aurora.git aurora
```

当前检查时的代码基线如下；本部署文档自身是控制器基线之上的新增文件。若使用更新版本，保存真实版本并重新核对配置和入口，不要仅为匹配表格丢弃修改：

| 仓库 | 检查时 HEAD |
|---|---|
| ai_weather_models | `85a4f294ab35c7064a56352f6f6571945e72ff73` |
| pangu-weather | `f65500904002079ba902d3977f8a93d45456a080` |
| fengwu | `73139fca2b2c02322d84264e2e3dd43294b89e23` |
| fuxi | `4d9e940e3ffadcab9dd7e3b83d6ce766df8017e7` |
| graphcast | `34c53fc96cab4c25c990e0f4141bcbab25c26d20` |
| aurora | `f39d8f959c21ed3951a92a077fe28e3d0595a0dc` |

## 5. 创建六个独立 Python 环境

在新 GPU 机器执行。控制器只有编排依赖，各模型继续使用独立环境。不要把 ONNX Runtime、JAX 和 Aurora 合并安装到同一个环境。

```bash
source "$WEATHER_WORKDIR/env.sh"
cd "$WEATHER_WORKDIR/src/ai_weather_models"
conda create -n weather-hub --override-channels -c conda-forge \
  python=3.11 pip pyyaml pytest -y
conda run -n weather-hub python -m pip install --no-deps -e .

conda env create -f "$WEATHER_WORKDIR/src/pangu-weather/environment.yml"
conda env create -f "$WEATHER_WORKDIR/src/fengwu/environment.yml"
conda env create -f "$WEATHER_WORKDIR/src/fuxi/environment.yml"
conda env create -f "$WEATHER_WORKDIR/src/graphcast/environment.yml"
conda env create -f "$WEATHER_WORKDIR/src/aurora/aurora.yml"
conda env list
```

预期环境名称：`weather-hub`、`pangu`、`fengwu`、`fuxi`、`graphcast`、`aurora`。环境创建失败时先处理该条命令，避免继续安装后误认为全部就绪。镜像有同名环境时核对版本；需要更新可使用 `conda env update -n ENV -f FILE`。保持仓库固定的主包版本，安装后运行各环境的 `python -m pip check`。

Pangu/FengWu/FuXi 的环境已经指定 `onnxruntime-gpu[cuda,cudnn]==1.22.0`，其代码会预加载 NVIDIA site-packages 中的运行库。不要额外安装 CPU 版 `onnxruntime`。这一安装方式见 [ONNX Runtime CUDA 说明](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#preload-dlls)。

Aurora 若安装到 CPU 版 PyTorch 或 `torch.version.cuda` 为 `None`，按 [PyTorch 2.8.0 官方安装矩阵](https://pytorch.org/get-started/previous-versions/#v280)安装固定版本的 CUDA wheel：

```bash
conda run -n aurora python -m pip install --force-reinstall \
  'torch==2.8.0' --index-url https://download.pytorch.org/whl/cu128
conda run -n aurora python -m pip check
```

## 6. 迁移模型权重

Git 不包含权重。现有机器已经有 Pangu、FengWu、FuXi、GraphCast 的资产，优先复制它们。在**旧机器**执行：

```bash
for task_repo in pangu-weather fengwu fuxi graphcast; do
  rsync -aP -e "$WEATHER_SSH" \
    "/scratch/hufeng/$task_repo/models/" \
    "$WEATHER_REMOTE:$WEATHER_REMOTE_ROOT/src/$task_repo/models/" || break
done
```

**FuXi 必须连同无扩展名的 external-data 权重文件一起复制。** 只复制 `.onnx` 会加载失败。FengWu 的现有目录可能还含 v2，本实验配置使用 v1。

| 模型 | 必需资产，相对于模型仓库根目录 |
|---|---|
| Pangu | `models/pangu_weather_{1,3,6,24}.onnx` |
| FengWu | `models/fengwu_v1.onnx`、`models/data_mean.npy`、`models/data_std.npy` |
| FuXi | `models/short.onnx` + `models/short`；`medium.onnx` + `medium`；`long.onnx` + `long` |
| GraphCast | `models/params/GraphCast_operational - ERA5-HRES 1979-2021 - resolution 0.25 - pressure levels 13 - mesh 2to6 - precipitation output only.npz`，以及 `models/stats/` 下三个统计文件 |
| Aurora | `checkpoints/aurora-0.25-pretrained.ckpt` |

GraphCast 三个统计文件是 `diffs_stddev_by_level.nc`、`mean_by_level.nc`、`stddev_by_level.nc`。新 GPU 机器下载 Aurora：

```bash
cd "$WEATHER_WORKDIR/src/aurora"
conda run --no-capture-output -n aurora python -m aurora_weather \
  download-model --config configs/default.yaml
```

该命令使用配置中固定的 Hugging Face revision。当前旧机器 Aurora checkpoints 目录为空，不能靠 rsync 获得其权重。

没有现成权重时，从 [Pangu 官方实现](https://github.com/198808xc/Pangu-Weather)、[FengWu 官方下载说明](https://github.com/OpenEarthLab/FengWu)、[FuXi 官方实现](https://github.com/tpys/FuXi)、[GraphCast 官方模型桶](https://console.cloud.google.com/storage/browser/dm_graphcast/graphcast)和 [Aurora checkpoint](https://huggingface.co/microsoft/aurora/blob/main/aurora-0.25-pretrained.ckpt)获取相应资产，再按表放置。FengWu 官方明确区分 ERA5 用 v1、业务分析用 v2；GraphCast 这里使用 operational 13 层版本。

为复制的权重记录 SHA-256 并在目的端比对。在旧机器生成校验文件后一起传输；以下仅针对权重及统计目录：

```bash
# 旧机器执行。完整读取十余 GB 权重，耗时取决于磁盘。
for task_repo in pangu-weather fengwu fuxi graphcast; do
  (
    cd "/scratch/hufeng/$task_repo"
    find models -type f -print0 | sort -z | xargs -0 sha256sum
  ) > "/tmp/weather-$task_repo-weights.sha256" || break
  rsync -aP -e "$WEATHER_SSH" "/tmp/weather-$task_repo-weights.sha256" \
    "$WEATHER_REMOTE:$WEATHER_REMOTE_ROOT/metadata/" || break
done
```

```bash
# 新 GPU 机器执行；四个模型的全部文件均应显示 OK。
for task_repo in pangu-weather fengwu fuxi graphcast; do
  (
    cd "$WEATHER_WORKDIR/src/$task_repo"
    sha256sum -c "$WEATHER_WORKDIR/metadata/weather-$task_repo-weights.sha256"
  ) || break
done
```

## 7. 上传并完整校验共享 ERA5

已有完整归档时，不必在租赁 GPU 上重新向 CDS 下载。在**旧机器**执行：

```bash
rsync -aP -e "$WEATHER_SSH" \
  /data/hufeng/ai_weather_models/era5_inputs/ \
  "$WEATHER_REMOTE:$WEATHER_REMOTE_ROOT/era5_inputs/"
```

若数据在移动硬盘，将源路径替换为硬盘上的 `era5_inputs/`。源路径结尾的 `/` 表示复制其内容；目标目录下面应该直接出现日期目录、`precipitation/`、`static.nc` 和 `static.json`。保留全部 `.json` 校验记录。

在新 GPU 机器先快速检查目录，再做一次完整 SHA-256 验证：

```bash
cd "$WEATHER_WORKDIR/src/ai_weather_models"
conda run --no-capture-output -n weather-hub python scripts/check_era5_integrity.py \
  --root "$WEATHER_INPUTS" --quick \
  --report-dir "$WEATHER_WORKDIR/metadata/era5-quick"

conda run --no-capture-output -n weather-hub python scripts/check_era5_integrity.py \
  --root "$WEATHER_INPUTS" \
  --report-dir "$WEATHER_WORKDIR/metadata/era5-full"
```

完整验证会读取约 159.35GiB。检查 `era5-full/report.json`：所有 1,207 个数据文件通过、`all_data_sha256_verified: true`，并检查 `issues.csv` 中是否有警告。`QUICK_CHECK_ONLY` 只证明文件大小符合预期。已有同名报告时换一个新的 `--report-dir`。清单采用原下载记录的 SHA-256，验证的是复制结果与该记录一致，不能单凭它声称已评估气象数据的科学准确性。

如果没有完整归档，可以在普通 CPU 机器按 [本地下载说明](README_ERA5_LOCAL.md)下载，再上传。新机器直接下载也可使用 `environment.era5.yml` 和 `scripts/download_era5_inputs.py --download --output-dir "$WEATHER_INPUTS"`，先配置 CDS 账号和数据集授权；不要让付费 GPU 长时间等待 CDS 排队。

## 8. 为新机器生成专用配置

本节生成五个 `configs/a100.yaml` 和控制器 `config/models.a100.yaml`，保留原默认配置。ERA5 使用显式路径，`download_missing: false`；断网不会触发隐式补下载。把输入设置写入实际 base config，使 `doctor` 检查的配置和推理一致。

```bash
cd "$WEATHER_WORKDIR/src/ai_weather_models"
conda run --no-capture-output -n weather-hub python - <<'PY'
import os
from pathlib import Path
import yaml
from weather_hub.adapters import SPECS
from weather_hub.types import ModelId

root = Path(os.environ['WEATHER_WORKDIR']).resolve()
inputs = Path(os.environ['WEATHER_INPUTS']).resolve()
hub = root / 'src' / 'ai_weather_models'
registry = yaml.safe_load((hub / 'config/models.yaml').read_text())
registry['controller_jobs_dir'] = str(root / 'runs')
target = registry['targets']['local']
target['jobs_dir'] = str(root / 'runs')
target['conda_executable'] = str(Path(os.environ['WEATHER_CONDA_BASE']) / 'bin/conda')
repos = {'pangu': 'pangu-weather', 'fengwu': 'fengwu', 'fuxi': 'fuxi',
         'graphcast': 'graphcast', 'aurora': 'aurora'}
for name, repo in repos.items():
    project = root / 'src' / repo
    source = project / 'configs/default.yaml'
    config = yaml.safe_load(source.read_text())
    for key in SPECS[ModelId(name)].path_keys:
        if config.get(key):
            path = Path(config[key]).expanduser()
            config[key] = str(path if path.is_absolute() else (source.parent / path).resolve())
    config.update(device='cuda', device_id=0, download_missing=False,
                  data_dir=str(root / 'unused-model-cache' / name),
                  surface_file=str(inputs / '{init:%Y%m%d}' / 'surface.nc'))
    if name == 'aurora':
        config.update(autocast=True, static_file=str(inputs / 'static.nc'),
                      atmospheric_file=str(inputs / '{init:%Y%m%d}' / 'upper.nc'))
    else:
        config.update(threads=1, upper_file=str(inputs / '{init:%Y%m%d}' / 'upper.nc'))
    if name == 'fuxi':
        config['precipitation_file'] = str(inputs / 'precipitation' / '{init:%Y%m%d}.nc')
    if name in ('pangu', 'fuxi'):
        config['max_sessions'] = 1
    destination = project / 'configs/a100.yaml'
    destination.write_text(yaml.safe_dump(config, sort_keys=False))
    installation = target['models'][name]
    installation.update(project_dir=str(project), base_config=str(destination), device_id=0)
    installation.pop('overrides', None)
destination = hub / 'config/models.a100.yaml'
destination.write_text(yaml.safe_dump(registry, sort_keys=False))
print(destination)
PY

conda activate weather-hub
weather-hub models
weather-hub doctor --model all --json > "$WEATHER_WORKDIR/metadata/doctor.json"
weather-hub doctor --model all --load-model --json \
  > "$WEATHER_WORKDIR/metadata/doctor-load.json"
```

确认命令退出码为 0、所有报告 `available: true`。`models` 的 ready 只表示仓库、配置和环境存在；`doctor --load-model` 也不能证明前向推理显存充足。

额外检查本批次会实际使用的 Pangu 24h 和 FuXi medium 模型：

```bash
cd "$WEATHER_WORKDIR/src/pangu-weather"
conda run --no-capture-output -n pangu python -m pangu_weather doctor \
  --config configs/a100.yaml --load-model 24
cd "$WEATHER_WORKDIR/src/fuxi"
conda run --no-capture-output -n fuxi python -m fuxi_weather doctor \
  --config configs/a100.yaml --load-model medium
cd "$WEATHER_WORKDIR/src/ai_weather_models"
```

固定批次最长 126h，FuXi 在 +126h 从 short 切到 medium；本批次不调用 long。仍保留完整权重目录，因为 FuXi doctor 会检查全部三级资产。

## 9. 上机验证：短时、最长时长、小批次

### 9.1 保存设备和版本记录

```bash
nvidia-smi > "$WEATHER_WORKDIR/metadata/nvidia-smi.txt"
uname -a > "$WEATHER_WORKDIR/metadata/uname.txt"
for task_repo in ai_weather_models pangu-weather fengwu fuxi graphcast aurora; do
  git -C "$WEATHER_WORKDIR/src/$task_repo" rev-parse HEAD \
    > "$WEATHER_WORKDIR/metadata/$task_repo.commit.txt"
  git -C "$WEATHER_WORKDIR/src/$task_repo" diff \
    > "$WEATHER_WORKDIR/metadata/$task_repo.diff.txt"
done
for task_env in weather-hub pangu fengwu fuxi graphcast aurora; do
  conda env export -n "$task_env" > "$WEATHER_WORKDIR/metadata/$task_env.environment.yaml"
  conda run -n "$task_env" python -m pip freeze \
    > "$WEATHER_WORKDIR/metadata/$task_env.pip.txt"
done
```

`git diff` 不包含未跟踪文件；回传时还要保留本文生成的实际 YAML、脚本和 `env.sh`。

### 9.2 检查冻结清单并生成验证输入

```bash
cd "$WEATHER_WORKDIR/src/ai_weather_models"
conda run --no-capture-output -n weather-hub python - <<'PY'
import csv, hashlib, json, os
from pathlib import Path

cases = Path('cases')
manifest = json.loads((cases / 'weather_hub_manifest_2022_2024_v2026-09-22.json').read_text())
for filename, record in manifest['artifacts'].items():
    assert hashlib.sha256((cases / filename).read_bytes()).hexdigest() == record['sha256'], filename
source = cases / 'weather_hub_cases_2022_2024_v2026-09-22.csv'
with source.open(newline='') as handle:
    reader = csv.DictReader(handle)
    fields, rows = reader.fieldnames, list(reader)
assert len(rows) == 570 and len({r['case_id'] for r in rows}) == 570
out = Path(os.environ['WEATHER_WORKDIR']) / 'metadata'
short = dict(rows[0], case_id=rows[0]['case_id'] + '-SMOKE6', forecast_hours='6')
long = max(rows, key=lambda r: int(r['forecast_hours']))
assert int(long['forecast_hours']) == 126
for filename, selected in [('smoke6.csv', [short]), ('long126.csv', [long]),
                            ('pilot10.csv', rows[:10])]:
    with (out / filename).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(selected)
print('Frozen cases verified; smoke6, long126, pilot10 created.')
PY

for task_model in pangu fengwu fuxi graphcast aurora; do
  weather-hub run --model "$task_model" \
    --cases cases/weather_hub_cases_2022_2024_v2026-09-22.csv --dry-run \
    > "$WEATHER_WORKDIR/logs/$task_model-dry-run.log" 2>&1 || break
done
```

五个 dry-run 均应成功。控制器 dry-run 会创建任务目录和计划状态；不进行预测，结果里的 `planned` 不能计为已完成。

### 9.3 在独立 tmux 会话执行真实验证

```bash
tmux new -s weather
source /data/weather-exp/env.sh
conda activate weather-hub
cd "$WEATHER_WORKDIR/src/ai_weather_models"
```

把上面的绝对路径改成实际目录。先逐模型跑 6h，再逐模型跑 126h。每条命令等待结束并检查 `succeeded`，失败时暂停该模型的全量启动：

```bash
for task_model in pangu fengwu fuxi graphcast aurora; do
  weather-hub run --model "$task_model" \
    --cases "$WEATHER_WORKDIR/metadata/smoke6.csv" --json \
    > "$WEATHER_WORKDIR/metadata/$task_model-smoke6-job.json" || break
done

for task_model in pangu fengwu fuxi graphcast aurora; do
  weather-hub run --model "$task_model" \
    --cases "$WEATHER_WORKDIR/metadata/long126.csv" --json \
    > "$WEATHER_WORKDIR/metadata/$task_model-long126-job.json" || break
done
```

这些 JSON 内含 `job_id`、`status`、`job_dir`。只有五个模型两类验证都通过，才进入五模型全量实验。6h 验证检查完整前向；126h 验证进一步检查反馈、持续输出、Pangu 24/6h 调度、FuXi short→medium 切换和磁盘写入。

在另一个终端观察设备、CPU 内存和日志。也可以持续记录 GPU 采样：

```bash
source /data/weather-exp/env.sh
nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu \
  --format=csv -l 2 > "$WEATHER_WORKDIR/logs/gpu-monitor.csv"
```

该命令持续运行，结束监控用 Ctrl+C；采样峰值可能漏掉短暂分配峰值。`Ctrl+B` 再按 `D` 离开 tmux，重新连接后用 `tmux attach -t weather`。tmux 和 detached worker 能承受 SSH 断线，不能承受机器关机、重启或实例释放。

### A100 40GB 上 GraphCast/Aurora 失败时

先确认 GPU 没有其他推理任务、完整显存可用。GraphCast 当前实现已经关闭 JAX 预分配，并使用 bfloat16 包装；Aurora 已启用 `autocast: true` 并逐时输出至 CPU。关闭预分配可能降低占用，也可能增加碎片，见 [JAX 显存说明](https://docs.jax.dev/en/latest/201/gpu-memory.html)。这些设置不能把任意模型压进 40GB。

若真实前向仍 OOM，保留错误日志，改用更大显存机器运行该模型。Pangu/FengWu/FuXi 可在通过验证后先完成各自 570 个 case，GraphCast/Aurora 在 80GB 机器使用同一冻结清单补齐。当前调度器没有把单个模型分布到多卡的功能；租两张 40GB 卡不会自动合并成单模型可用的 80GB。

## 10. 小批次估算租期与输出空间

对通过第 9 节的模型顺序运行 `pilot10.csv`，方式与上节相同：

```bash
for task_model in pangu fengwu fuxi graphcast aurora; do
  weather-hub run --model "$task_model" \
    --cases "$WEATHER_WORKDIR/metadata/pilot10.csv" --json \
    > "$WEATHER_WORKDIR/metadata/$task_model-pilot10-job.json" || break
done
```

不要仅凭一次 6h 推理估算全量费用。记录每个 pilot 的总耗时、GPU 峰值、主机内存峰值，以及对应 `job_dir/outputs/<model>/` 的实际磁盘占用。`du -sh` 默认不跟随 case 里的符号链接，适合统计整个模型实验目录；只看 case 目录会漏掉 `_forecasts/`。

作为初步外推，计算 pilot 的总输出时效数 `S_pilot = Σ(forecast_hours / 6)`：

- 每个模型全量时间约为 `pilot 总耗时 × 7135 / S_pilot`，另留安装、编译、I/O、重试余量。
- 每个模型全量输出空间约为 `pilot 实际输出字节 × 7135 / S_pilot`，再预留至少 30% 余量。

这是容量估算，受模型加载次数、JAX 首次编译、Pangu 调度、FuXi medium 切换、压缩率和存储吞吐影响。先用 126h 验证补充边界信息，运行中再校正估计。pilot 与全量 job 有独立输出，当前控制器不会自动跨 job 复用它们。

磁盘空间不足以容纳一个模型的 570 个 case 时，将冻结 CSV 按原顺序分成若干份，例如每份 50 行，用不同 job 顺序提交；保留 case_id 不变，每批完成即回传验收。最后合并各批 `result.json` 按第 12 节相同规则确认每个模型恰好覆盖 570 个 case。不要删掉仍在运行或准备 resume 的 job 输出。

## 11. 顺序启动全量实验

### 11.1 单个模型

```bash
cd "$WEATHER_WORKDIR/src/ai_weather_models"
weather-hub run --model pangu \
  --cases cases/weather_hub_cases_2022_2024_v2026-09-22.csv --detach --json \
  > "$WEATHER_WORKDIR/metadata/full-pangu.json"
```

`--detach` 成功只表示提交成功，**不是全部 case 推理成功**。从 JSON 读取 JOB_ID，使用：

```bash
weather-hub status JOB_ID --json
weather-hub logs JOB_ID --follow
weather-hub results JOB_ID --json
```

其他模型替换 `--model`。同一个 registry 的 `jobs_dir` 下、同一个 GPU ID 的 job 会被文件锁串行执行；直接启动模型 CLI、doctor 或另一个 jobs_dir 下的控制器任务不受这个锁保护。保持同一张卡每次只有一个真实模型任务。

### 11.2 自动顺序运行五个模型，并记录可恢复 job 映射

在新 GPU 机器保存以下脚本。脚本每次只提交一个模型，任务成功后才继续；重启脚本会读取原 job 映射，防止重复创建整批任务。失败先按第 13 节恢复，再重跑脚本。

```bash
cat > "$WEATHER_WORKDIR/metadata/run_all.py" <<'PY'
import os
from pathlib import Path
from weather_hub.api import Controller
from weather_hub.store import atomic_json
from weather_hub.types import ForecastRequest, ModelId, read_cases_csv
import json

root = Path(os.environ['WEATHER_WORKDIR'])
hub = root / 'src' / 'ai_weather_models'
cases = read_cases_csv(hub / 'cases/weather_hub_cases_2022_2024_v2026-09-22.csv')
assert len(cases) == 570
controller = Controller(os.environ['WEATHER_HUB_CONFIG'])
models = os.environ.get('WEATHER_MODELS', 'pangu fengwu fuxi graphcast aurora').split()
for model in models:
    model_id = ModelId(model)
    request = ForecastRequest(model=model_id, cases=cases)
    mapping = root / 'metadata' / f'full-{model}.json'
    if mapping.exists():
        record = controller.get_job(json.loads(mapping.read_text())['job_id'])
    else:
        record = controller.submit(request)
        atomic_json(mapping, record.to_dict())
    assert record.model == model_id, 'job mapping model mismatch'
    assert record.request_fingerprint == request.fingerprint, 'job mapping request mismatch'
    print(f'{model}: {record.job_id} {record.status.value}', flush=True)
    record = controller.wait(record.job_id, poll_interval=10)
    atomic_json(mapping, record.to_dict())
    print(f'{model}: {record.status.value} {record.error}', flush=True)
    if record.status.value != 'succeeded':
        raise SystemExit(f'Inspect/resume {record.job_id} before continuing.')
print('All selected full batches succeeded; run output acceptance next.', flush=True)
PY
```

在 tmux 中执行：

```bash
source /data/weather-exp/env.sh
conda activate weather-hub
python -u "$WEATHER_WORKDIR/metadata/run_all.py" \
  > "$WEATHER_WORKDIR/logs/full-batch.log" 2>&1
```

另一个终端可 `tail -f /data/weather-exp/logs/full-batch.log`，并用 `weather-hub logs JOB_ID --follow` 查看模型内部进度。

若 40GB 卡只通过前三个模型的验证，显式选择：

```bash
WEATHER_MODELS='pangu fengwu fuxi' python -u "$WEATHER_WORKDIR/metadata/run_all.py" \
  > "$WEATHER_WORKDIR/logs/full-onnx-batch.log" 2>&1
```

另一台更大显存机器只跑 `WEATHER_MODELS='graphcast aurora'`。每台机器独立保存配置和 job 映射，回传后汇总五个模型；前三个模型完成不等于整个五模型实验完成。

## 12. 验收：每个模型必须完整覆盖 570 个 case

先确认所有 full job 的 `status` 为 `succeeded`。`partial`、`failed`、`planned` 都不能通过全量验收。控制器退出码：0 成功、2 部分成功、1 失败或不可用、130 中断；`results` 的命令退出码不代表推理成功。

以下脚本逐模型核对冻结 case 集合、case 状态、两个结果文件、NetCDF 完整标志、变量、全球网格、13 层、全部预报时间坐标和单位，并抽查末时效数值。它不会完整扫描十余 TB 的全部数值；完整 SHA-256 回传校验在第 14 节。

```bash
cat > "$WEATHER_WORKDIR/metadata/accept_outputs.py" <<'PY'
import csv, importlib, json, os, sys
from datetime import datetime, timedelta
from pathlib import Path
import netCDF4
import numpy as np

root = Path(os.environ['WEATHER_WORKDIR'])
hub = root / 'src' / 'ai_weather_models'
with (hub / 'cases/weather_hub_cases_2022_2024_v2026-09-22.csv').open(newline='') as f:
    expected = {r['case_id']: r for r in csv.DictReader(f)}
repos = {'pangu': ('pangu-weather', 'pangu_weather'),
         'fengwu': ('fengwu', 'fengwu_weather'), 'fuxi': ('fuxi', 'fuxi_weather'),
         'graphcast': ('graphcast', 'graphcast_weather'), 'aurora': ('aurora', 'aurora_weather')}
models = os.environ.get('WEATHER_MODELS', 'pangu fengwu fuxi graphcast aurora').split()
report = {}
for model in models:
    repo, module = repos[model]
    sys.path.insert(0, str(root / 'src' / repo))
    constants = importlib.import_module(module + '.constants')
    mapping = json.loads((root / 'metadata' / f'full-{model}.json').read_text())
    job_dir = Path(mapping['job_dir'])
    job = json.loads((job_dir / 'job.json').read_text())
    index = json.loads((job_dir / 'result.json').read_text())
    assert job['status'] == 'succeeded', (model, job['status'])
    assert index['model'] == model and index['job_id'] == job['job_id']
    actual = index['cases']
    assert len(actual) == 570 and {r['case_id'] for r in actual} == set(expected)
    for result in actual:
        case = expected[result['case_id']]
        assert result['status'] in ('complete', 'reused'), result
        init = datetime.fromisoformat(case['init_time'].replace('Z', '+00:00'))
        leads = np.arange(6, int(case['forecast_hours']) + 1, 6)
        files = [('surface.nc', constants.SURFACE),
                 ('atmospheric.nc', constants.ATMOSPHERIC) if model == 'aurora'
                 else ('upper.nc', constants.UPPER)]
        for filename, variables in files:
            path = Path(result['result_dir']) / filename
            assert path.is_file() and path.stat().st_size > 0, path
            with netCDF4.Dataset(path) as ds:
                assert ds.status == 'complete', path
                np.testing.assert_array_equal(ds['lead_time'][:], leads)
                np.testing.assert_allclose(ds['latitude'][:], 90 - np.arange(721) * .25)
                np.testing.assert_allclose(ds['longitude'][:], np.arange(1440) * .25)
                ref = ds['forecast_reference_time']
                np.testing.assert_allclose(ref[...], netCDF4.date2num(init, ref.units))
                valid = ds['valid_time']
                times = [init + timedelta(hours=int(h)) for h in leads]
                np.testing.assert_allclose(valid[:], netCDF4.date2num(times, valid.units))
                upper = filename != 'surface.nc'
                if upper:
                    assert sorted(ds['level'][:].tolist()) == [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
                shape = (len(leads),) + ((13,) if upper else ()) + (721, 1440)
                dims = ('valid_time',) + (('level',) if upper else ()) + ('latitude', 'longitude')
                for name in variables:
                    var = ds[name]
                    assert var.shape == shape and var.dimensions == dims, (path, name)
                    assert var.dtype == np.dtype('float32') and var.units == constants.UNITS[name]
                    sample = var[-1, ..., ::72, ::144]
                    assert not np.ma.getmaskarray(sample).any() and np.isfinite(sample).all(), (path, name)
    report[model] = {'cases': 570, 'status': 'passed', 'job_id': job['job_id']}
    print(model, '570/570 accepted', flush=True)
destination = root / 'metadata' / ('acceptance-' + '-'.join(models) + '.json')
destination.write_text(json.dumps(report, indent=2))
print(destination)
PY

conda run --no-capture-output -n pangu python \
  "$WEATHER_WORKDIR/metadata/accept_outputs.py"
```

只验收某些模型时，像运行脚本一样设置 `WEATHER_MODELS`。分机器运行时，各自生成验收报告，最终合并后应为五个模型各 570/570。分片运行则需把每个模型多个 job 的 result cases 合并，检查总数、重复项和完整 case 集合，不能直接套用单 job 570 行的脚本。

## 13. 失败恢复与常见问题

正常中断、个别 case 失败或 SSH 断线后：

```bash
weather-hub status JOB_ID --json
weather-hub logs JOB_ID
weather-hub resume JOB_ID --detach --json
weather-hub logs JOB_ID --follow
```

**使用原 JOB_ID resume。** 再次运行 `weather-hub run` 会建立新 job 和独立输出，不能自动接上之前的结果。已完成且符合模型恢复规则的 case 会复用；未完成 case 从起报场重算，不是从失败的第 N 个时效继续。`succeeded` job 不需要也不能 resume。

恢复时保持原 `cases.csv`、`effective-config.yaml`、模型代码、权重和 ERA5 输入不变。控制器拒绝更改 effective config 的 resume；其他输入身份检查由各模型实现，严格程度不同，因此仍应保持这些资产固定。改模型身份、分辨率、精度策略或数据内容时，创建新实验并重新验证。

需要停止正在执行的任务时使用 `weather-hub cancel JOB_ID`。实例被强制终止或重启后，旧 job 可能仍显示 running/queued；先核对 `job.json` 的 worker/process PID 与实际进程。确认旧进程已经不存在后才把旧任务取消为 interrupted，再 resume；若 PID 已被其他进程复用，不要直接执行 cancel，应先处理陈旧的 PID 记录。

| 现象 | 处理 |
|---|---|
| `conda` 找不到或模块导入失败 | source `env.sh`，确认 registry 中 Conda 绝对路径、六个同级项目目录和环境名 |
| `libcudnn.so.9` / `libcublasLt.so.12` 缺失 | 检查该模型环境的 NVIDIA pip 依赖，确认 ORT 1.22.0；需要修复时重新安装固定的 `onnxruntime-gpu[cuda,cudnn]==1.22.0`，再做 load-model 和真实 6h 验证 |
| available providers 有 CUDA，但不能加载 | 以实际 load-model 和推理日志为准；provider 列表只是编译能力，不证明运行库可加载 |
| JAX 使用 CPU 或加载错误 CUDA | 保持 `jax[cuda12]==0.6.2`；排查全局 `LD_LIBRARY_PATH` 抢先加载旧库，参考 [JAX 安装说明](https://docs.jax.dev/en/latest/installation.html)；容器里的驱动库路径需要保留，不能盲目全部清空 |
| `torch.cuda.is_available()` 为 false | 确认驱动/GPU 暴露，核对 `torch.version.cuda`，必要时按第 5 节安装固定 CUDA wheel |
| CUDA OOM / `RESOURCE_EXHAUSTED` | 结束同卡其他任务，确认 `max_sessions: 1`、Aurora autocast；仍失败则换更大显存机器 |
| FuXi ONNX 加载时报 external data 缺失 | 重新复制同目录下的 `short/medium/long` 无扩展名资产 |
| ERA5 文件存在但缺少时次或变量 | 先查完整性报告；检查 T−6h 与 T 是否跨日，FuXi 降水目录是否完整；保持日期模板，不把所有 case 指向单日文件 |
| `No space left on device` | 扩容或先备份已完成的其他 job；留出足够空间后 resume 当前 job |
| doctor 找不到 `.cdsapirc` | 显式输入且禁止下载时，CDS 密钥无需部署；确认报告是否真的因此失败，而不是缺权重/依赖 |
| job 显示 partial | 查看各 case 的 error，修复后 resume，直至 succeeded 且 570/570 验收通过 |

## 14. 回传、验证备份并结束租赁

输出结构如下：

```text
runs/<job_id>/
├── job.json
├── request.json
├── effective-config.yaml
├── cases.csv
├── model.log
├── result.json
└── outputs/<model>/
    ├── summary.csv
    ├── batch.json
    ├── cases/<case_id>/
    │   ├── surface.nc
    │   └── upper.nc / atmospheric.nc
    └── _forecasts/<fingerprint>/   # Pangu/FengWu/FuXi/GraphCast 共享预报
```

前四个模型 case 中的 NetCDF 是相对符号链接，**必须保留整个模型实验目录及 `_forecasts/`**。推荐复制完整 job 或 `runs/`。使用 `rsync -a` 保留链接；仅复制 `cases/` 会产生断链。Aurora 输出直接位于 case 目录。迁移后 `result.json` 内的绝对路径仍指向原机器，作为原始来源保留；新机器/备份中按目录结构定位文件。

在新 GPU 机器为需要交付的预测文件生成校验清单。下面会读取 `runs/` 中真实 `.nc` 文件，包括验证批次；可将范围改为仅 full job：

```bash
cd "$WEATHER_WORKDIR"
find runs -type f -name '*.nc' -print0 | sort -z | xargs -0 -r sha256sum \
  > metadata/forecast-files.sha256
```

在接收备份的机器执行，先自行配置 `WEATHER_REMOTE`、`WEATHER_REMOTE_ROOT`、`WEATHER_SSH` 和备份路径：

```bash
export WEATHER_BACKUP=/path/to/persistent/weather-backup
mkdir -p "$WEATHER_BACKUP"
rsync -aP -e "$WEATHER_SSH" "$WEATHER_REMOTE:$WEATHER_REMOTE_ROOT/runs/" \
  "$WEATHER_BACKUP/runs/"
rsync -aP -e "$WEATHER_SSH" "$WEATHER_REMOTE:$WEATHER_REMOTE_ROOT/metadata/" \
  "$WEATHER_BACKUP/metadata/"
rsync -aP -e "$WEATHER_SSH" "$WEATHER_REMOTE:$WEATHER_REMOTE_ROOT/logs/" \
  "$WEATHER_BACKUP/logs/"
rsync -aP -e "$WEATHER_SSH" "$WEATHER_REMOTE:$WEATHER_REMOTE_ROOT/env.sh" \
  "$WEATHER_BACKUP/"
cd "$WEATHER_BACKUP"
sha256sum -c metadata/forecast-files.sha256
find runs -xtype l -print
```

SHA-256 全部通过，且最后一条命令没有断链输出。另将六个仓库的实际 `configs/a100.yaml` / `config/models.a100.yaml` 和源码快照保留；模型/输入资产已在旧机器保存的情况下可避免重复回传它们。结果、源码快照或环境清单可能很大，先确认备份空间和预计回传时间。

采用逐模型/分片释放空间的方案时，为每个已完成 job 单独生成校验清单、复制整个 job、在接收端验证并检查断链；保留这些清单和验收报告，再释放其源端空间。全量一次保留的方案则在五个模型验收完成后统一回传。

**结束租赁前确认：五个模型分别覆盖 570 个固定 case，NetCDF 验收通过，备份内容哈希匹配且无断链，配置/版本/日志/验收报告齐全。** 暂停机器与释放实例对数据盘的保留行为由平台决定；先完成可验证的持久备份，再停止计费。

## 15. 执行顺序速查

1. 选定显存、主机内存和持久数据盘，确认 40GB 的边界验证安排。
2. 安装工具/Conda，确定工作目录并保存 `env.sh`。
3. 上传六个源码仓库，创建六个独立 Python 环境。
4. 上传四组已有权重，下载固定 Aurora checkpoint，校验资产。
5. 上传 ERA5，完成 1,207 个文件的 SHA-256 校验。
6. 生成新机器配置，完成 doctor、实际加载和五个 dry-run。
7. 五个模型各跑 6h、126h 和 pilot10，确认显存及估算租期/空间。
8. 顺序运行五个全量批次；必要时按模型/分片回传释放空间。
9. 每个模型完成 570/570 验收，失败使用原 job resume。
10. 完整回传、哈希比对、检查符号链接，再结束租赁。
