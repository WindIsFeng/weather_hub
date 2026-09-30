# Windows 移动硬盘 ERA5 完整性检查

本工具检查本项目 570 个 case 所需的 ERA5 推理输入：397 个地面文件、397 个
高空文件、412 个降水文件和 1 个静态文件，共 **1,207 个 `.nc` 文件**。
原始文件总计 171,098,308,501 字节（159.35 GiB）。不包含模型权重或预测验证场。

## 运行方法

1. 将检查包解压到电脑上的任意文件夹，保持 `check_era5_integrity.py` 与
   `era5_integrity_manifest.json` 在同一目录。用 PowerShell 打开这个目录。
2. 需要 Python 3.10 或更新版本，不需要安装额外 Python 库，也不需要联网或 CDS 密钥。
3. 运行完整检查：

```powershell
py -3 .\check_era5_integrity.py --root "F:/"
```

如果电脑没有 `py` 命令，但已经安装 Python，可以替换成：

```powershell
python .\check_era5_integrity.py --root "F:/"
```

程序自动识别直接放在 `F:\`、`F:\era5_inputs`、
`F:\ai_weather_models\era5_inputs` 或
`F:\data\hufeng\ai_weather_models\era5_inputs` 的归档。
如果实际位置不同，或者同一硬盘有多份归档，请直接指定数据文件夹，例如：

```powershell
py -3 .\check_era5_integrity.py --root "F:/备份/era5_inputs"
```

正确的数据目录下面应直接包含日期目录、`precipitation` 目录和 `static.nc`：

```text
era5_inputs/
  20220127/surface.nc
  20220127/surface.json
  20220127/upper.nc
  20220127/upper.json
  precipitation/20220127.nc
  precipitation/20220127.json
  static.nc
  static.json
```

## 检查内容与结果

默认完整检查核对每个必需文件的存在性、准确字节数和 SHA-256。
**必须完整读取约 159.35 GiB 的数据**，耗时取决于硬盘和接口速度；运行时每
15 秒显示读取进度。检查期间保持硬盘连接，停止向数据目录复制或下载文件。
脚本只读取数据；它不会修复、删除或重新下载文件。

报告默认保存在用户目录中的 `era5_integrity_reports\日期时间\`，每次新建
一个文件夹。可以用 `--report-dir "C:/Users/你的用户名/Desktop/ERA5检查结果"`
指定一个新的结果目录。不要选择已经包含 `report.json` 或 `issues.csv` 的目录。

- `report.json`：结果、文件状态、异常统计和逐文件检查记录。
- `issues.csv`：缺失、损坏和其他异常，支持用 Excel 打开。
- `PASS`：1,207 个数据文件及其已复制的校验记录均与原始记录一致。
- `PASS_WITH_WARNINGS`：1,207 个数据文件均通过 SHA-256，但有警告。
  例如未复制 `.json` 校验记录、多余 `.nc` 或下载临时文件；详见异常清单。
  即使缺少 `.json`，独立参考清单仍能校验全部数据内容。
- `FAIL`：有缺失、字节数不符、内容不符、读取失败，或已复制的 `.json` 与原记录不符。
  优先查看 `issues.csv`。`all_data_sha256_verified` 单独标明所有数据文件是否通过。
- `INCOMPLETE`：按 Ctrl+C 中断，只能说明已经检查的部分，不能判断完整归档。

如需先快速找出缺文件，可以加 `--quick`，但该模式**不检查数据内容**，
结果是 `QUICK_CHECK_ONLY`；随后仍需运行不带 `--quick` 的完整检查。
退出码：0 表示所选检查无错误（快速检查仍不能证明内容完整）；1 表示发现异常；
2 表示路径、清单或报告写入等运行错误；130 表示用户中断。

## 参考清单来源与范围

参考清单来自服务器原始归档
`/data/hufeng/ai_weather_models/era5_inputs` 的下载校验记录，逐项匹配冻结 case
下载计划，并核对源文件当前大小。全部原下载记录均标记已通过数据值检查。
导出时没有重新读取 159.35 GiB 源数据计算哈希；清单沿用下载时记录的 SHA-256。
Windows 上计算出的 SHA-256 与这些原始记录比对，因此不依赖移动硬盘上的
`.json` 作为唯一参照。脚本还固定了参考清单本身的 SHA-256，避免误用修改后的清单。

通过表示复制的数据与原下载记录一致。它不评估天气数据的科学准确性，
也不检查这批推理输入以外的数据。将 `report.json` 和 `issues.csv` 发回即可进一步定位问题。
