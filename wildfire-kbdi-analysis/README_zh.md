# 多年冻土–野火研究：KBDI、冻土模拟与统计检验代码

本代码仓库是 **GitHub 发布前的待审核版本**。没有上传原始甲烷、气象、大气电场或实验观测数据，也没有虚构 DOI。项目使用**同一套25个 CMIP6 GCM 名称和4个 SSP 情景名称**，但两种核心计算的**输入物理变量完全不同**：

- 冻土分类：使用您原程序的逐年 `FI`（冻结指数）和 `TI`（融化指数）及青藏高原掩膜 `GTOPOP_v4.nc`；原程序一字不改地保留于 `src/permafrost/Permafrost_distribution_data_ORIGINAL.py`。
- **KBDI：完全独立地根据 Keetch–Byram 公式使用每日 `tasmax`（最高气温）+ `pr`（降水）+ 历史气候态年平均降水量**。逐个 GCM、逐个 SSP 递推，之后将各模型**算好的 KBDI**进行等权多模式集合平均。绝不使用 `FI` / `TI` 计算 KBDI，也不先平均气温降水再计算 KBDI。
- 统计检验：包含低中高纬过火面积 Mann–Kendall + Sen's slope；12个代表 CALM 站点 OLS / Pearson / t 检验；甲烷–火点负二项回归及年内时间平移置换检验。

## 快速运行

1. `python -m pip install -r requirements.txt`；`python -m pip install pytest`。
2. `PYTHONPATH=src pytest -q` 运行单元及合成测试（Windows PowerShell：`$env:PYTHONPATH="src"; pytest -q`）。
3. KBDI 输入布局默认 `{scenario}/{variable}_{model}_{year}.nc`，如 `historical/pr_ACCESS-CM2_1950.nc`、`historical/tasmax_ACCESS-CM2_1950.nc`；需要逐年文件及一致的 time/lat/lon。
4. `python src/kbdi/run_cmip6_kbdi.py --input-root /data/cmip6 --output-root /data/kbdi_by_gcm --historical-start 1950 --historical-end 2014 --baseline-start 1981 --baseline-end 2014 --start-year 2015 --end-year 2099`
5. `python src/kbdi/ensemble.py --model-output-root /data/kbdi_by_gcm --output-root /data/kbdi_mme --start-year 2015 --end-year 2099 --min-models 25`
6. 冻土：`python src/permafrost/frost_number_classification.py --input-root /data/frozen_indices --mask /data/frozen_indices/GTOPOP_v4.nc --output-root /data/frozen_classes --start-year 1950 --end-year 2099 --strict`。

更完整的代码说明、变量和单位、NetCDF 文件名调整、历史暖机和缺失值处理详见英文 `README.md`。

## KBDI 的物理和数据注意事项

- 每日降水先扣减上一日指数，再按 `tasmax` 和气候态年降水计算干燥增量；50°F（10°C）以下不增加干燥项，指数限制在0–800。
- 0.20英寸（5.08mm）的降水截留阈值，默认连续降雨过程只扣一次（`--rainfall-mode wet_spell`），另支持每日扣除（`daily`），方法须在论文中如实注明。
- 由1981—2014年每年完整日降水计算各个 GCM 独立的年平均降水量；不是使用 `FI` / `TI`。NEX-GDDP-CMIP6常见的 `K`、`kg m-2 s-1` 自动换算为°C、mm/day。
- 初始化默认1950年指数为0，仅为“初始土壤饱和”假设，需结合实际研究核实；2014年底历史状态接到2015年未来情景。
- 代码没有自动识别雨雪相态，也没有模拟积雪、土壤冻结或者甲烷点火机制。KBDI 仅反映气象干燥背景，不是点火概率。
- 25个模型必须都有完整输入才能生成 `min-models=25` 的集合结果；缺失时不自动编造或填补。

## 正式上传 GitHub 前

确认原始气象数据路径、降水基准时段、初始化、雨雪处理、暖季定义、冻土阈值与论文完全一致。代码测试使用合成数据，**尚未完成25个真实GCM的KBDI结果与文章图件对照**。请选用适当开源许可证、填入真实作者和 DOI；观测原始数据另存至具备使用许可的专业数据平台。详细见 `RELEASE_CHECKLIST.md`。