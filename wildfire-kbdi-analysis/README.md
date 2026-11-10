# Permafrost–wildfire: Frost Number, KBDI and statistical analysis

Research-code package prepared for review before public release. **Do not upload unpublished monitoring data or private experiment records to GitHub.** This repository contains **code and small synthetic examples only**. The original permafrost source supplied by the author is preserved verbatim; KBDI is an *independent reference implementation*, **not** the original KBDI code used to generate manuscript projections.

## What runs on what?

| Workflow | Input variables | Processing | Output |
|---|---|---|---|
| Frozen-ground classification | **annual** freezing index `FI`, thawing index `TI`; TP mask `GTOPOP_v4.nc/elevation` | Frost-number `sqrt(FI)/(sqrt(FI)+sqrt(TI))`, regional thresholds | Annual NetCDF `frozen`, coded 0/1/2/3 |
| **KBDI (separate process)** | **daily** `tasmax` (K or °C), **daily** `pr` (kg m⁻² s⁻¹ or mm/day), **historical mean annual** precipitation derived from daily `pr` | Keetch–Byram daily water-deficit recurrence | Per-GCM annual and optional daily NetCDF, values 0–800 |
| KBDI multi-model ensemble | Individual **computed KBDI fields** for each model, each SSP | Equal-weight cellwise average, with number of valid GCMs and inter-model SD | `MME_KBDI_*.nc` |
| Wildfire temporal trends | `Wildfire_data.xlsx` | Mann–Kendall / Sen slope on normalized index multiplied by 100 | CSV and PNG |
| ALT trends | `Thawing_Depths.xlsx` | OLS / Pearson r / two-sided t test | CSV and PNG |
| Methane–fire | `监测数据v3.xlsx` | Negative-binomial regression and within-year circular-shift permutation | CSV |

**The frost-number classification never uses `tasmax`/`pr` directly; the KBDI algorithm never uses `FI`/`TI`.** The two pipelines share only the 25-GCM / four-SSP model configuration listed in `src/cmip6_config.py`.

## Setup

Python 3.10+ is recommended. In the repo directory:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python -m pip install pytest
PYTHONPATH=src pytest -q
```

For Windows PowerShell, replace `PYTHONPATH=src pytest -q` with `$env:PYTHONPATH="src"; pytest -q`.

## Input file layout for the independent KBDI pipeline

Defaults assume **per-year** NetCDF files, with the following example layout:

```text
/data/cmip6/
  historical/
    tasmax_ACCESS-CM2_1950.nc   # variable tasmax, units K (or C)
    pr_ACCESS-CM2_1950.nc       # variable pr, units kg m-2 s-1 (or mm/day)
    ... all historical years for each model ...
  ssp126/
    tasmax_ACCESS-CM2_2015.nc
    pr_ACCESS-CM2_2015.nc
  ssp245/ ...
  ssp370/ ...
  ssp585/ ...
```

Each input is a **complete daily calendar year** with 1-D coordinates `time`, `lat`, `lon` (variable shape `time × lat × lon`). Both meteorological fields must share the same grid/calendar/timestamps; no silent regridding or interpolation is performed. `tasmax`/`pr` file naming differs across NEX-GDDP-CMIP6 downloads: **edit `--file-template` to match your actual file naming**. Example for a naming scheme `tasmax_day_MODEL_SCENARIO_YYYY.nc`:

```bash
python src/kbdi/run_cmip6_kbdi.py --input-root /data/cmip6 \
  --output-root /data/kbdi_by_gcm \
  --file-template '{scenario}/{variable}_day_{model}_{scenario}_{year}.nc' \
  --baseline-start 1981 --baseline-end 2014 \
  --historical-start 1950 --historical-end 2014 \
  --start-year 2015 --end-year 2099
```

**Important:** This program expects one daily NetCDF per variable, model and year. If your source uses multi-year NetCDF, split these into complete annual files (without changing values) before running, and document that preprocessing. The `--file-template` supports glob `*` only when it matches **exactly one** file. Variables in NEX-GDDP-CMIP6 commonly have units `K` (`tasmax`) and `kg m-2 s-1` (`pr`), automatically converted to °C and mm/day. Inspect your metadata first; use `--temp-units` / `--precip-units` only for verified corrections.

### KBDI formula and interpretation

The Keetch–Byram daily **post-rain** index `Q*` is computed as `max(0, Q_previous - 100 * effective_rain_inches)`, then

```
DeltaQ = (800 - Q*) * (0.968 * exp(0.0486 * Tmax_F) - 8.30) * 0.001
         / (1 + 10.88 * exp(-0.0441 * mean_annual_precip_inches))
Q_today = min(800, max(0, Q* + max(DeltaQ, 0)))
```

Drying is set to zero when `Tmax <= 50°F` (`10°C`). The standard **0.20 inch (5.08 mm) initial rainfall loss** is applied once per consecutive wet spell in the default `--rainfall-mode wet_spell` (preserves the threshold over adjacent rainy days); alternatively `--rainfall-mode daily` subtracts this threshold on **each individual wet day**. This choice affects results and is saved in NetCDF metadata. Both modes use `pr`, **not** frost indices.

The annual rainfall climatology is calculated **per GCM** from complete `pr` records in the historical baseline (default 1981–2014); it remains fixed for that GCM across the future scenarios to avoid redefining the calibration mid-run. The index is initialized to `0` on the first historical spin-up day (a **saturation assumption, not independently verified**), then run daily throughout the historical period with state carried across each subsequent calendar year. Future SSP runs each start from the same GCM's state at the end of 2014. For a different scientifically supported initialization or baseline, configure the CLI and disclose it.

**Limitations for cold-region interpretation:** KBDI is a simplified meteorological dryness proxy, not direct fire ignition probability, snowpack physics or frozen-soil hydrology. The supplied program does not distinguish rain from snow without precipitation-phase inputs and does not simulate soil freeze–thaw or methane. Interpret northern cold-season values cautiously. May–September `--warm-months` is an NH default; choose different months for SH regions. Missing meteorological values produce NaN at the affected cell/day and no numerical imputation; verify missingness and warm-season coverage before interpretation.

### Calculate each GCM × each SSP, then average the *KBDI values*

```bash
python src/kbdi/run_cmip6_kbdi.py \
  --input-root /data/cmip6 --output-root /data/kbdi_by_gcm \
  --file-template '{scenario}/{variable}_{model}_{year}.nc' \
  --historical-start 1950 --historical-end 2014 \
  --baseline-start 1981 --baseline-end 2014 \
  --start-year 2015 --end-year 2099 \
  --rainfall-mode wet_spell --warm-months 5 6 7 8 9

python src/kbdi/ensemble.py \
  --model-output-root /data/kbdi_by_gcm --output-root /data/kbdi_mme \
  --start-year 2015 --end-year 2099 --min-models 25
```

All **25 listed models** are required by default (`--min-models 25`), avoiding silently treating an incomplete ensemble as a 25-model ensemble. The MME is `mean(KBDI_model1,...,KBDI_model25)`, **not** `KBDI(mean tasmax, mean pr)`. Grids must align exactly. Ensemble outputs include equal-weight mean, inter-model sample SD, and `n_gcms` per grid cell; an inter-model SD is not a confidence interval. Computing more complex observational uncertainty or confidence intervals is beyond this repository.

### Run the author's original frozen-ground classification

**Original submitted script** (kept intact):

```bash
cd /path/to/fi_ti_data_root
python /path/to/repo/src/permafrost/Permafrost_distribution_data_ORIGINAL.py
```

It expects `GTOPOP_v4.nc` and scenario directories such as `SSP126/FI_ACCESS-CM2_1950.nc`, `SSP126/TI_ACCESS-CM2_1950.nc`. A reusable, validated adaptation, preserving the original class thresholds and IDs, is also provided:

```bash
python src/permafrost/frost_number_classification.py \
  --input-root /data/frozen_indices --mask /data/frozen_indices/GTOPOP_v4.nc \
  --output-root /data/frozen_classes \
  --start-year 1950 --end-year 2099 --strict
```

The author's code uses PF threshold **0.55 inside** TP (`mask == 1`) and **0.53 outside** TP, with other class thresholds of 0.01/0.10. These numbers are intentionally **not** overwritten by separate manuscript drafts; please check which thresholds generated the actual submitted figures. The original `FI_ensemble_mean` / `TI_ensemble_mean` classification is **not** a multi-model mean of categorical classes.

### Existing statistical scripts (user's attached workbooks, not part of GitHub package)

```bash
python src/statistics/wildfire_all_latitudes_mk_sen.py \
  --input /private/Wildfire_data.xlsx --output-dir /private/results/wildfire

python src/statistics/analyze_ALT_significance.py \
  --input /private/Thawing_Depths.xlsx --output-dir /private/results/alt

python src/statistics/reproduce_methane_fire_reported.py \
  --input /private/监测数据v3.xlsx --outdir /private/results/methane \
  --permutations 1000 --seed 20261109
```

The methane study's original analysis program (`src/statistics/wildfire_ch4_analysis.py`) also contains **exploratory analyses** not reported in the revised manuscript. Use `reproduce_methane_fire_reported.py` to reproduce **only** the negative-binomial regression and permutation statistics retained in Results. Re-check the early dates associated with sensor R-2 before public release.

**Do not commit Excel inputs:** Raw monitoring, site data and third-party records must receive separate rights/provenance review and appropriate public archival deposits, if permitted. The three statistics programs use a small built-in Python XLSX reader (`src/statistics/xlsx_reader.py`) for portable parsing; **cached** values in formula-driven cells are required, as formulas are not evaluated. The `.gitignore` excludes `.xlsx`, `.nc`, `.csv` and standard generated outputs; lightweight reference results in `reference_results/` are tracked as curated examples.

## Minimal synthetic end-to-end demonstration

```bash
python examples/make_synthetic_data.py --output examples/_synthetic_input
python src/kbdi/run_cmip6_kbdi.py \
  --input-root examples/_synthetic_input --output-root examples/_synthetic_results \
  --models MODEL-A MODEL-B --scenarios ssp126 ssp245 \
  --historical-start 2000 --historical-end 2001 \
  --baseline-start 2000 --baseline-end 2001 --start-year 2002 --end-year 2002
python src/kbdi/ensemble.py \
  --model-output-root examples/_synthetic_results \
  --output-root examples/_synthetic_mme \
  --models MODEL-A MODEL-B --scenarios ssp126 ssp245 \
  --start-year 2002 --end-year 2002 --min-models 2
```

**Synthetic datasets must never be described as observed data or climate projections.**

## Scientific and reproducibility notes

- The 25 GCMs and four SSP names are taken *verbatim* from the author's supplied script; their presence in a configuration does **not** establish that each historical and scenario file is actually available.
- The KBDI algorithm is reconstructed from published equations. Without original model forcing files and study choices, its numerical output has **not** been verified against manuscript KBDI figures.
- The CALM example workbooks cover 12 displayed sites, **not** the entire reported 168-site network.
- A relative change in 0–100 normalized burned-area *points* is not necessarily a relative percentage change in physical burned area.
- Trend P-values from standard MK and OLS assume appropriate serial independence under the null; significant autocorrelation may require more careful inference.

## Citations

- Keetch, J. J. & Byram, G. M. *A drought index for forest fire control*. USDA Forest Service Research Paper SE-38 (1968). Formula and implementation summaries: [USFS/Drought.gov](https://www.drought.gov/data-maps-tools/keetch-byram-drought-index); [WSL Fire Weather Indices Wiki](https://wikifire.wsl.ch/tiki-index908f.html?page=Keetch-Byram+drought+index).
- Mann, H. B. (1945). Nonparametric tests against trend. *Econometrica* 13:245–259. DOI: 10.2307/1907187.
- Sen, P. K. (1968). Estimates of the regression coefficient based on Kendall's tau. *JASA* 63:1379–1389. DOI: 10.1080/01621459.1968.10480934.
- Hilbe, J. M. *Negative Binomial Regression*, 2nd ed. CUP (2011). DOI: 10.1017/CBO9780511973420.

## Publication checklist

See [`RELEASE_CHECKLIST.md`](RELEASE_CHECKLIST.md), [`DATA_AVAILABILITY_TEMPLATE.md`](DATA_AVAILABILITY_TEMPLATE.md) and [`CODE_AVAILABILITY_TEMPLATE.md`](CODE_AVAILABILITY_TEMPLATE.md). **No repository URL, DOI or verified 25-model projection exists yet.** Choose a software license and confirm all contributors/rights before making the repository public.