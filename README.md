# AHSE v2: honest ensemble selection, archived weather forecasts and conformal intervals for PV forecasting

Code, data-preparation scripts and result tables of the manuscript

> **Honest ensemble selection with archived weather forecasts and conformal intervals for
> photovoltaic power forecasting: a curtailment-aware evaluation on two sites**
> Bilali Boureima Cissé, Ghamgeen Izat Rashed — Wuhan University
> *in preparation for Sustainable Energy Technologies and Assessments*

This repository is the revised version of [billciss/ahse-paper](https://github.com/billciss/ahse-paper).
The results of that first version are **withdrawn**: the Yulara series it used (the Desert Gardens
site) is curtailed around noon, the Folsom "PV power" was simulated from irradiance, and the
selection and skill-score code contained errors fixed here. The first repository is kept unchanged
as a record.

## What the study does

- **Curtailment-aware data.** At Yulara (off-grid mini-grid, Australia) the Desert Gardens site is
  curtailed around noon; about half of its available energy (31.7–59.1%) is not delivered. The
  target is the sum of the four uncurtailed sites (768 kW). Second site: NIST Ground array
  (Maryland, USA, 271 kW), from the public PVDAQ data lake.
- **AHSE selection.** For each block of lead times (1 h, 1–6 h, 6–24 h), a single model or an
  average of the best models, accepted only when the lower bound of a day-level bootstrap interval
  of its validation gain is positive. Members: LightGBM, XGBoost, PatchTST, N-HiTS, LSTM, GRU,
  persistence and day-ahead persistence.
- **Archived GFS forecasts without look-ahead** (NSF NCAR GDEX d084001): only the run published
  at least 5 h before the issue time is used.
- **Conformal prediction intervals**, calibrated out-of-fold with respect to the selection
  (Mondrian and adaptive conformal inference).
- **Evaluation in power** (kW and % of capacity), skill score against day-ahead persistence,
  day-level bootstrap and Diebold–Mariano tests, five seeds at Yulara.

## Main results (hourly forecasts, lead times 1–24 h, test periods)

| Test MAE of AHSE | without GFS | with GFS |
|---|---|---|
| Yulara (768 kW, 5 seeds) | 36.18 ± 0.32 kW (4.71%) | **31.32 ± 0.05 kW (4.08%)** |
| NIST Ground (271 kW) | 27.55 kW (10.17%) | **20.39 kW (7.52%)** |

- GFS lowers the error by 13% at Yulara (significant for every seed) and 26% at NIST.
- Without GFS at Yulara, AHSE beats the best single model of every seed by 7–10%; when one member
  dominates (GFS-fed XGBoost) it falls back to it. It is never significantly worse than the best
  member.
- 90% intervals reach 0.90–0.92 coverage; GFS narrows them by 17% at Yulara.

Details, including corrected errors and sensitivity analyses: [`results/yulara_neighbours_nwp/README.md`](results/yulara_neighbours_nwp/README.md)
and [`results/curtailment/README.md`](results/curtailment/README.md). All summary tables in kW:
[`results/power_summary/`](results/power_summary/).

## Repository layout

```
paper/                LaTeX manuscript (elsarticle) and its figures
src/
  data_engine/        preprocessing, clear-sky power envelope, k_t, gaps, GFS covariates (nwp_features.py)
  models/             base models, persistence references, honest_selection.py (AHSE block selector)
  evaluation/         metrics, conformal.py, statistical tests
  training/           neural and tree trainers
configs/              data configurations (Yulara uncurtailed, NIST; *_interleaved = season-balanced
                      validation) and model hyper-parameters
scripts/              data download/preparation, experiment chains, comparisons, figures (see below)
results/              result tables (CSV/JSON) of every run reported in the paper
tests/                unit tests (pytest)
main.py               training and evaluation pipeline
```

## Reproducing the results

Python 3.12 (tested with numpy 2.3, pandas 2.3, torch 2.9, LightGBM 4.6, XGBoost 3.1, pvlib 0.13).

```bash
pip install -r requirements.txt
pytest -q tests
```

**1. Data**

| Data | Source | Scripts |
|---|---|---|
| Yulara per-site 5-min data | [DKASC portal](https://dkasolarcentre.com.au/download?location=yulara) (manual download) | `scripts/prepare_yulara_sites.py`, curtailment: `quantify_curtailment.py`, `yulara_sites_analysis.py`, `censored_available_power.py` |
| NIST Ground array (PVDAQ system 4902) | [OEDI PVDAQ](https://doi.org/10.25984/1846021), no account needed | `scripts/fetch_pvdaq.py`, `convert_pvdaq_nist.py`, `prepare_nist.py --weather-source array` |
| GFS 0.25° archive | [NSF NCAR GDEX d084001](https://doi.org/10.5065/D65D8PWK), free account and API token in `~/.gdex_token` | `scripts/gdex_batch.py` |

The NIST pyranometer conversion uses the NIST archive `onemin-Ground-2016.zip` from
[pvdata.nist.gov](https://pvdata.nist.gov) (see `convert_pvdaq_nist.py`). Raw and processed data
are not tracked (`data/`, `data_external/`).

**2. Runs**

```bash
# Yulara, without and with GFS (seed 42)
python main.py --horizon 1h --models all --seed 42 \
    --data-config configs/data_config_yulara_neighbours.yaml \
    --data-file data/processed/yulara_neighbours_1h.csv --results-dir results/yulara_neighbours
python main.py --horizon 1h --models all --seed 42 \
    --data-config configs/data_config_yulara_neighbours.yaml \
    --data-file data/processed/yulara_neighbours_1h.csv \
    --gfs data_external/gfs/gfs_yulara_DSWRF.csv data_external/gfs/gfs_yulara_TCDC.csv \
    --results-dir results/yulara_neighbours_gfs
```

Experiment chains: `scripts/nwp_ahse_full.sh` (Yulara with GFS), `scripts/seeds_gfs.sh` (five
seeds), `scripts/nist_pvdaq_chain.sh` (NIST), `scripts/seasonal_val_chain.sh` (season-balanced
validation).

**3. Tables and figures**

```bash
python scripts/power_metrics.py --out results/power_summary          # all results in kW
python scripts/compare_nwp_runs.py --a <run A> --b <run B> --out <file>  # day-level tests
python scripts/conformal_intervals.py --cache <run>/tables/predictions_1h.npz --out <run>/tables
python scripts/make_paper_figures.py                                 # paper/figures
```

## Citation

To be added upon publication.

## License

MIT — see [LICENSE](LICENSE). The data remain under the licences of their providers (DKASC, NREL PVDAQ: CC-BY 4.0, NSF NCAR GDEX).
