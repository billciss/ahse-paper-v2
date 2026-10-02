# Yulara: value of archived GFS day-ahead forecasts

Experiment run on 2026-10-01 (`scripts/nwp_experiment.py`, commit 723a987).

## Data and protocol

- Target: summed output of the four uncurtailed Yulara sites (Service Station, Sails, Laundry,
  Connellan Airport), clearness index k_t = P / P_cs, 1-hour means labelled by interval start,
  24-step day-ahead horizon (`configs/data_config_yulara_neighbours.yaml`).
- Period: 2016-04-02 to 2018-12-31, the span covered by both GFS variables retrieved so far.
  Chronological split 60/20/20, sequences touching gaps longer than 1 h excluded, daytime
  targets only in the metrics. Test set: 3,842 forecasts (second half of 2018).
- GFS 0.25° (NSF NCAR GDEX d084001), all four daily runs, bilinear interpolation to the site:
  downward shortwave radiation (DSWRF, surface) and total cloud cover (TCDC, entire atmosphere).
  GFS window-start averages are de-accumulated to 3-hour means over (valid - 3 h, valid].
- **No look-ahead**: a forecast issued at t0 (label of the last observation) uses only the latest
  run with `init + 5 h <= t0` (GFS is usually published 3.5–4.5 h after its initial time).
  Each hourly target [tau, tau + 1 h) takes the 3-hour window holding its midpoint.
- Missing GFS values (run absent from the archive): train 0.03%, validation 14.9%, test 0%.

## Variants

| Variant | Inputs |
|---|---|
| `lgbm_no_nwp` | LightGBM (configs/model_params.yaml) on the flattened lookback window |
| `lgbm_nwp` | same + per target step: GFS DSWRF, TCDC, clear-sky GHI, GFS clearness index |
| `gfs_only` | GFS clearness index used directly as the k_t forecast |
| `SmartPersistence` | day-ahead persistence (reference of the skill score) |

The GFS clearness index divides the 3-hour GFS radiation by the clear-sky GHI averaged over the
**same** 3-hour window.

## Results (test, daytime targets)

| Model | MAE | RMSE | R² | Skill vs day-ahead persistence |
|---|---|---|---|---|
| **LightGBM + GFS** | **0.098** | **0.172** | **0.579** | **+0.588** |
| LightGBM without GFS | 0.133 | 0.207 | 0.394 | +0.407 |
| Day-ahead persistence | 0.146 | 0.268 | -0.021 | 0 |
| GFS clearness index alone | 0.167 | 0.280 | -0.113 | -0.090 |

- GFS reduces LightGBM's MAE by 27% and raises R² from 0.39 to 0.58.
- Raw GFS alone is worse than persistence: its value comes from the learned combination with
  recent observations, which motivates using it inside the ensemble rather than as a stand-alone
  forecast.
- The test period differs from the main Yulara runs (which extend to 2020), so these numbers must
  not be compared with `results/yulara_neighbours`; the like-for-like comparison of the full AHSE
  pipeline with and without GFS is in `results/yulara_neighbours_2018` and
  `results/yulara_neighbours_2018_gfs` (`scripts/nwp_ahse_chain.sh`).

## Record of a corrected error

`metrics_nwp_1h_v1_instant_clearsky.csv` is the first run, in which the GFS clearness index
divided the 3-hour mean radiation by the **instantaneous** clear-sky GHI at the target hour. This
distorts mornings and evenings (GFS alone: MAE 0.271, R² -1.25; best correlation with the
observations at a 2-hour lag). The GFS timing itself was verified to be correct: its mean diurnal
profile matches the clear-sky mean of each 3-hour window. LightGBM + GFS was already ahead in v1
(MAE 0.101).

## Full AHSE pipeline with and without GFS (same period, seed 42)

`scripts/nwp_ahse_chain.sh` -> `results/yulara_neighbours_2018` (A, no GFS) and
`results/yulara_neighbours_2018_gfs` (B). In B the tree models receive the GFS covariates and the
GFS clearness index is a candidate member; the deep-learning members are A's checkpoints.
Significance: losses averaged per issue day, day-bootstrap 95% interval and Diebold-Mariano with
Newey-West variance (`scripts/compare_nwp_runs.py`).

| Test MAE | A (no GFS) | B (GFS) | Gain B vs A [95% CI], DM |
|---|---|---|---|
| AHSE (blocks) | 0.1246 | **0.0968** | +0.0278 [+0.0216, +0.0344], 6.0 |
| AHSE_global (legacy selector) | 0.1227 | 0.1065 | +0.0162 [+0.0107, +0.0222], 4.6 |
| XGBoost | 0.1320 | 0.0954 | +0.0366 [+0.0301, +0.0435], 8.4 |
| LightGBM | 0.1334 | 0.0978 | +0.0357 [+0.0291, +0.0426], 8.3 |

Within runs (`within_run_tests.json`): without GFS both AHSE variants beat the best single model
significantly (blocks vs global not significantly different, p = 0.34); with GFS, AHSE (blocks)
beats AHSE_global (+0.0097, DM 5.9) but is slightly behind XGBoost alone (-0.0014, p = 0.002).
Cause identified: DSWRF of 2018-03 was missing from the archive (a downloader bug, fixed); that
month lies in the validation period, where LightGBM without GFS input has MAE 0.30, so the
selector under-rated the GFS-fed trees. The full-period run with a complete archive
(`scripts/nwp_ahse_full.sh`, `results/yulara_neighbours_gfs`) supersedes this comparison.

Conformal intervals (AHSE, out-of-fold calibration, 90% nominal, test):

| | coverage | mean width | interval score |
|---|---|---|---|
| A, split Mondrian | 0.911 | 0.578 | 0.985 |
| A, ACI | 0.902 | 0.553 | 0.989 |
| B, split Mondrian | 0.923 | 0.520 | 0.830 |
| B, ACI | 0.905 | **0.452** | 0.824 |
| B, split binned by the GFS clearness index | 0.927 | 0.516 | **0.789** |

GFS narrows the 90% intervals by about 18% at the same coverage (ACI). Split calibration
over-covers in B because validation residuals include the month without GFS; ACI corrects it.
Binning by the GFS forecast raises coverage on variable / overcast days (0.81 / 0.72 against
0.78 / 0.69 with predicted-k_t bins).

## Final comparison: full period 2016-04 to 2020-05, complete GFS archive

`scripts/nwp_ahse_full.sh`: `results/yulara_neighbours` (A, main run without GFS) against
`results/yulara_neighbours_gfs` (B, same data, split and seed; deep-learning checkpoints of A,
tree models retrained with GFS). Test set: 6,650 forecasts.

| Test | MAE A | MAE B | R² A | R² B | Gain [95% CI], DM |
|---|---|---|---|---|---|
| AHSE (blocks) | 0.1073 | **0.0914** | 0.506 | **0.651** | +0.0159 [+0.0114, +0.0206], 4.8 |
| XGBoost | 0.1199 | 0.0914 | 0.471 | 0.651 | +0.0285 [+0.0240, +0.0331], 8.4 |
| LightGBM | 0.1223 | 0.0943 | 0.448 | 0.633 | +0.0280 [+0.0235, +0.0327], 7.8 |
| Day-ahead persistence | 0.1203 | 0.1203 | 0.216 | 0.216 | - |

- Without GFS the selector combines several members and beats the best single model
  (N-HiTS, +0.0103, DM 8.0). With a complete GFS archive one member, the GFS-fed XGBoost,
  dominates and the statistical guard keeps it on the 1-6 h and 6-24 h blocks: AHSE equals the
  best member instead of degrading it. The protocol does not invent gains when the pool offers
  none.
- 90% conformal intervals for AHSE: coverage 0.902 / 0.907 (ACI / split), mean width 0.413 against
  0.495 without GFS (-17%). Binning by the GFS clearness index gives the best interval score
  (0.627) and the best coverage on variable days (0.82 against 0.77).

## Second site: NIST Ground array (Gaithersburg, MD; humid subtropical, Cfa)

Data: PVDAQ system 4902 = NIST_Ground_1 (271 kW DC, south-facing, 20 deg tilt), 1-min AC power and
array sensors, 2016-01-01 to 2017-10-10 (`scripts/fetch_pvdaq.py`, `convert_pvdaq_nist.py`,
`prepare_nist.py --weather-source array`; the horizontal pyranometer uses NIST's own conversion,
the series stops at the 2017-10 sensor change). GFS from GDEX (longitude-convention bug fixed
before this run). `scripts/nist_pvdaq_chain.sh` -> `results/nist` (A) and `results/nist_gfs` (B),
1 h, seed 42. Test: 3,069 forecasts (2017).

| Test | MAE A | MAE B | R² A | R² B | Gain [95% CI], DM |
|---|---|---|---|---|---|
| AHSE (blocks) | 0.2252 | **0.1600** | 0.500 | **0.728** | +0.0652 [+0.0526, +0.0790], 6.6 |
| XGBoost (= AHSE_global) | 0.2196 | 0.1618 | 0.507 | 0.722 | +0.0578 [+0.0452, +0.0718], 6.2 |
| LightGBM | 0.2226 | 0.1627 | 0.488 | 0.719 | +0.0599, 6.3 |
| Day-ahead persistence | 0.2598 | 0.2598 | 0.195 | 0.195 | - |

- GFS reduces AHSE's MAE by 29% at NIST (15% at Yulara): weather forecasts matter more in a
  cloudier climate, where k_t is far less persistent.
- Without GFS, the k_t MAE puts AHSE (blocks) significantly behind XGBoost alone (-0.0056, DM
  -3.9). **Correction (2026-10-02):** this is an artefact of the k_t metric, which weighs dawn and
  dusk hours (small power, noisy k_t) like midday hours. In power units AHSE is not behind
  XGBoost (27.55 vs 27.84 kW, n.s.) and beats LightGBM (DM 3.0); see "All results in power
  units" below. The earlier explanation (seasonal shift between validation and test) was tested
  with a season-balanced validation and is not supported: that validation does not improve the
  test error. With GFS, AHSE beats XGBoost in k_t (+0.0018, DM 2.2) and equals it in power.
- 90% conformal intervals (B): coverage 0.910 (ACI) / 0.921 (split), width 0.750 / 0.790; binning
  by the GFS clearness index gives the narrowest calibrated intervals (0.907, width 0.690, best
  interval score).
- The fitted clear-sky envelope finds tilt 35 deg (true 20 deg) with median error 6%: the
  orientation fit is only a power-envelope fit, not an identification of the geometry.

## Season-balanced validation (sensitivity analysis)

`data_split.validation: interleaved` (every 4th week of the train+val span is validation, embargo
on mixed windows; test block unchanged), `scripts/seasonal_val_chain.sh`, compared in power with
`scripts/compare_runs_power.py` (P_cs differs between the two validations).

| AHSE test MAE (kW) | chronological | seasonal | difference |
|---|---|---|---|
| Yulara, no GFS | **36.53** | 38.11 | -1.59 [-2.61, -0.54], p = 0.015 |
| Yulara, GFS | 31.38 | 30.86 | +0.52, n.s. |
| NIST, no GFS | **27.55** | 29.16 | -1.61 [-2.64, -0.67], p = 0.05 |
| NIST, GFS | 20.39 | 20.33 | n.s. |

Seasonal validation improves the single tree models (which have no early stopping in the
multi-output setting: interleaving moves training weeks up to the start of the test period, i.e.
more recent training data) but
the selector then keeps XGBoost alone and AHSE loses the benefit of combining members; with GFS
it changes nothing significant. Chronological validation stays the protocol; 90% conformal
intervals are equivalent (Yulara GFS, ACI: 0.907 / width 0.418 vs 0.902 / 0.413).

## All results in power units (primary metric)

`scripts/power_metrics.py` -> `results/power_summary/`: k_t forecasts x the run's P_cs, compared
with the measured power on daytime targets; nMAE = MAE / DC capacity (Yulara uncurtailed sites
768 kW, NIST Ground 271 kW). Yulara: mean +/- std over 5 seeds (42, 1-4); NIST: seed 42.

| Test, AHSE | MAE no GFS (kW) | MAE GFS (kW) | nMAE no GFS -> GFS | skill vs day-ahead persistence |
|---|---|---|---|---|
| Yulara | 36.18 +/- 0.32 | 31.32 +/- 0.05 | 4.71% -> 4.08% | 0.36 -> 0.54 |
| NIST | 27.55 | 20.39 | 10.17% -> 7.52% | 0.41 -> 0.68 |

- GFS gain for AHSE: Yulara +4.4 to +5.2 kW for every seed (DM 3.2-3.8); NIST +7.2 kW
  [+5.5, +8.9] (DM 5.7).
- Without GFS at Yulara, AHSE beats the best single model of every seed by 2.8 to 3.9 kW
  (-7% to -10%, DM 5.0-7.2). With GFS it equals the GFS-fed XGBoost (differences <= 0.05 kW).
  At NIST it equals XGBoost (n.s.) with and without GFS.
- In power, day-ahead persistence has a lower MAE (40.6 kW) than XGBoost and LightGBM without GFS
  at Yulara (41.3, 42.1 kW) but a much larger RMSE (78.9 vs 66 kW): in a sunny desert, k_t is very
  persistent and MAE alone flatters persistence; report MAE, RMSE and the MSE-based skill score.
