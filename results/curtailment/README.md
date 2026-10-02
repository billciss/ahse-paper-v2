# Yulara: curtailment of the Desert Gardens site

Analyses run on 2026-09-30 on the DKASC per-site 5-min downloads (April 2016 – February 2026).
Everything below is **inferred from the data**; the operator has not confirmed the control logic
(a data request to the DKA Solar Centre is pending).

## 1. What the original paper's series is

`data/raw/data_5min.csv` (2020–2023), used for all Yulara results of the AHSE paper, is **exactly
the Desert Gardens site** (1,058.4 kW DC), not the plant total: identical on 139,193 daytime rows,
least-squares weight 1.0 against 0 for every other site. Its measured GHI is the shared station.

## 2. Only Desert Gardens is curtailed

| Evidence | Numbers | Script / table |
|---|---|---|
| Output relative to the other sites, high irradiance, normalised to 08–09 h | Desert Gardens falls to **0.37 at noon**; Service Station, Sails, Laundry stay at 0.98–1.07 | `yulara_sites_analysis.py`, `site_to_reference_ratio_by_hour.csv` |
| Clear day 2023-06-04 | Desert Gardens 383 kW at 09 h → **170 kW at 12 h** → 325 kW at 16 h; plant total held at **615–630 kW from 11 h to 15 h** | same |
| Clear-sky instants below 80% of expected power (plant total, 2016–2026) | 78.9% overall, 94–96% at 11–13 h, 82–91% Apr–Oct, 61–63% Jan–Mar; every year between 56% and 94% | `quantify_curtailment.py`, `total_2016_2026_curtailed_share_month_by_hour.csv` |
| Load dependence | less curtailment on very hot days (66% when Tmax > 35 °C vs 82–90% otherwise), no weekday effect | same |

Interpretation: in the off-grid mini-grid (gas generation), total PV appears capped around noon
(presumably to respect minimum generator loading), and **Desert Gardens is the controllable site**
that absorbs the adjustment. Connellan Airport's afternoon decline is an orientation effect, not
curtailment (monotone, no midday dip).

## 3. Desert Gardens available power (reconstruction)

`censored_available_power.py` treats curtailment as censoring (delivered = min(available, cap)):

- clear-sky-index transfer from the four uncurtailed neighbours (NNLS weights, available
  neighbours only during outages) to a plane-of-array clear-sky envelope of Desert Gardens fitted
  on its uncurtailed rows; isotonic correction of Desert Gardens' different diffuse response;
- uncurtailed pool = before 09 h, from 17 h, or total < 350 kW under overcast, iteratively
  trimmed of points with delivered < 0.85 × available;
- leave-one-year-out validation on a fixed 07–08 h set.

| Metric (leave-one-year-out) | Value |
|---|---|
| MAE, all / above 400 kW | 30 kW (15.7%) / 28 kW (6.4%) |
| Midday instants with delivered > 1.05 × available + 5 kW | **0.8%** |
| Conformal 90% interval coverage, clear / variable / overcast | 0.89 / 0.90 / **0.77** |
| **Curtailed share of Desert Gardens' available energy** | **50.6%** (conservative bounds 31.7–59.1%) |
| By month | 36–39% Dec–Mar, 56–65% Apr–Sep |

Discarded variants and why (kept in the repository as a record):
`reconstruct_available_power.py` (LightGBM cannot extrapolate to midday output: 66% violations;
single-ratio model: 8.6% violations), `refine_available_power.py` (its "uncurtailed" pool contained
curtailed clear-sky instants at 09–15 h and 16 h).

## 4. Data quality events (`yulara_data_quality.py`, `site_data_quality.csv`)

- 2020-05-14 → 2020-11-06: the four neighbouring sites report zero while Desert Gardens operates;
- Oct 2021 → Jan 2022: Service Station and Laundry at zero;
- Jul → Sep 2025: intermittent zero output at all sites;
- 2024: 10–20% missing data depending on the site;
- multi-day gaps of GHI (2016) and temperature (2017–2018) in the weather station.

## 5. Consequences for the forecasting study

- The Desert Gardens series forecasts **dispatch-limited** power: a k_t target is not meaningful
  there (50% of daytime k_t at the 1.5 clip with the legacy envelope, 24% with the new one).
- The **sum of the four uncurtailed sites** (699 kW, `prepare_yulara_sites.py --target neighbours`,
  2016-04-02 → 2020-05-13) is a clean, weather-driven Yulara target: clear-sky envelope tilt 20°,
  north, median relative error 3.7%; 3.0% of daytime k_t at the clip.
- Open limitations: overcast interval coverage (0.77); no uncurtailed clear-sky midday instants
  exist for Desert Gardens, so midday accuracy of the reconstruction is only checked indirectly.
