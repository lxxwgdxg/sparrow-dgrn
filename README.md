# Code and Reproducibility Package (`03_Code_and_Reproducibility`)

**Manuscript Title:** *Preventing upstream spatial compensation in single-dry-year cross-basin streamflow routing via a SPARROW-informed differentiable graph network*  
**Authors:** Xue Li, Xiao-Yi Liu, Li Wang, and Jian Sha*  
**Journal:** *Journal of Hydrology*

---

## 1. Overview and Three-Layer Architecture

Because raw daily meteorological and hydrological station records from the *Hydrological Yearbook of the People's Republic of China* (Vol. 4, Yellow River Basin) are subject to national data-redistribution licenses, this reproducibility package is organized into a **three-layer architecture** that enables full methodological verification and empirical audit without redistributing restricted raw station records:

1. **Layer 1 — Open River-Network Topologies (`data/layer1_open_topologies/`)**:
   - Exact HydroSHEDS/HydroBASINS Level-12 directed acyclic graph (DAG) topologies (`topology.csv`), 10 static subbasin and reach attributes (`static_attributes.csv`), and basin metadata (`metadata.json`) for all eight Loess Plateau study basins (`jing`, `beiluo`, `shiwang`, `fenchuan`, `hulu`, `xinshui`, `wuding`, and `yanhe`).
   - Enables bit-exact verification of incremental drainage areas $A_j$, cumulative upstream drainage areas $A_j^{\text{up}}$, upstream routed-area fractions $\psi_j = (A_j^{\text{up}} - A_j)/A_j^{\text{up}} \in [0, 1)$, cumulative drainage shares $\eta_j = A_j^{\text{up}} / A_{\text{basin}} \in (0, 1]$, confluence-anchored corridor thresholds $\eta_{\text{thr}}(\mathcal{G}) > \eta_{\max,\text{trib}}(\mathcal{G})$, unit-mean corridor weights $\phi_{\text{scale},j}$, and upstream-aggregated internal gauge exposures $\bar{\phi}_{\text{scale}}(\mathcal{U}_g)$.
2. **Layer 2 — Surrogate Daily Benchmark (`data/layer2_surrogate_benchmark/`)**:
   - Self-contained daily hydroclimatic and hydrological surrogate time series (`surrogate_daily_benchmark.npz` and `gauges.csv`, 2006–2021, $T = 5{,}844$ days) on the real 11-node Fenchuan ($N=11$) and 35-node Hulu ($N=35$) river graphs.
   - Executes the complete 3-stage SPARROW-DGRN + SDKR-3T operator and Source-Pretrained GR4J + SDKR-3T baseline end to end, verifying machine-precision 4-tier water-balance closure, rank-3 basis commutation, and the $2 \times 2$ factorial separation of outlet volume recovery from upstream spatial compensation.
3. **Layer 3 — Verified Empirical Audit Ledgers (`data/layer3_verified_audit_tables/`)**:
   - Complete de-identified run-level and summary-level CSV audit tables (33 CSV files) covering all eight Loess Plateau basins across both meteorological forcings (local CMA station precipitation + Hargreaves PET vs. public $0.1^\circ$ `CHM_PRE` v1 daily precipitation + ChinaMet Penman–Monteith PET) and all experimental tiers (Tiers 1–3, $2 \times 2$ factorial ablation, `Uniform-Vol-Matched` control, `KDRD-3T`, `Source-Pretrained GR4J + SDKR-3T`, five Budyko prior regimes, four single-year calibration windows 2009–2012, and synthetic identifiability audits).

---

## 2. Exact Alignment Between Code (`src/sparrow_dgrn_sdkr3t.py`) and Manuscript Equations

The reference implementation in `src/sparrow_dgrn_sdkr3t.py` matches the manuscript and supplementary equations:
- **Stage 1 (Causal Soil-Moisture Accounting, Eq. 1a)**: Implements the availability-bounded causal soil-moisture operator $\mathcal{S}_{\text{soil}}(P_{j,t}, \text{PET}_{j,t}, \alpha_j u_{j,t}^0, w_{j,t}^{\text{ref}})$ (and supports capacity-bounded surrogate simulation when `w_ref_mm=None`), satisfying daily Tier-1 soil water-balance closure $P_{j,t} - e_{j,t} - u_{j,t} - (z_{j,t} - z_{j,t-1}) = 0$.
- **Stage 2 (Unit-Mass Three-Timescale Delivery Kernel, Eq. 2 & Eq. S2)**:
  - Profile ordinary least-squares (OLS) search over $\tau_{\text{base}} \in [1.0, 365.0]\text{ d}$ with daily recession rate $r = 1 / \tau_{\text{base}}$ and geometric kernel $g_{\tau,\text{base}}(\ell) \propto r(1-r)^{\ell-1}$.
  - Fixed corridor slow-recession timescale $\tau_{\text{corr}} = 60.0\text{ d}$ (`TAU_CORR_DAYS = 60.0`).
  - Upstream routed-area fraction $\psi_j = (A_j^{\text{up}} - A_j)/A_j^{\text{up}} \in [0, 1)$, bounded lag and slow weights $\phi_{\text{lag},j} = \phi_{\text{slow},j} = \psi_j$, and unit-mean confluence-anchored corridor basis $\phi_{\text{scale},j} = \psi_j \max(0, \eta_j - \eta_{\text{thr}}(\mathcal{G})) / \sum_{k=1}^N (A_k / A_{\text{basin}}) \psi_k \max(0, \eta_k - \eta_{\text{thr}}(\mathcal{G}))$ with $\eta_{\text{thr}}(\mathcal{G}) = 0.67$ for $N \le 15$ and $0.28$ for $N > 15$ (Eq. S2).
  - Non-negativity feasibility guard $\delta \in [0, \min(f_{\text{base}}, \Delta\tau_{\max})]$ with $\Delta\tau_{\max} = 0.35\text{ d}$ ensuring $h_j(\ell) \ge 0$ for all $(j,\ell)$.
- **Stage 3 (Directed Reach-Reservoir Routing, Eq. 1)**: Implements causal linear time-invariant reach routing with daily storage release fraction $\rho_j = 1 - \exp(-\Delta t / \tau_{\text{reach},j}) \in (0, 1)$ and same-day pass-through fraction $\kappa_j \in [0, \rho_j]$, satisfying exact rank-3 spatio-temporal commutation with Stage 2 (Eq. S1).
- **Five-Regime Budyko Corridor Prior (Section 2.3 & Supplement Section S2.2, Eq. S3)**:
  - `compute_budyko_prior_from_inputs()` implements both the zero-tuning **Flat Source-Only Prior** (`Regime_III`, $C_{\text{clim}}^*(b) \equiv C_{\text{clim},0}^* = 0.0365$, derived from the pre-2010 Jing and Beiluo source-basin mean) and the **Regime-Stratified Prior** (`Regime_IA`, Eq. S3), reproducing all 64 rows of `exact_prior_parameterization_audit.csv`.

---

## 3. One-Command Verification

Requires Python 3.9+ with `numpy`, `scipy`, and `pandas`. Run:

```bash
python run_reproducibility_suite.py
```

The script executes all four verification checks (`[Check 1]` through `[Check 4]`) and exits with code `0`.
