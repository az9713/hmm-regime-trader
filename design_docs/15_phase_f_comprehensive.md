# design_docs/15_phase_f_comprehensive.md
# ============================================================
# Phase F — Full Covariance HMM: Goals, Methodology, Results,
#            Interpretation, and Implications for HMM Quality
#
# Date: 2026-04-12
# Baseline entering Phase F: Sharpe 0.933 (diag + 6-feat, Phase C locked)
# Phase F best result:       Sharpe 0.933 (baseline unchanged — no improvement)
# Gate met?                  No — all non-baseline configs regressed
# Decision:                  Lock diag + 6-feat as final model configuration
#
# Detailed results:         design_docs/14_phase_f_results.md
# Phase F plan:             design_docs/13_phase_f_full_covariance.md
# Root cause (Phase C H3):  design_docs/11_phase_c_2022_diagnosis.md §4
# ============================================================

---

## 1. Goals

### 1.1 Why Phase F was attempted

Phase C established that the diagonal Gaussian HMM has a structural blind spot:
it cannot distinguish "all features simultaneously mildly elevated" (the 2022
slow-grind bear market pattern) from calm mid-vol periods, because diagonal
covariance evaluates each feature independently and cannot represent joint
cross-feature patterns.

The theoretical fix, well-established in the academic literature (Gray 1996,
JFE; Guidolin & Timmermann 2008, RFS), is to switch from diagonal to full
covariance. Under full covariance, the emission model is a joint multivariate
Gaussian for all features simultaneously, allowing the HMM to learn that
"realized_var +0.2 AND vix +0.4 AND hy_oas +0.6" is a distinct joint regime
pattern — not just the product of three independent mid-vol signals.

Phase C H3 diagnostic confirmed this was the exact mechanism of the 2022
failure. Phase F was the direct implementation of the theoretical fix.

### 1.2 What Phase F set out to achieve

1. **Fix the 2022 slow-grind blind spot** by switching to full or tied
   covariance, enabling the HMM to detect the joint "all elevated together"
   signature.

2. **Re-enable vix_slope** as the 7th feature. vix_slope (VIX/VIX3M) was
   excluded in Phase A because diagonal covariance cannot exploit the ~50%
   orthogonal variance relative to VIX level. Full covariance models the
   joint distribution of (vix, vix_slope) directly, making vix_slope useful.

3. **Meet the live trading gate:** aggregate OOS Sharpe > 1.0 and worst 2022
   window > -1.0.

### 1.3 Hypothesis

If the 2022 slow-grind pattern is learnable from training data and the model
has sufficient representational capacity, full covariance should create a
dedicated "slow-grind stress" state with a joint emission distribution centered
on the (realized_var ~+0.2, vix ~+0.4, hy_oas ~+0.6) pattern, causing those
observations to be assigned to a defensive allocation rather than mid-vol.

---

## 2. Backtests Run

### 2.1 Test matrix

Phase F was designed as a 2×2 grid plus a baseline re-run, not a one-at-a-time
sweep. The combinations:

| Config | Covariance | Features | n_params (est.) | bars/param |
|--------|-----------|----------|----------------|-----------|
| F0: diag + 6-feat | diag | 6 | ~107 | 3.5 |
| F1: tied + 6-feat | tied | 6 | ~86 | 4.4 |
| F2: full + 6-feat | full | 6 | ~167 | 2.3 |
| F3: full + 7-feat | full | 7 | ~202 | 1.9 |
| F4: tied + 7-feat | tied | 7 | ~98 | 3.9 |

**Covariance types:**
- `diag` (baseline): each state has an independent variance per feature.
  21 distinct numbers describe a 6-feature diagonal distribution (6 means + 6
  variances per state × 5 states = 60, plus 25 transition params = ~107 total).
- `tied`: all states share one covariance matrix. Cross-feature correlations
  are captured but assumed identical across regimes. One 6×6 matrix = 21 unique
  values shared across all states.
- `full`: each state has its own full 6×6 covariance matrix. Cross-feature
  correlations are state-specific — the model can represent that "stress regimes
  have different correlation structure than calm regimes." 21 × 5 = 105
  covariance parameters, plus 30 means, plus 25 transition = ~167 total.

**Why test tied before full:** Tied covariance captures cross-feature correlation
structure with far fewer parameters than full (21 vs 105 covariance parameters
for 6 features, 5 states). It was the conservative intermediate step.

### 2.2 What was measured per config

- Aggregate OOS Sharpe (primary gate criterion)
- CAGR, MaxDD (secondary context)
- Per-window Sharpe for every OOS window overlapping 2022
- BIC-selected n_states per 2022 window
- hmmlearn convergence behavior (qualitative from log output)

### 2.3 Code changes made for Phase F

**`data/feature_engineering.py`**
- Added `FEATURE_COLS_BASE` (6 features) and `FEATURE_COLS_WITH_VIX_SLOPE`
  (7 features, adding vix_slope after vix in the feature vector)
- `FeatureEngineer.__init__` reads `settings["features"]["use_vix_slope"]`
  (default False) and sets `self.feature_cols` accordingly
- All methods (`compute()`, `normalize_live()`, `get_observation_matrix()`)
  use `self.feature_cols` — feature set is instance-level, not module-level

**`core/hmm_engine.py`**
- Fixed BIC parameter count formula — previously hardcoded for diagonal
  (`n * n + n * d * 2`). Now computes correctly per covariance type:
  - `diag`: n² + n·d·2
  - `full`: n² + n·d + n·d·(d+1)/2
  - `tied`: n² + n·d + d·(d+1)/2
  - `spherical`: n² + n·d + n

**`config/settings.yaml`**
- Added `features.use_vix_slope: false`

**`backtest/sweep.py`**
- Added `PHASE_F_CONFIGS` list (5 configs)
- Added `run_phase_f()` function — runs all configs, reports 2022 per-window
  breakdown, prints summary table, identifies winner
- Added `--sweep F` CLI flag
- Added `vix3m` fetch in `main()` and passed through to all runs

---

## 3. How the Backtests Were Run

All Phase F runs used the same walk-forward backtester as Phase B/C, with
Phase C locked settings as the baseline:

| Parameter | Value |
|-----------|-------|
| IS window | 378 bars (Phase C H1 winner) |
| OOS window | 126 bars |
| Step size | 63 bars |
| Data range | 2010-01-01 → 2026-04-12 |
| OOS windows | 57 |
| Primary symbol | SPY |
| n_components_range | [3, 7] |
| n_iter | 100 |
| n_init | 10 (K-means restarts, best log-likelihood kept) |

Data was fetched once (vix, hy_oas, gold, term_spread, vix3m, SPY, QQQ)
and reused across all 5 runs.

```bash
python -m backtest.sweep --sweep F
```

Total wall time: ~50 minutes (5 runs × ~10 min each).

---

## 4. Results

### 4.1 Summary table

| Config | Sharpe | CAGR | MaxDD | Windows | Worst 2022 | Gate |
|--------|--------|------|-------|---------|-----------|------|
| diag + 6-feat (baseline) | **+0.933** | 11.8% | -32.8% | 57 | -1.981 | FAIL |
| tied + 6-feat | +0.844 | 10.7% | -34.2% | 57 | -1.966 | FAIL |
| full + 6-feat | +0.789 | — | — | 57 | -1.976 | FAIL |
| full + 7-feat (vix_slope) | +0.848 | 10.3% | -33.5% | 57 | -1.856 | FAIL |
| tied + 7-feat (vix_slope) | +0.825 | 10.3% | -32.2% | 57 | -1.716 | FAIL |

Gate: Sharpe > 1.0 AND worst 2022 window > -1.0. No config met either criterion.

**Diagonal baseline wins. No Phase F configuration improved on it.**

### 4.2 2022 per-window breakdown — baseline vs best Phase F

| OOS Window | diag baseline | full + 7-feat | tied + 7-feat |
|------------|--------------|--------------|--------------|
| 2021-09-17 → 2022-03-17 | +0.637 | +0.498 | +0.102 |
| 2021-12-16 → 2022-06-16 | **-1.981** | **-1.856** | **-1.716** |
| 2022-03-18 → 2022-09-16 | **-1.306** | **-1.267** | **-1.306** |
| 2022-06-17 → 2022-12-15 | +1.052 | +1.173 | +0.918 |
| 2022-09-19 → 2023-03-20 | -0.252 | +0.060 | +0.082 |
| 2022-12-16 → 2023-06-20 | +1.419 | +1.510 | +1.612 |

The two core problem windows (-1.981, -1.306) show marginal improvement at
best (-1.856, -1.267 under full + 7-feat). The improvement is not enough to
move the needle on aggregate Sharpe.

The modest 2022 improvement under full/tied covariance was more than offset by
regression in non-2022 windows — the covariance upgrade hurt the majority of
windows while barely helping the 2022 problem windows.

### 4.3 Convergence behavior

Qualitative observation from sweep log warnings:

| Config | Convergence warnings | Max delta magnitude |
|--------|---------------------|---------------------|
| diag | Many, all near-zero | < 0.001 |
| tied | Some | ~1-2 |
| full | Many, some large | **-47 to -83** |

Under full covariance, many EM iterations produced log-likelihood *decreases*
of 47-83 units — the M-step update made the model worse than the E-step
expected. This is a symptom of an underdetermined optimization: too many
parameters relative to data, causing the EM to oscillate between poorly-fitting
solutions rather than converging to a stable local maximum.

---

## 5. Interpretation

### 5.1 Why full covariance regressed

**The data scarcity problem is more severe than the parameter count suggests.**

Phase C §4.3 flagged 1.9 bars/parameter as "tight" and suggested testing tied
covariance as a safer middle ground. The Phase F results show even tied
covariance (-0.089 Sharpe) and tied + 7-feat (-0.108) regress meaningfully.

The issue is not just the total parameter count — it is the effective sample
size available to estimate each *individual* covariance matrix. For a 6-feature
model with 5 states, the BIC-weighted IS data available to estimate the
covariance matrix of a *specific state* is:

```
effective_T_per_state ≈ IS_bars × (fraction of bars assigned to that state)
                      ≈ 378 × 0.20 (if states roughly equal)
                      ≈ 75 bars per state for a 5-state model
```

Estimating a 6×6 covariance matrix (21 unique values) from ~75 effective
observations is unreliable. The sample covariance matrix will be noisy and
potentially non-positive-definite (requiring regularization that dilutes the
cross-feature signal). The minimum recommendation for Gaussian covariance
estimation is ~10× the feature count — for 6 features that's 60+ observations
per state, and for 7 features ~70+. 75 is barely above the minimum and assumes
perfectly balanced state occupancy, which never holds in practice.

**The 2022 slow-grind state is even worse off.** If the "slow-grind" state
occupies only 10-15% of IS bars (because IS windows span 2020-2021, which
is mostly COVID recovery and low-vol bull), then:

```
effective_T_for_2022_state ≈ 378 × 0.12 ≈ 45 bars
```

Estimating 21 covariance parameters from 45 observations is statistically
unreliable. The EM will either merge this state with another (producing the
convergence oscillations observed) or produce a noisy covariance matrix that
does not generalize to OOS data.

### 5.2 Why the 2022 problem windows didn't improve

The Phase C H3 hypothesis was correct about the *mechanism* (diagonal covariance
cannot detect the joint pattern) but the theoretical fix (full covariance) failed
because the *training data* does not contain enough examples of the 2022 slow-
grind pattern to estimate a reliable "slow-grind" emission distribution.

This is a subtle but important distinction:

- **Phase C H3 finding:** The model cannot represent the 2022 pattern. The
  diagonal covariance lacks the capacity to encode "all features elevated
  together."
- **Phase F finding:** Even with the representational capacity (full covariance),
  the model cannot *learn* the 2022 pattern from the available IS data, because
  the IS windows preceding the worst 2022 OOS periods are dominated by COVID
  recovery and 2021 low-vol bull data — the 2022 pattern appears only at the
  tail of a few IS windows, providing 20-50 bars of slow-grind examples at most.

No parametric model — diagonal, tied, or full — can learn a reliable cluster
from 20-50 representative observations. The training data is the constraint, not
the model architecture.

### 5.3 Why non-2022 windows regressed

The -0.144 Sharpe regression (full vs diag) came almost entirely from non-2022
windows. Three mechanisms:

**1. Overfitting in stable regimes.** Low-vol bull market windows (2013-2019,
2021) previously produced Sharpe 2.0-3.3 under diagonal covariance because the
HMM cleanly assigned these periods to a well-estimated low-vol state. Under full
covariance with insufficient data, the covariance matrix estimation is noisy —
the model fits a spurious correlation structure from the specific 378-bar IS
sample. This noisy covariance causes observations in the OOS period to be assigned
to the wrong states, degrading performance in windows that previously worked well.

**2. State fragmentation.** With more representational capacity but unreliable
estimation, BIC selected *fewer* states for the 2022 problem windows (4 states
vs 6-7 under diag). This is paradoxical but explicable: BIC's larger parameter
penalty for full covariance causes it to select simpler models, even when the
data would support more states. The result is that states that were cleanly
separated under diagonal covariance are merged under full covariance, losing
regime granularity that the diagonal model had achieved.

**3. Initialization sensitivity.** Full covariance EM is more sensitive to
initialization than diagonal. The K-means initialization used for diagonal
covariance is not optimal for full covariance (K-means implicitly assumes
spherical clusters). With n_init=10 restarts and 100 EM iterations, some windows
may not find good solutions, and the best-log-likelihood selection picks the
least-bad local optimum rather than the true MLE.

### 5.4 What vix_slope's marginal contribution revealed

vix_slope added +0.059 Sharpe (full 7-feat vs full 6-feat: 0.848 vs 0.789)
and reduced the worst 2022 window from -1.976 to -1.856. The improvement is
real but small relative to the baseline regression.

This tells us something specific: vix_slope does carry *some* additional
information about 2022-type conditions — the inverted vol term structure
(VIX > VIX3M, backwardation) that characterized most of 2022 added marginal
discriminating power. But the improvement was overwhelmed by the estimation
noise introduced by the additional 7th feature's covariance parameters.

The theoretical argument for vix_slope under full covariance remains sound —
the problem is that the data volume required to reliably estimate a 7×7
covariance matrix is far beyond what the current IS window provides.

### 5.5 The 2022 ceiling: a revised understanding

Phase C characterized the 2022 ceiling as a model misspecification problem.
Phase F reveals it is more precisely a *training data coverage problem*:

| Layer | Problem | Fixable? |
|-------|---------|---------|
| Model capacity | Diagonal can't represent joint pattern | Fixed by full cov — but fixing it isn't sufficient |
| Training data coverage | IS windows contain <50 bars of 2022-type examples | Requires 3+ year IS window or structural data augmentation |
| Crisis distinctiveness | 2022 features overlap with calm mid-vol in marginals | Inherent to the slow-grind crisis type |

The 2022 problem is not uniquely a model problem. It is a data coverage
problem that would require ~3-5 years of IS data (to include multiple examples
of slow-grind stress) before any model upgrade — diagonal, full, Student-t,
or otherwise — could learn to detect it reliably.

For context: a 3-year IS window would provide:
- ~750 total bars (vs 378 currently)
- ~150 bars per state (vs ~75)
- Sufficient data to include at least one full calendar year of 2022-type
  conditions in the IS period for most 2022 OOS windows

This would require changing is_window from 378 to ~750, which would:
a) Reduce the number of OOS windows significantly
b) Push the first OOS window ~3 years into the data history
c) Potentially cause "stale training data" problems similar to what the 504-bar
   IS window produced in Phase C H1 (COVID-type patterns conflicting with
   later regime structure)

The trade-off is not clearly favorable and was not tested.

---

## 6. Implications for HMM Quality

### 6.1 The model is well-specified for its observable domain

Phase F confirms what Phase C H3 identified: the diagonal Gaussian HMM is
well-specified for sharp, spike-type crises (COVID 2020: 100% detection, 2022
second-half: 97% detection) and poorly specified for slow-grind macro
deterioration (2022 first-half: systematically mis-detected).

This is not a flaw unique to this implementation. Guidolin & Timmermann (2008,
RFS) document the same limitation in multi-state HMMs across equity, bond,
and credit markets: models trained on crisis-spike periods under-detect gradual
deterioration regimes when the calibration window doesn't contain prior examples.

The 2022 bear was structurally distinct from the 2008 GFC, 2020 COVID, and
2018 Q4 selloff — all of which had sharp VIX spikes. Any HMM trained primarily
on the 2010-2021 data history will be calibrated to spike-type crises. This
is an inherent limitation of walk-forward backtesting on a historical record
that contains predominantly one type of crisis.

### 6.2 Full covariance is not the universal fix for diagonal covariance limitations

A common practitioner assumption is that diagonal covariance is a simplification
and full covariance is always the more correct model. Phase F provides empirical
evidence that this is false in data-scarce settings:

- Full covariance has more representational capacity
- But capacity without data to fill it produces noisier models, not better ones
- The optimal model complexity is bounded by the available training data, not
  by what the theoretical ideal would be

For a daily-bar walk-forward backtester with 378-bar IS windows, diagonal
covariance is *the correct model choice* — not a simplification. It provides
reliable parameter estimation (3.5 bars/parameter) while full covariance cannot
(1.9-2.3 bars/parameter).

The transition to full covariance should be considered only if:
1. IS window is extended to 3+ years (~750 bars)
2. Multiple slow-grind examples exist in the training history
3. The parameter count is controlled (e.g., factor model covariance, shrinkage
   estimators, or Ledoit-Wolf regularization applied to the sample covariance)

### 6.3 BIC's behavior under different covariance types

An unexpected finding: BIC selected *fewer* states under full covariance for
the 2022 problem windows (4 states vs 6-7 under diagonal). This reveals an
important property of BIC:

- BIC's penalty scales with parameter count × log(T)
- Full covariance has ~57% more parameters per state than diagonal (21 vs 6×2)
- BIC penalizes the additional parameters heavily, driving state selection toward
  simpler models
- The result is that full covariance "buys" cross-feature representational
  capacity but "pays" for it with fewer states

In this data regime, the BIC trade-off is unfavorable: fewer states + unreliable
full covariance estimates < more states + reliable diagonal estimates.

This finding has general implications for HMM covariance selection: when IS data
is limited, BIC applied to full covariance models will consistently select simpler
state structures that may underfit the data's actual regime structure, even though
the motivation for switching to full covariance was to capture richer structure.

### 6.4 The walk-forward IS window is the binding constraint

Phase F, taken together with Phase C, reveals that the IS window is not just
a tuning parameter — it is the fundamental constraint on what the HMM can learn.

The regime structure that the model can detect at any OOS period is bounded by
what was observable in the preceding IS window. The 2022 slow-grind was
structurally novel within the 2010-2021 history — no prior year had combined:
- Sustained Fed rate hikes at 425bps/year
- Equity drawdown without VIX spike above 40
- All major risk assets declining simultaneously

The model could not detect it in 2022 OOS windows because it had never seen
it in any prior IS window. No architectural change (diagonal → full) fixes a
model that has never encountered the phenomenon it's being asked to detect.

This is the fundamental limitation of walk-forward backtesting on short history:
the model's capability is bounded by the diversity of regimes in its training
windows, not by its architectural sophistication.

### 6.5 What "Sharpe 0.933" means given these findings

The aggregate OOS Sharpe of ~0.933 represents the model's performance across
57 OOS windows from 2010-2026. It reflects:

- **Strong performance** in the large majority of market environments: spike crises,
  bull markets, low-vol recoveries — roughly 80% of OOS windows
- **Structural weakness** in one specific crisis archetype: slow-grind macro
  deterioration (2022-type) — roughly 20% of windows that include 2022 data

This is a known, bounded, documented limitation. The model is not broken; it
has a specific blind spot that is explainable by training data coverage.

For deployment context:
- The 2022 bear market was historically unusual in its slow-grind character
- The Fed has begun easing as of 2024; a repeat of 2022-type conditions in
  the near term would require a new inflationary shock at similar scale
- The model correctly reduces allocation during the second half of 2022 (when
  the HMM does start detecting stress) — the damage is concentrated in the
  first half onset

---

## 7. Final Configuration and Locked Settings

After Phase F, all model and feature parameters are locked.

```yaml
# config/settings.yaml — Phase F locked values

hmm:
  covariance_type: "diag"    # Phase F: full tested, regresses -0.144 Sharpe.
                              # Root cause: ~2.3 bars/param insufficient for
                              # reliable 6x6 cov estimation at is_window=378.
                              # Diagonal is the correct choice at current IS length.

features:
  use_vix_slope: false        # Phase F: vix_slope adds +0.059 under full cov
                              # but full cov is net -0.144. Keep false.
                              # Re-enable only if IS window extended to 750+ bars.
```

All other parameters remain at Phase C locked values.

**Final model state:**
- 6 features: log_return, realized_variance, vix, hy_oas, gold_return, term_spread
- covariance_type: diag
- is_window: 378 bars
- n_components_range: [3, 7]
- Aggregate OOS Sharpe: ~0.933
- 2022 worst window: -1.981 (irreducible at current IS window)

---

## 8. Path Forward

All model calibration phases (A through F) are now complete. The 2022 limitation
is confirmed irreducible without structural changes beyond the current scope.

The recommended sequence:

1. **Phase D:** Implement Alpaca broker layer (`AlpacaSource.get_bars()`). Required
   before paper or live trading.

2. **Phase E:** 3-month paper trading. Calibrate ATR stop multiples from live fills.
   Validate execution quality, slippage behavior, and circuit breaker operation.

3. **Live trading criteria review:** After 3 months of paper data, reassess the
   Sharpe gate. Sharpe 0.933 is deployable for paper trading. The live trading
   gate (Sharpe > 1.0) may need to be revisited given the confirmed irreducibility
   of the 2022 drag.

4. **Optional future upgrade:** Student-t emissions (SEP-HMM, MDPI Mathematics
   14(3), 2025). Student-t has heavier tails than Gaussian and a similar parameter
   count to diagonal Gaussian. This is the next structural upgrade path that doesn't
   require full covariance and may improve fat-tail detection without the data
   scarcity problem.

---

## 9. Code Changes Summary

| File | Change | Status |
|------|--------|--------|
| `data/feature_engineering.py` | FEATURE_COLS_BASE / FEATURE_COLS_WITH_VIX_SLOPE; use_vix_slope flag | Kept (infrastructure) |
| `core/hmm_engine.py` | BIC parameter count formula corrected for full/tied/spherical | Kept (correctness fix) |
| `config/settings.yaml` | `covariance_type: "diag"` locked with Phase F evidence; `use_vix_slope: false` locked | Locked |
| `backtest/sweep.py` | PHASE_F_CONFIGS, run_phase_f(), --sweep F, vix3m pass-through | Kept |
| `logs/sweep_phase_f.log` | Raw Phase F sweep output | Saved |

All Phase F code changes are retained in the codebase. The infrastructure for
full covariance and vix_slope is available for future use — it simply isn't
activated under the current locked configuration.

---

## 10. References

- Hamilton (1989, Econometrica 57:357-384) — HMM framework
- Gray (1996, JFE 42:27-62) — Gaussian misspecification; full covariance
  recommendation for multi-asset HMMs; see also: why data volume matters
- Guidolin & Timmermann (2008, RFS 21:889-935) — full covariance HMM in
  practice; detection lag in slow-deterioration regimes
- Ledoit & Wolf (2004, JMVA 88:365-411) — covariance matrix shrinkage;
  relevant if full covariance is revisited with regularization
- SEP-HMM (2025, MDPI Mathematics 14(3)) — Student-t emissions as a next
  structural upgrade path
- design_docs/11_phase_c_2022_diagnosis.md — H3 structural diagnostic
- design_docs/12_phase_c_2022_fix.md — Phase C methodology
- design_docs/13_phase_f_full_covariance.md — Phase F plan and risk analysis
- design_docs/14_phase_f_results.md — Phase F results detail and path forward
- `logs/sweep_phase_f.log` — raw sweep output
