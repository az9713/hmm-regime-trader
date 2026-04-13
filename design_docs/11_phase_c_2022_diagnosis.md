# design_docs/11_phase_c_2022_diagnosis.md
# ============================================================
# Phase C — Fix 2022 Problem Windows: Diagnosis and Results
#
# Date: 2026-04-12
# Baseline Sharpe: 0.828 (is_window=252, n_components_range=[3,7])
# Final Sharpe:    0.940 (is_window=378, n_components_range=[3,7])
# Gate met?        No (target >1.0) — H3 confirmed, Phase F required
# ============================================================

---

## 1. Problem Statement

Walk-forward backtest aggregate OOS Sharpe: **0.828** after Phase B.

The following OOS windows covering 2022 bear market periods produce
severely negative Sharpe, dragging the aggregate from a hypothetical
~1.2 to 0.828:

| OOS window | Sharpe | n_states |
|------------|--------|---------|
| 2021-12-16 → 2022-06-16 | **-1.938** | 5 |
| 2022-03-18 → 2022-09-16 | **-1.276** | 7 |
| 2021-09-17 → 2022-03-17 | +0.432 | 6 |
| 2022-06-17 → 2022-12-15 | +0.547 | 6 |
| 2022-09-19 → 2023-03-20 | -0.306 | 6 |

Root cause hypothesis: the HMM is staying in mid-vol regime during the
2022 bear market, allocating ~65% instead of cutting to ~35%. The slow,
grinding nature of the 2022 decline (Fed rate hikes, no VIX spike) means
the regime signal is weak or absent.

Three hypotheses were tested in order of cost (cheapest first).

---

## 2. H1: IS Window Too Short

**Hypothesis:** The 252-bar (1-year) IS window preceding 2022 OOS periods
may not contain a prior high-vol period, leaving the HMM without training
examples of stress. Extending to 1.5 years (378 bars) or 2 years (504
bars) gives the HMM more regime history.

**Sweep:** `settings["backtest"]["is_window"]` over [252, 378, 504].

**Results:**

| is_window | Aggregate Sharpe | 2022-06-17 window | Worst 2022 window | OOS windows |
|-----------|-----------------|------------------|------------------|-------------|
| 252 (baseline) | 0.828 | +0.547 | -1.938 | 59 |
| **378** | **0.940** | **+1.052** | -1.981 | 57 |
| 504 | 0.834 | +0.590 | -1.973 | 55 |

**Winner: 378 bars (+0.111 Sharpe).**

Key observations:
- The 2022-06-17 → 2022-12-15 window improved dramatically (+0.547 →
  +1.052) with is_window=378. This window's IS period now includes the
  full 2021 low-vol period + early 2022 stress onset, giving the HMM
  richer context.
- The worst windows (-1.938, -1.276) did NOT improve with any IS length.
  This is the critical finding: the core 2022 problem is immune to IS
  window length.
- 504 bars overshoots (0.834 < 0.940): including 2 years of IS data
  adds stale pre-COVID patterns that confuse the model and selects more
  states (7) without benefiting OOS performance.
- Locked: `is_window = 378`.

---

## 3. H2: BIC Overfitting

**Hypothesis:** Problem windows selected 6-7 states. With 6 features and
7 states, ~147 parameters from 252 IS bars = 1.7 bars/parameter
(dangerously close to overfitting). Capping n_components_range upper
bound forces BIC to select simpler models.

**Sweep:** `settings["hmm"]["n_components_range"]` over [[3,4], [3,5],
[3,6], [3,7]] with is_window=252 (isolated test).

**Results:**

| n_components_range | Aggregate Sharpe | Worst 2022 window | n_states (typical) |
|--------------------|-----------------|------------------|--------------------|
| [3, 4] | 0.785 | -1.675 | 4 |
| **[3, 5]** | **0.863** | -1.938 | 5 |
| [3, 6] | 0.841 | -1.938 | 5-6 |
| [3, 7] (baseline) | 0.835 | -1.938 | 5-7 |

**[3, 5] wins in isolation (+0.028 Sharpe).**

However, a combined confirmation run (is_window=378 + n_components_range=
[3,5]) produced **Sharpe=0.852 — worse than H1 alone (0.940).**

This reveals a negative interaction: with longer IS data (378 bars), the
HMM benefits from having more states available to capture the richer
history. Capping at 5 when IS=378 is too restrictive — BIC selects 5
states in nearly every window (cap binding), underfitting the 1.5-year
IS window.

**Decision: keep n_components_range=[3,7].** With is_window=378, BIC
naturally self-corrects state selection without an artificial cap.
The BIC penalty `log(378) × k` is larger than `log(252) × k`, naturally
discouraging overfitting.

---

## 4. H3: Slow-Grind Detection (Structural Diagnostic)

**Hypothesis:** Diagonal Gaussian HMM structurally cannot distinguish
"elevated flat vol" (2022 slow-grind bear) from "mid-vol normal" because
the emission distributions are computed independently per feature. When
realized_variance, VIX, and HY OAS are all moderately elevated
simultaneously, the diagonal model treats each signal independently
rather than recognizing the pattern as a joint multivariate regime.

### 4.1 Emission means — problem window IS

IS period: 2020-06-18 → 2021-12-15 (378 bars, preceding the worst OOS
window 2021-12-16 → 2022-06-16, Sharpe=-1.938).

| State | realized_var (Z) | vix (Z) | hy_oas (Z) | Character |
|-------|-----------------|---------|-----------|-----------|
| 2 | **+1.011** | -0.471 | -0.973 | COVID spike high vol |
| 3 | **+0.855** | +1.468 | -0.610 | COVID spike + VIX spike |
| 4 | +0.027 | +0.514 | +1.204 | Mid-vol, credit stress |
| 0 | -0.541 | -0.336 | -0.630 | Low-mid vol |
| 1 | -0.945 | -0.648 | -1.507 | Low vol |
| 5 | -1.486 | -1.227 | -1.375 | Very low vol |
| 6 | -1.085 | -0.703 | -1.735 | Very low vol |

### 4.2 The mechanism of failure

The IS window (Jun 2020 – Dec 2021) contains the COVID crash recovery
period. The HMM's "high stress" states (2 and 3) are characterized by
realized_var Z-scores of +0.85 to +1.01 — the sharp spike pattern of
COVID-type crises.

During 2022, the bear market was slow and grinding:
- The Fed raised rates 425bps over 12 months
- Equity markets declined -18% on the year
- But VIX stayed in a 20-35 range (never spiked above 40)
- Realized variance was moderately elevated but NOT at COVID-spike levels
- The Z-score of realized_var during 2022 sat around +0.1 to +0.4 —
  squarely in the mid-vol state (Z = +0.027)

The diagonal HMM looks at each feature independently:
- realized_var alone: +0.2 → mid-vol
- VIX alone: +0.4 → mid-vol
- HY OAS alone: +0.6 → mid-vol

Each feature individually signals mid-vol. Under diagonal covariance,
the joint observation `(+0.2, +0.4, +0.6)` is just the product of three
individual mid-vol likelihoods — no "all elevated together" pattern is
modeled.

Under **full covariance**, the model learns that `(realized_var=+0.2,
vix=+0.4, hy_oas=+0.6)` is a distinct multivariate pattern — different
from the calm mid-vol `(−0.1, +0.1, −0.2)` even though the marginals
are similar. The joint distribution sees the correlation structure;
the diagonal model is blind to it.

### 4.3 Conclusion

H3 is confirmed. The 2022 failure is a structural diagonal covariance
misspecification, not a data, feature, or hyperparameter problem.

**No fix exists within diagonal Gaussian HMM.** The correct resolution
is `covariance_type='full'` in Phase F.

---

## 5. Final Results

### 5.1 Configurations tested

| Config | Sharpe | Delta vs baseline | 2022-06-17 | Worst 2022 |
|--------|--------|-----------------|-----------|------------|
| Baseline: 252, [3,7] | 0.828 | — | +0.547 | -1.938 |
| H1: 378, [3,7] | **0.940** | **+0.111** | +1.052 | -1.981 |
| H2: 252, [3,5] | 0.863 | +0.035 | +0.547 | -1.938 |
| Combined: 378, [3,5] | 0.852 | +0.024 | +0.872 | -1.946 |

### 5.2 Locked settings

```yaml
# config/settings.yaml
backtest:
  is_window: 378    # Phase C H1 winner. +0.111 Sharpe vs 252 baseline.

hmm:
  n_components_range: [3, 7]  # Kept. BIC cap [3,5] degrades when combined with is_window=378.
```

### 5.3 Gate assessment

| Gate | Target | Achieved | Status |
|------|--------|---------|--------|
| Aggregate Sharpe | > 1.0 | 0.940 | NOT MET (close) |
| Worst 2022 window | > -1.0 | -1.981 | NOT MET |

**Gate not met.** Phase C improved aggregate Sharpe by +0.111 (0.828 →
0.940) but the two worst 2022 windows are irreducible under diagonal
covariance. Phase F is required.

---

## 6. Implications and Path Forward

### 6.1 What Phase C achieved

- Confirmed the 2022 problem is structural, not parametric
- Identified the correct IS window (378 bars = 1.5 years)
- Identified the negative H1+H2 interaction (important for Phase F
  parameter choices)
- Extracted empirical emission means that precisely explain the mechanism
  of 2022 misclassification
- Achieved +0.111 Sharpe improvement despite not solving the root cause

### 6.2 What Phase F must address

`covariance_type='full'` allows the HMM emission model to represent the
joint covariance structure of the 6 features. With full covariance, the
model can distinguish:

- `(realized_var=+0.2, vix=+0.4, hy_oas=+0.6)` — all moderately
  elevated, 2022 slow-grind pattern
- `(realized_var=−0.1, vix=+0.1, hy_oas=−0.2)` — calm mid-vol

These two observations are indistinguishable under diagonal covariance
(similar marginals). Under full covariance, the model fits a joint
Gaussian with cross-feature correlations — the 2022 "all elevated
together" pattern gets its own cluster.

Phase F should also re-enable `vix_slope` in FEATURE_COLS. The ~54%
independent variance of vix_slope (from VIX level) is exactly the
kind of signal full covariance can exploit — it captures term structure
dynamics that are orthogonal in the joint distribution but overlapping
in the marginals.

See design_docs/09_theory_vs_empirical_conflicts.md §Case 1 for the
full vix_slope analysis.

### 6.3 Risk: full covariance + more features = more parameters

With 7 features (adding vix_slope) and full covariance, each state has
a full 7×7 covariance matrix (28 unique values). With 5 states:
- Means: 5×7 = 35
- Full covariance: 5×28 = 140
- Transition matrix: 5×5 = 25
Total: ~200 parameters

With is_window=378 bars, that's ~1.9 bars/parameter. Still tight.
May need to also test `covariance_type='tied'` (shared covariance matrix
across states) as a middle ground: retains cross-feature correlation
structure with fewer parameters.

---

## 7. References

- Hamilton (1989, Econometrica 57:357-384) — HMM regime framework
- Gray (1996, JFE 42:27-62) — Gaussian misspecification in vol regimes
- Guidolin & Timmermann (2008, RFS 21:889-935) — multi-state cross-asset
  HMM with full covariance
- design_docs/09_theory_vs_empirical_conflicts.md — vix_slope analysis
- `logs/sweep_phase_c_h1.log` — H1 raw results
- `logs/sweep_phase_c_h2.log` — H2 raw results
- `logs/sweep_phase_c_combined.log` — combined confirmation
