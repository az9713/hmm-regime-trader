# design_docs/13_phase_f_full_covariance.md
# ============================================================
# Phase F — Full Covariance HMM + vix_slope Re-enable
#
# Date: 2026-04-12
# Prerequisite: Phase C complete (Sharpe 0.940, gate NOT met)
# Root cause confirmed: design_docs/11_phase_c_2022_diagnosis.md §4
# Target: Sharpe > 1.0, worst 2022 window > -1.0
# ============================================================

---

## 1. Why Phase F Exists

Phase C H3 confirmed a structural model misspecification: the diagonal
Gaussian HMM cannot detect the 2022 slow-grind bear market. The root
cause is the conditional independence assumption embedded in diagonal
covariance — each feature is evaluated independently given the regime
state, and "all features simultaneously mildly elevated" is not a
representable joint pattern.

The two worst OOS windows are:

| Window | Sharpe | Character |
|--------|--------|-----------|
| 2021-12-16 → 2022-06-16 | -1.938 | First-half 2022 bear onset |
| 2022-03-18 → 2022-09-16 | -1.276 | Continued rate-hike drawdown |

Without these two windows, aggregate OOS Sharpe would be approximately
1.2+. With them, it is 0.940. The gap between current performance and
the live-trading gate (Sharpe > 1.0) is entirely attributable to 2022.

No parameter change can fix this — Phase B confirmed the ceiling is
structural, and Phase C H1/H2 confirmed the 2022 failure is immune to
IS window length and BIC cap. Phase F is the only remaining lever.

---

## 2. What Phase F Buys

### 2.1 2022 slow-grind detection

Under diagonal covariance, the emission probability for a 2022
observation `(realized_var Z +0.2, vix Z +0.4, hy_oas Z +0.6, ...)` is:

```
P(x | state=s) = ∏ₑ N(xₑ ; μₛₑ, σ²ₛₑ)
```

Each feature contributes independently. The mid-vol state has
`realized_var mean Z ≈ +0.027` — close to +0.2. VIX mean similarly
mid-vol. Each feature independently scores as mid-vol, so the joint
probability is highest for mid-vol → allocation stays at 65%.

Under full covariance, the emission is:

```
P(x | state=s) = N(x ; μₛ, Σₛ)
```

where `Σₛ` is a full 6×6 (or 7×7 with vix_slope) covariance matrix.
The model fits the joint distribution: "all features simultaneously
elevated and correlated in a specific pattern" is a learnable cluster.
If the 2022 pattern is distinct enough in the joint space — and the H3
emission means analysis suggests it is — BIC will allocate a dedicated
"slow-grind" state, and 2022 observations will be assigned to it with
a corresponding defensive allocation (~35%).

The 2022 bear had a specific cross-feature signature throughout:
- Realized variance: Z ~+0.1 to +0.4 (persistently)
- VIX: 20–35 range, never spiking above 40
- HY OAS: steadily widening (Z ~+0.6 to +0.8)
- Term spread: rapidly inverting
- Gold: elevated as flight-to-safety

This joint signature is invisible to diagonal covariance. It is
detectable under full covariance if training data contains a prior
example — which at is_window=378 it does, partially (the tail of the
COVID recovery and early 2022 onset overlap in some IS windows).

### 2.2 vix_slope becomes useful

vix_slope (VIX/VIX3M) is currently computed in
`data/feature_engineering.py` but excluded from FEATURE_COLS. The
exclusion was justified by Phase C H3 findings and the analysis in
`design_docs/09_theory_vs_empirical_conflicts.md §Case 1`.

The core problem under diagonal covariance: vix and vix_slope share
roughly 50% variance (VIX level drives both). Under diagonal covariance
this shared variance creates redundancy — the model sees two features
that are correlated but must treat them independently, inflating the
apparent weight of VIX-level information and producing unreliable
state assignments.

Under full covariance, the joint distribution of (vix, vix_slope) is
modeled directly. The model learns:
- The shared VIX-level variance → contributes to one axis of the
  joint distribution
- The orthogonal vix_slope variance (~50%) → contributes to a second,
  independent axis

In practical terms: inverted vol term structure (VIX > VIX3M,
vix_slope > 1.0) that preceded the 2022 bear is a detectable signal.
The vol term structure was in backwardation through most of 2022 —
near-term fear elevated relative to long-term — which is exactly the
"slow-grind stress" signature that vix_slope captures and VIX level
alone does not.

Reference: Egloff, Leippold & Wu (2010, JFEC 8(3):367-413) —
backwardation in vol term structure predicts future variance risk
premia. Carr & Wu (2006, JFE 79:1-33) — VIX term structure as a
pure measure of risk-neutral variance.

### 2.3 Unlocks the live trading path

The live trading criteria (per `design_docs/06_empirical_testing_plan.md`):
- OOS Sharpe > 1.0
- Max DD < 15%
- Hansen SPA p < 0.05
- 3-month paper trade: no unexpected blowups

At Sharpe 0.940 the first criterion is not met. Phase F is the only
remaining model lever that addresses the root cause of that shortfall.
Phases D (broker) and E (paper trading) are structurally ready but
there is no value in running 3 months of paper trading at a known-
deficient model when a targeted model fix may close the gap.

The correct sequence: Phase F → verify gate → Phase D → Phase E.

---

## 3. Implementation Plan

### 3.1 Two code changes (both already wired)

**Change 1 — `config/settings.yaml`:**
```yaml
# Before:
hmm:
  covariance_type: "diag"

# After (Phase F — option A, full):
hmm:
  covariance_type: "full"

# After (Phase F — option B, tied, test first):
hmm:
  covariance_type: "tied"
```

**Change 2 — `data/feature_engineering.py`:**
```python
# Before:
# vix_slope excluded from FEATURE_COLS — computed but not fed to HMM
# under diag covariance. Re-enable in Phase F with covariance_type='full'.
FEATURE_COLS = ["log_return", "realized_variance", "vix", "hy_oas",
                "gold_return", "term_spread"]

# After:
FEATURE_COLS = ["log_return", "realized_variance", "vix", "vix_slope",
                "hy_oas", "gold_return", "term_spread"]
```

All data-fetching wiring for vix_slope is already in place
(`market_data.py`, `DataManager.get_vix3m()`). No other changes needed.

### 3.2 Test sequence

Run in order, each contingent on the previous not regressing badly:

1. **`covariance_type='tied'` + 6 features (no vix_slope):**
   Tied covariance shares one covariance matrix across all states.
   Fewer parameters than full, more expressive than diagonal.
   Baseline for Phase F to isolate covariance type effect.

2. **`covariance_type='full'` + 6 features:**
   Full per-state covariance matrices. Maximum expressiveness.
   Primary Phase F hypothesis.

3. **`covariance_type='full'` + 7 features (vix_slope re-enabled):**
   Full covariance + vix_slope. Combined Phase F target.

4. **`covariance_type='tied'` + 7 features:**
   Fallback if full + 7 features overfits (>200 params, ~1.9 bars/param).

Run each as a full walk-forward backtest. Report aggregate Sharpe,
per-window Sharpe for 2022 windows, and BIC-selected state counts.

### 3.3 Gate decision

| Gate | Target | Baseline (Phase C) |
|------|--------|--------------------|
| Aggregate Sharpe | > 1.0 | 0.940 |
| Worst 2022 window | > -1.0 | -1.981 |
| Non-2022 Sharpe | no regression > 0.05 | ~1.2 (without 2022) |

If gate met: lock covariance_type and FEATURE_COLS, update settings.yaml,
write Phase F doc, proceed to Phase D.

If gate not met: document which windows improved and which did not,
assess whether Student-t emissions (SEP-HMM, MDPI Mathematics 14(3) 2025)
are warranted as a further structural upgrade.

---

## 4. Risk Analysis

### 4.1 Parameter count

| Config | States | Features | Cov params/state | Total params |
|--------|--------|----------|-----------------|-------------|
| Diagonal, 6 features | 5 | 6 | 6 | ~107 |
| Diagonal, 7 features | 5 | 7 | 7 | ~122 |
| Tied, 6 features | 5 | 6 | 21 (shared) | ~86 |
| Tied, 7 features | 5 | 7 | 28 (shared) | ~98 |
| **Full, 6 features** | **5** | **6** | **21 × 5 = 105** | **~167** |
| **Full, 7 features** | **5** | **7** | **28 × 5 = 140** | **~202** |

With is_window=378: full+7 features gives ~1.9 bars/parameter.
This is tight. However, BIC's log(378) penalty naturally suppresses
over-parameterized models. If BIC consistently selects 3-4 states
instead of 5 under full covariance, the effective parameter count
drops to ~120-160 — more comfortable.

Monitor: BIC-selected n_states per window. If consistently at 7 (cap),
the cap may be too loose for full covariance. Consider [3,5] cap
specifically for Phase F — re-test the H1/H2 interaction at full cov.

### 4.2 Numerical stability

Full covariance matrices must be positive definite. hmmlearn's
GaussianHMM with `covariance_type='full'` adds a regularization term
(`min_covar`) to the diagonal to prevent near-singular matrices. The
default `min_covar=1e-3` is usually sufficient for standardized
(Z-scored) features. If training fails with "covariance matrix not
positive definite" errors, increase `min_covar` to 1e-2.

### 4.3 Non-2022 regression risk

Full covariance adds expressiveness but also capacity for overfitting.
The low-vol 2014-2019 bull market windows (which currently produce
Sharpe 2.0-3.3) rely on the model cleanly assigning low-vol states.
A more complex model might fragment these into spurious sub-states,
degrading performance in calm periods.

Mitigation: track non-2022 aggregate Sharpe separately. If non-2022
Sharpe regresses more than 0.05 from its current ~1.2 level, the
covariance upgrade is net-negative and a fallback to `"tied"` or
additional regularization is needed.

---

## 5. What Phase F Does NOT Fix

- **Detection lag:** The HMM reacts to regime changes after they occur.
  This is intrinsic to any filter-based inference. The first bars of any
  new regime will always be partially misclassified. Full covariance
  reduces the steady-state misclassification but does not eliminate
  onset lag.

- **Training data mismatch:** If a future crisis has a signature unlike
  anything in the IS training window, the model will still misclassify
  it regardless of covariance type. Full covariance gives the model more
  expressive capacity, not foresight.

- **Truly novel regimes:** A hyperinflationary environment, a sovereign
  debt crisis, or a market structure change not present in 2010-2026 IS
  data will not be detected.

- **The SPA p-value problem:** Hansen SPA currently gives p = 0.908
  (not significant). This reflects Sharpe 0.828-0.940, not model
  structure. If Phase F raises Sharpe above 1.0, the SPA p-value will
  improve — but significance (p < 0.05) may require a longer OOS track
  record regardless of Sharpe.

---

## 6. Connection to Prior Work

| Document | Relevant section |
|----------|-----------------|
| design_docs/09_theory_vs_empirical_conflicts.md | §Case 1 — vix_slope exclusion rationale and Phase F re-enable checklist |
| design_docs/11_phase_c_2022_diagnosis.md | §4 — H3 structural diagnostic, emission means table, mechanism of failure |
| design_docs/12_phase_c_2022_fix.md | §6.3 — diagonal vs full covariance math; §7.5 — Phase F requirements |
| config/settings.yaml | `hmm.covariance_type` — the one-word change |
| data/feature_engineering.py | `FEATURE_COLS` — the one-line change |

---

## 7. References

- Hamilton (1989, Econometrica 57:357-384) — HMM framework
- Gray (1996, JFE 42:27-62) — Gaussian misspecification in vol regimes;
  principal recommendation for full covariance in multi-asset HMMs
- Guidolin & Timmermann (2008, RFS 21:889-935) — multi-state cross-asset
  HMM with full covariance; documents diagonal covariance limitations
- Egloff, Leippold & Wu (2010, JFEC 8(3):367-413) — vol term structure
  slope predicts variance risk premia (vix_slope theoretical basis)
- Carr & Wu (2006, JFE 79:1-33) — VIX term structure as risk-neutral
  variance measure (vix_slope orthogonality analysis)
- SEP-HMM (2025, MDPI Mathematics 14(3)) — Student-t emissions as
  further structural upgrade post-Phase F
