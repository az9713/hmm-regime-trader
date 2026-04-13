# design_docs/09_theory_vs_empirical_conflicts.md
# ============================================================
# Theory vs. Empirical Backtest: Where and Why They Diverge
#
# This document records cases where a feature or design choice
# has strong theoretical justification but weak or negative
# empirical backtest performance — and explains the mechanism
# of divergence in each case.
#
# Purpose: prevent mistaking an empirically-failed feature for
# a theoretically-wrong one (or vice versa). The correct
# response to theory-empirical conflict depends on *why* they
# conflict, not just that they conflict.
# ============================================================

---

## Case 1: VIX Term Structure Slope (vix_slope = VIX / VIX3M)

### 1.1 The Theoretical Argument

VIX term structure carries information about market expectations that
VIX level alone does not capture. This is well-established in the
volatility literature:

**Egloff, Leippold & Wu (2010, Journal of Financial Econometrics 8(3):367-413)**
show that the volatility term structure slope predicts future variance
risk premia. Backwardation (VIX > VIX3M) signals that the market
expects near-term vol to revert — a classic crisis signature.

**Eraker & Wu (2017, Journal of Financial Economics 123(3):431-463)**
document that the VIX term structure contains separate risk factors
from VIX level, with distinct compensation in the cross-section of
option returns.

**Simon & Campasano (2014, Journal of Futures Markets 34(2):129-153)**
show the VIX/VIX3M ratio is a stronger predictor of short-term
equity returns than VIX level in isolation.

The ratio VIX/VIX3M is preferred over the arithmetic spread (VIX3M-VIX)
because volatility surfaces are multiplicative (mean-reverting in
log-space), making the ratio scale-invariant across VIX regimes.
A ratio of 1.10 means the same thing whether VIX is 15 or 50; a spread
of +3 points means something very different.

**Directional consistency:** VIX/VIX3M > 1 (backwardation) = stress.
This is consistent with all other features: high realized_variance,
high VIX, high HY OAS, negative gold_return all point the same
direction. The arithmetic spread VIX3M-VIX inverts this (high spread =
calm), which would require the HMM to learn an inverted polarity for
one dimension only.

**Orthogonality:** Pairwise correlation with VIX = 0.68 (r²=0.46),
meaning ~54% of vix_slope variance is independent of VIX level.
This independent component captures term structure dynamics that are
genuinely separate from the level of implied volatility.

**Theoretical verdict:** Strong. Three independent top-tier papers,
a principled construction choice (ratio vs spread), directional
consistency, and measurable orthogonality (~54% independent variance).

---

### 1.2 The Empirical Backtest Result

**Baseline (6 features):** Sharpe = 0.831, CAGR = ~10%
**Phase A (7 features + vix_slope):** Sharpe = 0.828, CAGR = 10.1%

Net change: -0.003 Sharpe, within noise. Statistically indistinguishable.

**Problem windows specifically:**
| Window | Baseline Sharpe | Phase A Sharpe | Delta |
|--------|----------------|----------------|-------|
| 34 (2022 Q3) | -1.12 | -1.07 | +0.05 |
| 35 (2022 Q4) | +0.05 | -0.47 | -0.52 |
| 43 | — | -1.94 | very bad |
| 44 | — | -1.58 | very bad |

Window 35 is the most damning: the period where vix_slope was *supposed*
to help (2022 bear market) instead got worse.

**BIC state selection with 7 features:** averaged 6-7 states vs 5 with
6 features. More states = more parameters per IS window (252 bars).
Approximate parameter count with 7 features, 7 states:
- Transition matrix: 7×7 = 49
- Means: 7×7 = 49
- Variances (diag): 7×7 = 49
Total: ~147 parameters / 252 bars ≈ 1.7 bars per parameter.
This is dangerously close to the overfitting regime.

**Stress tests:** Both pass (2020: 100% HighVol, 2022: 98% HighVol).
Stress test detection is not sensitive to vix_slope addition — the model
already detects stress from the other 6 features.

**Empirical verdict:** Neutral-to-negative. Zero aggregate improvement,
one problem window degraded, BIC overparameterization risk.

---

### 1.3 Why They Conflict: Four Mechanisms

**Mechanism 1: Crisis type mismatch**

The theoretical evidence for vix_slope is built on spike-type crises:
sudden, sharp VIX surges (2008, 2020 COVID, Flash Crash). In these
events, the front end of the vol surface (VIX) spikes dramatically
while the 3-month point (VIX3M) moves less — producing strong
backwardation signal.

The 2022 bear market — our dominant problem — was a *slow grind*:
the Fed raised rates 425bps over 12 months, equity markets declined
steadily, but VIX stayed in a narrow elevated range (20-35) and
VIX3M tracked closely. Result: vix_slope ≈ 1.0 throughout 2022.
The feature the theory predicts should help was informationally flat
precisely during the period we need to improve.

This is not a failure of the theory — it is a scope mismatch. The
theory makes conditional claims about spike events. The data tests
across all regimes, including the slow-grind bear where the condition
doesn't hold.

**Mechanism 2: Diagonal covariance misspecification**

`GaussianHMM(covariance_type='diag')` assumes each feature is
conditionally independent given the regime state. VIX and vix_slope
have r=0.68 correlation. The diagonal model cannot represent this
dependency — it fits marginal distributions for each feature
independently.

In practice, this means the model is fitting two overlapping noisy
measurements of the same underlying construct (near-term implied vol
fear) and treating them as independent signals. The emission log-
likelihood for a high-VIX, high-vix_slope observation is computed as:

    log p(o|state) = log p(vix|state) + log p(vix_slope|state)   [diagonal]

vs. the correct:

    log p(o|state) = log p(vix, vix_slope | state)               [full cov]

The diagonal version cannot model the joint distribution. It inflates
the apparent information content of the 7-feature observation vector
because it assumes away the redundancy. The model thinks it has more
signal than it does.

The ~54% independent variance of vix_slope is theoretically real — but
the diagonal HMM cannot isolate it. The model sees both features,
treats them as independent, and the EM algorithm tries to fit both,
producing noisier parameter estimates.

**Mechanism 3: BIC overfitting via state explosion**

Adding a 7th feature increases the EM parameter space. BIC penalizes
parameter count, but the penalty is log(N)×k where N=252 bars (IS
window) and k grows as n_states × n_features for means and variances.

With 6 features, BIC selected ~5 states consistently.
With 7 features, BIC selected 6-7 states in later windows.
More states applied to the same 252-bar IS window = each state has
fewer bars to fit its distribution = noisier parameter estimates.

The 7th feature effectively encouraged the BIC-selector to pick more
complex models from the same data, compounding the overfitting risk
rather than adding clean signal.

**Mechanism 4: Low fire rate dilutes aggregate impact**

Backwardation (vix_slope > 1.0) occurred ~7.8% of the time
(approximately 320 days out of 4,040 in the test period).
Aggregate Sharpe is computed over all 7,375 OOS bars. Even a perfect
improvement in the 7.8% backwardation bars cannot materially move
the aggregate Sharpe because it is dominated by the 92.2% contango
bars where vix_slope hovers around 0.90 — adding nothing but noise.

---

### 1.4 Resolution

**Do not remove the infrastructure.** The theory is sound. The empirical
failure is explained by four specific mechanisms, each of which points
to a concrete fix. Removing the FRED pull and compute() wiring would
require re-implementing them in Phase F.

**Current state:** vix_slope is computed inside `FeatureEngineer.compute()`
but excluded from `FEATURE_COLS`. All call sites (main.py, backtester.py,
stress_test.py) already pass `vix3m=` correctly. The feature exists but
is not fed to the HMM.

**Re-enable in Phase F when `covariance_type='full'`:**
Full covariance eliminates Mechanism 2 — the model can now represent
the vix↔vix_slope joint distribution and extract the genuinely
independent 54% variance. This also tends to reduce state counts
(the model doesn't need extra states to implicitly capture correlations),
mitigating Mechanism 3.

Mechanism 1 (crisis type mismatch) and Mechanism 4 (low fire rate)
remain. But with full covariance handling the redundancy correctly,
the independent variance in vix_slope becomes extractable, and the
BIC state explosion pressure diminishes.

**Re-enable checklist for Phase F:**
1. Change `covariance_type='diag'` → `'full'` in `core/hmm_engine.py`
2. Add `"vix_slope"` back to `FEATURE_COLS` in `data/feature_engineering.py`
3. All vix3m= wiring is already in place — no other changes needed
4. Re-run walk-forward backtest; compare Sharpe and Window 35 specifically
5. If no improvement, vix_slope is empirically irrelevant for this strategy
   and can be permanently removed

---

## Template: Recording Future Theory-Empirical Conflicts

Use this structure for any future case where theory and data diverge:

```
### Case N: [Feature or design choice]

#### N.1 Theoretical Argument
- Source(s) and what they predict
- Why the prediction should apply here

#### N.2 Empirical Result
- What the backtest showed (numbers)
- Which windows/periods diverged most

#### N.3 Why They Conflict
- Scope mismatch (theory's conditions not met in data)
- Model misspecification (model can't exploit the signal)
- Statistical power (too few events / low fire rate)
- Data quality (measurement error, alignment issues)
- Overfitting (BIC selecting more complex models)

#### N.4 Resolution
- What to do now
- What condition would change the decision
- What infrastructure to keep vs remove
```

---

## References

- Egloff, Leippold & Wu (2010, JFE 8(3):367-413) — VIX term structure risk factors
- Eraker & Wu (2017, JFE 123(3):431-463) — VIX term structure separate from level
- Simon & Campasano (2014, JFM 34(2):129-153) — VIX/VIX3M as return predictor
- Gray (1996, JFE 42:27-62) — Gaussian emission misspecification in regime models
- Hamilton (1989, Econometrica 57:357-384) — HMM framework
- arXiv:2402.05272 (2024) — returns + realized vol are robust; extra features mostly noise
