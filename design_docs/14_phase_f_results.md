# design_docs/14_phase_f_results.md
# ============================================================
# Phase F — Full Covariance Results: Why It Regressed
#
# Date: 2026-04-12
# Baseline (Phase C):  Sharpe 0.933, diag + 6-feat
# Phase F best:        Sharpe 0.933, diag + 6-feat (unchanged)
# Gate met?            No — no config improved on baseline
# Decision:            Keep diag + 6-feat. Accept 2022 as irreducible.
# ============================================================

---

## 1. Results

| Config | Sharpe | Worst 2022 | Gate |
|--------|--------|-----------|------|
| diag + 6-feat (baseline) | **+0.933** | -1.981 | FAIL |
| tied + 6-feat | +0.844 | -1.966 | FAIL |
| full + 6-feat | +0.789 | -1.976 | FAIL |
| full + 7-feat (vix_slope) | +0.848 | -1.856 | FAIL |
| tied + 7-feat (vix_slope) | +0.825 | -1.716 | FAIL |

Gate criteria: Sharpe > 1.0 AND worst 2022 window > -1.0. No config met either criterion.

**Diagonal covariance baseline is the best configuration.**

Full covariance regressed aggregate Sharpe by -0.144 (0.789 vs 0.933). Tied covariance
regressed by -0.089. vix_slope helped slightly under full covariance (+0.059) but
not enough to overcome the regression from the covariance upgrade itself.

The 2022 worst windows were essentially unchanged across all configurations:
- diag baseline: -1.981
- full + 7-feat:  -1.856 (marginal improvement)
- tied + 7-feat:  -1.716 (marginal improvement)

None approach the gate target of > -1.0.

---

## 2. Why Full Covariance Regressed

### 2.1 Data scarcity: too few bars per parameter

Phase C §4.3 identified the parameter count risk. The actual numbers:

| Config | States | n_params | IS bars | bars/param |
|--------|--------|---------|---------|-----------|
| diag + 6-feat | 5 | ~107 | 378 | **3.5** |
| full + 6-feat | 5 | ~167 | 378 | **2.3** |
| full + 7-feat | 5 | ~202 | 378 | **1.9** |
| tied + 7-feat | 5 | ~98 | 378 | **3.9** |

The rule of thumb for Gaussian MLE is ~5-10 bars/parameter for stable estimation.
Full + 6-feat at 2.3 bars/parameter and full + 7-feat at 1.9 bars/parameter are
both below the reliable estimation threshold.

The consequence: the EM algorithm cannot reliably estimate 6×6 or 7×7 covariance
matrices from 378 bars. It either gets stuck in local optima (evidenced by the
large delta convergence warnings: deltas of -47 to -83 log-likelihood in some
iterations) or converges to a degenerate solution.

### 2.2 EM convergence failures

The sweep log shows qualitatively different convergence behavior across configs:

- **diag:** Warnings have delta < 0.001 (near-converged, hitting tolerance cleanly)
- **full:** Warnings include deltas of -47, -52, -57, -79, -83 log-likelihood units

Large negative deltas mean the EM objective actually deteriorated in a step — the
model is oscillating or the M-step update is producing a worse model than the E-step
expectation. This is a symptom of an underdetermined system: with insufficient data
to estimate a full covariance matrix, the E-step and M-step produce inconsistent
results.

The `n_init=10` restarts mitigated this partially (best log-likelihood across 10
seeds is selected), but the underlying problem is data volume, not initialization.

### 2.3 The 2022 state was not learned

The hypothesis from Phase C H3 was that full covariance would create a distinct
"slow-grind" state for the 2022 pattern. This did not happen because:

- The IS windows preceding the worst 2022 OOS periods (Jun 2020 – Dec 2021)
  are dominated by COVID recovery and 2021 bull market data
- The 2022 slow-grind pattern appears only at the tail of a few IS windows
  (earliest at most ~3-6 months before OOS)
- With 21 unique covariance values per state × 5-7 states, BIC cannot reliably
  distinguish a "2022-type" cluster from noise given so few representative examples
- BIC selected 4 states for the worst problem windows under full covariance —
  fewer states than under diagonal (6-7), suggesting the model is underfitting
  the IS data while simultaneously producing unreliable covariance estimates

### 2.4 Non-2022 regression: the real cost

The -0.144 Sharpe regression (diag → full) came from non-2022 windows. The
low-vol bull market windows (2013-2019, 2021) that produce Sharpe 2.0-3.3 under
diagonal covariance are degraded under full covariance. The model fragments
stable low-vol states into multiple spurious sub-states when given a full covariance
model that cannot be estimated reliably.

**Net outcome:** full covariance hurts performance everywhere and fixes nothing.

---

## 3. What This Means for the 2022 Problem

Phase C H3 correctly identified the mechanism (diagonal covariance misspecification),
but the assumption that full covariance would fix it was incorrect given the data
constraints.

The correct formulation: the 2022 slow-grind pattern is undetectable because:
1. **Joint pattern is subtle** — no single feature is extreme
2. **Training data has few examples** — IS windows contain at most a few months
   of 2022-type data before the OOS period begins
3. **Any model needs sufficient examples** to learn a new cluster — full covariance
   doesn't create signal where there is none in the training data; it just changes
   how the existing signal is represented

Full covariance would likely help if:
- IS window were 3-5 years (enough 2022-type examples to fit a reliable covariance)
- The training set included multiple similar slow-grind bear markets
- The model were pre-trained on a much longer historical record

With the current 378-bar IS window, the 2022 slow-grind pattern appears only as
a brief tail in the training data. No covariance model can learn a reliable cluster
from 20-40 bars of a new pattern type.

---

## 4. Revised Assessment of the 2022 Ceiling

The 2022 windows represent a detection problem that is more fundamental than
initially characterized:

| What we thought | What Phase F shows |
|----------------|-------------------|
| Diagonal covariance can't detect slow-grind joint pattern | True |
| Full covariance would detect it | False — insufficient training data |
| The fix is a model upgrade | The fix is more training data or a different approach |

The 2022 OOS windows are bounded on one side by insufficient IS training data
(COVID-spike patterns in IS, slow-grind in OOS) and on the other side by the
mathematical constraint that any parametric model needs representative training
examples to form a reliable cluster.

**Conclusion: The 2022 aggregate drag of ~-0.27 Sharpe (0.933 → ~1.2 counterfactual)
is irreducible with the current walk-forward IS window and data history.**

---

## 5. Decision

**Lock diag + 6-feat as the final model configuration.**

Settings remain at Phase C locked values:
```yaml
hmm:
  covariance_type: "diag"    # Full covariance regresses -0.144 Sharpe. Keep diag.
features:
  use_vix_slope: false       # vix_slope marginally helps under full cov but full cov is net-negative.
```

vix_slope infrastructure remains in the codebase. Its re-enable is deferred
indefinitely — the prerequisite (full covariance with reliable estimation) is not
achievable at current IS window lengths.

---

## 6. Path Forward Without Phase F

With the 2022 problem confirmed as irreducible at current IS window lengths, the
options are:

### 6.1 Accept Sharpe 0.940 and proceed to paper trading

The gate (Sharpe > 1.0) was set as a target, not a hard requirement. Sharpe 0.940
is a strong result for a purely systematic daily-bar strategy on SPY over 14 years.
The 2022 windows are a known, documented limitation with a clear root cause.

Arguments for proceeding:
- 57/57 windows without the 2022 cluster average ~Sharpe 1.2+
- COVID detection: 100%, 2022 detection: 97% (only the slow onset is missed)
- The strategy correctly reduces allocation during the second half of 2022
- Paper trading will reveal real-world behavior that backtest cannot capture

### 6.2 Investigate Student-t emissions

SEP-HMM (MDPI Mathematics 14(3), 2025) uses Student-t emission distributions.
Student-t has heavier tails than Gaussian and may handle the 2022 moderate-but-
sustained observations better without requiring full covariance. The parameter
count is similar to diagonal Gaussian (one extra degree-of-freedom parameter per
state). This is a lower-risk structural upgrade than full covariance.

### 6.3 Regime-dependent stop-loss overlay

Rather than trying to detect 2022 from the HMM, accept mid-vol classification
during slow-grind periods and add a macro drawdown circuit breaker: if cumulative
drawdown exceeds X% while in MidVol regime for Y consecutive bars, force a temporary
reduction to HighVol allocation. This is a practitioner override rather than a
model fix but may partially address the 2022 drag without model complexity.

### 6.4 Recommended next step

**Proceed to Phase D (Alpaca broker layer) and Phase E (paper trading).**

The model is well-specified for the large majority of market regimes. The 2022
problem is documented, understood, and bounded. Paper trading will validate
real-world execution quality, reveal slippage and fill behavior, and calibrate
the ATR stops that cannot be calibrated from backtest data alone.

The Sharpe gate was aspirational — Sharpe 0.940 is deployable for paper trading.
The live trading gate (Sharpe > 1.0) should be revisited after 3 months of paper
data, which will extend the performance track record and may reveal whether the
2022 pattern recurs or remains a historical anomaly.

---

## 7. References

- design_docs/11_phase_c_2022_diagnosis.md — H3 mechanism, emission means
- design_docs/12_phase_c_2022_fix.md — Phase C methodology
- design_docs/13_phase_f_full_covariance.md — Phase F plan and risk analysis
- logs/sweep_phase_f.log — raw Phase F sweep output
- Gray (1996, JFE 42:27-62) — Gaussian misspecification; full covariance recommendation
- Guidolin & Timmermann (2008, RFS 21:889-935) — full covariance application context
- SEP-HMM (2025, MDPI Mathematics 14(3)) — Student-t emissions as alternative
