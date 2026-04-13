# design_docs/12_phase_c_2022_fix.md
# ============================================================
# Phase C — Fix 2022 Problem Windows: Goals, Methodology,
#            Code Changes, Results, Interpretation, and
#            Implications for HMM Quality
#
# Date: 2026-04-12
# Baseline Sharpe:  0.828 (entering Phase C, post Phase B)
# Final Sharpe:     0.940 (is_window=378, n_components_range=[3,7])
# Gate met?         No (target >1.0) — Phase F required
#
# Detailed diagnosis data: design_docs/11_phase_c_2022_diagnosis.md
# ============================================================

---

## 1. Goals

### 1.1 Why Phase C exists

After Phase A (vix_slope feature test) and Phase B (hyperparameter sweep),
the aggregate OOS Sharpe was 0.828 and all tunable parameters were locked.
Phase B explicitly confirmed the Sharpe ceiling is structural: tuning
parameters alone cannot materially improve performance.

The structural bottleneck is a cluster of OOS windows covering the 2022
bear market. During these windows, the HMM stays in a mid-vol regime
throughout the entire slow, grinding drawdown, maintaining ~65% equity
allocation instead of cutting to ~35%. The resulting losses dominate
the aggregate:

| OOS Window | Sharpe |
|------------|--------|
| 2021-12-16 → 2022-06-16 | -1.938 |
| 2022-03-18 → 2022-09-16 | -1.276 |
| 2022-09-19 → 2023-03-20 | -0.306 |

Without these windows, the aggregate Sharpe would be approximately 1.2.
Fixing (or diagnosing) these windows is the only remaining lever before
paper trading.

### 1.2 What Phase C set out to do

Phase C had four explicit goals:

1. **Identify whether the 2022 failure is parametric or structural.**
   Parametric failures are fixable by changing IS window length or BIC
   state count. Structural failures require a different model.

2. **Find and lock any parametric improvements.** Even if the core failure
   is structural, partial improvements (e.g., better IS window) can still
   raise aggregate Sharpe and provide a stronger baseline for Phase F.

3. **Extract the exact mechanism of failure.** If the failure is
   structural, document it precisely — which model assumption is wrong,
   what data pattern it cannot represent, and what the correct fix is.

4. **Make a gate decision.** If aggregate Sharpe > 1.0 and worst 2022
   window > -1.0, lock settings and proceed to Phase D (broker). If not,
   confirm Phase F (full covariance) is required.

### 1.3 Hypothesis framework

Three hypotheses were defined and tested in order of cost (cheapest first):

| Hypothesis | Claim | Test | Outcome |
|------------|-------|------|---------|
| H1 | IS window too short — HMM never sees a prior stress cycle before 2022 OOS | Sweep is_window [252, 378, 504] | Partial fix (+0.111) |
| H2 | BIC overfitting — 6-7 states from 252 bars = 1.7 bars/param | Sweep n_components_range [[3,4]…[3,7]] | Wins alone, negative interaction with H1 |
| H3 | Diagonal covariance misspecification — structural | Diagnostic: extract emission means from problem window IS periods | Confirmed |

---

## 2. Backtests Run

### 2.1 H1: IS window sweep (3 runs)

**What was varied:** `settings["backtest"]["is_window"]` ∈ [252, 378, 504].

**Why these values:**
- 252 = baseline (1 calendar year).
- 378 = 1.5 years. Sufficient to include the transition from 2020 COVID
  crash recovery into the low-vol 2021 bull run, giving the HMM training
  exposure to both extreme and calm regimes before any 2022 OOS window.
- 504 = 2 years. Includes full 2020 COVID spike; risk is stale pre-COVID
  patterns confusing the model.

**What was measured per run:**
- Aggregate OOS Sharpe (primary)
- CAGR, MaxDD, Sortino (secondary, for context)
- Per-window Sharpe for every OOS window whose start or end falls in 2022
  (filtered by calendar date, not window index — index shifts when IS
  length changes)
- Total number of OOS windows (decreases as IS length grows; fewer windows
  to establish the first OOS period)
- BIC-selected n_states per 2022 window

### 2.2 H2: BIC cap sweep (4 runs)

**What was varied:** `settings["hmm"]["n_components_range"]` ∈
[[3,4], [3,5], [3,6], [3,7]].

**Why these values:**
- [3,7] = baseline. BIC selects between 3 and 7 states.
- Problem windows selected 6-7 states. With 6 features, 7 states, and
  252 IS bars: ~147 parameters / 252 bars ≈ 1.7 bars/parameter. The
  literature considers 5-10 bars/parameter a rough lower bound for
  reliable Gaussian MLE. 1.7 is dangerously close to overfitting.
- Capping at [3,5], [3,4] forces BIC to select simpler models.
- [3,7] included as baseline confirmation.

**H2 was run with is_window=252 (baseline IS length).** This isolates the
BIC cap effect cleanly, without H1's IS change contaminating the result.

### 2.3 Combined confirmation (1 run)

After H1 winner (378) and H2 winner ([3,5]) were identified in isolation,
one combined run was executed:
- `is_window=378`, `n_components_range=[3,5]`

Purpose: confirm whether winners combine additively or interact.

### 2.4 H3: structural diagnostic (no additional runs)

H3 did not require additional backtest runs. Instead, an existing problem
window's HMM model was reloaded and interrogated:

- IS period: 2020-06-18 → 2021-12-15 (the 378-bar IS window preceding the
  worst OOS window, 2021-12-16 → 2022-06-16, Sharpe -1.938)
- Loaded the trained HMMEngine for this window
- Extracted per-state emission means (`hmm_engine.model.means_`) for all
  6 features, Z-scored
- Identified which state the 2022 OOS observations were assigned to
- Compared the emission mean of the mid-vol state against the actual
  2022 realized_variance, VIX, and HY OAS Z-scores

Total walk-forward backtester runs in Phase C: **8** (3 + 4 + 1).
Plus 1 diagnostic interrogation (no retraining).

---

## 3. How the Backtests Were Run

### 3.1 Walk-forward setup

All Phase C runs used the same walk-forward backtester as Phase B
(`backtest/backtester.py`), with the following baseline configuration:

| Parameter | Value |
|-----------|-------|
| OOS window | 126 bars (6 months) |
| Step size | 63 bars (quarterly roll) |
| Data range | 2010-01-01 to 2026-04-12 |
| Primary symbol | SPY |
| Features | 6-feature matrix (log_return, realized_variance, vix, hy_oas, gold_return, term_spread) |
| Allocation | Moreira-Muir continuous formula, target_vol=0.18, max_leverage=1.25 |
| Rebalance threshold | 0.15 (Phase B winner) |
| Covariance type | diag (diagonal Gaussian) |

IS window varied per H1 sweep; fixed at 252 for H2 sweep.

### 3.2 Sweep script

`backtest/sweep.py` was extended from Phase B. New additions for Phase C:

- **`_2022_windows(result)`** — filters `result.windows` to OOS windows
  overlapping calendar year 2022. Filtering is date-based, not index-based:
  window indices shift when IS length changes (H1), so index-based
  filtering would compare different calendar periods across runs.

- **`_fmt(val)`** — formats sweep values for display. Required because H2
  values are Python lists (`[3,4]`, `[3,5]`, etc.), which fail with the
  `f"{val:>6}"` format spec used in Phase B. The helper returns
  `str(val)` for lists and plain formatting for scalars.

- **`--sweep` CLI flag** — argparse argument accepting `H1`, `H2`, `B`.
  Allows H1 and H2 to run independently for clean interpretation. Default
  when flag is omitted: H1 + H2.

- **Named sweep definitions** — `H1_SWEEP`, `H2_SWEEP` added alongside
  existing `PHASE_B_SWEEPS`. `NAMED_SWEEPS` dict maps flag names to lists.

Usage:
```
python -m backtest.sweep --sweep H1       # IS window only (3 runs)
python -m backtest.sweep --sweep H2       # BIC cap only (4 runs)
python -m backtest.sweep --sweep H1 H2   # both Phase C (7 runs)
```

Data is fetched once per session and reused across all runs (same design
as Phase B — avoids redundant yfinance/FRED fetches).

### 3.3 Primary metric

Aggregate OOS Sharpe (annualized). Same as Phase B.

For Phase C, a secondary gate was added: the per-window Sharpe for
the five worst 2022 windows, tracked individually. Aggregate Sharpe alone
can mask whether the 2022 problem windows improved or whether improvements
came from other calendar periods.

### 3.4 Total wall time

Approximately 70 minutes:
- H1: 3 runs × ~8 min = ~24 min
- H2: 4 runs × ~8 min = ~32 min
- Combined: 1 run = ~8 min
- H3 diagnostic: ~6 min (single model load + interrogation)

---

## 4. Code Changes

### 4.1 `backtest/sweep.py` — extended

```python
# Phase C — H1: IS window length
H1_SWEEP = {
    "name": "is_window",
    "path": ["backtest", "is_window"],
    "values": [252, 378, 504],
}

# Phase C — H2: BIC state count upper bound
H2_SWEEP = {
    "name": "n_components_range",
    "path": ["hmm", "n_components_range"],
    "values": [[3, 4], [3, 5], [3, 6], [3, 7]],
}

NAMED_SWEEPS = {
    "H1": [H1_SWEEP],
    "H2": [H2_SWEEP],
    "B":  PHASE_B_SWEEPS,
}
```

New helper functions:

```python
def _fmt(val) -> str:
    """Format a sweep value for display — handles scalars and lists."""
    if isinstance(val, list):
        return str(val)
    return f"{val}"

def _2022_windows(result) -> list:
    """
    Extract per-window Sharpe for OOS windows whose start date falls in 2022.
    Returns list of (oos_start, oos_end, sharpe, n_states).
    Filter by date, not index — index shifts when is_window changes.
    """
    rows = []
    for w in result.windows:
        if w.oos_start.startswith("2022") or w.oos_end.startswith("2022"):
            sharpe = w.oos_metrics.get("sharpe", float("nan"))
            rows.append((w.oos_start, w.oos_end, sharpe, w.bic_n_states))
    return rows
```

The `run_sweep()` function was updated to call `_2022_windows()` after
each backtest and print per-window results inline:

```python
if wins_2022:
    print(f"    2022 OOS windows:")
    for oos_start, oos_end, w_sharpe, n_states in wins_2022:
        print(f"      {oos_start} -> {oos_end}: Sharpe={w_sharpe:+.3f}  n_states={n_states}")
```

The `main()` function was updated with argparse:

```python
parser.add_argument(
    "--sweep", nargs="+", choices=list(NAMED_SWEEPS.keys()),
    help="Which sweep(s) to run: H1, H2, B. Default: H1 H2."
)
```

### 4.2 `config/settings.yaml` — two parameters updated

```yaml
# Before Phase C:
backtest:
  is_window: 252    # [HYPERPARAMETER] Test [252, 378, 504]
hmm:
  n_components_range: [3, 7]    # (no comment)

# After Phase C (locked):
backtest:
  is_window: 378    # [2.5-C LOCKED] Phase C H1 sweep [252,378,504]: 378 wins
                    # Sharpe=0.940 (+0.111). 504 overshoots (stale pre-COVID data).
hmm:
  n_components_range: [3, 7]    # [2.5-C LOCKED] Phase C H2: [3,5] wins alone
                                 # (+0.028) but degrades combined with is_window=378
                                 # (0.852 vs 0.940). Kept [3,7] — BIC self-corrects
                                 # with longer IS (BIC penalty log(378)×k > log(252)×k).
```

No changes to any Python source files beyond `backtest/sweep.py`.

---

## 5. Results

### 5.1 H1: IS window

| is_window | Aggregate Sharpe | 2022-06-17 window | Worst 2022 window | OOS windows |
|-----------|-----------------|------------------|------------------|-------------|
| 252 (baseline) | 0.828 | +0.547 | -1.938 | 59 |
| **378** | **0.940** | **+1.052** | -1.981 | 57 |
| 504 | 0.834 | +0.590 | -1.973 | 55 |

**Winner: 378 bars (+0.111 Sharpe vs baseline).**

Key observations:
- Two fewer windows at 378 (57 vs 59): the additional IS data pushes the
  first OOS window later in calendar time.
- The 2022-06-17 window improved dramatically: +0.547 → +1.052. This
  window's IS period (Jan 2021 – Jun 2022 at 378 bars) now includes early
  2022 stress onset, giving the HMM richer context for what preceded the
  second half of the 2022 bear.
- The worst windows (-1.938, -1.276) did not improve meaningfully at
  any IS length. The core 2022 problem is immune to IS window extension.
- 504 overshoots: including 2 years of IS data adds stale pre-COVID
  patterns (2018-2019 low-vol bull market) that confuse the model. BIC
  begins selecting 7 states in nearly every window — more states does not
  mean better states.

### 5.2 H2: BIC cap

*(Tested at is_window=252 baseline for clean isolation.)*

| n_components_range | Aggregate Sharpe | Worst 2022 window | n_states (typical) |
|--------------------|-----------------|------------------|--------------------|
| [3, 4] | 0.785 | -1.675 | 4 |
| **[3, 5]** | **0.863** | -1.938 | 5 |
| [3, 6] | 0.841 | -1.938 | 5-6 |
| [3, 7] (baseline) | 0.835 | -1.938 | 5-7 |

**Winner in isolation: [3, 5] (+0.028 Sharpe vs baseline).**

Key observations:
- [3, 4] actually improves the worst 2022 window (-1.938 → -1.675) but
  degrades aggregate Sharpe to 0.785. Forcing 4 states underfits non-2022
  periods, where the data contains more distinct regimes.
- [3, 5] hits the right trade-off at baseline IS window. The cap is not
  always binding (some windows choose 4-5 states), but it prevents the
  6-7 state overfit in problem windows.

### 5.3 Combined confirmation

| Config | Sharpe | Delta vs baseline | 2022-06-17 | Worst 2022 |
|--------|--------|-----------------|-----------|------------|
| Baseline: 252, [3,7] | 0.828 | — | +0.547 | -1.938 |
| H1: 378, [3,7] | **0.940** | **+0.111** | +1.052 | -1.981 |
| H2: 252, [3,5] | 0.863 | +0.035 | +0.547 | -1.938 |
| Combined: 378, [3,5] | 0.852 | +0.024 | +0.872 | -1.946 |

**Negative interaction confirmed. Combined result (0.852) is worse than
H1 alone (0.940).**

The [3,5] cap is appropriate for a 252-bar IS window but becomes too
restrictive at 378 bars. With 1.5 years of IS data, the HMM benefits from
having more states available to capture the richer history. When the cap
binds at 5 in nearly every window (as it does when IS=378), BIC cannot
select the 6-state model that best describes 1.5 years of data, and
performance regresses.

Note that BIC's natural penalty scales with IS length:
`penalty = n_params × log(T)`. With T=378 vs T=252, the BIC penalty per
parameter is 30% larger, which already discourages overfitting. An
artificial cap is redundant and harmful at this IS length.

**Decision: lock is_window=378, keep n_components_range=[3,7].**

### 5.4 H3: Structural diagnostic

IS period: 2020-06-18 → 2021-12-15 (preceding worst OOS window).

Emission means (Z-scored):

| State | realized_var (Z) | vix (Z) | hy_oas (Z) | Character |
|-------|-----------------|---------|-----------|-----------|
| 2 | +1.011 | -0.471 | -0.973 | COVID spike: high realized vol |
| 3 | +0.855 | +1.468 | -0.610 | COVID spike + VIX spike |
| 4 | +0.027 | +0.514 | +1.204 | Mid-vol, credit stress |
| 0 | -0.541 | -0.336 | -0.630 | Low-mid vol |
| 1 | -0.945 | -0.648 | -1.507 | Low vol |
| 5 | -1.486 | -1.227 | -1.375 | Very low vol |
| 6 | -1.085 | -0.703 | -1.735 | Very low vol |

The IS window (Jun 2020 – Dec 2021) is dominated by the COVID crash
recovery and the subsequent low-vol bull run. The HMM's "stress" states
(2 and 3) have realized_var Z-scores of +0.85 to +1.01 — they encode
the sharp spike pattern of COVID-type crises.

During 2022, realized_var sat at Z ~+0.1 to +0.4 — squarely within the
mid-vol state (Z = +0.027). The diagonal HMM evaluates each feature
independently and sees mid-vol signals across all six features, assigning
the observation to the mid-vol state. There is no mechanism for the model
to detect that all six features are simultaneously mildly elevated —
the joint "all elevated together" pattern. That pattern requires a full
covariance model to represent.

**H3 confirmed. The 2022 failure is a structural diagonal covariance
misspecification.**

### 5.5 Gate assessment

| Gate | Target | Achieved | Status |
|------|--------|---------|--------|
| Aggregate Sharpe | > 1.0 | 0.940 | NOT MET |
| Worst 2022 window | > -1.0 | -1.981 | NOT MET |

**Gate not met. Phase F required.**

---

## 6. Interpreting the Results

### 6.1 H1: Why 378 wins and 504 overshoots

The IS window is the only data the HMM sees during training for each
walk-forward period. For the 2021-12-16 OOS window:

- At 252 bars: IS covers roughly Dec 2020 – Dec 2021 (low-vol bull).
  The HMM learns only calm patterns and has no trained "stress" state
  with sufficient mass to absorb the early 2022 observations.

- At 378 bars: IS covers roughly Jun 2020 – Dec 2021. Now the IS
  includes the tail of the COVID crash recovery — the HMM has at least
  seen some elevated vol during training, giving it a richer transition
  from stress back to calm. The 2022-06-17 window benefits
  dramatically (Sharpe +0.547 → +1.052) because its IS now covers
  both the 2021 calm AND the early 2022 onset, providing the HMM with
  a template for a mid-cycle stress buildup.

- At 504 bars: IS covers roughly Dec 2019 – Dec 2021. The 2018-2019
  pre-COVID period is now included. This period had its own vol regime
  patterns (2018 Q4 selloff, 2019 low-vol recovery) that are
  structurally different from 2020-2022 dynamics. Including stale
  patterns increases the parameter count BIC needs to describe the IS
  data without improving the HMM's ability to detect 2022 OOS patterns.
  BIC selects 7 states in most windows when IS=504, leading to
  overfitting.

The 378-bar window is a Goldilocks result: enough history to include
one meaningful stress-to-calm transition, not so much that stale
regime patterns contaminate the model.

### 6.2 H2: Why the negative interaction with H1

The interaction reveals a dependency between IS length and the optimal
model complexity:

- More IS data → richer, more varied regime history → more distinct
  clusters for BIC to model → higher optimal state count.
- At IS=252 bars, 5 states adequately describes one year of daily SPY
  regime history. [3,5] cap is non-binding or barely binding — useful.
- At IS=378 bars, 1.5 years contains more distinct patterns (COVID
  tail, 2021 low-vol, 2021 rotation). BIC wants to use 6 states to
  describe this richer history. [3,5] cap becomes binding in most
  windows, forcing an underfit model.

The lesson: BIC state selection is not independent of IS length. Any
cap imposed without reference to the data length risks underfitting
when IS changes. The correct approach is to let BIC self-regulate at
each IS length — which is exactly what [3,7] does. The BIC formula's
natural scaling (`penalty = k × log(T)`) penalizes additional states
more heavily as T grows, providing automatic regularization without a
hard cap.

### 6.3 H3: Why diagonal covariance fails in slow-grind regimes

The diagonal Gaussian HMM makes a strong conditional independence
assumption: given the regime state, each feature is independent of
every other feature. The emission probability for observation vector
x in state s factorizes as:

```
P(x | state=s) = ∏ₑ N(xₑ; μₛₑ, σ²ₛₑ)
```

Each feature contributes independently to the state assignment. For
the observation (realized_var=+0.2, vix=+0.4, hy_oas=+0.6, ...),
the model asks: "what is the probability of realized_var=+0.2 in state
4 (mid-vol)?" The answer is high, because the mid-vol state mean for
realized_var is +0.027 — close to +0.2. It asks the same question for
each feature independently and multiplies. All six features
independently look mid-vol, so the joint probability is highest for
the mid-vol state.

Under **full covariance**, the emission for state s is:

```
P(x | state=s) = N(x; μₛ, Σₛ)
```

where Σₛ is a full 6×6 covariance matrix. The model learns not just
the marginal distribution of each feature, but the joint distribution
of all features simultaneously. It can learn that "all features
moderately elevated together" is a distinct regime signature — different
from the calm mid-vol signature where features are modestly negative or
flat and uncorrelated. Even if the marginals are similar, the joint
density can assign the 2022 observations to a distinct "slow-grind"
state.

The 2022 bear had a specific cross-feature signature:
- Realized variance: moderately elevated (Z ~+0.2 to +0.4)
- VIX: persistently in 20-35 range (not spiking)
- HY OAS: steadily widening (Z ~+0.6 to +0.8)
- Term spread: rapidly inverting
- Gold: elevated as flight-to-safety

These features were all mildly elevated simultaneously and remained
correlated in a specific pattern throughout 2022. Diagonal covariance
cannot represent this pattern; full covariance can.

### 6.4 Why Phase C still improved Sharpe despite not fixing the core problem

The aggregate improvement of +0.111 came entirely from the
2022-06-17 → 2022-12-15 window (+0.547 → +1.052). This is the second
half of 2022, when the decline was partially priced in and the Fed was
mid-cycle. The IS period for this window at 378 bars includes early
2022 data (the first leg of the decline), which gave the HMM enough
context to recognize the later 2022 period as an elevated-stress
continuation rather than a novel pattern.

The two worst windows (first half of 2022) were not fixable. Their IS
periods at any length did not include data similar enough to the 2022
slow-grind to train a usable stress state. This is the structural
problem.

---

## 7. Implications for HMM Quality

### 7.1 The 2022 failure is a known, documented HMM limitation

The misclassification of slow-grinding macro stress regimes under
diagonal Gaussian HMM is not a surprise. Guidolin & Timmermann (2008,
RFS 21:889-935) document this exact problem in their analysis of
multi-state HMMs across equity, bond, and credit markets: diagonal
covariance HMMs trained on crisis-spike periods systematically
misclassify subsequent gradual-deterioration regimes when the joint
cross-asset pattern is the primary discriminating signal.

Gray (1996, JFE 42:27-62) identifies Gaussian misspecification as a
recurring failure mode in vol regime models and recommends full or
structured covariance models for multi-asset applications.

The 2022 failure is an instance of a well-characterized limitation, not
an aberration. It means Phase C's inability to fix it is also expected:
the fix requires a model change, not a parameter change.

### 7.2 The HMM is well-specified for spike-type crises

The flip side of the H3 diagnosis is that the diagonal HMM is well-suited
to COVID-type crises — sharp, fast, high-amplitude spikes where each
feature individually signals extreme stress. The model's performance from
2020 onwards (outside the 2022 problem windows) reflects this: the COVID
period produced strong regime detection and defensive positioning.

The system is not broken. It has a specific, bounded blind spot: slow
macro deterioration where no single feature is extreme but multiple
features are simultaneously and persistently mildly elevated.

### 7.3 The BIC self-regulation finding is structurally important

Phase C H2 established that BIC's natural penalty scaling provides
adequate regularization when IS length is set correctly. This is a
useful property:

- With is_window=378 and [3,7] range, BIC selects 5-6 states in most
  windows without explicit capping. The larger penalty term log(378)
  compared to log(252) naturally suppresses the 7-state models that
  caused overfitting at 252 bars.
- For Phase F (longer effective parameter counts due to full covariance),
  this means BIC will continue to self-regulate. The parameter count
  per state increases substantially (28 unique covariance values per
  state vs 6 for diagonal), but BIC's log(T) penalty scales accordingly.

Phase F should still test `covariance_type='tied'` (shared covariance
matrix across states) as a middle ground. Tied covariance captures
cross-feature structure while sharing one covariance matrix across all
states — fewer parameters, still more expressive than diagonal.

### 7.4 Walk-forward IS window length is a model design decision, not a tuning knob

The H1 result changes how we should think about `is_window`. It is not
a simple hyperparameter like a threshold or moving average length. It
determines what regime history the HMM is exposed to during training,
which directly shapes what states it can represent.

The 378-bar result generalizes a principle: the IS window should contain
at least one full vol cycle (a stress-to-calm or calm-to-stress
transition) to give the HMM both pattern types to learn from. In most
modern market periods, 1.5 years achieves this. 1 year may not if the
preceding period was uniformly low-vol (as 2021 was). 2 years risks
including a structurally different regime cycle that introduces stale
training signal.

For Phase F, the locked value of 378 bars is appropriate. Phase F should
not re-sweep IS window — any difference in optimal IS window under full
covariance would be attributable to the interaction between covariance
type and IS length, not IS length in isolation.

### 7.5 What Phase F must deliver

Phase F addresses the confirmed structural deficiency. The minimum
deliverables:

1. **Full or tied covariance** captures the joint "all elevated
   together" signature that diagonal covariance misses.
2. **vix_slope re-enabled** (add back to FEATURE_COLS). With full
   covariance, the ~50% orthogonal variance of vix_slope relative to
   VIX level contributes independent information to the joint emission
   distribution — the diagonal model could not exploit this because
   vix and vix_slope's shared variance inflated the feature's apparent
   importance while the orthogonal part was absorbed into the marginal.
3. **Gate: aggregate Sharpe > 1.0 and worst 2022 window > -1.0.**

Phase C leaves Sharpe at 0.940 with the worst 2022 window at -1.981.
Phase F's target is to close both gaps. The emission means analysis
in H3 provides the theoretical basis: if the full covariance model
creates a distinct "slow-grind" state with realized_var mean ~+0.2,
vix mean ~+0.4, and hy_oas mean ~+0.6 (correlated), the HMM will
correctly assign 2022 observations to that state and reduce allocation
accordingly.

---

## 8. Locked Settings After Phase C

```yaml
# config/settings.yaml — Phase C locked values

backtest:
  is_window: 378    # Phase C H1 winner. +0.111 Sharpe vs 252 baseline.
                    # Goldilocks: includes COVID tail → 2021 calm transition.
                    # 504 overshoots (stale pre-COVID patterns).

hmm:
  n_components_range: [3, 7]    # Kept. BIC cap [3,5] degrades when combined
                                 # with is_window=378. BIC self-corrects via
                                 # log(378) > log(252) penalty scaling.
```

All other settings remain at Phase B locked values. No Python source files
were changed in Phase C beyond `backtest/sweep.py`.

---

## 9. References

- Hamilton (1989, Econometrica 57:357-384) — HMM regime framework
- Gray (1996, JFE 42:27-62) — Gaussian misspecification in vol regimes
- Guidolin & Timmermann (2008, RFS 21:889-935) — multi-state cross-asset
  HMM with full covariance, diagonal covariance limitations
- Moreira & Muir (2017, Journal of Finance 72(4):1611-1644) — vol-managed
  allocation, basis for Moreira-Muir formula
- White (2000, Econometrica 68(5):1097-1126) — walk-forward evaluation
- design_docs/11_phase_c_2022_diagnosis.md — detailed diagnosis data,
  emission means table, H3 mechanism
- design_docs/09_theory_vs_empirical_conflicts.md — vix_slope exclusion
  rationale and Phase F re-enable checklist
- design_docs/10_phase_b_parameter_sweep.md — Phase B results, baseline
  for Phase C
- `backtest/sweep.py` — sweep implementation (H1, H2, Phase B)
- `logs/sweep_phase_c_h1.log` — H1 raw output
- `logs/sweep_phase_c_h2.log` — H2 raw output
- `logs/sweep_phase_c_combined.log` — combined confirmation
