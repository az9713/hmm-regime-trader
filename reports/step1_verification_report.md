# Step 1 — Verification Report

**Date:** 2026-04-11  
**Machine:** Windows 11 Home 10.0.26200  
**Python:** 3.13.5  

---

## 1. Python Version

```
Python 3.13.5
```
**Status: PASS** — meets requirement (Python 3.10+)

---

## 2. Dependency Installation

Command: `pip install -r requirements.txt`

| Package | Version Installed | Status |
|---|---|---|
| hmmlearn | 0.3.3 | Already installed |
| pandas | 2.3.3 | Already installed |
| numpy | 2.2.6 | Already installed |
| yfinance | 1.1.0 | Already installed |
| pandas-datareader | 0.10.0 | Already installed |
| alpaca-py | 0.43.2 | **Freshly installed** |
| pyyaml | 6.0.2 | Already installed |
| python-dotenv | 1.2.1 | Already installed |
| scipy | 1.16.0 | Already installed |
| scikit-learn | 1.7.0 | Already installed |
| matplotlib | 3.9.3 | Already installed |
| pytest | 8.4.2 | Already installed |
| lxml | 6.0.0 | Already installed (required by pandas-datareader for FRED) |

Additional packages installed as dependencies of alpaca-py:
- `sseclient-py` 1.9.0 (SSE streaming for Alpaca WebSocket)

**Status: PASS** — all dependencies satisfied.

---

## 3. Import Verification

Command: `python -c "import yfinance, pandas_datareader, hmmlearn, alpaca, scipy, sklearn, matplotlib, yaml, dotenv; print('ALL IMPORTS OK')"`

```
ALL IMPORTS OK
```

**Status: PASS** — all modules import without error.

---

## 4. Test Suite

Command: `python -m pytest tests/ -v`

```
platform win32 -- Python 3.13.5, pytest-8.4.2, pluggy-1.6.0
rootdir: C:\Users\simon\Downloads\trading_bot_ai_advantage\regime-trader
collected 47 items
```

### Results by Module

**test_backtest.py** (10 tests)
```
TestPerformanceMetrics::test_sharpe_positive_returns          PASSED
TestPerformanceMetrics::test_sharpe_negative_returns          PASSED
TestPerformanceMetrics::test_max_drawdown_flat                PASSED
TestPerformanceMetrics::test_max_drawdown_50pct               PASSED
TestPerformanceMetrics::test_cagr_positive                    PASSED
TestPerformanceMetrics::test_regime_conditional_returns_structure PASSED
TestPerformanceMetrics::test_sortino_ignores_positive_returns PASSED
TestHansenSPATest::test_spa_structure                         PASSED
TestHansenSPATest::test_clearly_superior_strategy_low_pvalue  PASSED
TestWalkForwardStructure::test_oos_does_not_overlap_is        PASSED
```

**test_hmm.py** (10 tests)
```
TestHMMEngine::test_fit_returns_valid_state_count             PASSED
TestHMMEngine::test_state_labels_assigned                     PASSED
TestHMMEngine::test_needs_retrain_after_n_steps               PASSED
TestForwardAlgorithm::test_forward_uses_only_past_data        PASSED  ← look-ahead prevention verified
TestForwardAlgorithm::test_forward_output_shape               PASSED
TestForwardAlgorithm::test_confidence_sums_to_one             PASSED
TestRegimeLabeling::test_low_vol_state_has_lowest_variance    PASSED  ← variance labeling verified
TestStabilityFilters::test_persistence_delays_switch          PASSED
TestStabilityFilters::test_persistence_commits_after_3_bars   PASSED
TestStabilityFilters::test_confidence_floor_triggers_uncertainty PASSED
```

**test_risk.py** (12 tests)
```
TestCircuitBreakers::test_no_trigger_below_warn               PASSED
TestCircuitBreakers::test_warn_at_2pct                        PASSED
TestCircuitBreakers::test_pause_at_3pct                       PASSED
TestCircuitBreakers::test_halt_at_5pct                        PASSED
TestCircuitBreakers::test_stop_at_10pct                       PASSED
TestCircuitBreakers::test_thresholds_are_monotone             PASSED  ← breaker ordering verified
TestCorrelationGates::test_high_correlation_blocked           PASSED
TestCorrelationGates::test_low_correlation_approved           PASSED
TestCorrelationGates::test_no_existing_positions_approved     PASSED
TestKellySizing::test_positive_kelly_with_edge                PASSED
TestKellySizing::test_zero_shares_with_no_edge                PASSED
TestKellySizing::test_zero_shares_with_zero_price             PASSED
```

**test_signals.py** (6 tests)
```
TestAlwaysLong::test_low_vol_regime_is_long                   PASSED  ← always-long verified
TestAlwaysLong::test_high_vol_regime_no_short                 PASSED
TestAlwaysLong::test_uncertainty_regime_no_short              PASSED
TestSignalGenerator::test_trend_filter_blocks_downtrend       PASSED
TestSignalGenerator::test_max_positions_blocks_signal         PASSED
TestSignalGenerator::test_compute_atr_positive                PASSED
```

**test_strategies.py** (9 tests)
```
TestMoreiraMuirAllocation::test_low_vol_produces_high_allocation      PASSED  ← w=1.25 at σ=0.12
TestMoreiraMuirAllocation::test_high_vol_produces_low_allocation      PASSED  ← w=0.667 at σ=0.30
TestMoreiraMuirAllocation::test_target_vol_equals_realized_gives_full PASSED  ← w=1.0 at σ=target
TestMoreiraMuirAllocation::test_zero_vol_returns_zero                 PASSED
TestMoreiraMuirAllocation::test_allocation_bounded_by_max_leverage    PASSED  ← cap enforced
TestRegimeStrategies::test_all_strategies_instantiate                 PASSED
TestRegimeStrategies::test_get_strategy_dispatch                      PASSED
TestRegimeStrategies::test_uncertainty_strategy_blocks_new_entries    PASSED
TestRegimeStrategies::test_high_vol_lower_allocation_than_low_vol     PASSED
```

### Final Result

```
47 passed in 104.92s (0:01:44)
```

**Status: PASS — 47/47 tests passed. 0 failures. 0 errors.**

---

## 5. Summary

| Check | Result |
|---|---|
| Python version (≥3.10) | PASS — 3.13.5 |
| All deps installed | PASS — 13/13 packages |
| lxml present (FRED parsing) | PASS — 6.0.0 |
| alpaca-py present (broker) | PASS — 0.43.2 freshly installed |
| All imports succeed | PASS |
| Test suite | PASS — 47/47 |
| Look-ahead prevention test | PASS |
| Moreira-Muir formula tests | PASS |
| Circuit breaker ordering test | PASS |
| Always-long enforcement test | PASS |

**Environment is fully ready for Step 2 (baseline backtest).**

---

## Next Step (Step 2)

When ready, run:

```bash
cd C:\Users\simon\Downloads\trading_bot_ai_advantage\regime-trader
python main.py --backtest
```

Expected duration: 2–10 minutes (first run downloads data, then simulates 2010→2026).  
Results written to: `logs/backtest_YYYYMMDD_HHMMSS.jsonl`  
Save the console output to `reports/step2_backtest_results.txt` for review.
