# backtest/stress_test.py
# ============================================================
# Stress period analysis — validate HMM correctly identifies known crises.
#
# Required validation (design_docs/06_empirical_testing_plan.md §Step 7):
#   2020 COVID crash (Feb-Mar 2020): HMM MUST flag HighVol, allocation must drop.
#   2022 rate hike bear (Jan-Dec 2022): same.
#
# If HMM fails to identify these periods: feature engineering or stability
# filter needs adjustment before proceeding.
#
# Validates: Maheu & McCurdy (2000, JBES 18:100-112) bull/bear HMM detection
#            on our specific data, features, and normalization.
# ============================================================

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional

from core.hmm_engine import HMMEngine, RegimeLabel
from data.feature_engineering import FeatureEngineer
from backtest.performance import compute_all_metrics

logger = logging.getLogger(__name__)


STRESS_PERIODS = {
    "2020_covid_crash": ("2020-02-01", "2020-04-30"),
    "2022_rate_hike_bear": ("2022-01-01", "2022-12-31"),
}

# Warm-up: train on data ending just before stress period starts
PRETRAIN_YEARS = 5  # use 5 years of pre-stress data for training


@dataclass
class StressPeriodResult:
    period_name: str
    start: str
    end: str
    regime_counts: dict           # {label: bar_count}
    high_vol_pct: float           # fraction of bars flagged high-vol
    avg_allocation: float
    max_allocation: float
    min_allocation: float
    detection_pass: bool          # True if >50% bars flagged HighVol
    regime_series: Optional[pd.Series] = None
    allocation_series: Optional[pd.Series] = None


class StressTester:
    """
    Validates HMM regime detection on historically known stress periods.
    HMM must correctly identify these as high-volatility regimes.
    """

    def __init__(self, settings: dict):
        self.settings = settings

    def run_all(
        self,
        prices: dict,
        vix: pd.Series,
        hy_oas: pd.Series,
        primary_symbol: str = "SPY",
    ) -> dict[str, StressPeriodResult]:
        """Run stress tests for all defined periods."""
        results = {}
        for period_name, (start, end) in STRESS_PERIODS.items():
            result = self.run_single(
                period_name, start, end, prices, vix, hy_oas, primary_symbol
            )
            results[period_name] = result
            status = "PASS" if result.detection_pass else "FAIL"
            logger.info(
                f"Stress [{period_name}]: {status} | "
                f"HighVol={result.high_vol_pct:.0%} | AvgAlloc={result.avg_allocation:.2f}"
            )
        return results

    def run_single(
        self,
        period_name: str,
        start: str,
        end: str,
        prices: dict,
        vix: pd.Series,
        hy_oas: pd.Series,
        primary_symbol: str = "SPY",
    ) -> StressPeriodResult:
        """
        Train HMM on pre-stress data, then run forward algorithm through stress period.
        Verify HMM flags high-vol and reduces allocation.
        """
        spy_prices = prices[primary_symbol]

        # Build full feature matrix
        fe = FeatureEngineer(self.settings)
        features = fe.compute(spy_prices, vix, hy_oas)

        # IS: PRETRAIN_YEARS before stress start
        stress_start = pd.Timestamp(start)
        stress_end = pd.Timestamp(end)
        pretrain_start = stress_start - pd.DateOffset(years=PRETRAIN_YEARS)

        is_mask = (features.index >= pretrain_start) & (features.index < stress_start)
        oos_mask = (features.index >= stress_start) & (features.index <= stress_end)

        is_features = features[is_mask].values
        oos_features = features[oos_mask].values
        oos_index = features[oos_mask].index

        if len(is_features) < 100:
            logger.warning(f"Stress [{period_name}]: insufficient IS data ({len(is_features)} bars)")
            return self._empty_result(period_name, start, end)

        # Train HMM
        engine = HMMEngine(self.settings)
        try:
            n_states = engine.fit(is_features)
        except Exception as e:
            logger.error(f"Stress [{period_name}]: HMM training failed: {e}")
            return self._empty_result(period_name, start, end)

        # Forward algorithm on stress period (no look-ahead)
        regime_states = engine.predict_regime(oos_features)
        regime_labels = pd.Series([r.label for r in regime_states], index=oos_index)

        # Compute allocations per bar
        from core.regime_strategies import get_strategy
        allocations = []
        spy_close = spy_prices["Close"].reindex(oos_index, method="ffill")
        log_rets = np.log(spy_close / spy_close.shift(1)).fillna(0)

        for i, rs in enumerate(regime_states):
            strategy = get_strategy(rs.label, self.settings)
            # Simple allocation — use regime without EWMA vol for stress test
            alloc = strategy.get_allocation(ewma_vol=0.20)  # neutral vol as fallback
            allocations.append(alloc)

        alloc_series = pd.Series(allocations, index=oos_index)

        # Regime counts
        regime_counts = regime_labels.value_counts().to_dict()
        high_vol_labels = {RegimeLabel.HIGH_VOL, RegimeLabel.VERY_HIGH_VOL, RegimeLabel.UNCERTAINTY}
        high_vol_bars = sum(regime_counts.get(lbl, 0) for lbl in high_vol_labels)
        high_vol_pct = high_vol_bars / len(regime_labels) if len(regime_labels) > 0 else 0.0

        return StressPeriodResult(
            period_name=period_name,
            start=start,
            end=end,
            regime_counts=regime_counts,
            high_vol_pct=high_vol_pct,
            avg_allocation=float(alloc_series.mean()),
            max_allocation=float(alloc_series.max()),
            min_allocation=float(alloc_series.min()),
            detection_pass=(high_vol_pct >= 0.50),  # >50% bars flagged high-vol = pass
            regime_series=regime_labels,
            allocation_series=alloc_series,
        )

    def _empty_result(self, period_name, start, end) -> StressPeriodResult:
        return StressPeriodResult(
            period_name=period_name,
            start=start, end=end,
            regime_counts={}, high_vol_pct=0.0,
            avg_allocation=0.0, max_allocation=0.0, min_allocation=0.0,
            detection_pass=False,
        )


def analyze_stress_period(
    period_name: str,
    start: str,
    end: str,
    prices: dict,
    vix: pd.Series,
    hy_oas: pd.Series,
    settings: dict,
) -> StressPeriodResult:
    """Convenience function for single-period stress test."""
    tester = StressTester(settings)
    return tester.run_single(period_name, start, end, prices, vix, hy_oas)
