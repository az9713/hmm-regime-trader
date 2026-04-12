# tests/test_strategies.py

import pytest
import numpy as np


def make_settings():
    return {
        "allocation": {
            "use_continuous_formula": True,
            "target_vol": 0.20,
            "rebalance_threshold": 0.10,
            "low_vol": {"allocation": 0.95, "leverage": 1.25, "max_leverage": 1.25},
            "mid_vol": {"allocation": 0.65, "leverage": 1.00, "max_leverage": 1.25},
            "high_vol": {"allocation": 0.35, "leverage": 0.50, "max_leverage": 1.25},
            "uncertainty": {"allocation": 0.50, "leverage": 0.75},
        },
        "stops": {
            "low_vol": {"stop_atr": 3.0, "target_atr": 6.0},
            "mid_vol": {"stop_atr": 2.5, "target_atr": 5.0},
            "high_vol": {"stop_atr": 2.0, "target_atr": 4.0},
        },
        "sizing": {
            "risk_per_trade": 0.01,
            "kelly_fraction": {
                "low_vol": 0.333, "mid_vol": 0.500, "high_vol": 0.200, "uncertainty": 0.200
            },
        },
        "trend": {"fast_ema": 50, "slow_ema": 200},
    }


class TestMoreiraMuirAllocation:
    """
    Verify continuous formula: w = min(target_vol / realized_vol, max_leverage)
    Ref: Moreira & Muir (2017, Journal of Finance 72(4):1611-1644)
    """

    def test_low_vol_produces_high_allocation(self):
        from core.regime_strategies import moreira_muir_allocation
        # VIX ~12 → sigma ~0.12 → w = 0.20/0.12 = 1.67 → capped at 1.25
        alloc = moreira_muir_allocation(ewma_realized_vol=0.12, target_vol=0.20, max_leverage=1.25)
        assert alloc == 1.25, f"Expected 1.25 (capped), got {alloc}"

    def test_high_vol_produces_low_allocation(self):
        from core.regime_strategies import moreira_muir_allocation
        # VIX ~30 → sigma ~0.30 → w = 0.20/0.30 = 0.667
        alloc = moreira_muir_allocation(ewma_realized_vol=0.30, target_vol=0.20, max_leverage=1.25)
        assert abs(alloc - 0.667) < 0.01, f"Expected ~0.667, got {alloc}"

    def test_target_vol_equals_realized_gives_full_allocation(self):
        from core.regime_strategies import moreira_muir_allocation
        alloc = moreira_muir_allocation(ewma_realized_vol=0.20, target_vol=0.20, max_leverage=1.25)
        assert abs(alloc - 1.0) < 1e-9

    def test_zero_vol_returns_zero(self):
        from core.regime_strategies import moreira_muir_allocation
        alloc = moreira_muir_allocation(ewma_realized_vol=0.0, target_vol=0.20, max_leverage=1.25)
        assert alloc == 0.0

    def test_allocation_bounded_by_max_leverage(self):
        from core.regime_strategies import moreira_muir_allocation
        alloc = moreira_muir_allocation(ewma_realized_vol=0.01, target_vol=0.20, max_leverage=1.25)
        assert alloc <= 1.25


class TestRegimeStrategies:
    def test_all_strategies_instantiate(self):
        from core.regime_strategies import LowVolStrategy, MidVolStrategy, HighVolStrategy, UncertaintyStrategy
        s = make_settings()
        for cls in [LowVolStrategy, MidVolStrategy, HighVolStrategy, UncertaintyStrategy]:
            inst = cls(s)
            assert inst.get_stop_atr_multiple() > 0
            assert inst.get_target_atr_multiple() > 0

    def test_get_strategy_dispatch(self):
        from core.regime_strategies import get_strategy, LowVolStrategy, HighVolStrategy, UncertaintyStrategy
        from core.hmm_engine import RegimeLabel
        s = make_settings()
        assert isinstance(get_strategy(RegimeLabel.LOW_VOL, s), LowVolStrategy)
        assert isinstance(get_strategy(RegimeLabel.HIGH_VOL, s), HighVolStrategy)
        assert isinstance(get_strategy(RegimeLabel.UNCERTAINTY, s), UncertaintyStrategy)

    def test_uncertainty_strategy_blocks_new_entries(self):
        from core.regime_strategies import UncertaintyStrategy
        s = make_settings()
        strat = UncertaintyStrategy(s)
        # Trend filter always returns False in uncertainty mode
        assert not strat.trend_filter_passes(ema_fast=300, ema_slow=200)

    def test_high_vol_lower_allocation_than_low_vol(self):
        from core.regime_strategies import LowVolStrategy, HighVolStrategy
        s = make_settings()
        low_strat = LowVolStrategy(s)
        high_strat = HighVolStrategy(s)
        ewma_vol = 0.25  # same vol input
        low_alloc = low_strat.get_allocation(ewma_vol)
        high_alloc = high_strat.get_allocation(ewma_vol)
        # Both use same Moreira-Muir formula but different max_leverage caps
        # high_vol max_leverage = 1.25 same, so diff is only in discrete fallback
        # Test with discrete buckets
        s2 = dict(s)
        s2["allocation"] = dict(s["allocation"])
        s2["allocation"]["use_continuous_formula"] = False
        low_strat2 = LowVolStrategy(s2)
        high_strat2 = HighVolStrategy(s2)
        assert low_strat2.get_allocation(0.0) > high_strat2.get_allocation(0.0)
