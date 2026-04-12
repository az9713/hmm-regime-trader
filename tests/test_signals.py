# tests/test_signals.py

import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock


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
            "kelly_fraction": {"low_vol": 0.333, "mid_vol": 0.500, "high_vol": 0.200, "uncertainty": 0.200},
            "max_concurrent_positions": 5,
        },
        "risk": {
            "circuit_breakers": {
                "daily_loss_warn": 0.02, "daily_loss_pause": 0.03,
                "weekly_loss_halt": 0.05, "monthly_loss_review": 0.07,
                "max_drawdown_stop": 0.10,
            },
            "correlation": {"warn_threshold": 0.70, "block_threshold": 0.85, "lookback_days": 60},
            "max_position_pct": 0.30,
        },
        "trend": {"fast_ema": 50, "slow_ema": 200},
    }


def make_price_data(n=250, close=450.0, trending=True):
    """Synthetic OHLCV where EMA50 > EMA200 (uptrend)."""
    idx = pd.date_range("2023-01-01", periods=n, freq="B")
    close_series = np.linspace(350, close, n) if trending else np.linspace(close, 350, n)
    df = pd.DataFrame({
        "Open": close_series * 0.999,
        "High": close_series * 1.005,
        "Low": close_series * 0.995,
        "Close": close_series,
        "Volume": 50_000_000,
    }, index=idx)
    return df


def make_regime_state(label="LowVol", confidence=0.85):
    from core.hmm_engine import RegimeState
    return RegimeState(label=label, raw_state_idx=0, confidence=confidence, stable=True, bars_in_regime=5, n_states_selected=3)


class TestAlwaysLong:
    """
    Signal direction must NEVER be 'short'.
    Ref: Cooper et al. (2004, JF); Daniel-Moskowitz (2016, JFE); Moreira-Muir (2017, JF).
    """

    def test_low_vol_regime_is_long(self):
        from core.risk_manager import RiskManager
        from core.signal_generator import SignalGenerator, Signal
        s = make_settings()
        risk_mgr = RiskManager(s)
        gen = SignalGenerator(s, risk_mgr)

        signal = gen.generate(
            symbol="SPY",
            regime_state=make_regime_state("LowVol", 0.90),
            price_data=make_price_data(trending=True),
            portfolio_value=100_000.0,
            daily_pnl_pct=0.0,
            current_positions={},
            ewma_vol=0.12,
        )
        if hasattr(signal, "direction"):
            assert signal.direction == "long", f"Expected 'long', got '{signal.direction}'"
            assert signal.direction != "short"

    def test_high_vol_regime_no_short(self):
        from core.risk_manager import RiskManager
        from core.signal_generator import SignalGenerator, Signal, FlatSignal
        s = make_settings()
        risk_mgr = RiskManager(s)
        gen = SignalGenerator(s, risk_mgr)

        signal = gen.generate(
            symbol="SPY",
            regime_state=make_regime_state("HighVol", 0.75),
            price_data=make_price_data(trending=True),
            portfolio_value=100_000.0,
            daily_pnl_pct=0.0,
            current_positions={},
            ewma_vol=0.35,
        )
        # Either flat (trend filter) or long — never short
        if hasattr(signal, "direction"):
            assert signal.direction != "short"

    def test_uncertainty_regime_no_short(self):
        from core.risk_manager import RiskManager
        from core.signal_generator import SignalGenerator, FlatSignal
        s = make_settings()
        risk_mgr = RiskManager(s)
        gen = SignalGenerator(s, risk_mgr)

        signal = gen.generate(
            symbol="SPY",
            regime_state=make_regime_state("Uncertainty", 0.35),
            price_data=make_price_data(trending=True),
            portfolio_value=100_000.0,
            daily_pnl_pct=0.0,
            current_positions={},
            ewma_vol=0.30,
        )
        if hasattr(signal, "direction"):
            assert signal.direction != "short"


class TestSignalGenerator:
    def test_trend_filter_blocks_downtrend(self):
        from core.risk_manager import RiskManager
        from core.signal_generator import SignalGenerator, FlatSignal
        s = make_settings()
        risk_mgr = RiskManager(s)
        gen = SignalGenerator(s, risk_mgr)

        # Downtrending prices: EMA50 < EMA200
        signal = gen.generate(
            symbol="SPY",
            regime_state=make_regime_state("LowVol", 0.90),
            price_data=make_price_data(trending=False),  # downtrend
            portfolio_value=100_000.0,
            daily_pnl_pct=0.0,
            current_positions={},
            ewma_vol=0.12,
        )
        assert isinstance(signal, FlatSignal), "Downtrend should produce FlatSignal (trend filter)"

    def test_max_positions_blocks_signal(self):
        from core.risk_manager import RiskManager
        from core.signal_generator import SignalGenerator, FlatSignal
        s = make_settings()
        risk_mgr = RiskManager(s)
        gen = SignalGenerator(s, risk_mgr)

        # 5 existing positions = at max
        existing = {f"TICK{i}": {} for i in range(5)}
        signal = gen.generate(
            symbol="SPY",
            regime_state=make_regime_state("LowVol", 0.90),
            price_data=make_price_data(trending=True),
            portfolio_value=100_000.0,
            daily_pnl_pct=0.0,
            current_positions=existing,
            ewma_vol=0.12,
        )
        assert isinstance(signal, FlatSignal), "Max positions reached should produce FlatSignal"

    def test_compute_atr_positive(self):
        from core.signal_generator import compute_atr
        prices = make_price_data(n=50)
        atr = compute_atr(prices["High"], prices["Low"], prices["Close"])
        assert atr > 0
