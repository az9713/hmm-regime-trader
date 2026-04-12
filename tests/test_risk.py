# tests/test_risk.py

import pytest
import pandas as pd
import numpy as np


def make_settings():
    return {
        "risk": {
            "circuit_breakers": {
                "daily_loss_warn": 0.02,
                "daily_loss_pause": 0.03,
                "weekly_loss_halt": 0.05,
                "monthly_loss_review": 0.07,
                "max_drawdown_stop": 0.10,
            },
            "correlation": {
                "warn_threshold": 0.70,
                "block_threshold": 0.85,
                "lookback_days": 60,
            },
            "max_position_pct": 0.30,
        },
        "sizing": {
            "risk_per_trade": 0.01,
            "kelly_fraction": {"low_vol": 0.333, "mid_vol": 0.500, "high_vol": 0.200, "uncertainty": 0.200},
            "max_concurrent_positions": 5,
        },
    }


class TestCircuitBreakers:
    """
    Circuit breakers are RISK POLICY — test they fire correctly at fixed thresholds.
    These thresholds must NEVER be optimized on historical data.
    """

    def test_no_trigger_below_warn(self):
        from core.risk_manager import check_circuit_breakers, CircuitBreakerState
        s = make_settings()
        result = check_circuit_breakers(0.01, CircuitBreakerState.NORMAL, s)
        assert result.state == CircuitBreakerState.NORMAL

    def test_warn_at_2pct(self):
        from core.risk_manager import check_circuit_breakers, CircuitBreakerState
        s = make_settings()
        result = check_circuit_breakers(0.025, CircuitBreakerState.NORMAL, s)
        assert result.state == CircuitBreakerState.WARNING

    def test_pause_at_3pct(self):
        from core.risk_manager import check_circuit_breakers, CircuitBreakerState
        s = make_settings()
        result = check_circuit_breakers(0.035, CircuitBreakerState.NORMAL, s)
        assert result.state == CircuitBreakerState.PAUSED

    def test_halt_at_5pct(self):
        from core.risk_manager import check_circuit_breakers, CircuitBreakerState
        s = make_settings()
        result = check_circuit_breakers(0.055, CircuitBreakerState.NORMAL, s)
        assert result.state == CircuitBreakerState.HALTED

    def test_stop_at_10pct(self):
        from core.risk_manager import check_circuit_breakers, CircuitBreakerState
        s = make_settings()
        result = check_circuit_breakers(0.105, CircuitBreakerState.NORMAL, s)
        assert result.state == CircuitBreakerState.STOPPED

    def test_thresholds_are_monotone(self):
        """Higher loss → more severe state. Verify ordering."""
        from core.risk_manager import check_circuit_breakers, CircuitBreakerState
        s = make_settings()
        states = [
            check_circuit_breakers(loss, CircuitBreakerState.NORMAL, s).state
            for loss in [0.01, 0.025, 0.035, 0.055, 0.075, 0.105]
        ]
        severity = [
            CircuitBreakerState.NORMAL, CircuitBreakerState.WARNING,
            CircuitBreakerState.PAUSED, CircuitBreakerState.HALTED,
            CircuitBreakerState.REVIEW_REQUIRED, CircuitBreakerState.STOPPED,
        ]
        for i in range(len(states) - 1):
            assert severity.index(states[i]) <= severity.index(states[i+1]), (
                f"Circuit breaker states not monotone: {states}"
            )


class TestCorrelationGates:
    def test_high_correlation_blocked(self):
        from core.risk_manager import check_correlation
        s = make_settings()
        corr_matrix = pd.DataFrame(
            [[1.0, 0.92], [0.92, 1.0]],
            index=["SPY", "QQQ"],
            columns=["SPY", "QQQ"],
        )
        result = check_correlation("QQQ", ["SPY"], corr_matrix, s)
        assert not result["approved"], "High correlation should block the trade"

    def test_low_correlation_approved(self):
        from core.risk_manager import check_correlation
        s = make_settings()
        corr_matrix = pd.DataFrame(
            [[1.0, 0.30], [0.30, 1.0]],
            index=["SPY", "GLD"],
            columns=["SPY", "GLD"],
        )
        result = check_correlation("GLD", ["SPY"], corr_matrix, s)
        assert result["approved"]

    def test_no_existing_positions_approved(self):
        from core.risk_manager import check_correlation
        s = make_settings()
        result = check_correlation("SPY", [], pd.DataFrame(), s)
        assert result["approved"]


class TestKellySizing:
    def test_positive_kelly_with_edge(self):
        from core.risk_manager import kelly_position_size
        shares = kelly_position_size(
            win_rate=0.55, avg_win=200.0, avg_loss=100.0,
            kelly_fraction=0.333, capital=100_000, price=450.0
        )
        assert shares > 0

    def test_zero_shares_with_no_edge(self):
        from core.risk_manager import kelly_position_size
        shares = kelly_position_size(
            win_rate=0.40, avg_win=100.0, avg_loss=200.0,
            kelly_fraction=0.333, capital=100_000, price=450.0
        )
        assert shares == 0, "No edge → Kelly = 0 → no position"

    def test_zero_shares_with_zero_price(self):
        from core.risk_manager import kelly_position_size
        shares = kelly_position_size(
            win_rate=0.60, avg_win=100.0, avg_loss=50.0,
            kelly_fraction=0.5, capital=100_000, price=0.0
        )
        assert shares == 0
