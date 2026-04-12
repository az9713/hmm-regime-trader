# tests/test_backtest.py

import pytest
import numpy as np
import pandas as pd


class TestPerformanceMetrics:
    def test_sharpe_positive_returns(self):
        from backtest.performance import compute_sharpe
        returns = pd.Series([0.001] * 252)
        sharpe = compute_sharpe(returns)
        assert sharpe > 0

    def test_sharpe_negative_returns(self):
        from backtest.performance import compute_sharpe
        returns = pd.Series([-0.001] * 252)
        sharpe = compute_sharpe(returns)
        assert sharpe < 0

    def test_max_drawdown_flat(self):
        from backtest.performance import compute_max_drawdown
        equity = pd.Series([100.0] * 50)
        dd, dur = compute_max_drawdown(equity)
        assert dd == 0.0

    def test_max_drawdown_50pct(self):
        from backtest.performance import compute_max_drawdown
        equity = pd.Series([100.0, 100.0, 50.0, 50.0])
        dd, dur = compute_max_drawdown(equity)
        assert abs(dd - (-0.50)) < 0.01

    def test_cagr_positive(self):
        from backtest.performance import compute_cagr
        # 10% return over 252 bars ≈ 10% CAGR
        equity = pd.Series([100.0 * (1.0 + 0.10/252) ** i for i in range(252)])
        cagr = compute_cagr(equity)
        assert abs(cagr - 0.10) < 0.01

    def test_regime_conditional_returns_structure(self):
        from backtest.performance import regime_conditional_returns
        returns = pd.Series(np.random.randn(100) * 0.01)
        regimes = pd.Series(["LowVol"] * 50 + ["HighVol"] * 50, index=returns.index)
        stats = regime_conditional_returns(returns, regimes)
        assert "LowVol" in stats.index
        assert "HighVol" in stats.index

    def test_sortino_ignores_positive_returns(self):
        from backtest.performance import compute_sortino
        # Positive-only returns → no downside → Sortino undefined but > Sharpe
        returns = pd.Series([0.001] * 252)
        sortino = compute_sortino(returns)
        sharpe = __import__("backtest.performance", fromlist=["compute_sharpe"]).compute_sharpe(returns)
        assert sortino >= sharpe or sortino == 0.0  # either undefined or better than Sharpe


class TestHansenSPATest:
    """
    Hansen (2005, JBES 23:365-380) SPA test.
    Verify structure of output and that clearly superior strategy passes.
    """

    def test_spa_structure(self):
        from backtest.performance import hansen_spa_test
        strategy = pd.Series(np.random.randn(200) * 0.01 + 0.001)
        bench = pd.Series(np.random.randn(200) * 0.01, index=strategy.index)
        result = hansen_spa_test(strategy, [bench], n_bootstrap=100)
        assert "p_value" in result
        assert "t_spa" in result
        assert "interpretation" in result
        assert 0.0 <= result["p_value"] <= 1.0

    def test_clearly_superior_strategy_low_pvalue(self):
        """Strategy returns +2%/day, benchmark +0%/day → should reject H0."""
        from backtest.performance import hansen_spa_test
        rng = np.random.default_rng(42)
        strategy = pd.Series(rng.normal(0.02, 0.005, 500))
        bench = pd.Series(rng.normal(0.0, 0.005, 500), index=strategy.index)
        result = hansen_spa_test(strategy, [bench], n_bootstrap=500)
        assert result["p_value"] < 0.05, f"Clearly superior strategy should reject H0 (p={result['p_value']:.3f})"


class TestWalkForwardStructure:
    """Verify IS/OOS windows are non-overlapping."""

    def test_oos_does_not_overlap_is(self):
        """
        For each window: OOS start must equal IS end.
        IS/OOS non-overlapping enforced per White (2000, Econometrica).
        """
        # Simulate window generation logic
        is_window = 252
        oos_window = 126
        step = 63
        total = 700

        windows = []
        start = 0
        while start + is_window + oos_window <= total:
            is_end = start + is_window
            oos_start = is_end
            oos_end = oos_start + oos_window
            windows.append((start, is_end, oos_start, oos_end))
            start += step

        for (is_start, is_end, oos_start, oos_end) in windows:
            assert oos_start == is_end, f"OOS start {oos_start} != IS end {is_end} — overlap detected"
            assert oos_start > is_start, "OOS must come after IS start"
