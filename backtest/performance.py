# backtest/performance.py
# ============================================================
# Performance metrics. Report OOS only — IS metrics meaningless.
# Ref: White (2000, Econometrica 68:1097-1126): IS performance = overfitting.
# ============================================================

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def compute_cagr(equity_curve: pd.Series) -> float:
    """Compound Annual Growth Rate."""
    if len(equity_curve) < 2:
        return 0.0
    years = len(equity_curve) / 252.0
    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0]
    return float(total_return ** (1.0 / years) - 1.0)


def compute_sharpe(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Annualized Sharpe ratio. Target: >1.0 OOS."""
    excess = returns - risk_free_rate / 252.0
    if excess.std() == 0:
        return 0.0
    return float(excess.mean() / excess.std() * np.sqrt(252))


def compute_sortino(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Annualized Sortino ratio (downside deviation only)."""
    excess = returns - risk_free_rate / 252.0
    downside = excess[excess < 0]
    if len(downside) == 0 or downside.std() == 0:
        return 0.0
    return float(excess.mean() / downside.std() * np.sqrt(252))


def compute_max_drawdown(equity_curve: pd.Series) -> tuple:
    """
    Returns (max_drawdown_pct, max_drawdown_duration_bars).
    Target: max_drawdown < 15% in OOS periods.
    """
    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve - rolling_max) / rolling_max
    max_dd = float(drawdown.min())

    # Duration: longest consecutive period below peak
    in_dd = drawdown < 0
    max_dur = 0
    cur_dur = 0
    for flag in in_dd:
        if flag:
            cur_dur += 1
            max_dur = max(max_dur, cur_dur)
        else:
            cur_dur = 0

    return max_dd, max_dur


def compute_calmar(returns: pd.Series, equity_curve: pd.Series) -> float:
    """CAGR / |max drawdown|."""
    cagr = compute_cagr(equity_curve)
    max_dd, _ = compute_max_drawdown(equity_curve)
    if max_dd == 0:
        return 0.0
    return float(cagr / abs(max_dd))


def regime_conditional_returns(returns: pd.Series, regimes: pd.Series) -> pd.DataFrame:
    """
    Mean and std of daily returns per regime label.
    Validate: high-vol regime should have lower mean + higher std.
    """
    df = pd.DataFrame({"return": returns, "regime": regimes}).dropna()
    stats = df.groupby("regime")["return"].agg(["mean", "std", "count"])
    stats["mean_annualized"] = stats["mean"] * 252
    stats["std_annualized"] = stats["std"] * np.sqrt(252)
    return stats


def compute_all_metrics(equity_curve: pd.Series, returns: pd.Series = None) -> dict:
    """Compute full metric suite for one OOS window."""
    if returns is None:
        returns = equity_curve.pct_change().dropna()
    max_dd, max_dd_dur = compute_max_drawdown(equity_curve)
    return {
        "cagr": compute_cagr(equity_curve),
        "sharpe": compute_sharpe(returns),
        "sortino": compute_sortino(returns),
        "calmar": compute_calmar(returns, equity_curve),
        "max_drawdown": max_dd,
        "max_drawdown_duration_bars": max_dd_dur,
        "total_return": float(equity_curve.iloc[-1] / equity_curve.iloc[0] - 1),
        "n_bars": len(equity_curve),
    }


def hansen_spa_test(
    strategy_returns: pd.Series,
    benchmark_returns_list: list[pd.Series],
    n_bootstrap: int = 1000,
    random_state: int = 42,
) -> dict:
    """
    Hansen (2005, JBES 23:365-380) Superior Predictive Ability test.
    Tests H0: strategy has no superior predictive ability over ANY benchmark.
    p < 0.05 → reject null → genuine predictive ability demonstrated.

    Without this test, performance differences could be due to luck.
    Ref: White (2000, Econometrica 68:1097-1126): data snooping reality check.
         Romano & Wolf (2005, Econometrica 73:1237-1282): stepwise correction.

    Simplified implementation using stationary bootstrap.
    For rigorous production use, consider arch or statsmodels SPA implementation.
    """
    rng = np.random.default_rng(random_state)

    # De-duplicate indices (overlapping OOS windows in walk-forward produce duplicate dates)
    strategy_returns = strategy_returns[~strategy_returns.index.duplicated(keep="first")]
    T = len(strategy_returns)

    # Loss differential: d_k,t = L(benchmark_k,t) - L(strategy,t)
    # Using negative return as loss function
    loss_strategy = -strategy_returns.values

    d_matrices = []
    for bench in benchmark_returns_list:
        bench_dedup = bench[~bench.index.duplicated(keep="first")]
        bench_aligned = bench_dedup.reindex(strategy_returns.index).fillna(0)
        loss_bench = -bench_aligned.values
        d_matrices.append(loss_bench - loss_strategy)  # positive = strategy beats bench

    if not d_matrices:
        return {"p_value": None, "t_stat": None, "note": "No benchmarks provided"}

    d_matrix = np.column_stack(d_matrices)  # (T, n_benchmarks)
    d_bar = d_matrix.mean(axis=0)

    # Bootstrap distribution of max mean loss differential
    block_size = max(1, int(T ** (1/3)))
    bootstrap_max = []
    for _ in range(n_bootstrap):
        # Stationary block bootstrap
        idx = []
        while len(idx) < T:
            start = rng.integers(0, T)
            block = list(range(start, min(start + block_size, T)))
            idx.extend(block)
        idx = np.array(idx[:T])
        boot_d_bar = d_matrix[idx].mean(axis=0)
        bootstrap_max.append(np.max(boot_d_bar - d_bar))

    t_spa = float(np.max(d_bar))
    p_value = float(np.mean(np.array(bootstrap_max) >= t_spa))

    return {
        "t_spa": t_spa,
        "p_value": p_value,
        "n_benchmarks": len(benchmark_returns_list),
        "n_bootstrap": n_bootstrap,
        "interpretation": (
            "Reject H0 — genuine predictive ability" if p_value < 0.05
            else "Fail to reject H0 — performance may be luck"
        ),
    }
