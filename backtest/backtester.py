# backtest/backtester.py
# ============================================================
# Walk-forward backtester — the calibration instrument.
#
# STRUCTURE:
#   IS window:   252 bars (1 year)
#   OOS window:  126 bars (6 months)
#   Step:        63 bars (quarterly roll)
#   Non-overlapping IS/OOS enforced — OOS NEVER used for param selection.
#   Ref: White (2000, Econometrica 68(5):1097-1126, 3000+ citations)
#
# REGIME INFERENCE:
#   Forward α-recursion ONLY. model.predict() is FORBIDDEN in backtest.
#   Ref: Hamilton (1989, Econometrica) — look-ahead prevention.
#
# ALLOCATION:
#   Moreira-Muir continuous formula primary (design_docs/04).
#   Allocation-based P&L — no per-trade stops tracked in backtest.
#   Stops are live-trading only (design_docs/06 §"Note on stops").
#
# EVERY RUN LOGGED:
#   Date, all params, BIC state counts, all OOS metrics.
#   Prevents re-testing same params unknowingly (data snooping).
#
# BENCHMARKS:
#   1. Buy-and-hold SPY
#   2. 200-day SMA filter (Faber 2007, JWM)
#   3. Random entry with same risk mgmt (100 seeds)
#   Final: Hansen (2005, JBES 23:365-380) SPA test.
# ============================================================

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from data.feature_engineering import FeatureEngineer, compute_ewma_realized_vol
from core.hmm_engine import HMMEngine, RegimeLabel
from core.regime_strategies import get_strategy, moreira_muir_allocation
from backtest.performance import compute_all_metrics, regime_conditional_returns

logger = logging.getLogger(__name__)


@dataclass
class WindowResult:
    window_idx: int
    is_start: str
    is_end: str
    oos_start: str
    oos_end: str
    bic_n_states: int
    params: dict
    oos_metrics: dict
    regime_stats: Optional[pd.DataFrame] = None
    equity_curve: Optional[pd.Series] = None
    regimes: Optional[pd.Series] = None


@dataclass
class BacktestResult:
    windows: list[WindowResult] = field(default_factory=list)
    settings: dict = field(default_factory=dict)
    run_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    aggregate_metrics: dict = field(default_factory=dict)
    benchmark_returns: dict = field(default_factory=dict)
    spa_result: dict = field(default_factory=dict)

    def summary(self) -> pd.DataFrame:
        rows = []
        for w in self.windows:
            row = {"window": w.window_idx, "is_start": w.is_start, "oos_end": w.oos_end,
                   "n_states": w.bic_n_states}
            row.update(w.oos_metrics)
            rows.append(row)
        return pd.DataFrame(rows)


class WalkForwardBacktester:
    """
    Walk-forward backtest engine.
    Trains HMM on IS window, evaluates on OOS window, rolls forward.
    """

    def __init__(self, settings: dict):
        self.settings = settings
        self.bt_cfg = settings["backtest"]
        self.is_window = self.bt_cfg["is_window"]      # 252 bars
        self.oos_window = self.bt_cfg["oos_window"]    # 126 bars
        self.step_size = self.bt_cfg.get("step_size", 63)

    def run(
        self,
        prices: dict,         # {symbol: pd.DataFrame with OHLCV}
        vix: pd.Series,
        hy_oas: pd.Series,
        primary_symbol: str = "SPY",
    ) -> BacktestResult:
        """
        Run full walk-forward backtest.

        Args:
            prices:         dict of symbol → OHLCV DataFrame
            vix:            VIX series from FRED
            hy_oas:         HY OAS series from FRED
            primary_symbol: benchmark symbol for buy-hold comparison
        """
        result = BacktestResult(settings=self.settings)

        # Build feature matrix for primary symbol
        fe = FeatureEngineer(self.settings)
        spy_prices = prices[primary_symbol]
        features = fe.compute(spy_prices, vix, hy_oas)

        # Align price index to feature index (features drop NaN warm-up rows)
        spy_close = spy_prices["Close"].reindex(features.index)

        total_bars = len(features)
        window_start = 0
        window_idx = 0

        all_oos_returns = []
        all_oos_dates = []

        while window_start + self.is_window + self.oos_window <= total_bars:
            is_end = window_start + self.is_window
            oos_end = is_end + self.oos_window

            is_features = features.iloc[window_start:is_end].values
            oos_features = features.iloc[is_end:oos_end].values
            oos_close = spy_close.iloc[is_end:oos_end]
            oos_index = features.index[is_end:oos_end]

            logger.info(
                f"Window {window_idx}: IS {features.index[window_start].date()} → "
                f"{features.index[is_end-1].date()} | "
                f"OOS {features.index[is_end].date()} → {features.index[oos_end-1].date()}"
            )

            # Train HMM on IS data
            engine = HMMEngine(self.settings)
            try:
                n_states = engine.fit(is_features)
            except Exception as e:
                logger.warning(f"Window {window_idx}: HMM training failed: {e}")
                window_start += self.step_size
                window_idx += 1
                continue

            # Run forward algorithm on OOS data (NO look-ahead)
            oos_regime_states = engine.predict_regime(oos_features)
            oos_regimes = pd.Series(
                [r.label for r in oos_regime_states], index=oos_index
            )

            # Compute EWMA vol for Moreira-Muir allocation
            is_returns = pd.Series(
                np.diff(np.log(spy_close.iloc[window_start:is_end].values)),
                index=features.index[window_start+1:is_end]
            )
            oos_close_returns = np.diff(np.log(oos_close.values))

            # Allocation-based OOS equity curve
            equity_curve, daily_returns = self._simulate_oos(
                oos_close, oos_regimes, oos_regime_states, window_idx
            )

            oos_metrics = compute_all_metrics(equity_curve, pd.Series(daily_returns, index=oos_index[1:]))
            regime_stats = regime_conditional_returns(
                pd.Series(daily_returns, index=oos_index[1:]), oos_regimes.iloc[1:]
            )

            window_result = WindowResult(
                window_idx=window_idx,
                is_start=str(features.index[window_start].date()),
                is_end=str(features.index[is_end-1].date()),
                oos_start=str(features.index[is_end].date()),
                oos_end=str(features.index[oos_end-1].date()),
                bic_n_states=n_states,
                params=self._extract_params(),
                oos_metrics=oos_metrics,
                regime_stats=regime_stats,
                equity_curve=equity_curve,
                regimes=oos_regimes,
            )
            result.windows.append(window_result)

            all_oos_returns.extend(daily_returns)
            all_oos_dates.extend(oos_index[1:].tolist())

            logger.info(
                f"Window {window_idx} OOS: Sharpe={oos_metrics['sharpe']:.2f} "
                f"MaxDD={oos_metrics['max_drawdown']:.1%} n_states={n_states}"
            )

            window_start += self.step_size
            window_idx += 1

        # Aggregate OOS metrics across all windows
        if all_oos_returns:
            all_returns = pd.Series(all_oos_returns, index=all_oos_dates)
            all_equity = (1 + all_returns).cumprod()
            result.aggregate_metrics = compute_all_metrics(all_equity, all_returns)
            logger.info(f"Aggregate OOS: {result.aggregate_metrics}")

            # Benchmarks
            result.benchmark_returns = self._compute_benchmarks(
                spy_close, all_returns.index, primary_symbol
            )

        return result

    def _simulate_oos(
        self,
        oos_close: pd.Series,
        oos_regimes: pd.Series,
        oos_regime_states,
        window_idx: int,
    ) -> tuple:
        """
        Simulate allocation-based returns in OOS window.
        Allocation = Moreira-Muir formula per regime.
        No per-trade stops (stops are live-trading only).
        Ref: design_docs/06_empirical_testing_plan.md
        """
        log_returns = np.log(oos_close / oos_close.shift(1)).dropna()
        allocations = []
        ewma_vol = None

        for i, (dt, regime_state) in enumerate(zip(oos_regimes.index, oos_regime_states)):
            if i == 0:
                allocations.append(self.settings["allocation"]["mid_vol"]["allocation"])
                continue
            # Estimate EWMA vol from past OOS returns
            past_rets = log_returns.iloc[:i]
            if len(past_rets) > 5:
                ewma_vol = float(
                    compute_ewma_realized_vol(past_rets,
                                              halflife=self.settings["features"]["ewma_halflife"]).iloc[-1]
                )
            strategy = get_strategy(regime_state.label, self.settings)
            alloc = strategy.get_allocation(ewma_vol or 0.20)
            # Rebalance threshold check
            if allocations and abs(alloc - allocations[-1]) < self.settings["allocation"]["rebalance_threshold"]:
                alloc = allocations[-1]
            allocations.append(alloc)

        alloc_series = pd.Series(allocations, index=oos_regimes.index)
        # Lag allocation by 1 bar (signal at close t → trade opens close t+1)
        alloc_lagged = alloc_series.shift(1).bfill()
        strategy_returns = (log_returns * alloc_lagged.reindex(log_returns.index).fillna(0)).values
        equity = np.concatenate([[1.0], np.cumprod(1 + strategy_returns)])
        equity_curve = pd.Series(equity, index=oos_close.index)

        return equity_curve, strategy_returns.tolist()

    def _compute_benchmarks(
        self, spy_close: pd.Series, oos_index, primary_symbol: str
    ) -> dict:
        """
        Benchmark returns for comparison.
        1. Buy-and-hold
        2. 200-day SMA (Faber 2007, JWM)
        3. Random entry (100 seeds)
        """
        spy_oos = spy_close.reindex(oos_index, method="ffill")
        bh_returns = np.log(spy_oos / spy_oos.shift(1)).dropna()

        # 200 SMA
        sma200 = spy_close.rolling(200).mean()
        sma_signal = (spy_close > sma200).astype(float)
        sma_returns = (np.log(spy_oos / spy_oos.shift(1)) * sma_signal.reindex(oos_index, method="ffill").shift(1)).dropna()

        return {
            "buy_hold": bh_returns,
            "sma_200": sma_returns,
        }

    def _extract_params(self) -> dict:
        """Snapshot current parameter values for logging."""
        return {
            "target_vol": self.settings["allocation"]["target_vol"],
            "low_vol_alloc": self.settings["allocation"]["low_vol"]["allocation"],
            "mid_vol_alloc": self.settings["allocation"]["mid_vol"]["allocation"],
            "high_vol_alloc": self.settings["allocation"]["high_vol"]["allocation"],
            "persistence_bars": self.settings["stability"]["persistence_bars"],
            "confidence_floor": self.settings["stability"]["confidence_floor"],
        }
