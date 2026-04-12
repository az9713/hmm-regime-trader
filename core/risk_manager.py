# core/risk_manager.py
# ============================================================
# Risk management gates independent of HMM (defense in depth).
#
# CIRCUIT BREAKERS [RISK POLICY — NOT hyperparameters]:
#   2/3/5/7/10% thresholds are fixed safety policy.
#   Optimizing these on historical data defeats their purpose —
#   it would create the very look-ahead bias they prevent.
#   Ref: design_docs/06_empirical_testing_plan.md §Group D
#
# CORRELATION GATES [MVP — static thresholds]:
#   0.70 warn / 0.85 block. Academically insufficient.
#   Static thresholds fail in market stress (correlations spike exactly
#   when protection matters most).
#   Correct upgrade: Meucci (2010, SSRN 1358533) Effective Number of Bets
#     ENB = exp(-Σ p_i · log p_i)  where p_i = PCA risk contribution
#   Deferred: 2-ticker portfolio (SPY, QQQ) doesn't justify complexity.
#   Upgrade when expanding to more tickers.
#
# POSITION SIZING:
#   1% risk per trade [RISK POLICY — not optimized].
#   Kelly fractional sizing [hyperparameters per regime — calibrate post-build].
# ============================================================

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class CircuitBreakerState(str, Enum):
    NORMAL = "normal"
    WARNING = "warning"
    PAUSED = "paused"         # no new trades for rest of day
    HALTED = "halted"         # no new trades
    REVIEW_REQUIRED = "review_required"
    STOPPED = "stopped"       # fully liquidate, halt


@dataclass
class RiskDecision:
    approved: bool
    reason: str
    circuit_state: CircuitBreakerState = CircuitBreakerState.NORMAL


class RiskManager:
    """
    Risk gates applied to every potential signal before execution.
    Circuit breakers, correlation, position limits, Kelly sizing.
    """

    def __init__(self, settings: dict):
        self.settings = settings
        self.risk_cfg = settings["risk"]
        self.sizing_cfg = settings["sizing"]
        self._circuit_state = CircuitBreakerState.NORMAL
        self._daily_pnl = 0.0
        self._weekly_pnl = 0.0
        self._monthly_pnl = 0.0
        self._peak_equity = None

    def evaluate(
        self,
        symbol: str,
        portfolio_value: float,
        daily_pnl_pct: float,
        current_positions: dict,
        correlation_matrix: pd.DataFrame = None,
    ) -> RiskDecision:
        """
        Gate check before accepting any signal.
        Returns RiskDecision with approved=True only if all gates pass.
        """
        # 1. Circuit breakers (RISK POLICY — immutable thresholds)
        circuit_check = check_circuit_breakers(
            daily_loss_pct=abs(min(daily_pnl_pct, 0)),
            circuit_state=self._circuit_state,
            settings=self.settings,
        )
        if circuit_check.state != CircuitBreakerState.NORMAL:
            self._circuit_state = circuit_check.state
            logger.warning(f"Circuit breaker active: {circuit_check.state} | {circuit_check.message}")
            if circuit_check.state in (CircuitBreakerState.HALTED,
                                        CircuitBreakerState.REVIEW_REQUIRED,
                                        CircuitBreakerState.STOPPED):
                return RiskDecision(False, circuit_check.message, circuit_check.state)

        # 2. Max concurrent positions
        max_pos = self.sizing_cfg["max_concurrent_positions"]
        if len(current_positions) >= max_pos:
            return RiskDecision(
                False,
                f"Max positions reached ({len(current_positions)}/{max_pos})",
                self._circuit_state,
            )

        # 3. Correlation check (static MVP thresholds)
        if correlation_matrix is not None and symbol in correlation_matrix.columns:
            existing_symbols = list(current_positions.keys())
            corr_check = check_correlation(symbol, existing_symbols, correlation_matrix, self.settings)
            if not corr_check["approved"]:
                return RiskDecision(False, corr_check["reason"], self._circuit_state)

        return RiskDecision(True, "All risk gates passed", self._circuit_state)

    def compute_position_size(
        self,
        capital: float,
        entry_price: float,
        stop_price: float,
        kelly_fraction: float,
    ) -> int:
        """
        Position size in shares.
        Risk per trade: 1% of capital [RISK POLICY].
        Scaled by Kelly fraction [HYPERPARAMETER].
        """
        risk_per_trade = self.sizing_cfg["risk_per_trade"]
        dollar_risk = capital * risk_per_trade * kelly_fraction
        risk_per_share = abs(entry_price - stop_price)
        if risk_per_share <= 0:
            return 0
        shares = int(dollar_risk / risk_per_share)
        return max(0, shares)

    def reset_daily(self):
        """Call at market open each day."""
        self._daily_pnl = 0.0
        # Circuit breaker state resets at day open for PAUSED (not HALTED/STOPPED)
        if self._circuit_state == CircuitBreakerState.PAUSED:
            self._circuit_state = CircuitBreakerState.NORMAL


@dataclass
class CircuitBreakerCheck:
    state: CircuitBreakerState
    message: str


def check_circuit_breakers(
    daily_loss_pct: float,
    circuit_state: CircuitBreakerState,
    settings: dict,
) -> CircuitBreakerCheck:
    """
    Evaluate circuit breaker thresholds.
    [RISK POLICY] — thresholds are FIXED. Never optimize on historical data.
    Ref: design_docs/06_empirical_testing_plan.md §Group D

    Thresholds (from settings.yaml):
      2%  → WARNING (log, alert)
      3%  → PAUSED (no new trades rest of day)
      5%  → HALTED (halt trading for rest of day)
      7%  → REVIEW_REQUIRED (manual intervention required)
      10% → STOPPED (liquidate all positions, full stop)
    """
    cb = settings["risk"]["circuit_breakers"]

    if daily_loss_pct >= cb["max_drawdown_stop"]:
        return CircuitBreakerCheck(
            CircuitBreakerState.STOPPED,
            f"STOPPED: daily loss {daily_loss_pct:.1%} ≥ {cb['max_drawdown_stop']:.0%} — LIQUIDATE ALL"
        )
    if daily_loss_pct >= cb["monthly_loss_review"]:
        return CircuitBreakerCheck(
            CircuitBreakerState.REVIEW_REQUIRED,
            f"REVIEW REQUIRED: daily loss {daily_loss_pct:.1%} ≥ {cb['monthly_loss_review']:.0%}"
        )
    if daily_loss_pct >= cb["weekly_loss_halt"]:
        return CircuitBreakerCheck(
            CircuitBreakerState.HALTED,
            f"HALTED: daily loss {daily_loss_pct:.1%} ≥ {cb['weekly_loss_halt']:.0%}"
        )
    if daily_loss_pct >= cb["daily_loss_pause"]:
        return CircuitBreakerCheck(
            CircuitBreakerState.PAUSED,
            f"PAUSED: daily loss {daily_loss_pct:.1%} ≥ {cb['daily_loss_pause']:.0%} — no new trades"
        )
    if daily_loss_pct >= cb["daily_loss_warn"]:
        return CircuitBreakerCheck(
            CircuitBreakerState.WARNING,
            f"WARNING: daily loss {daily_loss_pct:.1%} ≥ {cb['daily_loss_warn']:.0%}"
        )

    return CircuitBreakerCheck(CircuitBreakerState.NORMAL, "OK")


def check_correlation(
    candidate_symbol: str,
    existing_symbols: list,
    correlation_matrix: pd.DataFrame,
    settings: dict,
) -> dict:
    """
    MVP static correlation gate.
    warn_threshold=0.70, block_threshold=0.85.

    KNOWN LIMITATION: Static thresholds fail during stress (correlations spike).
    Future upgrade: Meucci (2010, SSRN 1358533) Effective Number of Bets.
    Ref: design_docs/04_research_allocation_signals.md §Section 4
    """
    cfg = settings["risk"]["correlation"]
    warn = cfg["warn_threshold"]
    block = cfg["block_threshold"]

    if not existing_symbols or candidate_symbol not in correlation_matrix.columns:
        return {"approved": True, "reason": "no correlation data"}

    for sym in existing_symbols:
        if sym not in correlation_matrix.columns:
            continue
        corr = abs(correlation_matrix.loc[candidate_symbol, sym])
        if corr >= block:
            return {
                "approved": False,
                "reason": f"Correlation {candidate_symbol}/{sym} = {corr:.2f} ≥ block threshold {block}"
            }
        if corr >= warn:
            logger.warning(f"Correlation {candidate_symbol}/{sym} = {corr:.2f} ≥ warn threshold {warn}")

    return {"approved": True, "reason": "correlation check passed"}


def kelly_position_size(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    kelly_fraction: float,
    capital: float,
    price: float,
) -> int:
    """
    Fractional Kelly position size in shares.
    Full Kelly: f* = (p·b - q) / b  where b = avg_win/avg_loss, q = 1-p
    Use kelly_fraction (0 < f ≤ 1) to scale down for safety.
    General fractional Kelly principle is grounded; regime-specific fractions
    are [HYPERPARAMETER] — calibrate post-build.
    """
    if avg_loss <= 0 or price <= 0:
        return 0
    b = avg_win / avg_loss
    q = 1.0 - win_rate
    full_kelly = (win_rate * b - q) / b
    full_kelly = max(0.0, full_kelly)  # never negative
    position_value = capital * full_kelly * kelly_fraction
    shares = int(position_value / price)
    return max(0, shares)
