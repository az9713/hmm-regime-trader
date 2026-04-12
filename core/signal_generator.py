# core/signal_generator.py
# ============================================================
# Combines HMM regime + strategy + risk gates → Signal objects.
#
# REGIME-CONDITIONAL DISPATCH:
#   Cooper, Gutierrez & Hameed (2004, JF 59:1345-1365, 1000+ citations):
#   Momentum = +0.93%/mo after up-markets, -0.37%/mo after down-markets.
#   → Apply full strategy in continuation regimes; reduce/skip in transition/bear.
#
# CONTINUATION VS TRANSITION STATES:
#   Asem & Tian (2010, JFQA 45:1549-1562):
#   Profits highest in bull→bull, bear→bear. Losses during transitions.
#   → 3-bar persistence filter (in HMMEngine) reduces transition-state exposure.
#
# ALWAYS LONG — three converging top-tier papers:
#   1. Cooper et al. (2004, JF): shorting in down-markets captures -0.37%/mo.
#   2. Daniel & Moskowitz (2016, JFE): momentum crash wipes short positions at
#      bull start. HMM detects 2-3 bars late — max loss on short side.
#   3. Moreira & Muir (2017, JF): long-term equity drift +5-7% real.
#      Correct response to high vol: reduce long allocation, NOT flip short.
#
# META-LABELING [DEFERRED]:
#   López de Prado (2018) is a book chapter, not peer-reviewed.
#   Hudson & Thames (2022, JFDS): OOS precision gain 0.17→0.20 only.
#   Post-MVP enhancement.
# ============================================================

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from core.hmm_engine import RegimeLabel, RegimeState
from core.regime_strategies import get_strategy, BaseStrategy
from core.risk_manager import RiskManager, RiskDecision

logger = logging.getLogger(__name__)


@dataclass
class Signal:
    symbol: str
    direction: str              # "long" | "flat" — NEVER "short"
    size_shares: int            # number of shares
    size_dollars: float         # dollar value of position
    entry_price: float
    stop_price: float
    target_price: float
    allocation: float           # portfolio weight [0, max_leverage]
    confidence: float           # HMM posterior [0, 1]
    regime: str                 # RegimeLabel
    strategy: str               # strategy class name
    risk_decision: Optional[str] = None
    timestamp: Optional[datetime] = None
    atr: float = 0.0


@dataclass
class FlatSignal:
    """Signals no action — risk gate blocked or uncertainty."""
    symbol: str
    reason: str
    regime: str
    confidence: float
    timestamp: Optional[datetime] = None


class SignalGenerator:
    """
    Generates trade signals from regime state + market data.
    Applies risk manager gates before emitting any signal.
    """

    def __init__(self, settings: dict, risk_manager: RiskManager):
        self.settings = settings
        self.risk_manager = risk_manager

    def generate(
        self,
        symbol: str,
        regime_state: RegimeState,
        price_data: pd.DataFrame,
        portfolio_value: float,
        daily_pnl_pct: float,
        current_positions: dict,
        ewma_vol: float,
        correlation_matrix: pd.DataFrame = None,
    ) -> Signal | FlatSignal:
        """
        Main entry point. Returns Signal (enter/hold) or FlatSignal (no action).

        Args:
            symbol:            ticker
            regime_state:      output from HMMEngine
            price_data:        recent OHLCV bars
            portfolio_value:   current portfolio value in dollars
            daily_pnl_pct:     today's P&L as fraction of portfolio
            current_positions: {symbol: position_dict} currently held
            ewma_vol:          EWMA realized vol for Moreira-Muir formula
            correlation_matrix: rolling correlation for correlation gate
        """
        ts = datetime.now()

        # 1. Get strategy for regime
        strategy = get_strategy(regime_state.label, self.settings)

        # 2. Compute technical levels
        atr = compute_atr(price_data["High"], price_data["Low"], price_data["Close"])
        if atr <= 0:
            return FlatSignal(symbol, "ATR is zero — insufficient price history", regime_state.label, regime_state.confidence, ts)

        entry = float(price_data["Close"].iloc[-1])
        stop = entry - atr * strategy.get_stop_atr_multiple()
        target = entry + atr * strategy.get_target_atr_multiple()

        # 3. Trend filter
        ema_fast = price_data["Close"].ewm(span=self.settings["trend"]["fast_ema"]).mean().iloc[-1]
        ema_slow = price_data["Close"].ewm(span=self.settings["trend"]["slow_ema"]).mean().iloc[-1]
        if not strategy.trend_filter_passes(ema_fast, ema_slow):
            return FlatSignal(symbol, f"Trend filter failed ({strategy.name}): EMA{self.settings['trend']['fast_ema']}={ema_fast:.2f} < EMA{self.settings['trend']['slow_ema']}={ema_slow:.2f}", regime_state.label, regime_state.confidence, ts)

        # 4. Allocation (Moreira-Muir primary)
        allocation = strategy.get_allocation(ewma_vol)

        # 5. Risk manager gates
        risk_decision = self.risk_manager.evaluate(
            symbol=symbol,
            portfolio_value=portfolio_value,
            daily_pnl_pct=daily_pnl_pct,
            current_positions=current_positions,
            correlation_matrix=correlation_matrix,
        )
        if not risk_decision.approved:
            return FlatSignal(symbol, f"Risk gate: {risk_decision.reason}", regime_state.label, regime_state.confidence, ts)

        # 6. Position size
        kelly_frac = strategy.get_kelly_fraction()
        shares = self.risk_manager.compute_position_size(
            capital=portfolio_value,
            entry_price=entry,
            stop_price=stop,
            kelly_fraction=kelly_frac,
        )
        if shares <= 0:
            return FlatSignal(symbol, "Position size computed to 0 shares", regime_state.label, regime_state.confidence, ts)

        # 7. Direction is ALWAYS long — never short
        # Ref: Cooper et al. (2004); Daniel-Moskowitz (2016); Moreira-Muir (2017)
        direction = "long"

        logger.info(
            f"Signal: {symbol} {direction} | regime={regime_state.label} | conf={regime_state.confidence:.2f} | "
            f"alloc={allocation:.2f} | entry={entry:.2f} stop={stop:.2f} target={target:.2f} | "
            f"shares={shares} | strategy={strategy.name}"
        )

        return Signal(
            symbol=symbol,
            direction=direction,
            size_shares=shares,
            size_dollars=shares * entry,
            entry_price=entry,
            stop_price=stop,
            target_price=target,
            allocation=allocation,
            confidence=regime_state.confidence,
            regime=regime_state.label,
            strategy=strategy.name,
            risk_decision=risk_decision.reason,
            timestamp=ts,
            atr=atr,
        )


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> float:
    """
    Average True Range.
    True Range = max(H-L, |H-prev_C|, |L-prev_C|)
    ATR multiples are [HYPERPARAMETER] — see design_docs/06_empirical_testing_plan.md §C.
    """
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean().iloc[-1]
    return float(atr) if not np.isnan(atr) else 0.0


def compute_position_size(capital: float, risk_pct: float, entry: float, stop: float) -> int:
    """
    Basic position size from dollar risk.
    Shares = (capital × risk_pct) / |entry - stop|
    risk_pct = 1% [RISK POLICY — not optimized].
    """
    dollar_risk = capital * risk_pct
    risk_per_share = abs(entry - stop)
    if risk_per_share <= 0:
        return 0
    return int(dollar_risk / risk_per_share)
