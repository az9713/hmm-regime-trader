# core/regime_strategies.py
# ============================================================
# Strategy classes — one per volatility regime.
#
# ALLOCATION:
#   Primary: Moreira-Muir continuous formula
#     w_t = min(target_vol / ewma_realized_vol, max_leverage)
#     Ref: Moreira & Muir (2017, Journal of Finance 72(4):1611-1644)
#     Barroso & Santa-Clara (2015, JFE 116(1):111-120): vol-managed momentum ~2× Sharpe.
#   Fallback: discrete buckets from settings.yaml
#     These are practitioner discretizations of Moreira-Muir at representative VIX levels.
#     NOT independently validated in any top-tier paper.
#     [HYPERPARAMETER] — calibrate post-build via walk-forward.
#
# LEVERAGE:
#   Low-vol: up to 1.25× — direction from Asness, Frazzini & Pedersen (2012, FAJ 68).
#   High-vol: de-lever — Moreira & Muir (2017).
#   Exact multipliers are Alpaca constraint + practitioner defaults, not research.
#
# TREND FILTER:
#   50 EMA > 200 EMA — Faber (2007, Journal of Wealth Management 9(4):69-79).
#   Practitioner-tier journal. Evidence-supported heuristic.
#
# STOP LOSS / TAKE PROFIT:
#   ATR multiples — [HYPERPARAMETER]. NO top-tier academic foundation.
#   ATR from Wilder (1978) practitioner manual.
#   Lo & Remorov (2015, SSRN 2695383): stop effectiveness is asset/regime-specific.
#   Calibrate via 3-month paper trading. See design_docs/06_empirical_testing_plan.md §C.
#
# ALWAYS LONG:
#   Cooper, Gutierrez & Hameed (2004, JF 59:1345-1365): −0.37%/mo shorting in down-mkts.
#   Daniel & Moskowitz (2016, JFE): momentum crash hits short side hardest.
#   Moreira & Muir (2017, JF): long-term equity drift +5-7% real; reduce longs, not flip.
# ============================================================

import numpy as np
import logging

logger = logging.getLogger(__name__)


class BaseStrategy:
    """Base class for all regime strategies."""

    name: str = "Base"

    def __init__(self, settings: dict):
        self.settings = settings

    def get_allocation(self, ewma_vol: float) -> float:
        """
        Returns portfolio weight [0, max_leverage].
        Primary: Moreira-Muir continuous if use_continuous_formula=True.
        Fallback: discrete bucket from settings.
        """
        raise NotImplementedError

    def get_stop_atr_multiple(self) -> float:
        raise NotImplementedError

    def get_target_atr_multiple(self) -> float:
        raise NotImplementedError

    def get_kelly_fraction(self) -> float:
        raise NotImplementedError

    def trend_filter_passes(self, ema_fast: float, ema_slow: float) -> bool:
        """
        50 EMA > 200 EMA → trend intact → allow entry.
        Ref: Faber (2007, JWM).
        """
        return ema_fast > ema_slow


class LowVolStrategy(BaseStrategy):
    """
    Low-volatility regime: full allocation, optional leverage.
    Leverage direction: Asness, Frazzini & Pedersen (2012, FAJ).
    """

    name = "LowVolStrategy"

    def get_allocation(self, ewma_vol: float) -> float:
        cfg = self.settings["allocation"]
        if cfg.get("use_continuous_formula", True) and ewma_vol > 0:
            return moreira_muir_allocation(
                ewma_vol,
                target_vol=cfg["target_vol"],
                max_leverage=cfg["low_vol"]["max_leverage"],
            )
        return cfg["low_vol"]["allocation"]

    def get_stop_atr_multiple(self) -> float:
        return self.settings["stops"]["low_vol"]["stop_atr"]

    def get_target_atr_multiple(self) -> float:
        return self.settings["stops"]["low_vol"]["target_atr"]

    def get_kelly_fraction(self) -> float:
        return self.settings["sizing"]["kelly_fraction"]["low_vol"]


class MidVolStrategy(BaseStrategy):
    """
    Mid-volatility regime: partial allocation, no leverage, trend filter required.
    """

    name = "MidVolStrategy"

    def get_allocation(self, ewma_vol: float) -> float:
        cfg = self.settings["allocation"]
        if cfg.get("use_continuous_formula", True) and ewma_vol > 0:
            return moreira_muir_allocation(
                ewma_vol,
                target_vol=cfg["target_vol"],
                max_leverage=cfg["mid_vol"]["max_leverage"],
            )
        return cfg["mid_vol"]["allocation"]

    def get_stop_atr_multiple(self) -> float:
        return self.settings["stops"]["mid_vol"]["stop_atr"]

    def get_target_atr_multiple(self) -> float:
        return self.settings["stops"]["mid_vol"]["target_atr"]

    def get_kelly_fraction(self) -> float:
        return self.settings["sizing"]["kelly_fraction"]["mid_vol"]


class HighVolStrategy(BaseStrategy):
    """
    High-volatility regime: reduced allocation, no new longs unless trend very strong.
    Ref: Daniel & Moskowitz (2016, JFE) — momentum crashes in high-vol.
         Moreira & Muir (2017, JF) — de-lever in high vol, don't flip short.
    """

    name = "HighVolStrategy"

    def get_allocation(self, ewma_vol: float) -> float:
        cfg = self.settings["allocation"]
        if cfg.get("use_continuous_formula", True) and ewma_vol > 0:
            return moreira_muir_allocation(
                ewma_vol,
                target_vol=cfg["target_vol"],
                max_leverage=cfg["high_vol"]["max_leverage"],
            )
        return cfg["high_vol"]["allocation"]

    def get_stop_atr_multiple(self) -> float:
        return self.settings["stops"]["high_vol"]["stop_atr"]

    def get_target_atr_multiple(self) -> float:
        return self.settings["stops"]["high_vol"]["target_atr"]

    def get_kelly_fraction(self) -> float:
        return self.settings["sizing"]["kelly_fraction"]["high_vol"]

    def trend_filter_passes(self, ema_fast: float, ema_slow: float) -> bool:
        """In high-vol, require trend to be intact before any entry."""
        return ema_fast > ema_slow


class UncertaintyStrategy(BaseStrategy):
    """
    Activated when HMM confidence is low or regime is flickering.
    Reduced allocation, no new entries, hold existing with tighter risk.
    """

    name = "UncertaintyStrategy"

    def get_allocation(self, ewma_vol: float) -> float:
        return self.settings["allocation"]["uncertainty"]["allocation"]

    def get_stop_atr_multiple(self) -> float:
        return self.settings["stops"]["high_vol"]["stop_atr"]  # conservative

    def get_target_atr_multiple(self) -> float:
        return self.settings["stops"]["high_vol"]["target_atr"]

    def get_kelly_fraction(self) -> float:
        return self.settings["sizing"]["kelly_fraction"]["uncertainty"]

    def trend_filter_passes(self, ema_fast: float, ema_slow: float) -> bool:
        """No new entries in uncertainty mode."""
        return False


def moreira_muir_allocation(
    ewma_realized_vol: float,
    target_vol: float = 0.20,
    max_leverage: float = 1.25,
) -> float:
    """
    Continuous inverse-volatility allocation.
    w_t = min(target_vol / realized_vol, max_leverage)

    Ref: Moreira & Muir (2017, Journal of Finance 72(4):1611-1644)
    "Volatility-Managed Portfolios" — produces ~25% Sharpe improvement across factors.

    Args:
        ewma_realized_vol: EWMA realized vol (10-day half-life), annualized.
        target_vol: target portfolio volatility (default 20%).
        max_leverage: cap (default 1.25x, Alpaca paper-trading limit).

    Returns:
        allocation float in [0, max_leverage]
    """
    if ewma_realized_vol <= 0:
        logger.warning("moreira_muir_allocation: ewma_realized_vol <= 0, returning 0")
        return 0.0
    raw = target_vol / ewma_realized_vol
    return float(min(raw, max_leverage))


def get_strategy(regime_label: str, settings: dict) -> BaseStrategy:
    """Dispatch regime label to strategy class."""
    from core.hmm_engine import RegimeLabel
    mapping = {
        RegimeLabel.LOW_VOL: LowVolStrategy,
        RegimeLabel.MID_VOL: MidVolStrategy,
        RegimeLabel.HIGH_VOL: HighVolStrategy,
        RegimeLabel.VERY_HIGH_VOL: HighVolStrategy,  # treat 4th state as high-vol
        RegimeLabel.UNCERTAINTY: UncertaintyStrategy,
    }
    cls = mapping.get(regime_label, UncertaintyStrategy)
    return cls(settings)
