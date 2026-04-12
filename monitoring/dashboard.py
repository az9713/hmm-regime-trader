# monitoring/dashboard.py
# ============================================================
# Real-time dashboard using matplotlib.
# Shows: P&L, current regime, open positions, regime history.
# ============================================================

import logging
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime

logger = logging.getLogger(__name__)

REGIME_COLORS = {
    "LowVol": "#2ecc71",       # green
    "MidVol": "#f39c12",       # orange
    "HighVol": "#e74c3c",      # red
    "VeryHighVol": "#8e44ad",  # purple
    "Uncertainty": "#95a5a6",  # grey
}


class Dashboard:
    """Matplotlib-based live dashboard."""

    def __init__(self, settings: dict):
        self.settings = settings
        self._equity_history = []
        self._regime_history = []
        self._dates = []
        plt.ion()
        self._fig = None

    def update(
        self,
        timestamp: datetime,
        portfolio_value: float,
        regime: str,
        positions: pd.DataFrame,
        confidence: float,
    ):
        """Update dashboard with latest state."""
        self._equity_history.append(portfolio_value)
        self._regime_history.append(regime)
        self._dates.append(timestamp)

        self._render(positions, confidence)

    def _render(self, positions: pd.DataFrame, confidence: float):
        if self._fig is None:
            self._fig, self._axes = plt.subplots(3, 1, figsize=(12, 8))
            self._fig.suptitle("Regime Trader — Live Dashboard")

        ax_eq, ax_regime, ax_pos = self._axes

        # Equity curve
        ax_eq.clear()
        ax_eq.plot(self._dates, self._equity_history, color="#3498db", linewidth=1.5)
        ax_eq.set_title("Portfolio Value")
        ax_eq.set_ylabel("$")

        # Regime history
        ax_regime.clear()
        for i, (dt, reg) in enumerate(zip(self._dates, self._regime_history)):
            color = REGIME_COLORS.get(reg, "#95a5a6")
            ax_regime.axvspan(
                i - 0.5, i + 0.5, color=color, alpha=0.6
            )
        current_regime = self._regime_history[-1] if self._regime_history else "—"
        ax_regime.set_title(f"Regime: {current_regime} (conf={confidence:.2f})")
        ax_regime.set_xlim(0, max(1, len(self._dates)))

        # Positions
        ax_pos.clear()
        if positions is not None and not positions.empty:
            positions["unrealized_pnl"].plot(kind="bar", ax=ax_pos, color="#27ae60")
            ax_pos.set_title("Open Positions — Unrealized P&L")
        else:
            ax_pos.set_title("No Open Positions")

        plt.tight_layout()
        plt.pause(0.01)

    def save(self, path: str = "dashboard.png"):
        if self._fig:
            self._fig.savefig(path, dpi=150)
            logger.info(f"Dashboard saved: {path}")
