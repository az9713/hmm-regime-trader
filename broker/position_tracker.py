# broker/position_tracker.py
# ============================================================
# Tracks open positions. Reconciles local state with Alpaca every tick.
# Alpaca is the source of truth — never trust local state alone.
# ============================================================

import logging
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime

from broker.alpaca_client import AlpacaClient

logger = logging.getLogger(__name__)


@dataclass
class Position:
    symbol: str
    qty: float
    avg_entry: float
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    stop_price: Optional[float] = None
    target_price: Optional[float] = None
    regime_at_entry: Optional[str] = None
    entered_at: Optional[datetime] = None


class PositionTracker:
    """
    Maintains in-memory position state synchronized with Alpaca.
    Reconcile on every bar to catch fills, stops, and target hits.
    """

    def __init__(self, client: AlpacaClient):
        self.client = client
        self._positions: dict[str, Position] = {}
        self._local_metadata: dict[str, dict] = {}  # stop/target/regime from signal

    def reconcile(self) -> dict[str, Position]:
        """
        Sync local state with Alpaca. Alpaca is source of truth.
        Call on every bar update.
        """
        try:
            broker_positions = self.client.get_positions()
        except Exception as e:
            logger.error(f"Position reconcile failed: {e}")
            return self._positions

        # Update existing / add new
        updated = {}
        for symbol, data in broker_positions.items():
            meta = self._local_metadata.get(symbol, {})
            updated[symbol] = Position(
                symbol=symbol,
                qty=data["qty"],
                avg_entry=data["avg_entry"],
                market_value=data["market_value"],
                unrealized_pnl=data["unrealized_pnl"],
                unrealized_pnl_pct=data["unrealized_pnl_pct"],
                stop_price=meta.get("stop_price"),
                target_price=meta.get("target_price"),
                regime_at_entry=meta.get("regime"),
                entered_at=meta.get("entered_at"),
            )

        # Detect closed positions (existed locally but not in broker)
        closed = set(self._positions.keys()) - set(updated.keys())
        for sym in closed:
            logger.info(f"Position closed: {sym}")
            self._local_metadata.pop(sym, None)

        self._positions = updated
        return self._positions

    def register_signal(self, symbol: str, stop: float, target: float, regime: str):
        """Store metadata for a signal (stop/target/regime) before order fills."""
        self._local_metadata[symbol] = {
            "stop_price": stop,
            "target_price": target,
            "regime": regime,
            "entered_at": datetime.now(),
        }

    @property
    def positions(self) -> dict[str, Position]:
        return self._positions

    def has_position(self, symbol: str) -> bool:
        return symbol in self._positions

    def total_market_value(self) -> float:
        return sum(p.market_value for p in self._positions.values())

    def daily_pnl_pct(self, portfolio_value: float) -> float:
        if portfolio_value <= 0:
            return 0.0
        total_pnl = sum(p.unrealized_pnl for p in self._positions.values())
        return total_pnl / portfolio_value

    def to_dataframe(self) -> pd.DataFrame:
        if not self._positions:
            return pd.DataFrame()
        rows = []
        for pos in self._positions.values():
            rows.append({
                "symbol": pos.symbol,
                "qty": pos.qty,
                "avg_entry": pos.avg_entry,
                "market_value": pos.market_value,
                "unrealized_pnl": pos.unrealized_pnl,
                "unrealized_pnl_pct": pos.unrealized_pnl_pct,
                "stop": pos.stop_price,
                "target": pos.target_price,
                "regime": pos.regime_at_entry,
            })
        return pd.DataFrame(rows).set_index("symbol")
