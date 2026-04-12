# broker/order_executor.py
# ============================================================
# Converts Signal objects into Alpaca bracket orders.
# Bracket = entry market/limit + stop loss + take profit (OCO).
#
# Stop/target prices from ATR multiples in Signal.
# [HYPERPARAMETER] ATR multiples — calibrate via 3-month paper trading.
# See design_docs/06_empirical_testing_plan.md §Group C.
# ============================================================

import logging
from dataclasses import dataclass
from typing import Optional

from broker.alpaca_client import AlpacaClient
from core.signal_generator import Signal

logger = logging.getLogger(__name__)


@dataclass
class OrderResult:
    submitted: bool
    order_id: Optional[str]
    symbol: str
    side: str
    qty: int
    entry: float
    stop: float
    target: float
    message: str


class OrderExecutor:
    """
    Submits bracket orders to Alpaca.
    Bracket: entry + stop-loss + take-profit as one-cancels-other.
    """

    def __init__(self, client: AlpacaClient, settings: dict):
        self.client = client
        self.settings = settings

    def submit(self, signal: Signal) -> OrderResult:
        """Submit bracket order for a Signal. Returns OrderResult."""
        if signal.direction != "long":
            # Safety check — should never be reached given always-long design
            raise ValueError(f"Short orders not permitted. Got direction='{signal.direction}'.")

        try:
            from alpaca.trading.requests import (
                MarketOrderRequest, LimitOrderRequest,
                TakeProfitRequest, StopLossRequest
            )
            from alpaca.trading.enums import OrderSide, TimeInForce

            order_data = MarketOrderRequest(
                symbol=signal.symbol,
                qty=signal.size_shares,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY,
                order_class="bracket",
                take_profit=TakeProfitRequest(limit_price=round(signal.target_price, 2)),
                stop_loss=StopLossRequest(stop_price=round(signal.stop_price, 2)),
            )

            order = self.client._trading_client.submit_order(order_data)
            logger.info(
                f"Bracket order submitted: {signal.symbol} BUY {signal.size_shares}sh "
                f"| stop={signal.stop_price:.2f} target={signal.target_price:.2f} "
                f"| order_id={order.id}"
            )
            return OrderResult(
                submitted=True,
                order_id=str(order.id),
                symbol=signal.symbol,
                side="long",
                qty=signal.size_shares,
                entry=signal.entry_price,
                stop=signal.stop_price,
                target=signal.target_price,
                message="submitted",
            )

        except Exception as e:
            logger.error(f"Order submission failed for {signal.symbol}: {e}")
            return OrderResult(
                submitted=False,
                order_id=None,
                symbol=signal.symbol,
                side="long",
                qty=signal.size_shares,
                entry=signal.entry_price,
                stop=signal.stop_price,
                target=signal.target_price,
                message=str(e),
            )

    def cancel_all(self) -> int:
        """Cancel all open orders. Returns count cancelled."""
        try:
            result = self.client._trading_client.cancel_orders()
            logger.warning(f"Cancelled {len(result)} open orders")
            return len(result)
        except Exception as e:
            logger.error(f"cancel_all failed: {e}")
            return 0

    def liquidate_all(self) -> int:
        """Close all positions. Used by circuit breaker STOPPED state."""
        try:
            result = self.client._trading_client.close_all_positions(cancel_orders=True)
            logger.warning(f"LIQUIDATED all positions ({len(result)} closed)")
            return len(result)
        except Exception as e:
            logger.error(f"liquidate_all failed: {e}")
            return 0
