# broker/alpaca_client.py
# ============================================================
# Alpaca Markets API wrapper using alpaca-py SDK.
#
# Data source note (design_docs/05_data_sources.md):
#   Live trading uses Alpaca's official CTA/UTP feeds — same source as execution.
#   This ensures price consistency (no subtle slippage from data source mismatch).
#   yfinance used only for historical backtest.
#
# Paper trading is default mode. Live requires --live flag + "CONFIRM" prompt in main.py.
# ============================================================

import logging
import os
import pandas as pd
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class AlpacaClient:
    """
    Thin wrapper around alpaca-py SDK.
    Provides unified interface for both paper and live trading.
    """

    def __init__(self, settings: dict):
        self.settings = settings
        self.mode = settings["broker"]["mode"]  # "paper" | "live"
        self._trading_client = None
        self._data_client = None
        self._stream_client = None
        self._connected = False

    def connect(self):
        """Initialize Alpaca SDK clients from credentials."""
        try:
            from alpaca.trading.client import TradingClient
            from alpaca.data.historical import StockHistoricalDataClient
            from alpaca.data.live import StockDataStream

            api_key = os.getenv("ALPACA_API_KEY")
            secret_key = os.getenv("ALPACA_SECRET_KEY")

            if not api_key or not secret_key:
                raise EnvironmentError(
                    "ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in environment."
                )

            paper = (self.mode == "paper")
            self._trading_client = TradingClient(api_key, secret_key, paper=paper)
            self._data_client = StockHistoricalDataClient(api_key, secret_key)
            self._stream_client = StockDataStream(api_key, secret_key)
            self._connected = True
            logger.info(f"Alpaca connected (mode={self.mode})")

        except ImportError as e:
            raise ImportError("alpaca-py not installed. Run: pip install alpaca-py") from e

    def get_account(self) -> dict:
        self._require_connection()
        account = self._trading_client.get_account()
        return {
            "portfolio_value": float(account.portfolio_value),
            "buying_power": float(account.buying_power),
            "cash": float(account.cash),
            "equity": float(account.equity),
        }

    def get_positions(self) -> dict:
        """Returns {symbol: {qty, avg_entry, market_value, unrealized_pnl}}"""
        self._require_connection()
        positions = self._trading_client.get_all_positions()
        result = {}
        for pos in positions:
            result[pos.symbol] = {
                "qty": float(pos.qty),
                "avg_entry": float(pos.avg_entry_price),
                "market_value": float(pos.market_value),
                "unrealized_pnl": float(pos.unrealized_pl),
                "unrealized_pnl_pct": float(pos.unrealized_plpc),
            }
        return result

    def get_bars(self, symbol: str, start: str, end: str = None) -> pd.DataFrame:
        """Historical OHLCV bars from Alpaca (used during live trading for warmup)."""
        self._require_connection()
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame

        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Day,
            start=start,
            end=end,
        )
        bars = self._data_client.get_stock_bars(request)
        df = bars.df
        if hasattr(df.index, "levels"):
            df = df.xs(symbol, level=0)
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df = df[["open", "high", "low", "close", "volume"]].rename(columns={
            "open": "Open", "high": "High", "low": "Low",
            "close": "Close", "volume": "Volume"
        })
        return df

    def subscribe_bars(self, symbols: list, callback):
        """Subscribe to live bar stream via WebSocket."""
        self._require_connection()
        self._stream_client.subscribe_bars(callback, *symbols)

    def run_stream(self):
        """Block and run the WebSocket stream."""
        self._stream_client.run()

    def _require_connection(self):
        if not self._connected:
            raise RuntimeError("AlpacaClient not connected. Call connect() first.")
