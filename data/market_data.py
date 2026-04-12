# data/market_data.py
# ============================================================
# Unified data interface — same API for backtest and live.
# Swap underlying source without changing strategy code.
#
# MVP stack (design_docs/05_data_sources.md):
#   Price:  yfinance (repair=True) — free, adequate for development
#   Macro:  FRED via pandas-datareader — gold standard, always free
#   Live:   Alpaca WebSocket (same broker used for execution)
#
# CREDIT SPREAD CORRECTION:
#   Do NOT use log(HYG/LQD). Duration mismatch (LQD ~7.7yr vs HYG ~3.9yr)
#   confounds interest rate risk with credit risk.
#   Use FRED BAMLH0A0HYM2 (ICE BofA HY OAS, duration-adjusted).
#   See design_docs/05_data_sources.md §"Critical Correction".
# ============================================================

import os
import logging
import pandas as pd
import numpy as np
import yfinance as yf
from pandas_datareader import data as pdr

logger = logging.getLogger(__name__)


class DataManager:
    """
    Unified data interface. mode='backtest' uses yfinance + FRED.
    mode='live' uses Alpaca WebSocket + FRED.
    """

    def __init__(self, settings: dict, mode: str = "backtest"):
        self.settings = settings
        self.mode = mode
        self.cache_dir = settings["data"].get("cache_dir", "data/cache")
        os.makedirs(self.cache_dir, exist_ok=True)

        if mode == "backtest":
            self.price_source = YFinanceSource(repair=settings["data"].get("repair", True))
        else:
            self.price_source = AlpacaSource(settings)

        self.macro_source = FREDSource()

    def get_bars(self, symbol: str, start: str, end: str = None) -> pd.DataFrame:
        """OHLCV bars for a symbol. Validated before return."""
        df = self.price_source.get_bars(symbol, start, end)
        validate_price_data(df, symbol)
        return df

    def get_vix(self, start: str, end: str = None) -> pd.Series:
        """VIX daily close from FRED VIXCLS."""
        series = self.macro_source.get_series(
            self.settings["data"]["fred_series"]["vix"], start, end
        )
        return series.dropna()

    def get_hy_oas(self, start: str, end: str = None) -> pd.Series:
        """
        ICE BofA High Yield Option-Adjusted Spread from FRED BAMLH0A0HYM2.
        Duration-adjusted — pure credit signal, no interest rate contamination.
        """
        series = self.macro_source.get_series(
            self.settings["data"]["fred_series"]["hy_oas"], start, end
        )
        return series.dropna()

    def get_all_features(self, symbols: list, start: str, end: str = None) -> dict:
        """
        Fetch prices + macro for all symbols.
        Returns dict: symbol → {prices: df, vix: series, hy_oas: series}
        """
        vix = self.get_vix(start, end)
        hy_oas = self.get_hy_oas(start, end)
        validate_macro_data(vix, hy_oas)

        result = {}
        for sym in symbols:
            prices = self.get_bars(sym, start, end)
            result[sym] = {"prices": prices, "vix": vix, "hy_oas": hy_oas}
        return result


class YFinanceSource:
    """
    yfinance wrapper with repair=True and sanity checks.
    Acceptable for MVP and backtesting. NOT for live production.
    See design_docs/05_data_sources.md for known issues.
    """

    def __init__(self, repair: bool = True):
        self.repair = repair

    def get_bars(self, symbol: str, start: str, end: str = None) -> pd.DataFrame:
        logger.info(f"Fetching {symbol} via yfinance ({start} -> {end or 'today'})")
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start, end=end, repair=self.repair, auto_adjust=True)

        if df.empty:
            raise ValueError(f"yfinance returned empty data for {symbol}")

        # Normalize column names
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()

        # yfinance auto_adjust applies dividend/split factors per column independently.
        # This can produce Close > High or Close < Low on ex-dividend days (known artifact).
        # Clamp to ensure OHLC consistency before validation.
        df["High"] = df[["Open", "High", "Low", "Close"]].max(axis=1)
        df["Low"] = df[["Open", "High", "Low", "Close"]].min(axis=1)

        return df


class AlpacaSource:
    """
    Alpaca data feed. Used for live trading to ensure price consistency
    with execution feed (CTA/UTP official feeds).
    """

    def __init__(self, settings: dict):
        self.settings = settings

    def get_bars(self, symbol: str, start: str, end: str = None) -> pd.DataFrame:
        raise NotImplementedError("Alpaca live source — implement in Phase 5")


class FREDSource:
    """
    FRED data via pandas-datareader.
    Gold standard source — official Federal Reserve data.
    VIX: VIXCLS (1990-present). HY OAS: BAMLH0A0HYM2 (1997-present).
    """

    def get_series(self, series_id: str, start: str, end: str = None) -> pd.Series:
        logger.info(f"Fetching FRED series {series_id} ({start} -> {end or 'today'})")
        try:
            df = pdr.get_data_fred(series_id, start=start, end=end)
            series = df.iloc[:, 0]
            series.index = pd.to_datetime(series.index).tz_localize(None)
            return series
        except Exception as e:
            raise RuntimeError(f"FRED fetch failed for {series_id}: {e}") from e


def validate_price_data(df: pd.DataFrame, ticker: str) -> bool:
    """
    Mandatory validation before any backtest.
    Catches data errors early. Run on every fetch.
    Ref: design_docs/05_data_sources.md §"Data Validation Protocol"
    """
    if df.empty:
        raise ValueError(f"{ticker}: empty price data")

    # 1. OHLC sanity
    if not (df["Low"] <= df["Close"]).all():
        bad = df[df["Low"] > df["Close"]]
        raise AssertionError(f"{ticker}: Close < Low on {len(bad)} days:\n{bad.head()}")
    if not (df["Close"] <= df["High"]).all():
        bad = df[df["Close"] > df["High"]]
        raise AssertionError(f"{ticker}: Close > High on {len(bad)} days:\n{bad.head()}")
    if not (df["Low"] <= df["High"]).all():
        bad = df[df["Low"] > df["High"]]
        raise AssertionError(f"{ticker}: Low > High on {len(bad)} days:\n{bad.head()}")

    # 2. No zero or negative prices
    if not (df["Close"] > 0).all():
        raise AssertionError(f"{ticker}: zero or negative Close prices detected")

    # 3. Suspicious flat days (all OHLC equal — likely missing data)
    flat = (df["Open"] == df["High"]) & (df["High"] == df["Low"])
    if flat.sum() > len(df) * 0.01:
        logger.warning(f"{ticker}: {flat.sum()} flat days ({flat.sum()/len(df)*100:.1f}%) — possible missing data")

    # 4. Extreme daily moves (>10% for SPY/QQQ is extraordinary)
    daily_ret = df["Close"].pct_change().dropna()
    extreme = abs(daily_ret) > 0.10
    if extreme.sum() > 0:
        logger.warning(f"{ticker}: {extreme.sum()} days with >10% daily move — verify data")

    # 5. No gaps >5 trading days (missing data)
    if len(df) > 5:
        date_diffs = df.index.to_series().diff().dt.days.dropna()
        large_gaps = date_diffs[date_diffs > 7]  # 7 calendar days > 5 trading days
        if len(large_gaps) > 0:
            logger.warning(f"{ticker}: {len(large_gaps)} date gaps > 7 days:\n{large_gaps.head()}")

    logger.debug(f"{ticker}: validation passed ({len(df)} bars, {df.index[0].date()} -> {df.index[-1].date()})")
    return True


def validate_macro_data(vix_series: pd.Series, hy_oas_series: pd.Series) -> bool:
    """
    Validate FRED macro features.
    VIX range [5, 90], HY OAS range [1%, 25%] historically.
    Ref: design_docs/05_data_sources.md §"Data Validation Protocol"
    """
    if vix_series.empty:
        raise ValueError("VIX series is empty")
    if hy_oas_series.empty:
        raise ValueError("HY OAS series is empty")

    vix_clean = vix_series.dropna()
    oas_clean = hy_oas_series.dropna()

    if not (vix_clean > 5).all():
        raise AssertionError(f"VIX values below 5 detected — data error")
    if not (vix_clean < 90).all():
        raise AssertionError(f"VIX values above 90 detected — data error")

    if not (oas_clean > 1.0).all():
        raise AssertionError(f"HY OAS below 1% detected — data error")
    if not (oas_clean < 25.0).all():
        raise AssertionError(f"HY OAS above 25% detected — data error")

    logger.info(
        f"VIX: {len(vix_clean)} obs, {vix_clean.index[0].date()} -> {vix_clean.index[-1].date()}"
        f" | HY OAS: {len(oas_clean)} obs, {oas_clean.index[0].date()} -> {oas_clean.index[-1].date()}"
    )
    return True
