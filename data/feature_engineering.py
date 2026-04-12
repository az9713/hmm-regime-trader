# data/feature_engineering.py
# ============================================================
# Feature construction for HMM observation matrix.
#
# Feature set — all peer-reviewed sources (design_docs/03_research_hmm_engine.md):
#
#   1. Log returns
#      Ref: Hamilton (1989, Econometrica 57:357-384)
#           Turner, Startz & Nelson (1989, JFE 25:3-22)
#
#   2. Realized variance (20-day rolling)
#      Ref: Turner et al. (1989, JFE) — variance differences DOMINATE mean differences
#           as the regime-identification mechanism. Include as observation to partially
#           compensate for Gaussian emission misspecification.
#           Gray (1996, JFE 42:27-62) — within-regime vol clustering exists.
#
#   3. VIX daily close (FRED VIXCLS)
#      Ref: Guidolin & Timmermann (2008, RFS 21:889-935) — cross-asset observations
#           materially improve regime separation in multi-state equity models.
#
#   4. HY OAS (FRED BAMLH0A0HYM2)
#      Same Guidolin-Timmermann justification. Duration-adjusted: pure credit signal.
#      NOT HYG/LQD ratio (duration mismatch confounds rate + credit risk).
#
# Normalization: 60-day rolling Z-score.
#   arXiv:2402.05272 (2024): returns + realized vol are the robust features;
#   extra features marginal. Keep feature set small.
#   CRITICAL: persist scaler fitted on training window → apply same at inference.
#   Fitting new scaler on live data is a data leak.
# ============================================================

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

# Feature column order — must be consistent across train and live
FEATURE_COLS = ["log_return", "realized_variance", "vix", "hy_oas"]


class FeatureEngineer:
    """
    Builds the (T, 4) observation matrix for HMM training and inference.
    Persists normalization scaler — apply identical transform at live inference.
    """

    def __init__(self, settings: dict):
        self.realized_vol_window = settings["features"]["realized_vol_window"]
        self.norm_window = settings["features"]["normalization_window"]
        self._scaler_params: dict = {}   # {col: {mean_series, std_series}} stored after fit

    def compute(
        self,
        prices: pd.DataFrame,
        vix: pd.Series,
        hy_oas: pd.Series,
    ) -> pd.DataFrame:
        """
        Full pipeline: raw OHLCV + macro → normalized (T, 4) feature matrix.
        Returns DataFrame with FEATURE_COLS columns, same index as prices.
        """
        df = pd.DataFrame(index=prices.index)

        # 1. Log returns
        df["log_return"] = compute_log_returns(prices["Close"])

        # 2. Realized variance (20-day rolling sum of squared log returns)
        df["realized_variance"] = compute_realized_variance(
            df["log_return"], window=self.realized_vol_window
        )

        # 3. VIX — forward-fill up to 3 days for weekends / FRED gaps
        vix_aligned = vix.reindex(prices.index, method="ffill", limit=3)
        df["vix"] = vix_aligned

        # 4. HY OAS — same alignment
        oas_aligned = hy_oas.reindex(prices.index, method="ffill", limit=3)
        df["hy_oas"] = oas_aligned

        df = df[FEATURE_COLS].copy()
        df = df.dropna()

        # 5. Rolling Z-score normalization
        df_norm = self._normalize(df)

        logger.debug(f"Feature matrix: {df_norm.shape}, {df_norm.index[0].date()} → {df_norm.index[-1].date()}")
        return df_norm

    def _normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Rolling Z-score: (x - rolling_mean) / rolling_std over norm_window.
        Stores scaler params for live reuse.
        """
        normalized = pd.DataFrame(index=df.index, columns=df.columns, dtype=float)
        for col in df.columns:
            roll_mean = df[col].rolling(window=self.norm_window, min_periods=self.norm_window // 2).mean()
            roll_std = df[col].rolling(window=self.norm_window, min_periods=self.norm_window // 2).std()
            roll_std = roll_std.replace(0, np.nan)  # avoid divide-by-zero
            normalized[col] = (df[col] - roll_mean) / roll_std
            self._scaler_params[col] = {"mean": roll_mean, "std": roll_std}
        return normalized.dropna()

    def normalize_live(self, obs: pd.Series) -> np.ndarray:
        """
        Normalize a single live observation using stored scaler params.
        MUST call compute() first to fit scaler.
        """
        if not self._scaler_params:
            raise RuntimeError("Scaler not fitted. Call compute() on training data first.")
        result = []
        for col in FEATURE_COLS:
            params = self._scaler_params[col]
            mean = params["mean"].iloc[-1]
            std = params["std"].iloc[-1]
            result.append((obs[col] - mean) / std if std > 0 else 0.0)
        return np.array(result)

    def get_observation_matrix(self, features_df: pd.DataFrame) -> np.ndarray:
        """Return (T, 4) numpy array for hmmlearn."""
        return features_df[FEATURE_COLS].values


def compute_log_returns(close: pd.Series) -> pd.Series:
    """
    Log returns: r_t = log(P_t / P_{t-1}).
    Ref: Hamilton (1989); Turner et al. (1989).
    """
    return np.log(close / close.shift(1))


def compute_realized_variance(returns: pd.Series, window: int = 20) -> pd.Series:
    """
    Rolling realized variance: (1/window) * Σ r_i²
    Ref: Turner, Startz & Nelson (1989, JFE 25:3-22).
    Variance differences across states are the primary HMM identification mechanism.
    Annualized by multiplying by 252.
    """
    return returns.pow(2).rolling(window=window).mean() * 252


def compute_ewma_realized_vol(returns: pd.Series, halflife: int = 10) -> pd.Series:
    """
    EWMA realized volatility for Moreira-Muir allocation formula.
    w_t = min(target_vol / ewma_vol, max_leverage)
    Ref: Moreira & Muir (2017, Journal of Finance 72(4):1611-1644).
    halflife=10 days as in original paper (monthly realized vol with 10d half-life).
    Returns annualized volatility.
    """
    ewma_var = returns.pow(2).ewm(halflife=halflife).mean()
    return np.sqrt(ewma_var * 252)


def align_macro_features(
    prices_df: pd.DataFrame,
    vix_series: pd.Series,
    hy_oas_series: pd.Series,
) -> pd.DataFrame:
    """
    Align FRED macro series to trading-day index of price data.
    Forward-fill up to 3 days for weekends and FRED reporting lags.
    """
    idx = prices_df.index
    vix = vix_series.reindex(idx, method="ffill", limit=3)
    oas = hy_oas_series.reindex(idx, method="ffill", limit=3)
    return pd.DataFrame({"vix": vix, "hy_oas": oas}, index=idx)


def rolling_zscore(series: pd.Series, window: int = 60) -> pd.Series:
    """Standalone rolling Z-score utility."""
    roll_mean = series.rolling(window=window, min_periods=window // 2).mean()
    roll_std = series.rolling(window=window, min_periods=window // 2).std()
    return (series - roll_mean) / roll_std.replace(0, np.nan)
