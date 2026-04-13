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
#   5. Gold log returns (FRED GOLDAMGBD228NLBM)
#      Flight-to-safety cross-asset signal independent of VIX/OAS.
#      Ref: Baur & Lucey (2010, Financial Review 46(1):217-229) — gold is a
#      hedge and safe haven during extreme equity market conditions. Adds
#      information not captured by implied vol (VIX) or credit spreads (OAS).
#
#   6. Term spread — 10yr minus 2yr yield (FRED T10Y2Y)
#      Yield curve shape as macro cycle indicator. Negative = inverted = recession.
#      Ref: Estrella & Mishkin (1998, Review of Economics and Statistics 80(1):45-61).
#      Captures rate/macro regime; does not overlap with HY OAS (credit) or VIX (implied vol).
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

# Base 6-feature set — all peer-reviewed, covariance-type agnostic.
FEATURE_COLS_BASE = ["log_return", "realized_variance", "vix", "hy_oas", "gold_return", "term_spread"]

# 7-feature set — adds vix_slope (VIX/VIX3M). Only useful under covariance_type='full'
# because diagonal covariance cannot exploit the ~50% orthogonal variance in vix_slope
# relative to VIX level. Enable via settings.features.use_vix_slope = true (Phase F).
# See design_docs/09_theory_vs_empirical_conflicts.md §Case 1.
FEATURE_COLS_WITH_VIX_SLOPE = ["log_return", "realized_variance", "vix", "vix_slope",
                                "hy_oas", "gold_return", "term_spread"]

# Module-level alias — reflects current settings at import time.
# Use FeatureEngineer.feature_cols for the instance-level selection.
FEATURE_COLS = FEATURE_COLS_BASE


class FeatureEngineer:
    """
    Builds the (T, 6) or (T, 7) observation matrix for HMM training and inference.
    Feature set controlled by settings.features.use_vix_slope (default False).
    Persists normalization scaler — apply identical transform at live inference.
    """

    def __init__(self, settings: dict):
        self.realized_vol_window = settings["features"]["realized_vol_window"]
        self.norm_window = settings["features"]["normalization_window"]
        use_vix_slope = settings["features"].get("use_vix_slope", False)
        self.feature_cols = FEATURE_COLS_WITH_VIX_SLOPE if use_vix_slope else FEATURE_COLS_BASE
        self._scaler_params: dict = {}   # {col: {mean_series, std_series}} stored after fit

    def compute(
        self,
        prices: pd.DataFrame,
        vix: pd.Series,
        hy_oas: pd.Series,
        gold: pd.Series = None,
        term_spread: pd.Series = None,
        vix3m: pd.Series = None,
    ) -> pd.DataFrame:
        """
        Full pipeline: raw OHLCV + macro → normalized (T, 6) feature matrix.
        Returns DataFrame with FEATURE_COLS columns, same index as prices.

        gold, term_spread, vix3m are optional for backward compatibility;
        if omitted, those columns use neutral fallbacks (no regime signal).
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

        # 5. Gold log returns — flight-to-safety cross-asset signal
        if gold is not None:
            gold_aligned = gold.reindex(prices.index, method="ffill", limit=3)
            df["gold_return"] = compute_log_returns(gold_aligned)
        else:
            df["gold_return"] = 0.0

        # 6. Term spread (10yr-2yr) — yield curve shape / macro cycle
        if term_spread is not None:
            ts_aligned = term_spread.reindex(prices.index, method="ffill", limit=3)
            df["term_spread"] = ts_aligned
        else:
            df["term_spread"] = 0.0

        # 7. VIX slope = VIX / VIX3M — term structure shape signal.
        #    > 1 = backwardation (crisis), < 1 = contango (calm, normal).
        #    Ratio is scale-invariant unlike the arithmetic spread VIX3M-VIX.
        #    Fallback 1.0 = flat term structure (neutral, not extreme contango).
        #    Ref: Egloff et al. (2010); Eraker & Wu (2017); Simon & Campasano (2014).
        if vix3m is not None:
            vix3m_aligned = vix3m.reindex(prices.index, method="ffill", limit=3)
            df["vix_slope"] = vix_aligned / vix3m_aligned
        else:
            df["vix_slope"] = 1.0

        df = df[self.feature_cols].copy()
        df = df.dropna()

        # 8. Rolling Z-score normalization
        df_norm = self._normalize(df)

        logger.debug(f"Feature matrix: {df_norm.shape}, {df_norm.index[0].date()} -> {df_norm.index[-1].date()}")
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
        for col in self.feature_cols:
            params = self._scaler_params[col]
            mean = params["mean"].iloc[-1]
            std = params["std"].iloc[-1]
            result.append((obs[col] - mean) / std if std > 0 else 0.0)
        return np.array(result)

    def get_observation_matrix(self, features_df: pd.DataFrame) -> np.ndarray:
        """Return (T, n_features) numpy array for hmmlearn."""
        return features_df[self.feature_cols].values


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
