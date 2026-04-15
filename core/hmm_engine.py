# core/hmm_engine.py
# ============================================================
# HMM regime detection engine.
#
# ARCHITECTURE:
#   Model:    hmmlearn GaussianHMM, covariance_type='diag'
#             Gaussian is practical default. Theoretically misspecified per
#             Gray (1996, JFE 42:27-62): within-regime vol-clustering exists.
#             SEP-HMM (2025, MDPI Mathematics 14(3)): Student-t handles tails better.
#             Keep Gaussian for MVP; Student-t is principled upgrade path.
#
#   States:   BIC sweep n_components ∈ [3,7], pick min BIC.
#             Ang & Bekaert (2002, RFS 15:1137-1187); Guidolin & Timmermann (2008, RFS).
#             Top-tier literature concentrates at 2-4 states. Never hardcode 5.
#
#   Init:     K-means, n_init=10, pick best log-likelihood.
#             Baum & Petrie (1966, Ann Math Stat 37:1554-1563).
#
#   Retrain:  Every 20 bars on rolling 500-bar window.
#             Markets are non-stationary: Ang & Bekaert (2002).
#
#   Labels:   Sort states by mean realized variance (ascending) → LowVol ... HighVol.
#             Turner, Startz & Nelson (1989, JFE 25:3-22): variance differences
#             dominate mean differences as the primary identification signal.
#
# LOOK-AHEAD PREVENTION (critical for valid backtesting):
#   sklearn model.predict() = Viterbi over FULL observation sequence.
#   This uses future data → look-ahead bias → invalid backtest.
#   We implement manual forward α-recursion in log-space.
#   At time t: uses ONLY observations o_1, ..., o_t.
#
#   Forward algorithm (Hamilton 1989, Econometrica 57:357-384;
#                      Baum & Petrie 1966; Rabiner 1989, Proc IEEE 77:257-286):
#
#     Initialization:
#       log α_1(j) = log π_j + log b_j(o_1)
#
#     Recursion (t = 2, ..., T):
#       log α_t(j) = log b_j(o_t) + logsumexp_i( log α_{t-1}(i) + log A_ij )
#
#     Regime at time t:
#       raw_t = argmax_j α_t(j)
#
#     Confidence:
#       conf_t = max_j softmax(log α_t)
#
# STABILITY FILTERS:
#   3-bar persistence (Asem & Tian 2010, JFQA 45:1549-1562):
#     Commit to regime switch only after 3 consecutive bars confirm new regime.
#     Momentum highest in continuation states; lowest in transition states.
#
#   Flicker detector (heuristic, direction from Asem & Tian):
#     >4 switches in 20 bars → UNCERTAINTY mode.
#
#   Confidence floor (engineering judgment):
#     HMM posterior < 0.40 → UNCERTAINTY mode.
# ============================================================

import logging
import numpy as np
import pandas as pd
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from hmmlearn import hmm
from scipy.special import logsumexp

logger = logging.getLogger(__name__)


class RegimeLabel:
    LOW_VOL = "LowVol"
    MID_VOL = "MidVol"
    HIGH_VOL = "HighVol"
    VERY_HIGH_VOL = "VeryHighVol"    # used if BIC selects 4+ states
    UNCERTAINTY = "Uncertainty"


# Ordered sequence for n-state labeling (ascending vol)
REGIME_SEQUENCE = [
    RegimeLabel.LOW_VOL,
    RegimeLabel.MID_VOL,
    RegimeLabel.HIGH_VOL,
    RegimeLabel.VERY_HIGH_VOL,
]


@dataclass
class RegimeState:
    label: str                        # RegimeLabel value
    raw_state_idx: int                # HMM state index (before sorting)
    confidence: float                 # posterior probability [0, 1]
    stable: bool = True               # False during uncertainty
    bars_in_regime: int = 0
    n_states_selected: int = 0        # BIC-selected state count


class HMMEngine:
    """
    HMM-based regime detector with look-ahead-free inference.
    """

    def __init__(self, settings: dict):
        cfg = settings["hmm"]
        stab = settings["stability"]

        self.n_components_range = tuple(cfg["n_components_range"])
        self.covariance_type = cfg["covariance_type"]
        self.n_iter = cfg["n_iter"]
        self.tol = cfg["tol"]
        self.n_init = cfg["n_init"]
        self.training_window = cfg["training_window"]
        self.retrain_every = cfg["retrain_every"]
        self.random_state = cfg.get("random_state", 42)

        self.persistence_bars = stab["persistence_bars"]
        self.flicker_window = stab["flicker_window"]
        self.flicker_threshold = stab["flicker_threshold"]
        self.confidence_floor = stab["confidence_floor"]

        self.model: Optional[hmm.GaussianHMM] = None
        self.n_states: int = 0
        self.state_labels: dict = {}      # raw_idx → RegimeLabel
        self.variance_feature_idx: int = 1  # index of realized_variance in feature vector

        # Stability filter state
        self._pending_regime: Optional[str] = None
        self._pending_count: int = 0
        self._confirmed_regime: str = RegimeLabel.UNCERTAINTY
        self._recent_switches: deque = deque(maxlen=self.flicker_window)
        self._bars_in_regime: int = 0
        self._bars_since_retrain: int = 0

        # Latest log α vector (n_states,), set by step()/predict_regime()
        # for downstream consumers such as RegimeForecaster.
        self._last_log_alpha: Optional[np.ndarray] = None

    def fit(self, observations: np.ndarray) -> int:
        """
        Train HMM. BIC sweep selects optimal state count.
        Returns selected n_components.
        Ref: Universal ML/econometric standard; lit concentrates 2-4 states.
        """
        n_min, n_max = self.n_components_range
        best_bic = np.inf
        best_n = n_min
        best_model = None

        d = observations.shape[1]
        for n in range(n_min, n_max + 1):
            try:
                model = self._train_single(observations, n)
                log_lik = model.score(observations)
                # Parameter count depends on covariance type:
                #   diag:     n*n (trans) + n*d (means) + n*d (variances)
                #   full:     n*n (trans) + n*d (means) + n*d*(d+1)/2 (full cov per state)
                #   tied:     n*n (trans) + n*d (means) + d*(d+1)/2 (one shared cov)
                #   spherical: n*n (trans) + n*d (means) + n (one variance per state)
                cov = self.covariance_type
                if cov == "full":
                    n_params = n * n + n * d + n * d * (d + 1) // 2
                elif cov == "tied":
                    n_params = n * n + n * d + d * (d + 1) // 2
                elif cov == "spherical":
                    n_params = n * n + n * d + n
                else:  # diag (default)
                    n_params = n * n + n * d * 2
                bic = -2 * log_lik + n_params * np.log(len(observations))
                logger.debug(f"  n={n}: log_lik={log_lik:.2f}, BIC={bic:.2f}")
                if bic < best_bic:
                    best_bic = bic
                    best_n = n
                    best_model = model
            except Exception as e:
                logger.warning(f"HMM fit failed for n={n}: {e}")
                continue

        if best_model is None:
            raise RuntimeError("HMM training failed for all state counts")

        self.model = best_model
        self.n_states = best_n
        self.state_labels = label_states_by_variance(best_model, observations, self.variance_feature_idx)
        logger.info(f"HMM trained: n_states={best_n} (BIC={best_bic:.1f}), labels={self.state_labels}")
        return best_n

    def _train_single(self, observations: np.ndarray, n_components: int) -> hmm.GaussianHMM:
        """
        Train one GaussianHMM with K-means init, n_init restarts.
        Pick best log-likelihood across restarts.
        """
        best_score = -np.inf
        best_model = None
        for seed in range(self.n_init):
            try:
                model = hmm.GaussianHMM(
                    n_components=n_components,
                    covariance_type=self.covariance_type,
                    n_iter=self.n_iter,
                    tol=self.tol,
                    init_params="kmeans",
                    random_state=self.random_state + seed,
                )
                model.fit(observations)
                score = model.score(observations)
                if score > best_score:
                    best_score = score
                    best_model = model
            except Exception:
                continue
        if best_model is None:
            raise RuntimeError(f"All {self.n_init} restarts failed for n={n_components}")
        return best_model

    def predict_regime(self, observations: np.ndarray) -> list[RegimeState]:
        """
        Run forward α-recursion over observation sequence.
        Returns list of RegimeState per time step.
        NEVER calls model.predict() — that uses Viterbi over full sequence (look-ahead).
        """
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        log_alphas, confidences = forward_algorithm_log(self.model, observations)
        # Persist the final log α for any downstream forecaster
        self._last_log_alpha = log_alphas[-1].copy()
        raw_regimes = np.argmax(log_alphas, axis=1)

        results = []
        for t, (raw, conf) in enumerate(zip(raw_regimes, confidences)):
            label = self.state_labels.get(int(raw), RegimeLabel.UNCERTAINTY)
            stable_label = self._apply_stability_filters(label, conf)
            results.append(RegimeState(
                label=stable_label,
                raw_state_idx=int(raw),
                confidence=float(conf),
                stable=(stable_label != RegimeLabel.UNCERTAINTY),
                bars_in_regime=self._bars_in_regime,
                n_states_selected=self.n_states,
            ))
        return results

    def step(self, observation: np.ndarray) -> RegimeState:
        """
        Single-bar inference for live trading.
        Updates internal filter state incrementally.
        """
        if self.model is None:
            raise RuntimeError("Model not fitted.")

        # Single-step forward pass
        log_emission = _log_emission_probs(self.model, observation.reshape(1, -1))
        # For a single new observation we just use the prior + emission
        log_init = np.log(self.model.startprob_ + 1e-300)
        log_alpha = log_init + log_emission[0]
        self._last_log_alpha = log_alpha.copy()
        conf = float(np.exp(log_alpha - logsumexp(log_alpha)).max())
        raw = int(np.argmax(log_alpha))
        label = self.state_labels.get(raw, RegimeLabel.UNCERTAINTY)
        stable_label = self._apply_stability_filters(label, conf)

        self._bars_since_retrain += 1

        return RegimeState(
            label=stable_label,
            raw_state_idx=raw,
            confidence=conf,
            stable=(stable_label != RegimeLabel.UNCERTAINTY),
            bars_in_regime=self._bars_in_regime,
            n_states_selected=self.n_states,
        )

    def forecast(self, horizon: int = 20, risk_horizon: int = 5):
        """
        Build a forward-looking regime forecast from the latest filtered
        posterior. Must be called *after* `step()` or `predict_regime()`
        so that the internal log α vector is populated.

        Returns a `RegimeForecast` with k-step regime distributions,
        expected sojourn time, transition risk, and stationary distribution.
        See core/regime_forecaster.py for the full API.
        """
        if self.model is None:
            raise RuntimeError("Model not fitted.")
        if self._last_log_alpha is None:
            raise RuntimeError(
                "No posterior available — call step() or predict_regime() first."
            )
        # Local import to avoid a circular import at module load time.
        from core.regime_forecaster import RegimeForecaster
        forecaster = RegimeForecaster(
            transmat=self.model.transmat_,
            state_labels=self.state_labels,
            horizon=horizon,
            risk_horizon=risk_horizon,
        )
        return forecaster.forecast_from_log_alpha(self._last_log_alpha, horizon=horizon)

    def needs_retrain(self) -> bool:
        return self._bars_since_retrain >= self.retrain_every

    def reset_retrain_counter(self):
        self._bars_since_retrain = 0

    def _apply_stability_filters(self, raw_label: str, confidence: float) -> str:
        """
        Apply 3-bar persistence and flicker detection.
        Ref: Asem & Tian (2010, JFQA): momentum highest in continuation states.
        """
        # Confidence floor
        if confidence < self.confidence_floor:
            self._bars_in_regime = 0
            return RegimeLabel.UNCERTAINTY

        # 3-bar persistence filter
        if raw_label == self._confirmed_regime:
            self._pending_regime = None
            self._pending_count = 0
            self._bars_in_regime += 1
        else:
            if raw_label == self._pending_regime:
                self._pending_count += 1
            else:
                self._pending_regime = raw_label
                self._pending_count = 1

            if self._pending_count >= self.persistence_bars:
                # Track switch for flicker detection
                self._recent_switches.append(1)
                self._confirmed_regime = raw_label
                self._pending_regime = None
                self._pending_count = 0
                self._bars_in_regime = 1
            else:
                # Not yet confirmed — stay in current regime
                self._bars_in_regime += 1
                return self._confirmed_regime

        # Flicker detection (>threshold switches in window → uncertainty)
        recent_switch_count = sum(self._recent_switches)
        if recent_switch_count > self.flicker_threshold:
            return RegimeLabel.UNCERTAINTY

        return self._confirmed_regime


def bic_state_selection(observations: np.ndarray, n_range: tuple = (3, 7),
                         n_iter: int = 100, n_init: int = 10,
                         random_state: int = 42) -> tuple:
    """
    Standalone BIC sweep. Returns (best_n, best_model, bic_scores dict).
    """
    bic_scores = {}
    best_bic = np.inf
    best_n = n_range[0]
    best_model = None

    for n in range(n_range[0], n_range[1] + 1):
        try:
            model = hmm.GaussianHMM(
                n_components=n, covariance_type="diag",
                n_iter=n_iter, init_params="kmeans", random_state=random_state
            )
            model.fit(observations)
            log_lik = model.score(observations)
            n_params = n * n + n * observations.shape[1] * 2
            bic = -2 * log_lik + n_params * np.log(len(observations))
            bic_scores[n] = bic
            if bic < best_bic:
                best_bic = bic
                best_n = n
                best_model = model
        except Exception as e:
            logger.warning(f"BIC selection: n={n} failed: {e}")

    return best_n, best_model, bic_scores


def forward_algorithm_log(model: hmm.GaussianHMM, observations: np.ndarray) -> tuple:
    """
    Look-ahead-free regime inference via forward α-recursion in log-space.

    NEVER use model.predict() in backtesting. That method runs Viterbi
    over the complete observation sequence, using future data at each step.

    This function uses ONLY o_1, ..., o_t to compute regime at time t.

    Algorithm (Hamilton 1989, Econometrica 57:357-384;
               Baum & Petrie 1966, Ann Math Stat 37:1554-1563;
               Rabiner 1989, Proc IEEE 77:257-286):

      Initialization:
        log α_1(j) = log π_j + log b_j(o_1)

      Recursion (t = 2, ..., T):
        log α_t(j) = log b_j(o_t) + logsumexp_i( log α_{t-1}(i) + log A_ij )

    Returns:
      log_alphas:   (T, n_states) array of log forward probabilities
      confidences:  (T,) array of max posterior probability (confidence)
    """
    T, n_features = observations.shape
    n_states = model.n_components

    log_A = np.log(model.transmat_ + 1e-300)
    log_pi = np.log(model.startprob_ + 1e-300)
    log_B = _log_emission_probs(model, observations)  # (T, n_states)

    log_alphas = np.zeros((T, n_states))

    # Initialization
    log_alphas[0] = log_pi + log_B[0]

    # Recursion
    for t in range(1, T):
        for j in range(n_states):
            log_alphas[t, j] = log_B[t, j] + logsumexp(log_alphas[t-1] + log_A[:, j])

    # Posterior: normalize each row
    log_normalizer = logsumexp(log_alphas, axis=1, keepdims=True)
    log_posterior = log_alphas - log_normalizer
    posterior = np.exp(log_posterior)
    confidences = posterior.max(axis=1)

    return log_alphas, confidences


def _log_emission_probs(model: hmm.GaussianHMM, observations: np.ndarray) -> np.ndarray:
    """
    Compute log b_j(o_t) for all states j and observations t.
    Uses hmmlearn's internal _compute_log_likelihood to handle all covariance types
    correctly and avoid shape/broadcasting issues across hmmlearn versions.
    Returns (T, n_states).
    """
    # hmmlearn internal method handles diag/full/spherical/tied correctly
    return model._compute_log_likelihood(observations)


def label_states_by_variance(
    model: hmm.GaussianHMM,
    observations: np.ndarray,
    variance_feature_idx: int = 1,
) -> dict:
    """
    Assign regime labels by sorting HMM states on mean realized variance (ascending).
    State with lowest mean realized variance → LowVol.
    State with highest → HighVol (or VeryHighVol if 4+ states).

    Ref: Turner, Startz & Nelson (1989, JFE 25:3-22).
    Variance differences are the primary identification mechanism.
    DO NOT label by mean return — return differences are near-indistinguishable.

    Label mapping by n_states (endpoints always LOW and HIGH):
      n=2: [LowVol, HighVol]
      n=3: [LowVol, MidVol, HighVol]
      n=4: [LowVol, MidVol, HighVol, VeryHighVol]

    Returns: dict mapping raw state index → RegimeLabel string
    """
    n_states = model.n_components
    means = model.means_[:, variance_feature_idx]  # realized variance column
    sorted_idx = np.argsort(means)  # ascending variance order

    # Build n-point label set ensuring LOW is first and HIGH is last
    if n_states == 2:
        label_set = [RegimeLabel.LOW_VOL, RegimeLabel.HIGH_VOL]
    elif n_states == 3:
        label_set = [RegimeLabel.LOW_VOL, RegimeLabel.MID_VOL, RegimeLabel.HIGH_VOL]
    else:
        # 4+: LOW, MID, HIGH, VERY_HIGH (and any extra get HIGH_VOL)
        label_set = [RegimeLabel.LOW_VOL, RegimeLabel.MID_VOL,
                     RegimeLabel.HIGH_VOL, RegimeLabel.VERY_HIGH_VOL]

    labels = {}
    for rank, raw_idx in enumerate(sorted_idx):
        if rank < len(label_set):
            labels[int(raw_idx)] = label_set[rank]
        else:
            labels[int(raw_idx)] = RegimeLabel.HIGH_VOL  # fallback for n>4

    logger.debug(f"State labels by variance rank: {labels}")
    return labels


# Re-export for convenience
REGIME_SEQUENCE = [
    RegimeLabel.LOW_VOL,
    RegimeLabel.MID_VOL,
    RegimeLabel.HIGH_VOL,
    RegimeLabel.VERY_HIGH_VOL,
]
