# core/regime_forecaster.py
# ============================================================
# Forward-looking regime forecasting from a trained HMM.
#
# WHAT IT DOES:
#   The HMM's transition matrix A and the current filtered posterior π_t
#   carry forward-looking information that the per-bar `step()` loop
#   discards. This module makes that information first-class:
#
#     1. k-step regime distribution     π_{t+k} = π_t · A^k
#     2. Expected sojourn time          E[duration | S=i] = 1/(1 - A_ii)
#     3. Transition risk score          P(exit current regime within H bars)
#     4. Most-likely next regime        argmax_{j != i} A_ij  (renormalized)
#     5. Stationary distribution        π* solving π* A = π*
#     6. Forecast horizon by label      aggregates raw states into LowVol / MidVol / HighVol
#
# WHY IT'S USEFUL:
#   The stability filter (persistence_bars, flicker detector) is INTENTIONALLY
#   lagging — it only commits to a regime change after several confirming
#   bars. That is correct for *acting*, but it leaves a gap: by the time
#   the filter flips, the posterior may already have moved meaningfully.
#
#   Transition risk is a LEAD signal:
#     - The confirmed regime is still "LowVol"
#     - But forecaster says P(exit within 5 bars) = 0.62
#     - => downshift allocation, tighten stops, send alert
#
#   This is purely descriptive of the trained HMM. There are no free
#   parameters to tune, so no overfitting risk vs. backtest Sharpe.
#
# REFERENCES:
#   Hamilton (1989, Econometrica 57:357-384) — forward algorithm, Markov chain forecasts
#   Rabiner (1989, Proc IEEE 77:257-286) — HMM evaluation problems
#   Norris (1997) — "Markov Chains" — sojourn time, stationary distribution
#   Ang & Bekaert (2002, RFS 15:1137-1187) — regime-switching forecast horizons
#   Guidolin & Timmermann (2007, JEDC 31:3503-3544) — forecasting with HMM regime
#
# DESIGN NOTES:
#   - All math operates on the HMM's RAW state space (n_components),
#     not on labels. Labels are aggregated at the API boundary so
#     callers can ask "P(HighVol) in 5 bars".
#   - We deliberately use the RAW posterior here (softmax of log_alpha),
#     NOT the stability-filtered confirmed_regime. The filter is for
#     execution; the forecast is for situational awareness.
#   - Stationary distribution is computed via left eigenvector method
#     with a power-iteration fallback for numerical robustness.
# ============================================================

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.special import logsumexp

from core.hmm_engine import RegimeLabel

logger = logging.getLogger(__name__)


@dataclass
class RegimeForecast:
    """One-shot forecast snapshot computed from posterior + transition matrix."""

    # Inputs reflected back for logging
    horizon: int
    current_label: str
    current_label_prob: float

    # Raw-state forward distributions: shape (horizon+1, n_states); row 0 = posterior now
    state_path: np.ndarray

    # Aggregated label probabilities over the horizon: dict[label] -> np.ndarray (horizon+1,)
    label_path: dict

    # Expected sojourn time in current dominant raw state, in bars
    expected_sojourn_bars: float
    # Variance of sojourn-time distribution (geometric)
    sojourn_variance: float

    # P(exit current LABEL within `horizon` bars) — uses label aggregation
    transition_risk: float

    # Most-likely next LABEL after exit (normalized over non-current labels)
    next_label: Optional[str]
    next_label_prob: float

    # Stationary distribution by label
    stationary_by_label: dict

    # Quantile bars-to-exit for current label (P(exit by k) >= q)
    exit_quantiles: dict = field(default_factory=dict)


class RegimeForecaster:
    """
    Computes forward-looking regime statistics from a trained HMM.

    Construction is cheap (no copies); call `forecast()` per bar after
    `HMMEngine.step()` if you want forward visibility.
    """

    def __init__(
        self,
        transmat: np.ndarray,
        state_labels: dict,
        horizon: int = 20,
        risk_horizon: int = 5,
    ):
        """
        Parameters
        ----------
        transmat : (n_states, n_states) HMM transition matrix.  Rows must sum to 1.
        state_labels : dict mapping raw_state_idx -> RegimeLabel string.
        horizon : default forward horizon (bars) to compute when not specified.
        risk_horizon : default horizon used for `transition_risk` summary scalar.
        """
        A = np.asarray(transmat, dtype=float)
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError(f"transmat must be square; got shape {A.shape}")
        row_sums = A.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-6):
            raise ValueError(f"transmat rows must sum to 1; got {row_sums}")
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1; got {horizon}")
        if risk_horizon < 1 or risk_horizon > horizon:
            raise ValueError(
                f"risk_horizon must be in [1, horizon]; got {risk_horizon} (horizon={horizon})"
            )

        self.A = A
        self.n_states = A.shape[0]
        self.state_labels = dict(state_labels)
        self.horizon = int(horizon)
        self.risk_horizon = int(risk_horizon)

        # Pre-compute stationary distribution (depends only on A)
        self._stationary_raw = stationary_distribution(self.A)

    # ── Construction helpers ────────────────────────────────────

    @classmethod
    def from_engine(cls, engine, horizon: int = 20, risk_horizon: int = 5) -> "RegimeForecaster":
        """Build a forecaster from a fitted HMMEngine."""
        if engine.model is None:
            raise RuntimeError("HMMEngine not fitted — cannot build forecaster.")
        return cls(
            transmat=engine.model.transmat_,
            state_labels=engine.state_labels,
            horizon=horizon,
            risk_horizon=risk_horizon,
        )

    # ── Public API ──────────────────────────────────────────────

    def forecast_from_log_alpha(
        self, log_alpha: np.ndarray, horizon: Optional[int] = None
    ) -> RegimeForecast:
        """
        Build a forecast from the latest log α vector (the same internal
        representation HMMEngine.step() computes).
        """
        log_alpha = np.asarray(log_alpha, dtype=float).ravel()
        if log_alpha.shape[0] != self.n_states:
            raise ValueError(
                f"log_alpha length {log_alpha.shape[0]} != n_states {self.n_states}"
            )
        posterior = np.exp(log_alpha - logsumexp(log_alpha))
        return self.forecast(posterior, horizon=horizon)

    def forecast(
        self, posterior: np.ndarray, horizon: Optional[int] = None
    ) -> RegimeForecast:
        """
        Build a forecast given an explicit posterior distribution over raw states.

        Parameters
        ----------
        posterior : (n_states,) probability vector.  Must sum to 1.
        horizon : optional override for forecast horizon.
        """
        H = int(horizon) if horizon is not None else self.horizon
        if H < 1:
            raise ValueError(f"horizon must be >= 1; got {H}")

        post = np.asarray(posterior, dtype=float).ravel()
        if post.shape[0] != self.n_states:
            raise ValueError(
                f"posterior length {post.shape[0]} != n_states {self.n_states}"
            )
        s = post.sum()
        if not np.isfinite(s) or s <= 0:
            raise ValueError(f"posterior has non-positive sum: {s}")
        post = post / s  # defensive renormalization

        # ── 1. State path: π_{t+k} = π_t · A^k, computed iteratively ──
        state_path = np.zeros((H + 1, self.n_states))
        state_path[0] = post
        for k in range(1, H + 1):
            state_path[k] = state_path[k - 1] @ self.A

        # ── 2. Aggregate by label ──
        label_path = self._aggregate_labels(state_path)

        # ── 3. Identify dominant current label ──
        current_label_probs = {lbl: arr[0] for lbl, arr in label_path.items()}
        current_label = max(current_label_probs, key=current_label_probs.get)
        current_label_prob = float(current_label_probs[current_label])

        # ── 4. Expected sojourn in dominant raw state ──
        dominant_raw = int(np.argmax(post))
        a_ii = float(self.A[dominant_raw, dominant_raw])
        expected_sojourn, sojourn_var = sojourn_moments(a_ii)

        # ── 5. Transition risk ──
        # P(exit current LABEL within risk_horizon) using the label path:
        #   risk = 1 - P(label_path = current at t+risk_horizon)
        # Note this is the *marginal* probability, not the path probability of
        # never visiting another label. For our purposes (early warning) the
        # marginal is the right metric: "where will I be in 5 bars".
        H_risk = min(self.risk_horizon, H)
        stay_prob = float(label_path[current_label][H_risk])
        transition_risk = max(0.0, 1.0 - stay_prob)

        # ── 6. Most-likely next label after exit ──
        next_label, next_label_prob = self._most_likely_next_label(
            label_path, current_label, H_risk
        )

        # ── 7. Stationary distribution by label ──
        stationary_by_label = self._aggregate_stationary()

        # ── 8. Bars-to-exit quantiles (10/25/50/75/90%) ──
        exit_quantiles = self._exit_quantiles(label_path[current_label])

        return RegimeForecast(
            horizon=H,
            current_label=current_label,
            current_label_prob=current_label_prob,
            state_path=state_path,
            label_path=label_path,
            expected_sojourn_bars=expected_sojourn,
            sojourn_variance=sojourn_var,
            transition_risk=transition_risk,
            next_label=next_label,
            next_label_prob=next_label_prob,
            stationary_by_label=stationary_by_label,
            exit_quantiles=exit_quantiles,
        )

    # ── Internals ───────────────────────────────────────────────

    def _aggregate_labels(self, state_path: np.ndarray) -> dict:
        """Sum raw-state probabilities into label-keyed time series."""
        out: dict = {}
        for raw_idx, label in self.state_labels.items():
            col = state_path[:, raw_idx]
            if label in out:
                out[label] = out[label] + col
            else:
                out[label] = col.copy()
        return out

    def _aggregate_stationary(self) -> dict:
        """Sum stationary probability mass by label."""
        out: dict = {}
        for raw_idx, label in self.state_labels.items():
            v = float(self._stationary_raw[raw_idx])
            out[label] = out.get(label, 0.0) + v
        return out

    def _most_likely_next_label(
        self, label_path: dict, current_label: str, k: int
    ) -> tuple:
        """
        Among labels that aren't `current_label`, pick the one with highest
        marginal probability at horizon `k`. Returns (label, normalized_prob).
        Probabilities are renormalized over non-current labels so the user
        sees a conditional distribution: "given that we exit, where do we go?"
        """
        candidates = {
            lbl: float(arr[k]) for lbl, arr in label_path.items() if lbl != current_label
        }
        if not candidates:
            return None, 0.0
        total = sum(candidates.values())
        if total <= 0:
            return None, 0.0
        next_label = max(candidates, key=candidates.get)
        return next_label, candidates[next_label] / total

    def _exit_quantiles(self, stay_path: np.ndarray) -> dict:
        """
        For each quantile q in {0.10, 0.25, 0.50, 0.75, 0.90}, return the
        smallest k such that P(in current label at t+k) <= 1 - q. That is,
        the smallest horizon at which cumulative exit probability >= q.
        Returns dict {q -> k or None if never reached within horizon}.
        """
        out: dict = {}
        H = stay_path.shape[0] - 1
        for q in (0.10, 0.25, 0.50, 0.75, 0.90):
            target = 1.0 - q
            hit = None
            for k in range(1, H + 1):
                if stay_path[k] <= target + 1e-12:
                    hit = k
                    break
            out[q] = hit
        return out


# ── Module-level numerics (testable in isolation) ──────────────


def stationary_distribution(A: np.ndarray, tol: float = 1e-10, max_iter: int = 10_000) -> np.ndarray:
    """
    Solve π A = π,  Σ π = 1 for the stationary distribution of an irreducible
    aperiodic Markov chain.

    Strategy
    --------
    1. Eigendecompose A.T; pick the eigenvector for eigenvalue closest to 1.
    2. If eigenvector has mixed signs (numerical issues / reducible chain),
       fall back to power iteration on a uniform start.

    Returns
    -------
    π : (n,) probability vector summing to 1, with all entries >= 0.
    """
    n = A.shape[0]
    # Eigendecomposition route
    try:
        eigvals, eigvecs = np.linalg.eig(A.T)
        idx = int(np.argmin(np.abs(eigvals - 1.0)))
        v = np.real(eigvecs[:, idx])
        # If eigenvector has mostly one sign, flip and normalize.
        if np.all(v <= 0):
            v = -v
        if np.all(v >= -1e-9):
            v = np.clip(v, 0.0, None)
            s = v.sum()
            if s > 0:
                return v / s
    except np.linalg.LinAlgError:
        pass

    # Power-iteration fallback
    pi = np.ones(n) / n
    for _ in range(max_iter):
        nxt = pi @ A
        if np.linalg.norm(nxt - pi, ord=1) < tol:
            return nxt / nxt.sum()
        pi = nxt
    return pi / pi.sum()


def sojourn_moments(a_ii: float) -> tuple:
    """
    Geometric distribution moments for time spent in a self-loop with
    self-transition probability a_ii.

    Mean    = 1 / (1 - a_ii)
    Var     = a_ii / (1 - a_ii)^2

    Edge cases
    ----------
    a_ii >= 1 - eps : absorbing state; return (np.inf, np.inf)
    a_ii <= 0       : single-bar regime; return (1.0, 0.0)
    """
    if not np.isfinite(a_ii):
        return float("inf"), float("inf")
    if a_ii >= 1.0 - 1e-12:
        return float("inf"), float("inf")
    if a_ii <= 0.0:
        return 1.0, 0.0
    mean = 1.0 / (1.0 - a_ii)
    var = a_ii / (1.0 - a_ii) ** 2
    return float(mean), float(var)
