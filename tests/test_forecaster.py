# tests/test_forecaster.py
# ============================================================
# Tests for RegimeForecaster.
#
# Strategy: use *analytical* ground truth wherever possible.
# All forecaster math is closed-form, so we don't need fixtures
# or random seeds — every test compares to a hand-computable value.
# ============================================================

import numpy as np
import pytest

from core.hmm_engine import RegimeLabel
from core.regime_forecaster import (
    RegimeForecaster,
    sojourn_moments,
    stationary_distribution,
)


# ── Helpers ────────────────────────────────────────────────


def make_simple_3state():
    """
    3-state chain with a clear LowVol/MidVol/HighVol structure.
    Persistent diagonal (sticky), small off-diagonal.
    """
    # Rows must sum to 1.
    A = np.array([
        [0.90, 0.08, 0.02],   # LowVol — tends to stay or drift to MidVol
        [0.10, 0.80, 0.10],   # MidVol — symmetric
        [0.05, 0.15, 0.80],   # HighVol — usually decays via MidVol
    ])
    labels = {
        0: RegimeLabel.LOW_VOL,
        1: RegimeLabel.MID_VOL,
        2: RegimeLabel.HIGH_VOL,
    }
    return A, labels


# ── Module-level numerics ─────────────────────────────────


class TestSojournMoments:
    def test_geometric_mean_formula(self):
        # E[T] = 1/(1-p) for self-loop probability p.
        for p in (0.1, 0.5, 0.8, 0.95):
            mean, var = sojourn_moments(p)
            assert mean == pytest.approx(1.0 / (1.0 - p))
            assert var == pytest.approx(p / (1.0 - p) ** 2)

    def test_absorbing_state(self):
        mean, var = sojourn_moments(1.0)
        assert mean == float("inf")
        assert var == float("inf")

    def test_zero_self_loop(self):
        # No self-transitions ⇒ exits in 1 bar deterministically.
        mean, var = sojourn_moments(0.0)
        assert mean == 1.0
        assert var == 0.0


class TestStationaryDistribution:
    def test_uniform_on_doubly_stochastic(self):
        # Doubly-stochastic chain → uniform stationary dist.
        A = np.array([
            [0.5, 0.3, 0.2],
            [0.2, 0.5, 0.3],
            [0.3, 0.2, 0.5],
        ])
        pi = stationary_distribution(A)
        np.testing.assert_allclose(pi, np.ones(3) / 3, atol=1e-8)

    def test_sums_to_one(self):
        A, _ = make_simple_3state()
        pi = stationary_distribution(A)
        assert pi.sum() == pytest.approx(1.0, abs=1e-10)
        assert (pi >= -1e-12).all()

    def test_invariance_under_transition(self):
        # π A = π by definition.
        A, _ = make_simple_3state()
        pi = stationary_distribution(A)
        np.testing.assert_allclose(pi @ A, pi, atol=1e-8)


# ── Forecaster API ─────────────────────────────────────────


class TestRegimeForecasterValidation:
    def test_rejects_non_square_transmat(self):
        with pytest.raises(ValueError, match="square"):
            RegimeForecaster(np.zeros((2, 3)), {0: "x", 1: "y"})

    def test_rejects_unnormalized_transmat(self):
        A = np.array([[0.5, 0.3], [0.4, 0.4]])
        with pytest.raises(ValueError, match="rows must sum to 1"):
            RegimeForecaster(A, {0: "x", 1: "y"})

    def test_rejects_bad_horizon(self):
        A, labels = make_simple_3state()
        with pytest.raises(ValueError, match="horizon must be >= 1"):
            RegimeForecaster(A, labels, horizon=0)

    def test_rejects_risk_horizon_above_horizon(self):
        A, labels = make_simple_3state()
        with pytest.raises(ValueError, match="risk_horizon"):
            RegimeForecaster(A, labels, horizon=5, risk_horizon=10)


class TestForecastShape:
    def test_state_path_dimensions(self):
        A, labels = make_simple_3state()
        fc = RegimeForecaster(A, labels, horizon=10, risk_horizon=3)
        post = np.array([1.0, 0.0, 0.0])
        result = fc.forecast(post)
        assert result.state_path.shape == (11, 3)  # horizon+1 rows
        # Each row sums to 1 (probability conservation).
        np.testing.assert_allclose(result.state_path.sum(axis=1), 1.0, atol=1e-10)

    def test_label_path_keys(self):
        A, labels = make_simple_3state()
        fc = RegimeForecaster(A, labels, horizon=5)
        result = fc.forecast(np.array([0.6, 0.3, 0.1]))
        assert set(result.label_path.keys()) == {
            RegimeLabel.LOW_VOL, RegimeLabel.MID_VOL, RegimeLabel.HIGH_VOL
        }
        for arr in result.label_path.values():
            assert arr.shape == (6,)


class TestForecastMath:
    """Compare forecaster output to closed-form HMM math."""

    def test_state_path_equals_matrix_powers(self):
        A, labels = make_simple_3state()
        fc = RegimeForecaster(A, labels, horizon=4, risk_horizon=2)
        post = np.array([0.5, 0.3, 0.2])
        result = fc.forecast(post)

        # Reference: explicit power iteration.
        Ak = np.eye(3)
        for k in range(5):
            np.testing.assert_allclose(result.state_path[k], post @ Ak, atol=1e-10)
            Ak = Ak @ A

    def test_long_horizon_converges_to_stationary(self):
        """
        After many steps, π_{t+k} → stationary π* regardless of start.
        This is the ergodic theorem; our forecaster output must reflect it.
        """
        A, labels = make_simple_3state()
        fc = RegimeForecaster(A, labels, horizon=200, risk_horizon=2)
        post = np.array([1.0, 0.0, 0.0])
        result = fc.forecast(post)
        pi_star = stationary_distribution(A)
        np.testing.assert_allclose(result.state_path[-1], pi_star, atol=1e-6)

    def test_expected_sojourn_matches_formula(self):
        """E[duration | S=i] = 1/(1 - A_ii) for the dominant raw state."""
        A, labels = make_simple_3state()
        fc = RegimeForecaster(A, labels, horizon=10)
        # Posterior concentrated on state 0 (a_00 = 0.90 ⇒ E[T] = 10).
        post = np.array([0.99, 0.005, 0.005])
        result = fc.forecast(post)
        assert result.expected_sojourn_bars == pytest.approx(10.0)
        assert result.sojourn_variance == pytest.approx(0.90 / 0.01)

    def test_transition_risk_increases_with_horizon(self):
        """
        Transition risk must be monotone non-decreasing in horizon —
        more time ⇒ more chance of exiting (or equal, never less).
        """
        A, labels = make_simple_3state()
        post = np.array([0.99, 0.005, 0.005])
        prev = -1.0
        for h in (1, 2, 5, 10, 20):
            fc = RegimeForecaster(A, labels, horizon=h, risk_horizon=h)
            result = fc.forecast(post)
            assert result.transition_risk >= prev - 1e-12
            prev = result.transition_risk

    def test_transition_risk_zero_for_absorbing_state(self):
        """If current state never exits (a_ii = 1), transition risk = 0."""
        A = np.array([
            [1.0, 0.0, 0.0],
            [0.1, 0.8, 0.1],
            [0.0, 0.1, 0.9],
        ])
        labels = {0: RegimeLabel.LOW_VOL, 1: RegimeLabel.MID_VOL, 2: RegimeLabel.HIGH_VOL}
        fc = RegimeForecaster(A, labels, horizon=50, risk_horizon=20)
        post = np.array([1.0, 0.0, 0.0])
        result = fc.forecast(post)
        assert result.transition_risk == pytest.approx(0.0, abs=1e-10)
        assert result.expected_sojourn_bars == float("inf")

    def test_next_label_picks_highest_off_diagonal(self):
        """
        From LowVol (state 0), short-horizon exit goes preferentially to
        MidVol (a_01=0.08) over HighVol (a_02=0.02). next_label == MidVol.
        """
        A, labels = make_simple_3state()
        fc = RegimeForecaster(A, labels, horizon=5, risk_horizon=1)
        post = np.array([1.0, 0.0, 0.0])
        result = fc.forecast(post)
        assert result.current_label == RegimeLabel.LOW_VOL
        assert result.next_label == RegimeLabel.MID_VOL
        # Conditional probability of MidVol given exit at h=1:
        # = 0.08 / (0.08 + 0.02) = 0.8
        assert result.next_label_prob == pytest.approx(0.8, abs=1e-10)

    def test_label_aggregation_with_duplicate_labels(self):
        """
        When BIC selects more raw states than label slots (e.g. n=5 with
        VeryHighVol falling back to HighVol), label aggregation must sum
        the raw probabilities, not pick one.
        """
        # 4 raw states but two are both labeled HighVol (n>4 fallback case).
        A = np.array([
            [0.85, 0.10, 0.03, 0.02],
            [0.10, 0.80, 0.05, 0.05],
            [0.05, 0.10, 0.70, 0.15],
            [0.02, 0.08, 0.20, 0.70],
        ])
        labels = {
            0: RegimeLabel.LOW_VOL,
            1: RegimeLabel.MID_VOL,
            2: RegimeLabel.HIGH_VOL,
            3: RegimeLabel.HIGH_VOL,   # duplicate
        }
        fc = RegimeForecaster(A, labels, horizon=3, risk_horizon=1)
        post = np.array([0.0, 0.0, 0.5, 0.5])
        result = fc.forecast(post)
        # At t=0, P(HighVol) must be 1.0 (sum of states 2 and 3).
        assert result.label_path[RegimeLabel.HIGH_VOL][0] == pytest.approx(1.0)
        # Stationary mass for HighVol is the sum of stationary states 2+3.
        pi = stationary_distribution(A)
        assert result.stationary_by_label[RegimeLabel.HIGH_VOL] == pytest.approx(
            pi[2] + pi[3], abs=1e-10
        )

    def test_exit_quantiles_monotone(self):
        """Quantiles must be non-decreasing in q (later quantile, later exit)."""
        A, labels = make_simple_3state()
        fc = RegimeForecaster(A, labels, horizon=200, risk_horizon=5)
        post = np.array([1.0, 0.0, 0.0])
        result = fc.forecast(post)
        ks = [result.exit_quantiles[q] for q in (0.10, 0.25, 0.50, 0.75, 0.90)]
        ks = [k for k in ks if k is not None]
        for a, b in zip(ks, ks[1:]):
            assert a <= b


class TestForecastFromLogAlpha:
    def test_log_alpha_path_matches_posterior_path(self):
        """forecast_from_log_alpha must produce the same result as forecast()
        applied to the corresponding posterior."""
        A, labels = make_simple_3state()
        fc = RegimeForecaster(A, labels, horizon=4, risk_horizon=2)
        # Build a log α that softmaxes to a known posterior.
        post = np.array([0.6, 0.25, 0.15])
        log_alpha = np.log(post)  # constant offset is absorbed by softmax
        r1 = fc.forecast(post)
        r2 = fc.forecast_from_log_alpha(log_alpha)
        np.testing.assert_allclose(r1.state_path, r2.state_path, atol=1e-10)
        assert r1.transition_risk == pytest.approx(r2.transition_risk)
        assert r1.current_label == r2.current_label


# ── Integration with HMMEngine ────────────────────────────


def make_engine_settings():
    return {
        "hmm": {
            "n_components_range": [2, 3],
            "covariance_type": "diag",
            "n_iter": 50,
            "tol": 1e-4,
            "n_init": 3,
            "training_window": 200,
            "retrain_every": 20,
            "random_state": 42,
        },
        "stability": {
            "persistence_bars": 3,
            "flicker_window": 20,
            "flicker_threshold": 4,
            "confidence_floor": 0.40,
        },
    }


def make_synthetic_obs(T=300, seed=42):
    rng = np.random.default_rng(seed)
    split = T // 2
    obs = np.zeros((T, 4))
    obs[:split] = rng.normal(
        loc=[0.001, 0.01, 15.0, 3.5],
        scale=[0.01, 0.005, 2.0, 0.3],
        size=(split, 4),
    )
    obs[split:] = rng.normal(
        loc=[-0.002, 0.05, 35.0, 6.0],
        scale=[0.02, 0.02, 5.0, 0.8],
        size=(T - split, 4),
    )
    return obs


class TestEngineForecastIntegration:
    def test_engine_forecast_after_step(self):
        from core.hmm_engine import HMMEngine
        engine = HMMEngine(make_engine_settings())
        obs = make_synthetic_obs(T=250)
        engine.fit(obs[:200])
        engine.step(obs[200])
        result = engine.forecast(horizon=10, risk_horizon=3)
        # Sanity: structural invariants.
        assert result.state_path.shape == (11, engine.n_states)
        np.testing.assert_allclose(result.state_path.sum(axis=1), 1.0, atol=1e-10)
        assert 0.0 <= result.transition_risk <= 1.0
        # Sum of stationary label probabilities must be 1.
        assert sum(result.stationary_by_label.values()) == pytest.approx(1.0, abs=1e-8)

    def test_engine_forecast_before_step_raises(self):
        from core.hmm_engine import HMMEngine
        engine = HMMEngine(make_engine_settings())
        obs = make_synthetic_obs(T=250)
        engine.fit(obs[:200])
        with pytest.raises(RuntimeError, match="No posterior"):
            engine.forecast(horizon=10)

    def test_engine_forecast_consistent_with_predict_regime(self):
        """After predict_regime() the forecaster's t=0 marginal label probs
        must match the softmax of the final log α."""
        from core.hmm_engine import HMMEngine
        from scipy.special import softmax
        engine = HMMEngine(make_engine_settings())
        obs = make_synthetic_obs(T=250)
        engine.fit(obs[:200])
        engine.predict_regime(obs[200:220])

        result = engine.forecast(horizon=5, risk_horizon=2)
        post = softmax(engine._last_log_alpha)
        # Aggregate posterior by label and compare to t=0 of label_path.
        expected = {}
        for raw_idx, lbl in engine.state_labels.items():
            expected[lbl] = expected.get(lbl, 0.0) + post[raw_idx]
        for lbl, p in expected.items():
            assert result.label_path[lbl][0] == pytest.approx(p, abs=1e-10)
