# tests/test_hmm.py
# ============================================================
# Tests for HMM engine.
# Critical: verify look-ahead prevention in forward algorithm.
# ============================================================

import numpy as np
import pytest
from unittest.mock import MagicMock


def make_settings():
    return {
        "hmm": {
            "n_components_range": [2, 4],
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
        "features": {
            "realized_vol_window": 20,
            "normalization_window": 60,
            "ewma_halflife": 10,
        },
    }


def make_synthetic_observations(T=300, n_features=4, seed=42):
    """Synthetic (T, 4) observation matrix for testing."""
    rng = np.random.default_rng(seed)
    # Two regimes: low-vol (first half) and high-vol (second half)
    split = T // 2
    obs = np.zeros((T, n_features))
    obs[:split] = rng.normal(
        loc=[0.001, 0.01, 15.0, 3.5],
        scale=[0.01, 0.005, 2.0, 0.3],
        size=(split, n_features)
    )
    obs[split:] = rng.normal(
        loc=[-0.002, 0.05, 35.0, 6.0],
        scale=[0.02, 0.02, 5.0, 0.8],
        size=(T - split, n_features)
    )
    return obs


class TestHMMEngine:
    def test_fit_returns_valid_state_count(self):
        from core.hmm_engine import HMMEngine
        settings = make_settings()
        engine = HMMEngine(settings)
        obs = make_synthetic_observations()
        n = engine.fit(obs)
        assert 2 <= n <= 4, f"BIC selected {n} — outside expected [2,4] range"

    def test_state_labels_assigned(self):
        from core.hmm_engine import HMMEngine, RegimeLabel
        settings = make_settings()
        engine = HMMEngine(settings)
        obs = make_synthetic_observations()
        engine.fit(obs)
        assert len(engine.state_labels) == engine.n_states
        labels = list(engine.state_labels.values())
        # LOW_VOL always assigned (lowest variance state)
        assert RegimeLabel.LOW_VOL in labels, "LowVol label must always be assigned"
        # With n>=2, HIGH_VOL must also be assigned (endpoints of scale)
        assert RegimeLabel.HIGH_VOL in labels, (
            f"HighVol label must be assigned for n={engine.n_states} states. "
            f"Labels: {labels}. Check label_states_by_variance() n=2 case."
        )

    def test_needs_retrain_after_n_steps(self):
        from core.hmm_engine import HMMEngine
        settings = make_settings()
        engine = HMMEngine(settings)
        obs = make_synthetic_observations()
        engine.fit(obs)
        assert not engine.needs_retrain()
        for _ in range(20):
            engine.step(obs[-1])
        assert engine.needs_retrain()


class TestForwardAlgorithm:
    """
    Verify no look-ahead bias in forward α-recursion.
    Key property: regime at time t must be identical whether
    computed on obs[:t+1] or obs[:t+1+k] for any k > 0.
    """

    def test_forward_uses_only_past_data(self):
        """
        Compute regimes on prefix obs[:T] and on full sequence obs[:T+k].
        Regimes at positions 0..T-1 must be identical.
        Ref: Hamilton (1989) — forward recursion is causal.
        """
        from core.hmm_engine import HMMEngine, forward_algorithm_log
        settings = make_settings()
        engine = HMMEngine(settings)
        obs = make_synthetic_observations(T=100)
        engine.fit(obs[:80])  # train on first 80

        T_prefix = 15
        log_alphas_short, conf_short = forward_algorithm_log(engine.model, obs[:T_prefix])
        log_alphas_long, conf_long = forward_algorithm_log(engine.model, obs[:T_prefix + 10])

        # Regimes at positions 0..T_prefix-1 must match
        regimes_short = np.argmax(log_alphas_short, axis=1)
        regimes_long = np.argmax(log_alphas_long[:T_prefix], axis=1)
        np.testing.assert_array_equal(
            regimes_short, regimes_long,
            err_msg="Forward algorithm is not causal — regime changed when appending future data!"
        )

    def test_forward_output_shape(self):
        from core.hmm_engine import HMMEngine, forward_algorithm_log
        settings = make_settings()
        engine = HMMEngine(settings)
        obs = make_synthetic_observations(T=100)
        engine.fit(obs)
        log_alphas, confidences = forward_algorithm_log(engine.model, obs)
        assert log_alphas.shape == (100, engine.n_states)
        assert confidences.shape == (100,)
        assert (confidences >= 0).all() and (confidences <= 1).all()

    def test_confidence_sums_to_one(self):
        from core.hmm_engine import HMMEngine, forward_algorithm_log
        import scipy.special
        settings = make_settings()
        engine = HMMEngine(settings)
        obs = make_synthetic_observations(T=50)
        engine.fit(obs)
        log_alphas, _ = forward_algorithm_log(engine.model, obs)
        # Posterior probabilities (softmax of log_alphas) must sum to 1 per row
        log_norm = scipy.special.logsumexp(log_alphas, axis=1, keepdims=True)
        posteriors = np.exp(log_alphas - log_norm)
        row_sums = posteriors.sum(axis=1)
        np.testing.assert_allclose(row_sums, np.ones(50), atol=1e-6)


class TestRegimeLabeling:
    """States labeled by realized variance, not mean return."""

    def test_low_vol_state_has_lowest_variance(self):
        """
        LowVol state must correspond to HMM state with lowest mean realized variance.
        Ref: Turner, Startz & Nelson (1989, JFE).
        """
        from core.hmm_engine import HMMEngine, RegimeLabel
        settings = make_settings()
        engine = HMMEngine(settings)
        obs = make_synthetic_observations()
        engine.fit(obs)

        # Find state labeled LowVol
        low_vol_state = [k for k, v in engine.state_labels.items() if v == RegimeLabel.LOW_VOL]
        high_vol_state = [k for k, v in engine.state_labels.items() if v == RegimeLabel.HIGH_VOL]

        if not low_vol_state or not high_vol_state:
            pytest.skip("BIC selected fewer than 2 distinguishable states")

        var_idx = engine.variance_feature_idx
        low_mean_var = engine.model.means_[low_vol_state[0]][var_idx]
        high_mean_var = engine.model.means_[high_vol_state[0]][var_idx]

        assert low_mean_var < high_mean_var, (
            f"LowVol state has higher realized variance ({low_mean_var:.4f}) than "
            f"HighVol state ({high_mean_var:.4f}) — labeling is wrong"
        )


class TestStabilityFilters:
    def test_persistence_delays_switch(self):
        """3-bar persistence: single-bar regime change must not immediately commit."""
        from core.hmm_engine import HMMEngine, RegimeLabel
        settings = make_settings()
        engine = HMMEngine(settings)
        obs = make_synthetic_observations()
        engine.fit(obs)

        # Force a known stable low-vol regime
        engine._confirmed_regime = RegimeLabel.LOW_VOL
        engine._pending_regime = None
        engine._pending_count = 0

        # Single bar of high-vol signal — should NOT switch yet
        result = engine._apply_stability_filters(RegimeLabel.HIGH_VOL, 0.90)
        assert result == RegimeLabel.LOW_VOL, (
            "Single-bar regime change committed immediately — persistence filter not working"
        )

    def test_persistence_commits_after_3_bars(self):
        """After 3 consecutive bars of same new regime → commit switch."""
        from core.hmm_engine import HMMEngine, RegimeLabel
        settings = make_settings()
        engine = HMMEngine(settings)
        obs = make_synthetic_observations()
        engine.fit(obs)

        engine._confirmed_regime = RegimeLabel.LOW_VOL
        engine._pending_regime = None
        engine._pending_count = 0

        # 3 consecutive bars of HIGH_VOL
        for _ in range(3):
            result = engine._apply_stability_filters(RegimeLabel.HIGH_VOL, 0.90)

        assert result == RegimeLabel.HIGH_VOL, "Regime should have committed after 3 bars"

    def test_confidence_floor_triggers_uncertainty(self):
        """Low confidence → UNCERTAINTY regardless of predicted regime."""
        from core.hmm_engine import HMMEngine, RegimeLabel
        settings = make_settings()
        engine = HMMEngine(settings)
        obs = make_synthetic_observations()
        engine.fit(obs)
        engine._confirmed_regime = RegimeLabel.LOW_VOL

        result = engine._apply_stability_filters(RegimeLabel.LOW_VOL, 0.20)  # below 0.40 floor
        assert result == RegimeLabel.UNCERTAINTY
