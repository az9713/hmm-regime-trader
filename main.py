# main.py
# ============================================================
# Entry point for regime-trader.
#
# Modes:
#   --backtest   Walk-forward backtest on historical data (Phase 6)
#   --paper      Live paper trading via Alpaca (safe default)
#   --live       Live trading (explicit flag + "CONFIRM" prompt required)
#
# Per-bar loop (paper/live):
#   1. Fetch latest bar (Alpaca WebSocket or polling)
#   2. Update features (log returns, realized var, VIX, HY OAS)
#   3. HMM forward step → regime + confidence (forward α-recursion)
#   4. Apply stability filters (persistence, flicker, confidence floor)
#   5. Generate signal via regime strategy + Moreira-Muir allocation
#   6. Risk manager gates (circuit breakers, correlation, position limits)
#   7. Execute bracket order via Alpaca
#   8. Log decision + update dashboard
#   9. Reconcile positions with Alpaca (Alpaca is source of truth)
# ============================================================

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import yaml
from dotenv import load_dotenv

from monitoring.logger import setup_logging, TradeLogger
from monitoring.alerts import AlertManager
from monitoring.dashboard import Dashboard


def load_settings(config_path: str = "config/settings.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_credentials(creds_path: str = "config/credentials.yaml") -> dict:
    if not Path(creds_path).exists():
        return {}
    with open(creds_path) as f:
        return yaml.safe_load(f) or {}


def parse_args():
    parser = argparse.ArgumentParser(description="Regime Trader — HMM-based volatility regime trading bot")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--backtest", action="store_true", help="Walk-forward backtest on historical data")
    group.add_argument("--paper", action="store_true", help="Paper trading via Alpaca")
    group.add_argument("--live", action="store_true", help="LIVE trading — real money. Requires CONFIRM.")
    group.add_argument("--test-connection", action="store_true", help="Test Alpaca connectivity and exit")
    group.add_argument("--dry-run", action="store_true", help="Smoke test: warmup + HMM fit + regime step, then exit")
    parser.add_argument("--config", default="config/settings.yaml", help="Path to settings.yaml")
    parser.add_argument("--start", default=None, help="Backtest start date (YYYY-MM-DD)")
    return parser.parse_args()


def _bridge_alpaca_credentials(credentials: dict):
    """
    Load Alpaca keys from credentials.yaml into env vars if env vars are not already set.
    Env vars take precedence (useful for CI / Docker environments).
    """
    if not os.getenv("ALPACA_API_KEY"):
        alpaca_creds = credentials.get("alpaca", {})
        api_key = alpaca_creds.get("api_key")
        secret_key = alpaca_creds.get("secret_key")
        if api_key and secret_key:
            os.environ["ALPACA_API_KEY"] = api_key
            os.environ["ALPACA_SECRET_KEY"] = secret_key
            logging.getLogger("main").info("Alpaca credentials loaded from credentials.yaml")


def run_backtest(settings: dict, start_date: str = None):
    """Run walk-forward backtest with current settings."""
    from data.market_data import DataManager
    from backtest.backtester import WalkForwardBacktester
    from backtest.stress_test import StressTester
    from backtest.performance import hansen_spa_test
    import pandas as pd

    log = TradeLogger()
    logger = logging.getLogger("main")

    start = start_date or settings["data"]["start_date"]
    symbols = settings["data"]["symbols"]

    logger.info(f"Starting backtest: {symbols}, {start} -> today")

    # Fetch data
    dm = DataManager(settings, mode="backtest")
    vix = dm.get_vix(start)
    hy_oas = dm.get_hy_oas(start)
    gold = dm.get_gold(start)
    term_spread = dm.get_term_spread(start)
    vix3m = dm.get_vix3m(start)
    prices = {sym: dm.get_bars(sym, start) for sym in symbols}

    # Walk-forward
    backtester = WalkForwardBacktester(settings)
    result = backtester.run(prices, vix, hy_oas, primary_symbol="SPY",
                            gold=gold, term_spread=term_spread, vix3m=vix3m)

    # Log all windows
    for window in result.windows:
        log.log_backtest_window(window)

    # SPA test
    if result.aggregate_metrics and result.benchmark_returns:
        strategy_oos = _concat_oos_returns(result)
        bench_list = [v for v in result.benchmark_returns.values()]
        result.spa_result = hansen_spa_test(strategy_oos, bench_list)

    log.log_backtest_summary(result)

    # Print summary
    summary = result.summary()
    print("\n=== Walk-Forward Backtest Summary ===")
    print(summary.to_string(index=False))
    print(f"\nAggregate OOS metrics: {result.aggregate_metrics}")
    print(f"SPA test: {result.spa_result}")

    # Stress tests
    logger.info("Running stress period tests...")
    stress = StressTester(settings)
    stress_results = stress.run_all(prices, vix, hy_oas, gold=gold, term_spread=term_spread, vix3m=vix3m)
    for name, sr in stress_results.items():
        status = "PASS" if sr.detection_pass else "FAIL"
        print(f"\nStress [{name}]: {status} | HighVol={sr.high_vol_pct:.0%} | AvgAlloc={sr.avg_allocation:.2f}")


def run_paper(settings: dict, credentials: dict = None):
    """Live paper trading loop."""
    from broker.alpaca_client import AlpacaClient
    from broker.order_executor import OrderExecutor
    from broker.position_tracker import PositionTracker
    from data.market_data import DataManager
    from data.feature_engineering import FeatureEngineer, compute_ewma_realized_vol
    from core.hmm_engine import HMMEngine
    from core.signal_generator import SignalGenerator
    from core.risk_manager import RiskManager
    import pandas as pd

    logger = logging.getLogger("main")
    log = TradeLogger()
    alerts = AlertManager(settings, credentials or {})

    # Dashboard state tracking
    _equity_history = []
    _regime_history = []
    _timestamps = []
    _prev_regime = None
    _session_start = datetime.now().strftime("%H:%M ET")
    _session_open_value = None

    logger.info("Starting paper trading session...")

    # Credentials — env vars take precedence, credentials.yaml as fallback
    _bridge_alpaca_credentials(credentials or {})

    # Broker
    client = AlpacaClient(settings)
    client.connect()
    executor = OrderExecutor(client, settings)
    tracker = PositionTracker(client)

    # Data + features — share the same AlpacaClient (one connection)
    dm = DataManager(settings, mode="live", alpaca_client=client)
    fe = FeatureEngineer(settings)

    # Risk + signals
    risk_mgr = RiskManager(settings)
    signal_gen = SignalGenerator(settings, risk_mgr)

    # Initial HMM training on historical warmup data
    symbols = settings["data"]["symbols"]
    warmup_start = settings["data"]["start_date"]
    vix = dm.get_vix(warmup_start)
    hy_oas = dm.get_hy_oas(warmup_start)
    gold = dm.get_gold(warmup_start)
    term_spread = dm.get_term_spread(warmup_start)
    vix3m = dm.get_vix3m(warmup_start)
    spy_prices = dm.get_bars("SPY", warmup_start)

    engine = HMMEngine(settings)
    features = fe.compute(spy_prices, vix, hy_oas, gold=gold, term_spread=term_spread, vix3m=vix3m)
    engine.fit(features.values)
    logger.info(f"HMM trained on warmup data: n_states={engine.n_states}")

    # --- Live bar handler ---
    async def on_bar(bar):
        symbol = bar.symbol
        try:
            # Fetch latest bars for feature computation
            recent_prices = client.get_bars(symbol, start=warmup_start)
            recent_vix = dm.get_vix(warmup_start)
            recent_oas = dm.get_hy_oas(warmup_start)
            recent_gold = dm.get_gold(warmup_start)
            recent_ts = dm.get_term_spread(warmup_start)
            recent_vix3m = dm.get_vix3m(warmup_start)

            # Features
            feat = fe.compute(recent_prices, recent_vix, recent_oas,
                              gold=recent_gold, term_spread=recent_ts, vix3m=recent_vix3m)
            if len(feat) < 10:
                return

            # Retrain if needed (uses already-computed feat which includes all 6 features)
            if engine.needs_retrain():
                engine.fit(feat.values[-engine.training_window:])
                engine.reset_retrain_counter()
                alerts.hmm_retrain(engine.n_states, 0.0, str(feat.index[-1].date()))

            # Regime (forward step)
            regime_state = engine.step(feat.values[-1])
            log.log_regime(symbol, regime_state)

            # EWMA vol for Moreira-Muir
            from data.feature_engineering import compute_ewma_realized_vol
            import numpy as np
            ret_series = feat["log_return"]
            ewma_vol = float(compute_ewma_realized_vol(ret_series).iloc[-1])

            # Risk / signal
            account = client.get_account()
            positions = tracker.reconcile()
            daily_pnl_pct = tracker.daily_pnl_pct(account["portfolio_value"])

            # Circuit breaker check
            from core.risk_manager import check_circuit_breakers, CircuitBreakerState
            cb = check_circuit_breakers(abs(min(daily_pnl_pct, 0)), risk_mgr._circuit_state, settings)
            if cb.state == CircuitBreakerState.STOPPED:
                alerts.circuit_breaker(cb.state, cb.message, account["portfolio_value"])
                executor.liquidate_all()
                return
            if cb.state in (CircuitBreakerState.HALTED, CircuitBreakerState.REVIEW_REQUIRED):
                alerts.circuit_breaker(cb.state, cb.message, account["portfolio_value"])
                return

            signal = signal_gen.generate(
                symbol=symbol,
                regime_state=regime_state,
                price_data=recent_prices.tail(50),
                portfolio_value=account["portfolio_value"],
                daily_pnl_pct=daily_pnl_pct,
                current_positions=positions,
                ewma_vol=ewma_vol,
            )

            from core.signal_generator import Signal as TradeSignal, FlatSignal
            if isinstance(signal, TradeSignal) and not tracker.has_position(symbol):
                tracker.register_signal(symbol, signal.stop_price, signal.target_price, signal.regime)
                order_result = executor.submit(signal)
                log.log_signal(signal, regime_state)
                if order_result.submitted:
                    alerts.order_filled(
                        symbol=symbol,
                        side="buy",
                        qty=order_result.qty,
                        fill_price=order_result.entry,
                        stop=order_result.stop,
                        target=order_result.target,
                        regime=regime_state.label,
                    )
                else:
                    alerts.broker_error(symbol, order_result.message)

            # ── Regime flip detection ──────────────────────────────
            nonlocal _prev_regime
            if _prev_regime is not None and regime_state.label != _prev_regime:
                alerts.regime_flip(
                    symbol=symbol,
                    prev_regime=_prev_regime,
                    new_regime=regime_state.label,
                    confidence=regime_state.confidence,
                    allocation=signal.allocation if hasattr(signal, "allocation") else 0.0,
                )
            _prev_regime = regime_state.label

            # ── Dashboard state update ─────────────────────────────
            nonlocal _session_open_value
            pv = account["portfolio_value"]
            if _session_open_value is None:
                _session_open_value = pv
            _equity_history.append(pv)
            _regime_history.append(regime_state.label)
            _timestamps.append(datetime.now().isoformat())

            open_pos = [
                {"symbol": p, "regime": tracker._local_metadata.get(p, {}).get("regime", "—")}
                for p in positions.keys()
            ]
            log.update_dashboard_state(
                regime=regime_state.label,
                confidence=regime_state.confidence,
                allocation=signal.allocation if hasattr(signal, "allocation") else 0.0,
                portfolio_value=pv,
                daily_pnl=pv - _session_open_value,
                daily_pnl_pct=daily_pnl_pct,
                positions=open_pos,
                equity_history=_equity_history,
                regime_history=_regime_history,
                timestamps=_timestamps,
                session_start=_session_start,
            )

        except Exception as e:
            logger.exception(f"Bar handler error for {symbol}: {e}")
            alerts.broker_error(symbol, str(e))

    # Subscribe and run
    client.subscribe_bars(symbols, on_bar)
    logger.info(f"Subscribed to live bars: {symbols}")
    client.run_stream()


def run_dry_run(settings: dict, credentials: dict = None):
    """
    Smoke test — exercises every component of the paper trading warmup path
    without blocking on the WebSocket stream. Safe to run any time of day.

    Checks:
      1. Alpaca credentials + connection
      2. Historical bar fetch (SPY via Alpaca REST)
      3. FRED macro data fetch (VIX, HY OAS, term_spread, vix3m)
      4. Gold via yfinance
      5. Feature engineering (shape, date range, no NaN)
      6. HMM fit (BIC state selection, convergence)
      7. Regime step (forward α-recursion on latest bar)
      8. EWMA realized vol computation
    """
    from broker.alpaca_client import AlpacaClient
    from data.market_data import DataManager
    from data.feature_engineering import FeatureEngineer, compute_ewma_realized_vol
    from core.hmm_engine import HMMEngine
    import numpy as np

    logger = logging.getLogger("main")
    results = {}
    passed = 0
    failed = 0

    def check(name, fn):
        nonlocal passed, failed
        try:
            val = fn()
            results[name] = ("OK", val)
            passed += 1
            return val
        except Exception as e:
            results[name] = ("FAIL", str(e))
            failed += 1
            logger.error(f"[DRY RUN] {name} FAILED: {e}")
            return None

    print("\n=== Paper Trading Dry Run (Smoke Test) ===")

    # 1. Credentials + connection
    _bridge_alpaca_credentials(credentials or {})
    settings["broker"]["mode"] = "paper"
    client = check("Alpaca connect", lambda: (AlpacaClient(settings).__class__,
                                               AlpacaClient(settings).connect() or True))

    # Re-create properly (lambda above doesn't hold state)
    _bridge_alpaca_credentials(credentials or {})
    alpaca = AlpacaClient(settings)
    try:
        alpaca.connect()
    except Exception as e:
        print(f"  FATAL: Alpaca connection failed: {e}")
        sys.exit(1)

    # 2. Account
    account = check("Alpaca account", alpaca.get_account)

    # 3. Data
    warmup_start = settings["data"]["start_date"]
    dm = DataManager(settings, mode="live", alpaca_client=alpaca)

    spy_prices = check("SPY bars (Alpaca)", lambda: dm.get_bars("SPY", warmup_start))
    vix        = check("VIX (FRED)",        lambda: dm.get_vix(warmup_start))
    hy_oas     = check("HY OAS (FRED)",     lambda: dm.get_hy_oas(warmup_start))
    gold       = check("Gold (yfinance)",   lambda: dm.get_gold(warmup_start))
    term_sp    = check("Term spread (FRED)",lambda: dm.get_term_spread(warmup_start))
    vix3m      = check("VIX3M (FRED)",      lambda: dm.get_vix3m(warmup_start))

    # 4. Features
    fe = FeatureEngineer(settings)
    features = None
    if all(x is not None for x in [spy_prices, vix, hy_oas, gold, term_sp, vix3m]):
        features = check(
            "Feature engineering",
            lambda: fe.compute(spy_prices, vix, hy_oas, gold=gold, term_spread=term_sp, vix3m=vix3m)
        )

    # 5. HMM fit
    engine = None
    if features is not None:
        engine = HMMEngine(settings)
        n_states = check("HMM fit (BIC)", lambda: engine.fit(features.values))

    # 6. Regime step
    regime_state = None
    if engine is not None and features is not None:
        regime_state = check("Regime step", lambda: engine.step(features.values[-1]))

    # 7. EWMA vol
    ewma_vol = None
    if features is not None and "log_return" in features.columns:
        ewma_vol = check(
            "EWMA vol",
            lambda: float(compute_ewma_realized_vol(features["log_return"]).iloc[-1])
        )

    # ── Print summary ──────────────────────────────────────────
    print(f"\n  {'Component':<25} {'Status':<6} {'Detail'}")
    print(f"  {'-'*60}")
    for name, (status, val) in results.items():
        if status == "OK":
            if name == "Alpaca account" and val:
                detail = f"${val['portfolio_value']:,.0f} portfolio"
            elif name == "SPY bars (Alpaca)" and val is not None:
                detail = f"{len(val)} bars  {val.index[0].date()} to {val.index[-1].date()}"
            elif name == "Feature engineering" and val is not None:
                detail = f"shape={val.shape}  NaN={val.isna().sum().sum()}"
            elif name == "HMM fit (BIC)" and val is not None:
                detail = f"n_states={val}"
            elif name == "Regime step" and val is not None:
                detail = f"{val.label}  conf={val.confidence:.0%}"
            elif name == "EWMA vol" and val is not None:
                detail = f"annualized={val:.3f}"
            elif hasattr(val, '__len__'):
                detail = f"{len(val)} obs"
            else:
                detail = str(val)[:40]
        else:
            detail = val[:60]
        marker = "+" if status == "OK" else "!"
        print(f"  {marker} {name:<24} {status:<6} {detail}")

    print(f"\n  {'-'*60}")
    if failed == 0:
        print(f"  ALL SYSTEMS GO -- {passed}/{passed+failed} checks passed")
        print(f"  Ready for: python main.py --paper")
    else:
        print(f"  {failed} CHECK(S) FAILED — fix before running --paper")
    print("==========================================\n")

    sys.exit(0 if failed == 0 else 1)


def _concat_oos_returns(result) -> "pd.Series":
    import pandas as pd
    all_returns = []
    all_idx = []
    for w in result.windows:
        if w.equity_curve is not None and len(w.equity_curve) > 1:
            rets = w.equity_curve.pct_change().dropna()
            all_returns.extend(rets.values)
            all_idx.extend(rets.index.tolist())
    if all_returns:
        return pd.Series(all_returns, index=all_idx)
    return pd.Series()


def main():
    load_dotenv()
    args = parse_args()
    settings = load_settings(args.config)
    credentials = load_credentials()

    log_cfg = settings.get("monitoring", {})
    setup_logging(
        log_file=log_cfg.get("log_file", "logs/regime_trader.log"),
        level=log_cfg.get("log_level", "INFO"),
    )
    logger = logging.getLogger("main")

    if args.dry_run:
        run_dry_run(settings, credentials)

    elif args.test_connection:
        from broker.alpaca_client import AlpacaClient
        _bridge_alpaca_credentials(credentials)
        settings["broker"]["mode"] = "paper"
        client = AlpacaClient(settings)
        client.connect()
        account = client.get_account()
        print("\n=== Alpaca Connection Test ===")
        print(f"  Status:          OK")
        print(f"  Mode:            paper")
        print(f"  Portfolio value: ${account['portfolio_value']:,.2f}")
        print(f"  Buying power:    ${account['buying_power']:,.2f}")
        print(f"  Cash:            ${account['cash']:,.2f}")
        print("==============================\n")
        sys.exit(0)

    elif args.live:
        print("\n" + "="*60)
        print("WARNING: LIVE TRADING MODE — REAL MONEY AT RISK")
        print("="*60)
        confirm = input("Type 'CONFIRM' to proceed with live trading: ")
        if confirm.strip() != "CONFIRM":
            print("Aborted.")
            sys.exit(0)
        logger.warning("LIVE trading mode started by user confirmation")
        # Live uses same paper loop but with settings["broker"]["mode"] = "live"
        settings["broker"]["mode"] = "live"
        run_paper(settings, credentials)

    elif args.paper:
        settings["broker"]["mode"] = "paper"
        run_paper(settings, credentials)

    elif args.backtest:
        run_backtest(settings, start_date=args.start)


if __name__ == "__main__":
    main()
