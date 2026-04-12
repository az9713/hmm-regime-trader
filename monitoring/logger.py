# monitoring/logger.py
# ============================================================
# Structured JSON logging for all regime decisions and trades.
# Every backtest run logged with: timestamp, all params, all metrics,
# data range, BIC-selected state counts.
# Creates audit trail — prevents re-testing same params unknowingly.
# Ref: design_docs/06_empirical_testing_plan.md §"Log everything"
# ============================================================

import json
import logging
import os
from datetime import datetime
from pathlib import Path


def setup_logging(log_file: str = "logs/regime_trader.log", level: str = "INFO"):
    """Configure root logger with file + console handlers."""
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    log_level = getattr(logging, level.upper(), logging.INFO)

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger("regime_trader")


class TradeLogger:
    """
    Structured logger for trade decisions and backtest results.
    Every entry logged as JSON for easy parsing and audit.
    """

    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._logger = logging.getLogger("trade_logger")
        self._run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._backtest_log = self.log_dir / f"backtest_{self._run_id}.jsonl"
        self._trade_log = self.log_dir / f"trades_{self._run_id}.jsonl"

    def log_signal(self, signal, regime_state):
        """Log a generated signal."""
        entry = {
            "ts": datetime.now().isoformat(),
            "type": "signal",
            "symbol": signal.symbol,
            "direction": signal.direction,
            "size_shares": signal.size_shares,
            "entry": signal.entry_price,
            "stop": signal.stop_price,
            "target": signal.target_price,
            "regime": signal.regime,
            "confidence": round(signal.confidence, 4),
            "allocation": round(signal.allocation, 4),
            "strategy": signal.strategy,
        }
        self._write(self._trade_log, entry)

    def log_regime(self, symbol: str, regime_state):
        """Log HMM regime detection for each bar."""
        entry = {
            "ts": datetime.now().isoformat(),
            "type": "regime",
            "symbol": symbol,
            "regime": regime_state.label,
            "confidence": round(regime_state.confidence, 4),
            "stable": regime_state.stable,
            "bars_in_regime": regime_state.bars_in_regime,
            "n_states": regime_state.n_states_selected,
        }
        self._write(self._trade_log, entry)

    def log_circuit_breaker(self, state: str, message: str):
        """Log circuit breaker events — always at WARNING level."""
        entry = {
            "ts": datetime.now().isoformat(),
            "type": "circuit_breaker",
            "state": state,
            "message": message,
        }
        self._logger.warning(f"CIRCUIT BREAKER: {state} | {message}")
        self._write(self._trade_log, entry)

    def log_backtest_window(self, window_result):
        """Log one walk-forward window result."""
        entry = {
            "ts": datetime.now().isoformat(),
            "type": "backtest_window",
            "window_idx": window_result.window_idx,
            "is_start": window_result.is_start,
            "is_end": window_result.is_end,
            "oos_start": window_result.oos_start,
            "oos_end": window_result.oos_end,
            "bic_n_states": window_result.bic_n_states,
            "params": window_result.params,
            "metrics": window_result.oos_metrics,
        }
        self._write(self._backtest_log, entry)

    def log_backtest_summary(self, result):
        """Log aggregate backtest result."""
        entry = {
            "ts": datetime.now().isoformat(),
            "type": "backtest_summary",
            "run_id": self._run_id,
            "n_windows": len(result.windows),
            "aggregate_metrics": result.aggregate_metrics,
            "spa_result": result.spa_result,
        }
        self._write(self._backtest_log, entry)
        self._logger.info(f"Backtest complete. Log: {self._backtest_log}")

    def update_dashboard_state(
        self,
        regime: str,
        confidence: float,
        allocation: float,
        portfolio_value: float,
        daily_pnl: float,
        daily_pnl_pct: float,
        positions: list,
        equity_history: list,
        regime_history: list,
        timestamps: list,
        session_start: str,
    ):
        """
        Write latest snapshot to logs/dashboard_state.json.
        Streamlit dashboard reads this file every 10 seconds.
        """
        state = {
            "regime": regime,
            "confidence": round(confidence, 4),
            "allocation": round(allocation, 4),
            "portfolio_value": round(portfolio_value, 2),
            "daily_pnl": round(daily_pnl, 2),
            "daily_pnl_pct": round(daily_pnl_pct, 6),
            "positions": positions,
            "equity_history": [round(v, 2) for v in equity_history[-500:]],
            "regime_history": regime_history[-500:],
            "timestamps": [str(t) for t in timestamps[-500:]],
            "session_start": session_start,
            "last_updated": datetime.now().strftime("%H:%M:%S ET"),
        }
        path = self.log_dir / "dashboard_state.json"
        with open(path, "w") as f:
            json.dump(state, f)

    def _write(self, path: Path, entry: dict):
        with open(path, "a") as f:
            f.write(json.dumps(entry) + "\n")
