# monitoring/alerts.py
# ============================================================
# Push alerts via Telegram + optional email.
#
# Telegram events (agreed 2026-04-12):
#   🔴 Critical  — circuit breaker trip, broker error
#   🟠 Important — regime flip, bracket order filled
#   🟡 Daily     — end-of-day P&L summary (16:05 ET)
#   ⚪ Suppressed — HMM retrains (too frequent, noise)
#
# Setup: credentials.yaml → telegram.bot_token + telegram.chat_id
# Test:  python -m monitoring.alerts
# ============================================================

import logging
import smtplib
import urllib.request
import urllib.parse
import json
from datetime import datetime
from email.mime.text import MIMEText

logger = logging.getLogger(__name__)

REGIME_EMOJI = {
    "LowVol":      "🟢",
    "MidVol":      "🟡",
    "HighVol":     "🔴",
    "VeryHighVol": "🟣",
    "Uncertainty": "⚪",
}


class AlertManager:
    """
    Sends push alerts via Telegram + optional email.
    Telegram is the primary channel — instant, phone-delivered.
    """

    def __init__(self, settings: dict, credentials: dict = None):
        self.settings = settings
        self._email = settings.get("monitoring", {}).get("alert_email")
        self._tg_enabled = settings.get("telegram", {}).get("enabled", False)
        self._bot_token = None
        self._chat_id = None

        if credentials and self._tg_enabled:
            tg = credentials.get("telegram", {})
            self._bot_token = tg.get("bot_token")
            self._chat_id = tg.get("chat_id")
            if self._bot_token and self._chat_id:
                logger.info("Telegram alerts: enabled")
            else:
                logger.warning("Telegram alerts: credentials missing — check credentials.yaml")

    # ── 🔴 Critical ────────────────────────────────────────────

    def circuit_breaker(self, state: str, message: str, portfolio_value: float):
        msg = (
            f"🚨 *CIRCUIT BREAKER — {state}*\n"
            f"{message}\n"
            f"Portfolio: *${portfolio_value:,.0f}*\n"
            f"_{datetime.now().strftime('%H:%M:%S ET')}_"
        )
        logger.critical(f"[CIRCUIT BREAKER] {state} | {message} | ${portfolio_value:,.0f}")
        self._telegram(msg)
        self._send_email("CIRCUIT BREAKER TRIGGERED", message)

    def broker_error(self, symbol: str, error: str):
        msg = (
            f"⚠️ *Broker Error — {symbol}*\n"
            f"`{error}`\n"
            f"_{datetime.now().strftime('%H:%M:%S ET')}_"
        )
        logger.error(f"[BROKER ERROR] {symbol}: {error}")
        self._telegram(msg)
        self._send_email("Broker Error", f"{symbol}: {error}")

    # ── 🟠 Important ───────────────────────────────────────────

    def regime_flip(
        self,
        symbol: str,
        prev_regime: str,
        new_regime: str,
        confidence: float,
        allocation: float,
    ):
        prev_emoji = REGIME_EMOJI.get(prev_regime, "⚪")
        new_emoji = REGIME_EMOJI.get(new_regime, "⚪")
        msg = (
            f"{prev_emoji}→{new_emoji} *Regime Flip — {symbol}*\n"
            f"{prev_regime} → *{new_regime}*\n"
            f"Confidence: {confidence:.0%} | Allocation: {allocation:.0%}\n"
            f"_{datetime.now().strftime('%H:%M:%S ET')}_"
        )
        logger.info(f"[REGIME FLIP] {symbol}: {prev_regime} → {new_regime} conf={confidence:.2f}")
        self._telegram(msg)

    def order_filled(
        self,
        symbol: str,
        side: str,
        qty: int,
        fill_price: float,
        stop: float,
        target: float,
        regime: str,
    ):
        emoji = REGIME_EMOJI.get(regime, "⚪")
        msg = (
            f"✅ *Order Filled — {symbol}*\n"
            f"{side.upper()} {qty} shares @ ${fill_price:.2f}\n"
            f"Stop: ${stop:.2f} | Target: ${target:.2f}\n"
            f"{emoji} Regime: {regime}\n"
            f"_{datetime.now().strftime('%H:%M:%S ET')}_"
        )
        logger.info(f"[ORDER FILLED] {symbol} {side} {qty}@{fill_price:.2f}")
        self._telegram(msg)

    # ── 🟡 Daily summary ───────────────────────────────────────

    def daily_summary(
        self,
        portfolio_value: float,
        daily_pnl: float,
        daily_pnl_pct: float,
        regime: str,
        confidence: float,
        n_trades: int,
        open_positions: list,
    ):
        pnl_emoji = "📈" if daily_pnl >= 0 else "📉"
        emoji = REGIME_EMOJI.get(regime, "⚪")
        positions_str = (
            "\n".join(f"  • {p}" for p in open_positions)
            if open_positions else "  None"
        )
        msg = (
            f"{pnl_emoji} *Daily Summary — {datetime.now().strftime('%Y-%m-%d')}*\n\n"
            f"Portfolio: *${portfolio_value:,.0f}*\n"
            f"Day P&L: *{'+' if daily_pnl >= 0 else ''}"
            f"${daily_pnl:,.0f} ({daily_pnl_pct:+.2%})*\n\n"
            f"{emoji} Regime: {regime} ({confidence:.0%} conf)\n"
            f"Trades today: {n_trades}\n\n"
            f"*Open positions:*\n{positions_str}"
        )
        logger.info(f"[DAILY SUMMARY] pnl={daily_pnl:+.0f} ({daily_pnl_pct:+.2%}) regime={regime}")
        self._telegram(msg)

    # ── ⚪ Suppressed (log only) ────────────────────────────────

    def hmm_retrain(self, n_states: int, bic: float, window_start: str):
        logger.info(f"[HMM RETRAIN] window={window_start} n_states={n_states} BIC={bic:.1f}")
        # No Telegram — retrains every 20 bars, would be noise

    def data_gap(self, symbol: str, gap_details: str):
        logger.warning(f"[DATA GAP] {symbol}: {gap_details}")

    # ── Internal ───────────────────────────────────────────────

    def _telegram(self, text: str):
        """Send a Markdown message via Telegram Bot API."""
        if not (self._tg_enabled and self._bot_token and self._chat_id):
            return
        try:
            url = f"https://api.telegram.org/bot{self._bot_token}/sendMessage"
            payload = json.dumps({
                "chat_id": self._chat_id,
                "text": text,
                "parse_mode": "Markdown",
            }).encode()
            req = urllib.request.Request(
                url, data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status != 200:
                    logger.warning(f"Telegram API returned {resp.status}")
        except Exception as e:
            logger.warning(f"Telegram alert failed: {e}")

    def _send_email(self, subject: str, body: str):
        if not self._email:
            return
        try:
            msg = MIMEText(f"{body}\n\nTimestamp: {datetime.now().isoformat()}")
            msg["Subject"] = f"[RegimeTrader] {subject}"
            msg["From"] = self._email
            msg["To"] = self._email
            with smtplib.SMTP("localhost") as smtp:
                smtp.send_message(msg)
        except Exception as e:
            logger.warning(f"Alert email failed: {e}")


if __name__ == "__main__":
    # Quick smoke test — sends a real Telegram message if credentials present
    import yaml, os
    creds_path = os.path.join(os.path.dirname(__file__), "../config/credentials.yaml")
    settings_path = os.path.join(os.path.dirname(__file__), "../config/settings.yaml")
    with open(settings_path) as f:
        settings = yaml.safe_load(f)
    credentials = {}
    if os.path.exists(creds_path):
        with open(creds_path) as f:
            credentials = yaml.safe_load(f)
    alerts = AlertManager(settings, credentials)
    alerts.regime_flip("SPY", "LowVol", "HighVol", confidence=0.87, allocation=0.45)
    print("Test alert sent. Check your Telegram.")
