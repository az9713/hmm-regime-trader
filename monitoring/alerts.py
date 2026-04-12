# monitoring/alerts.py
# ============================================================
# Alerts on: circuit breaker trips, broker errors, data gaps.
# ============================================================

import logging
import smtplib
from email.mime.text import MIMEText
from datetime import datetime

logger = logging.getLogger(__name__)


class AlertManager:
    """
    Sends alerts via logging + optional email.
    Configure alert_email in settings.yaml to enable email.
    """

    def __init__(self, settings: dict):
        self.settings = settings
        self._email = settings.get("monitoring", {}).get("alert_email")

    def circuit_breaker(self, state: str, message: str, portfolio_value: float):
        msg = f"[CIRCUIT BREAKER] {state} | {message} | Portfolio: ${portfolio_value:,.0f}"
        logger.critical(msg)
        self._send_email("CIRCUIT BREAKER TRIGGERED", msg)

    def broker_error(self, symbol: str, error: str):
        msg = f"[BROKER ERROR] {symbol}: {error}"
        logger.error(msg)
        self._send_email("Broker Error", msg)

    def data_gap(self, symbol: str, gap_details: str):
        msg = f"[DATA GAP] {symbol}: {gap_details}"
        logger.warning(msg)

    def hmm_retrain(self, n_states: int, bic: float, window_start: str):
        logger.info(f"[HMM RETRAIN] window={window_start} n_states={n_states} BIC={bic:.1f}")

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
