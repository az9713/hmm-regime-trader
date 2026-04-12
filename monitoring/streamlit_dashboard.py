# monitoring/streamlit_dashboard.py
# ============================================================
# Live Streamlit dashboard for paper/live trading sessions.
#
# Layout (one screen, no scroll):
#   Top banner  — current regime + confidence gauge
#   Left panel  — equity curve (since session start)
#   Right panel — regime timeline (last 60 bars, colour bands)
#   Bottom left — open positions table
#   Bottom right— today's alerts log
#
# Run:
#   streamlit run monitoring/streamlit_dashboard.py
#
# The dashboard reads from two files written by the main loop:
#   logs/dashboard_state.json   — latest snapshot (every bar)
#   logs/regime_trader.log      — alert feed
# ============================================================

import json
import os
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

STATE_FILE = Path("logs/dashboard_state.json")
LOG_FILE = Path("logs/regime_trader.log")

REGIME_COLORS = {
    "LowVol":      "#2ecc71",
    "MidVol":      "#f39c12",
    "HighVol":     "#e74c3c",
    "VeryHighVol": "#8e44ad",
    "Uncertainty": "#95a5a6",
}
REGIME_EMOJI = {
    "LowVol":      "🟢",
    "MidVol":      "🟡",
    "HighVol":     "🔴",
    "VeryHighVol": "🟣",
    "Uncertainty": "⚪",
}

st.set_page_config(
    page_title="HMM Regime Trader",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS — dark, clean ────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 1rem; padding-bottom: 0.5rem; }
    .regime-banner {
        padding: 1rem 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        font-size: 1.4rem;
        font-weight: 700;
        color: white;
    }
    .metric-card {
        background: #1e1e2e;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        text-align: center;
        color: white;
    }
    .metric-label { font-size: 0.75rem; color: #aaa; margin-bottom: 0.2rem; }
    .metric-value { font-size: 1.4rem; font-weight: 700; }
    .metric-positive { color: #2ecc71; }
    .metric-negative { color: #e74c3c; }
    .alert-row { font-size: 0.82rem; border-bottom: 1px solid #333; padding: 0.3rem 0; }
</style>
""", unsafe_allow_html=True)


def load_state() -> dict:
    if not STATE_FILE.exists():
        return {}
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except Exception:
        return {}


def load_recent_alerts(n: int = 20) -> list[str]:
    if not LOG_FILE.exists():
        return []
    try:
        lines = LOG_FILE.read_text(encoding="utf-8").splitlines()
        keywords = ["CIRCUIT BREAKER", "BROKER ERROR", "REGIME FLIP", "ORDER FILLED",
                    "DATA GAP", "DAILY SUMMARY"]
        alerts = [l for l in lines if any(k in l for k in keywords)]
        return alerts[-n:][::-1]  # most recent first
    except Exception:
        return []


def render():
    state = load_state()

    regime = state.get("regime", "—")
    confidence = state.get("confidence", 0.0)
    portfolio_value = state.get("portfolio_value", 0.0)
    daily_pnl = state.get("daily_pnl", 0.0)
    daily_pnl_pct = state.get("daily_pnl_pct", 0.0)
    allocation = state.get("allocation", 0.0)
    equity_history = state.get("equity_history", [])
    regime_history = state.get("regime_history", [])
    timestamps = state.get("timestamps", [])
    positions = state.get("positions", [])
    last_updated = state.get("last_updated", "—")
    session_start = state.get("session_start", "—")

    color = REGIME_COLORS.get(regime, "#95a5a6")
    emoji = REGIME_EMOJI.get(regime, "⚪")

    # ── Top banner ──────────────────────────────────────────
    st.markdown(
        f'<div class="regime-banner" style="background:{color}">'
        f'{emoji} Regime: {regime} &nbsp;|&nbsp; '
        f'Confidence: {confidence:.0%} &nbsp;|&nbsp; '
        f'Allocation: {allocation:.0%} &nbsp;|&nbsp; '
        f'<span style="font-size:0.9rem;font-weight:400">Updated: {last_updated}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Confidence bar ──────────────────────────────────────
    st.progress(confidence, text="")

    # ── Metric row ──────────────────────────────────────────
    m1, m2, m3, m4, m5 = st.columns(5)
    pnl_class = "metric-positive" if daily_pnl >= 0 else "metric-negative"
    pnl_sign = "+" if daily_pnl >= 0 else ""

    with m1:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">Portfolio Value</div>'
            f'<div class="metric-value">${portfolio_value:,.0f}</div>'
            f'</div>', unsafe_allow_html=True)
    with m2:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">Day P&L</div>'
            f'<div class="metric-value {pnl_class}">'
            f'{pnl_sign}${daily_pnl:,.0f}</div>'
            f'</div>', unsafe_allow_html=True)
    with m3:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">Day P&L %</div>'
            f'<div class="metric-value {pnl_class}">'
            f'{pnl_sign}{daily_pnl_pct:.2%}</div>'
            f'</div>', unsafe_allow_html=True)
    with m4:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">Open Positions</div>'
            f'<div class="metric-value">{len(positions)}</div>'
            f'</div>', unsafe_allow_html=True)
    with m5:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">Session Start</div>'
            f'<div class="metric-value" style="font-size:1rem">{session_start}</div>'
            f'</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Main panels ─────────────────────────────────────────
    left, right = st.columns(2)

    with left:
        st.subheader("Equity Curve")
        if len(equity_history) > 1:
            eq_df = pd.DataFrame({
                "Value": equity_history,
            }, index=range(len(equity_history)))
            st.line_chart(eq_df, use_container_width=True, height=250)
        else:
            st.info("Waiting for data — equity curve will appear after the first bar.")

    with right:
        st.subheader("Regime Timeline (last 60 bars)")
        if regime_history:
            display = regime_history[-60:]
            reg_df = pd.DataFrame({
                "Regime": display,
                "Color": [REGIME_COLORS.get(r, "#95a5a6") for r in display],
                "Bar": list(range(len(display))),
            })
            # Render as colored band chart using Streamlit bar_chart
            # Map regimes to numeric values for height (all equal)
            reg_df["Val"] = 1
            regime_counts = reg_df.groupby("Regime")["Val"].sum()
            st.bar_chart(regime_counts, use_container_width=True, height=250)
            # Legend
            legend_parts = [
                f"{REGIME_EMOJI.get(r,'⚪')} {r}"
                for r in ["LowVol", "MidVol", "HighVol", "VeryHighVol", "Uncertainty"]
                if r in display
            ]
            st.caption("  |  ".join(legend_parts))
        else:
            st.info("Regime history will appear after the first bar.")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Bottom panels ───────────────────────────────────────
    bot_left, bot_right = st.columns(2)

    with bot_left:
        st.subheader("Open Positions")
        if positions:
            pos_df = pd.DataFrame(positions)
            st.dataframe(
                pos_df,
                use_container_width=True,
                hide_index=True,
                height=200,
            )
        else:
            st.info("No open positions.")

    with bot_right:
        st.subheader("Today's Alerts")
        alerts = load_recent_alerts(20)
        if alerts:
            for alert in alerts:
                st.markdown(
                    f'<div class="alert-row">{alert}</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.info("No alerts yet today.")

    # ── Auto-refresh ────────────────────────────────────────
    st.markdown("---")
    st.caption(f"Auto-refreshing every 10s · {datetime.now().strftime('%H:%M:%S')}")
    time.sleep(10)
    st.rerun()


render()
