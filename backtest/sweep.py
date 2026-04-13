# backtest/sweep.py
# ============================================================
# Parameter sweep — one-at-a-time sensitivity analysis.
#
# Phase B sweeps (all locked):
#   normalization_window, rebalance_threshold,
#   flicker_window, flicker_threshold
#
# Phase C sweeps (H1 + H2):
#   H1: is_window         [252, 378, 504]
#   H2: n_components_range [[3,4],[3,5],[3,6],[3,7]]
#
# Usage:
#   python -m backtest.sweep                  # all sweeps
#   python -m backtest.sweep --sweep H1       # IS window only
#   python -m backtest.sweep --sweep H2       # BIC cap only
#   python -m backtest.sweep --sweep H1 H2    # both Phase C
# ============================================================

import argparse
import copy
import logging
import sys
from pathlib import Path

import pandas as pd
import yaml

# silence INFO noise from sub-modules during sweep
logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s | %(name)s | %(message)s",
    stream=sys.stdout,
)
sweep_log = logging.getLogger("sweep")
sweep_log.setLevel(logging.INFO)

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.market_data import DataManager
from backtest.backtester import WalkForwardBacktester


# ── Sweep definitions ────────────────────────────────────────
# Phase B — all locked, kept for reference
PHASE_B_SWEEPS = [
    {"name": "normalization_window", "path": ["features", "normalization_window"], "values": [45, 60, 90]},
    {"name": "rebalance_threshold",  "path": ["allocation", "rebalance_threshold"],  "values": [0.05, 0.08, 0.10, 0.15]},
    {"name": "flicker_window",       "path": ["stability", "flicker_window"],        "values": [10, 15, 20, 25]},
    {"name": "flicker_threshold",    "path": ["stability", "flicker_threshold"],     "values": [3, 4, 5, 6]},
]

# Phase C — H1: IS window length
H1_SWEEP = {
    "name": "is_window",
    "path": ["backtest", "is_window"],
    "values": [252, 378, 504],
}

# Phase C — H2: BIC state count upper bound
H2_SWEEP = {
    "name": "n_components_range",
    "path": ["hmm", "n_components_range"],
    "values": [[3, 4], [3, 5], [3, 6], [3, 7]],
}

NAMED_SWEEPS = {
    "H1": [H1_SWEEP],
    "H2": [H2_SWEEP],
    "B":  PHASE_B_SWEEPS,
}


def _get(d: dict, path: list):
    for k in path:
        d = d[k]
    return d


def _set(d: dict, path: list, value):
    for k in path[:-1]:
        d = d[k]
    d[path[-1]] = value


def _fmt(val) -> str:
    """Format a sweep value for display — handles scalars and lists."""
    if isinstance(val, list):
        return str(val)
    return f"{val}"


def _2022_windows(result) -> list:
    """
    Extract per-window Sharpe for OOS windows whose start date falls in 2022.
    Returns list of (oos_start, sharpe, n_states).
    Filter by date, not index — index shifts when is_window changes.
    """
    rows = []
    for w in result.windows:
        if w.oos_start.startswith("2022") or w.oos_end.startswith("2022"):
            sharpe = w.oos_metrics.get("sharpe", float("nan"))
            rows.append((w.oos_start, w.oos_end, sharpe, w.bic_n_states))
    return rows


def run_sweep(sweeps: list, base_settings: dict, prices: dict, vix, hy_oas, gold, term_spread) -> dict:
    """
    Run one-at-a-time sweep over given sweep list.
    Returns dict of {param_name: pd.DataFrame}.
    """
    all_results = {}

    for sweep in sweeps:
        param = sweep["name"]
        path = sweep["path"]
        values = sweep["values"]
        baseline = _get(base_settings, path)

        print(f"\n{'='*60}")
        print(f"SWEEP: {param}  (baseline={_fmt(baseline)})")
        print(f"{'='*60}")

        rows = []
        for val in values:
            settings = copy.deepcopy(base_settings)
            _set(settings, path, val)

            bt = WalkForwardBacktester(settings)
            try:
                result = bt.run(
                    prices, vix, hy_oas,
                    primary_symbol="SPY",
                    gold=gold,
                    term_spread=term_spread,
                )
                m = result.aggregate_metrics
                sharpe  = m.get("sharpe",        float("nan"))
                cagr    = m.get("cagr",          float("nan"))
                max_dd  = m.get("max_drawdown",  float("nan"))
                sortino = m.get("sortino",       float("nan"))
                n_wins  = len(result.windows)

                # 2022-specific window breakdown
                wins_2022 = _2022_windows(result)

            except Exception as e:
                sweep_log.warning(f"  {param}={_fmt(val)} failed: {e}")
                sharpe = cagr = max_dd = sortino = float("nan")
                n_wins = 0
                wins_2022 = []

            is_baseline = (_fmt(val) == _fmt(baseline))
            tag = "  <-- baseline" if is_baseline else ""

            print(
                f"  {param}={_fmt(val):>12}: "
                f"Sharpe={sharpe:+.3f}  CAGR={cagr:.1%}  MaxDD={max_dd:.1%}  "
                f"Sortino={sortino:.3f}  windows={n_wins}"
                f"{tag}"
            )

            if wins_2022:
                print(f"    2022 OOS windows:")
                for oos_start, oos_end, w_sharpe, n_states in wins_2022:
                    print(f"      {oos_start} -> {oos_end}: Sharpe={w_sharpe:+.3f}  n_states={n_states}")

            # Use string key for list values so DataFrame index works
            row_key = _fmt(val)
            rows.append({
                param: row_key,
                "sharpe": sharpe,
                "cagr": cagr,
                "max_dd": max_dd,
                "sortino": sortino,
                "n_windows": n_wins,
                "baseline": is_baseline,
            })

        df = pd.DataFrame(rows).set_index(param)
        all_results[param] = df

        best_idx = df["sharpe"].idxmax()
        best_sharpe = df.loc[best_idx, "sharpe"]
        base_key = _fmt(baseline)
        base_sharpe = df.loc[base_key, "sharpe"] if base_key in df.index else float("nan")
        delta = best_sharpe - base_sharpe

        print(f"\n  Winner: {param}={best_idx}  Sharpe={best_sharpe:.3f}  (delta vs baseline: {delta:+.3f})")

    return all_results


def print_summary(sweeps: list, all_results: dict, base_settings: dict, label: str = "SWEEP"):
    """Print consolidated winner table."""
    print(f"\n{'='*60}")
    print(f"{label} SUMMARY")
    print(f"{'='*60}")
    print(f"{'Param':<25} {'Baseline':>14} {'Winner':>14} {'Sharpe':>8} {'Delta':>8}")
    print("-" * 73)

    for sweep in sweeps:
        param = sweep["name"]
        path = sweep["path"]
        baseline = _get(base_settings, path)
        df = all_results.get(param)
        if df is None:
            continue
        best_idx = df["sharpe"].idxmax()
        best_sharpe = df.loc[best_idx, "sharpe"]
        base_key = _fmt(baseline)
        base_sharpe = df.loc[base_key, "sharpe"] if base_key in df.index else float("nan")
        delta = best_sharpe - base_sharpe
        changed = "*" if best_idx != base_key else ""
        print(f"{param:<25} {base_key:>14} {str(best_idx):>14} {best_sharpe:>8.3f} {delta:>+8.3f} {changed}")

    print("\n* = change from baseline recommended")


def main():
    parser = argparse.ArgumentParser(description="Regime-trader parameter sweep")
    parser.add_argument(
        "--sweep", nargs="+", choices=list(NAMED_SWEEPS.keys()),
        help="Which sweep(s) to run: H1 (is_window), H2 (BIC cap), B (Phase B params). Default: H1 H2."
    )
    args = parser.parse_args()

    selected = args.sweep or ["H1", "H2"]
    sweeps = []
    for name in selected:
        sweeps.extend(NAMED_SWEEPS[name])

    config_path = Path(__file__).parent.parent / "config" / "settings.yaml"
    with open(config_path) as f:
        base_settings = yaml.safe_load(f)

    start = base_settings["data"]["start_date"]
    symbols = base_settings["data"]["symbols"]
    label = "PHASE C" if set(selected) <= {"H1", "H2"} else "SWEEP"

    print(f"{label} Parameter Sweep — {selected}")
    print(f"Data: {symbols}, {start} -> today")
    print("Fetching data (once, reused across all runs)...")

    dm = DataManager(base_settings, mode="backtest")
    vix        = dm.get_vix(start)
    hy_oas     = dm.get_hy_oas(start)
    gold       = dm.get_gold(start)
    term_spread = dm.get_term_spread(start)
    prices     = {sym: dm.get_bars(sym, start) for sym in symbols}

    n_runs = sum(len(s["values"]) for s in sweeps)
    print(f"Data loaded. Running {n_runs} backtest runs...\n")

    all_results = run_sweep(sweeps, base_settings, prices, vix, hy_oas, gold, term_spread)
    print_summary(sweeps, all_results, base_settings, label=label)


if __name__ == "__main__":
    main()
