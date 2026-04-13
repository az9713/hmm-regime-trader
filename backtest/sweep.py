# backtest/sweep.py
# ============================================================
# Parameter sweep — one-at-a-time sensitivity analysis (OAT)
# plus Phase F 2x2 grid sweep.
#
# Phase B sweeps (all locked):
#   normalization_window, rebalance_threshold,
#   flicker_window, flicker_threshold
#
# Phase C sweeps (H1 + H2):
#   H1: is_window         [252, 378, 504]
#   H2: n_components_range [[3,4],[3,5],[3,6],[3,7]]
#
# Phase F sweep (2x2 grid):
#   covariance_type x use_vix_slope:
#   F1: tied + 6-feat  |  F2: full + 6-feat
#   F3: full + 7-feat  |  F4: tied + 7-feat
#
# Usage:
#   python -m backtest.sweep                  # default: H1 H2
#   python -m backtest.sweep --sweep H1       # IS window only
#   python -m backtest.sweep --sweep H2       # BIC cap only
#   python -m backtest.sweep --sweep H1 H2    # both Phase C
#   python -m backtest.sweep --sweep F        # Phase F full covariance grid
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

# Phase F — 2x2 grid: covariance_type × use_vix_slope
# Not OAT — runs all 4 combinations explicitly.
# Baseline for Phase F: diag + 6-feat (Phase C result, Sharpe 0.940).
PHASE_F_CONFIGS = [
    {"label": "diag + 6-feat (baseline)", "covariance_type": "diag",  "use_vix_slope": False},
    {"label": "tied + 6-feat",            "covariance_type": "tied",  "use_vix_slope": False},
    {"label": "full + 6-feat",            "covariance_type": "full",  "use_vix_slope": False},
    {"label": "full + 7-feat (vix_slope)","covariance_type": "full",  "use_vix_slope": True},
    {"label": "tied + 7-feat (vix_slope)","covariance_type": "tied",  "use_vix_slope": True},
]


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


def run_sweep(sweeps: list, base_settings: dict, prices: dict, vix, hy_oas, gold, term_spread, vix3m=None) -> dict:
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
                    vix3m=vix3m,
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


def run_phase_f(base_settings: dict, prices: dict, vix, hy_oas, gold, term_spread, vix3m) -> list:
    """
    Phase F: 2x2 grid sweep over covariance_type x use_vix_slope.
    Returns list of result dicts for summary table.
    """
    rows = []

    print(f"\n{'='*60}")
    print("PHASE F SWEEP: covariance_type x feature set")
    print(f"Baseline (Phase C): diag + 6-feat, Sharpe=0.940")
    print(f"Gate: Sharpe > 1.0 and worst-2022-window > -1.0")
    print(f"{'='*60}")

    for cfg in PHASE_F_CONFIGS:
        settings = copy.deepcopy(base_settings)
        settings["hmm"]["covariance_type"] = cfg["covariance_type"]
        settings["features"]["use_vix_slope"] = cfg["use_vix_slope"]
        n_feat = 7 if cfg["use_vix_slope"] else 6

        print(f"\n  Running: {cfg['label']} ...")

        bt = WalkForwardBacktester(settings)
        try:
            result = bt.run(
                prices, vix, hy_oas,
                primary_symbol="SPY",
                gold=gold,
                term_spread=term_spread,
                vix3m=vix3m,
            )
            m = result.aggregate_metrics
            sharpe  = m.get("sharpe",       float("nan"))
            cagr    = m.get("cagr",         float("nan"))
            max_dd  = m.get("max_drawdown", float("nan"))
            n_wins  = len(result.windows)
            wins_2022 = _2022_windows(result)
            worst_2022 = min((s for _, _, s, _ in wins_2022), default=float("nan"))
        except Exception as e:
            sweep_log.warning(f"  {cfg['label']} failed: {e}")
            sharpe = cagr = max_dd = worst_2022 = float("nan")
            n_wins = 0
            wins_2022 = []

        gate = "PASS" if sharpe > 1.0 and worst_2022 > -1.0 else "---"

        print(
            f"  {cfg['label']:<30}: "
            f"Sharpe={sharpe:+.3f}  CAGR={cagr:.1%}  MaxDD={max_dd:.1%}  "
            f"windows={n_wins}  worst-2022={worst_2022:+.3f}  gate={gate}"
        )
        if wins_2022:
            print(f"    2022 OOS windows:")
            for oos_start, oos_end, w_sharpe, n_states in wins_2022:
                print(f"      {oos_start} -> {oos_end}: Sharpe={w_sharpe:+.3f}  n_states={n_states}")

        rows.append({
            "config": cfg["label"],
            "covariance_type": cfg["covariance_type"],
            "n_features": n_feat,
            "sharpe": sharpe,
            "cagr": cagr,
            "max_dd": max_dd,
            "n_windows": n_wins,
            "worst_2022": worst_2022,
            "gate": gate,
        })

    print(f"\n{'='*60}")
    print("PHASE F SUMMARY")
    print(f"{'='*60}")
    print(f"{'Config':<32} {'Sharpe':>8} {'worst-2022':>12} {'Gate':>6}")
    print("-" * 62)
    for r in rows:
        print(f"{r['config']:<32} {r['sharpe']:>+8.3f} {r['worst_2022']:>+12.3f} {r['gate']:>6}")

    # Find winner (best Sharpe among gate-passing configs; else best Sharpe overall)
    passing = [r for r in rows if r["gate"] == "PASS"]
    winner = max(passing, key=lambda r: r["sharpe"]) if passing else max(rows, key=lambda r: r["sharpe"])
    print(f"\n  Winner: {winner['config']}  Sharpe={winner['sharpe']:.3f}  gate={winner['gate']}")

    return rows


def main():
    all_choices = list(NAMED_SWEEPS.keys()) + ["F"]
    parser = argparse.ArgumentParser(description="Regime-trader parameter sweep")
    parser.add_argument(
        "--sweep", nargs="+", choices=all_choices,
        help="Which sweep(s) to run: H1 (is_window), H2 (BIC cap), B (Phase B), F (Phase F full cov). Default: H1 H2."
    )
    args = parser.parse_args()

    selected = args.sweep or ["H1", "H2"]

    config_path = Path(__file__).parent.parent / "config" / "settings.yaml"
    with open(config_path) as f:
        base_settings = yaml.safe_load(f)

    start = base_settings["data"]["start_date"]
    symbols = base_settings["data"]["symbols"]

    print(f"Parameter Sweep — {selected}")
    print(f"Data: {symbols}, {start} -> today")
    print("Fetching data (once, reused across all runs)...")

    dm = DataManager(base_settings, mode="backtest")
    vix         = dm.get_vix(start)
    hy_oas      = dm.get_hy_oas(start)
    gold        = dm.get_gold(start)
    term_spread = dm.get_term_spread(start)
    vix3m       = dm.get_vix3m(start)
    prices      = {sym: dm.get_bars(sym, start) for sym in symbols}

    print("Data loaded.\n")

    if "F" in selected:
        run_phase_f(base_settings, prices, vix, hy_oas, gold, term_spread, vix3m)

    uat_selected = [s for s in selected if s != "F"]
    if uat_selected:
        sweeps = []
        for name in uat_selected:
            sweeps.extend(NAMED_SWEEPS[name])
        label = "PHASE C" if set(uat_selected) <= {"H1", "H2"} else "SWEEP"
        n_runs = sum(len(s["values"]) for s in sweeps)
        print(f"Running {n_runs} OAT backtest runs...\n")
        all_results = run_sweep(sweeps, base_settings, prices, vix, hy_oas, gold, term_spread, vix3m)
        print_summary(sweeps, all_results, base_settings, label=label)


if __name__ == "__main__":
    main()
