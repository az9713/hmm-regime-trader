# Regime Trader — Quickstart Guide

## What This System Is

An automated trading bot that detects market volatility regimes using a Hidden Markov Model (HMM) and adjusts position sizing accordingly. No LLM involved — the "AI" is classical statistical pattern recognition (Hamilton 1989).

**Three modes:**
- `--backtest` — walk-forward historical simulation (calibration instrument)
- `--paper` — live paper trading via Alpaca (no real money)
- `--live` — live trading with real money (requires explicit confirmation)

**Recommended sequence:** backtest → paper → live

---

## Requirements

### Python Environment

```bash
Python 3.10+ required (tested on 3.13)
```

Install dependencies:
```bash
cd regime-trader
pip install -r requirements.txt
```

`requirements.txt` includes:
```
hmmlearn       # Hidden Markov Model engine
pandas         # Data manipulation
numpy          # Numerical computation
yfinance       # Historical price data (SPY, QQQ)
pandas-datareader  # FRED macro data (VIX, HY OAS)
alpaca-py      # Broker API (paper + live trading)
pyyaml         # Config loading
python-dotenv  # API key management
scipy          # Statistical functions (Hansen SPA test)
scikit-learn   # K-means HMM initialization
matplotlib     # Live dashboard
pytest         # Test suite
```

Additional dependency sometimes needed:
```bash
pip install lxml   # Required by pandas-datareader for FRED XML parsing
```

### API Keys

**Backtesting:** No API keys required. yfinance and FRED are both public.

**Paper/Live trading:** Alpaca account required (free).

1. Sign up at alpaca.markets
2. Go to Paper Trading → API Keys → Generate Key
3. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
4. Fill in `.env`:
   ```
   ALPACA_API_KEY=your_key_here
   ALPACA_SECRET_KEY=your_secret_here
   ALPACA_BASE_URL=https://paper-api.alpaca.markets
   ```

### Internet Connection

Required at runtime for data fetching. First run downloads ~3,600 bars per symbol. Subsequent runs use local cache (`data/cache/`).

---

## Data Sources

All data fetched automatically at runtime. No manual downloads needed.

| Data | Source | Series | History | Cost |
|---|---|---|---|---|
| SPY daily OHLCV | yfinance | SPY | 2010–today | Free |
| QQQ daily OHLCV | yfinance | QQQ | 2010–today | Free |
| VIX daily close | FRED | `VIXCLS` | 1990–today | Free |
| HY Credit Spread | FRED | `BAMLH0A0HYM2` | 1997–today | Free |

**Why these sources:**
- yfinance: adequate for backtesting, 20+ years history, `repair=True` fixes known issues
- FRED: gold standard for macro data (Federal Reserve official source)

**Credit spread note:** We use FRED `BAMLH0A0HYM2` (ICE BofA HY Option-Adjusted Spread), NOT the `log(HYG/LQD)` ratio the video implied. The ETF ratio confounds interest rate risk with credit risk due to duration mismatch (LQD ~7.7yr vs HYG ~3.9yr). The FRED series is duration-adjusted — pure credit signal.

**Data validation runs automatically** before any backtest:
- OHLC sanity checks (Close ≤ High, Low ≤ Close, etc.)
- No zero/negative prices
- Flat day detection (suspicious missing data)
- Extreme move detection (>10% daily — verify before proceeding)
- Date gap detection (>7 calendar days)

---

## Step 1 — Verify Installation

```bash
cd regime-trader
python -c "import yfinance, pandas_datareader, hmmlearn, alpaca; print('All imports OK')"
```

Run the test suite to confirm all logic is working:
```bash
python -m pytest tests/ -v
```

Expected output: **47 passed** in ~10 seconds. No internet required for tests.

---

## Step 2 — Backtest

### Command

```bash
python main.py --backtest
```

Or with a custom start date:
```bash
python main.py --backtest --start 2015-01-01
```

### What Happens

1. **Data fetch** (~30–60 seconds first run, instant after cache)
   - Downloads SPY + QQQ daily bars via yfinance
   - Downloads VIX + HY OAS from FRED
   - Validates all data (OHLC checks, gap detection)
   - Caches to `data/cache/`

2. **Walk-forward simulation** (~2–5 minutes)
   - Rolls a 252-bar IS window + 126-bar OOS window in 63-bar steps
   - Each IS window: trains fresh HMM (BIC sweeps n_states ∈ [3,7])
   - Each OOS window: forward α-recursion only (no look-ahead)
   - Allocation per bar: Moreira-Muir formula `w = min(0.20/ewma_vol, 1.25)`
   - No per-trade stops in backtest (stops are live-trading only)

3. **Stress period tests**
   - 2020 COVID crash (Feb–Apr 2020): checks HMM flagged HighVol
   - 2022 rate hike bear (Jan–Dec 2022): checks HMM flagged HighVol

4. **Benchmark comparison**
   - Buy-and-hold SPY
   - 200-day SMA filter (Faber 2007)

5. **Hansen SPA test** (p-value vs benchmarks)

6. **Logs written to** `logs/backtest_YYYYMMDD_HHMMSS.jsonl`

### Expected Output

```
=== Walk-Forward Backtest Summary ===
window  is_start    oos_end     n_states  sharpe  max_drawdown  cagr
0       2010-01-04  2011-09-30  3         ...     ...           ...
1       2010-04-01  2011-12-30  3         ...     ...           ...
...

Aggregate OOS metrics: {
  'cagr': 0.xx,
  'sharpe': x.xx,
  'sortino': x.xx,
  'calmar': x.xx,
  'max_drawdown': -0.xx,
  ...
}

SPA test: {
  'p_value': 0.xx,
  'interpretation': '...'
}

Stress [2020_covid_crash]: PASS ✓ | HighVol=xx% | AvgAlloc=0.xx
Stress [2022_rate_hike_bear]: PASS ✓ | HighVol=xx% | AvgAlloc=0.xx
```

### What "Good" Looks Like

| Metric | Target | Meaning |
|---|---|---|
| OOS Sharpe | > 1.0 | Risk-adjusted return better than buy-hold |
| Max drawdown | < 15% | Drawdown protection working |
| 2020 HighVol% | > 50% | HMM detected COVID crash |
| 2022 HighVol% | > 50% | HMM detected rate hike bear |
| SPA p-value | < 0.05 | Genuine predictive ability (not luck) |
| BIC n_states | 2–4 (stable) | Consistent regime structure found |

### What "Bad" Looks Like — and What to Do

| Problem | Likely Cause | Action |
|---|---|---|
| Sharpe < 0.5 consistently | HMM not detecting regimes | Check stress test results; consider Student-t emission |
| 2020/2022 not flagged HighVol | Feature normalization issue or wrong FRED series | Check feature matrix output; verify VIX + HY OAS aligned |
| BIC selects wildly different n per window | HMM unstable | Fix n_components to the most frequent BIC selection |
| IS Sharpe >> OOS Sharpe | Overfitting | Reduce parameters being tuned; check IS/OOS separation |
| SPA p > 0.10 consistently | No genuine edge | Architecture problem; revisit feature engineering |

### After Baseline — Calibration (design_docs/06)

Do NOT tune parameters until baseline run is complete. Then:

1. **BIC state count review**: what n_states does BIC select most often? Stable?
2. **Continuous vs discrete allocation**: baseline uses Moreira-Muir continuous. Compare vs discrete 95/65/35 buckets (set `use_continuous_formula: false` in settings.yaml).
3. **One-at-a-time sensitivity**: change ONE parameter, rerun, compare Sharpe. Never jointly optimize multiple parameters simultaneously (overfitting).
4. **Lock best parameters** in `config/settings.yaml` with comment justifying each value.

**Do NOT tune:** circuit breakers, risk_per_trade, correlation thresholds.

---

## Step 3 — Paper Trading

### Prerequisites

- Alpaca paper account + API keys in `.env`
- Baseline backtest completed (parameters calibrated)
- Market hours: NYSE 9:30am–4:00pm ET on trading days

### Command

```bash
python main.py --paper
```

### What Happens

1. Connects to Alpaca paper trading endpoint
2. Downloads historical warmup data for initial HMM training (same start date as settings)
3. Subscribes to live bar stream via Alpaca WebSocket
4. On each bar:
   - Computes features (log return, realized variance, VIX, HY OAS)
   - Runs HMM forward step → regime + confidence
   - Applies stability filters (persistence, flicker, confidence floor)
   - Generates signal (Moreira-Muir allocation → position size)
   - Risk manager gates (circuit breakers, correlation, position limits)
   - Submits bracket order (entry + stop + target as OCO)
   - Logs decision to `logs/trades_YYYYMMDD_HHMMSS.jsonl`
5. Retrains HMM every 20 bars automatically

### What to Monitor

- **Console output**: regime transitions, signal generation, order fills
- **Alpaca dashboard** (paper.alpaca.markets): position reconciliation source of truth
- **Log file**: every decision structured as JSON — grep for `regime`, `circuit_breaker`, `signal`
- **Circuit breaker trips**: logged at CRITICAL level, console and log

### Stop Loss / Take Profit Note

ATR-based stops (3.0×/2.5×/2.0× ATR) and targets (6×/5×/4× ATR) are **uncalibrated defaults**. These are the values from the video — not academically grounded. Paper trading for 3 months is the calibration instrument for stops. Observe:
- Are stops triggering before momentum plays out? → widen stops
- Are stops too wide (large losses before triggering)? → tighten stops
- Adjust one regime at a time

### Paper Trading Goals (3 months)

- Confirm regime detection in live conditions matches backtest quality
- Calibrate stop/target ATR multiples (backtest cannot do this)
- Verify circuit breakers trigger correctly in adverse conditions
- Confirm Alpaca bracket orders fill as expected
- Identify any data gaps between market open and first bar

---

## Step 4 — Live Trading

### Prerequisites

- 3-month paper trading completed
- All parameters locked in `config/settings.yaml` with justification comments
- Stops calibrated from paper trading observation
- Switch Alpaca keys to live account in `.env`:
  ```
  ALPACA_BASE_URL=https://api.alpaca.markets
  ```

### Command

```bash
python main.py --live
```

You will see:
```
============================================================
WARNING: LIVE TRADING MODE — REAL MONEY AT RISK
============================================================
Type 'CONFIRM' to proceed with live trading:
```

Type `CONFIRM` exactly. Any other input aborts.

---

## Configuration Reference

All parameters in `config/settings.yaml`. Key sections:

```yaml
# Hyperparameters — calibrate via walk-forward backtest
allocation:
  target_vol: 0.20          # Moreira-Muir target vol — test [0.15, 0.18, 0.20, 0.22, 0.25]
  low_vol.allocation: 0.95  # test [0.80, 0.85, 0.90, 0.95, 1.00]

stability:
  persistence_bars: 3       # test [2, 3, 4, 5]
  confidence_floor: 0.40    # test [0.30, 0.35, 0.40, 0.45, 0.50]

stops:                      # calibrate via paper trading ONLY
  low_vol.stop_atr: 3.0     # test [2.0, 2.5, 3.0, 3.5, 4.0]

# Risk policy — DO NOT tune on historical data
risk.circuit_breakers:
  daily_loss_warn: 0.02     # fixed
  max_drawdown_stop: 0.10   # fixed
```

Full parameter list with calibration ranges: `design_docs/06_empirical_testing_plan.md`

---

## File Structure

```
regime-trader/
├── config/
│   ├── settings.yaml          # All parameters (annotated)
│   └── credentials.yaml.example
├── core/
│   ├── hmm_engine.py          # HMM training, forward α-recursion, stability filters
│   ├── regime_strategies.py   # LowVol/MidVol/HighVol/Uncertainty strategies
│   ├── risk_manager.py        # Circuit breakers, correlation gates, Kelly sizing
│   └── signal_generator.py   # Signal + FlatSignal generation
├── broker/
│   ├── alpaca_client.py       # Alpaca SDK wrapper
│   ├── order_executor.py      # Bracket order submission
│   └── position_tracker.py   # Position reconciliation
├── data/
│   ├── market_data.py         # DataManager, yfinance + FRED sources, validation
│   └── feature_engineering.py # Log returns, realized variance, Z-score normalization
├── backtest/
│   ├── backtester.py          # Walk-forward engine (252/126/63 IS/OOS/step)
│   ├── performance.py         # Sharpe, Sortino, Calmar, MaxDD, Hansen SPA test
│   └── stress_test.py         # 2020 + 2022 crisis validation
├── monitoring/
│   ├── logger.py              # Structured JSON logging
│   ├── dashboard.py           # Matplotlib live dashboard
│   └── alerts.py              # Circuit breaker + broker error alerts
├── tests/                     # 47 tests, all passing
├── logs/                      # Created at runtime
├── data/cache/                # Created at runtime
├── main.py                    # Entry point (--backtest / --paper / --live)
├── requirements.txt
└── .env.example
```

---

## Design Documents

All research, decisions, and calibration protocols:

| File | Contents |
|---|---|
| `design_docs/00_session_overview.md` | Master index and session narrative |
| `design_docs/01_project_architecture.md` | Complete system spec, settings.yaml schema |
| `design_docs/02_gap_analysis.md` | Research gaps, two rounds of research, final status |
| `design_docs/03_research_hmm_engine.md` | HMM papers, forward algorithm math, feature list |
| `design_docs/04_research_allocation_signals.md` | Moreira-Muir, always-long rationale, validation |
| `design_docs/05_data_sources.md` | Data source comparison, FRED series, validation code |
| `design_docs/06_empirical_testing_plan.md` | Calibration protocol, hyperparameter ranges, guardrails |
| `design_docs/07_video_comparison.md` | Our build vs. video, corrections, honest gaps |

---

## What This System Is NOT

- **Not an LLM**: no GPT, no language model, no neural network. The HMM is classical statistics.
- **Not a guaranteed profit system**: backtest targets (Sharpe > 1.0) are validation thresholds, not promises.
- **Not calibrated**: ATR stop multiples, exact allocation buckets, Kelly fractions are starting defaults from the video. They require empirical calibration via your own walk-forward run and paper trading. See `design_docs/06_empirical_testing_plan.md`.
- **Not production-ready for live trading** until: backtest targets met + 3-month paper trading completed + parameters locked.
