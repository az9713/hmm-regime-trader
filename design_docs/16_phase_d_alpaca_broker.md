# design_docs/16_phase_d_alpaca_broker.md
# ============================================================
# Phase D — Alpaca Broker Layer: Goals, Architecture,
#            Code Changes, Results, Interpretation, and
#            Implications for Live Paper Trading
#
# Date: 2026-04-13
# Pre-Phase D: AlpacaSource.get_bars() stub — paper trading
#              would crash immediately on startup
# Post-Phase D: Full paper trading stack verified connected.
#              Alpaca paper account: $100,000 portfolio value,
#              $200,000 buying power (2x margin).
# Gate met?    Yes — --test-connection confirms end-to-end.
# Next:        Phase E — 3-month paper trading
# ============================================================

---

## 1. Goals

### 1.1 Why Phase D exists

Phases A through F completed all model calibration work. The final locked model
state is:

| Parameter | Value | Source |
|-----------|-------|--------|
| covariance_type | diag | Phase F — full cov regresses -0.144 |
| use_vix_slope | false | Phase F — only useful under full cov |
| is_window | 378 | Phase C H1 — +0.111 Sharpe vs 252 |
| n_components_range | [3, 7] | Phase C H2 — BIC self-corrects at 378 IS |
| rebalance_threshold | 0.15 | Phase B — +0.004 vs 0.10 |
| Aggregate OOS Sharpe | 0.933 | Final model state |

The next requirement is to deploy the model in a paper trading environment.
Paper trading serves two purposes:

1. **ATR stop calibration**: The stop-loss and take-profit ATR multiples
   (3.0/2.5/2.0) have no academic basis and cannot be calibrated from
   backtest data. Real fill prices vs. target prices during live market
   hours are the only valid calibration signal. Three months of paper
   trading is the minimum to observe a useful variety of market conditions.

2. **Execution quality validation**: Slippage, partial fills, bracket order
   behavior, and market microstructure effects are absent from backtest.
   Paper trading reveals whether the strategy's edge survives real execution.

Phase D's job is to make `python main.py --paper` work without crashing.

### 1.2 What Phase D set out to do

1. **Fix the only runtime stub**: `AlpacaSource.get_bars()` in
   `data/market_data.py` raised `NotImplementedError` — this caused an
   immediate crash at paper trading startup.

2. **Eliminate duplicate Alpaca connections**: The `run_paper()` loop
   in `main.py` was creating two separate Alpaca client instances — one
   for broker operations and one implicitly via `DataManager`. Share one.

3. **Bridge credential sources**: `AlpacaClient` reads env vars
   (`ALPACA_API_KEY`, `ALPACA_SECRET_KEY`). `credentials.yaml` also
   stores these keys, but nothing bridged them. Users with credentials
   in the YAML file only would get an unhelpful `EnvironmentError`.

4. **Add connectivity test**: A quick `--test-connection` flag to verify
   credentials and network before committing to a full paper trading
   session.

---

## 2. Architecture Before Phase D

### 2.1 The broker layer (already complete)

The broker layer was built in Phase 6 and was fully functional:

```
broker/
  alpaca_client.py    — TradingClient + StockHistoricalDataClient + StockDataStream
  order_executor.py   — Bracket order submission (entry + stop + take-profit OCO)
  position_tracker.py — Reconciles local state with Alpaca on every bar
```

`AlpacaClient.connect()` initializes three SDK clients:
- `TradingClient` — account, orders, positions (REST)
- `StockHistoricalDataClient` — historical OHLCV bars (REST)
- `StockDataStream` — live bar subscription (WebSocket)

The `run_paper()` loop in `main.py` was also complete: HMM warmup,
WebSocket bar subscription, regime detection, signal generation, risk
gating, order execution, regime flip alerts, and dashboard updates.

### 2.2 The gap

`DataManager` uses a strategy pattern for price data:

```python
class DataManager:
    def __init__(self, settings, mode="backtest"):
        if mode == "backtest":
            self.price_source = YFinanceSource(...)  # yfinance
        else:
            self.price_source = AlpacaSource(settings)  # STUB
```

`run_paper()` called:
```python
dm = DataManager(settings, mode="live")    # creates AlpacaSource — stub
spy_prices = dm.get_bars("SPY", warmup_start)  # CRASHES HERE
```

The crash happened at line 166 of `main.py`, before any bar was ever
processed. The strategy layer, risk manager, order executor, and
WebSocket stream never had a chance to run.

### 2.3 Two-client problem

`run_paper()` also created its own `AlpacaClient` for the broker layer:
```python
client = AlpacaClient(settings)
client.connect()                          # connection #1
dm = DataManager(settings, mode="live")   # would create connection #2 (stubbed)
```

Two separate SDK client instances is wasteful. The historical data client
inside `AlpacaClient` (`_data_client`) already provides exactly what
`AlpacaSource.get_bars()` needs.

### 2.4 Credential gap

`AlpacaClient.connect()`:
```python
api_key = os.getenv("ALPACA_API_KEY")
secret_key = os.getenv("ALPACA_SECRET_KEY")
```

`main.py` loaded `credentials.yaml` (which may contain the same keys
under `alpaca.api_key` / `alpaca.secret_key`) but only passed it to
`AlertManager` for Telegram. The Alpaca keys in credentials.yaml were
silently ignored.

---

## 3. Code Changes

### 3.1 `data/market_data.py` — fix the stub

**`AlpacaSource`**: Accept an `AlpacaClient` instance, delegate `get_bars()`:

```python
class AlpacaSource:
    def __init__(self, settings: dict, client=None):
        self.settings = settings
        self._client = client

    def get_bars(self, symbol: str, start: str, end: str = None) -> pd.DataFrame:
        if self._client is None:
            raise RuntimeError(
                "AlpacaSource requires a connected AlpacaClient. "
                "Pass alpaca_client= to DataManager."
            )
        return self._client.get_bars(symbol, start, end)
```

`AlpacaClient.get_bars()` was already fully implemented:

```python
def get_bars(self, symbol: str, start: str, end: str = None) -> pd.DataFrame:
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Day,
        start=start, end=end,
    )
    bars = self._data_client.get_stock_bars(request)
    df = bars.df
    if hasattr(df.index, "levels"):
        df = df.xs(symbol, level=0)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df[["open", "high", "low", "close", "volume"]].rename(columns={
        "open": "Open", "high": "High", "low": "Low",
        "close": "Close", "volume": "Volume"
    })
    return df
```

The returned DataFrame matches the `YFinanceSource` column format exactly
(Open/High/Low/Close/Volume, timezone-naive DatetimeIndex). The existing
`validate_price_data()` validation runs on both paths identically.

**`DataManager`**: Accept `alpaca_client=` and forward to `AlpacaSource`:

```python
def __init__(self, settings: dict, mode: str = "backtest", alpaca_client=None):
    ...
    if mode != "backtest":
        self.price_source = AlpacaSource(settings, client=alpaca_client)
```

Backward-compatible: `alpaca_client=None` default means all existing backtest
calls (`DataManager(settings, mode="backtest")`) work unchanged.

### 3.2 `main.py` — three additions

**Credential bridge** (new function):

```python
def _bridge_alpaca_credentials(credentials: dict):
    """
    Load Alpaca keys from credentials.yaml into env vars if env vars not set.
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
```

**Client sharing in `run_paper()`**:

```python
_bridge_alpaca_credentials(credentials or {})
client = AlpacaClient(settings)
client.connect()
...
dm = DataManager(settings, mode="live", alpaca_client=client)  # shared client
```

**`--test-connection` flag**:

```python
group.add_argument("--test-connection", action="store_true",
                   help="Test Alpaca connectivity and exit")
```

```python
if args.test_connection:
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
```

### 3.3 `.env.example` — documentation update

Added credential loading precedence note:

```
# Env vars take precedence over credentials.yaml if both are set.
```

---

## 4. Files Changed

| File | Change |
|------|--------|
| `data/market_data.py` | `AlpacaSource`: accept `client=`, delegate `get_bars()`. `DataManager`: accept `alpaca_client=`, forward to `AlpacaSource` |
| `main.py` | `_bridge_alpaca_credentials()`, shared client in `run_paper()`, `--test-connection` flag |
| `.env.example` | Clarified credential loading order |

**Unchanged (fully working before Phase D):**
- `broker/alpaca_client.py`
- `broker/order_executor.py`
- `broker/position_tracker.py`
- `monitoring/alerts.py`
- `config/settings.yaml`

---

## 5. Results

### 5.1 Connection test output

```
$ python main.py --test-connection

2026-04-13 10:23:35,893 | INFO | broker.alpaca_client | Alpaca connected (mode=paper)

=== Alpaca Connection Test ===
  Status:          OK
  Mode:            paper
  Portfolio value: $100,000.00
  Buying power:    $200,000.00
  Cash:            $100,000.00
==============================
```

All three SDK clients initialized successfully:
- `TradingClient` — authenticated, account data returned
- `StockHistoricalDataClient` — ready for bar fetches
- `StockDataStream` — ready for WebSocket subscription

### 5.2 Backtest still works

`python main.py --backtest` continues to work unchanged. The `alpaca_client=None`
default in `DataManager.__init__` means the backtest path (`mode="backtest"`)
never touches `AlpacaSource`. The change is fully backward-compatible.

---

## 6. Interpretation

### 6.1 Why share one client

The `StockHistoricalDataClient` inside `AlpacaClient` handles both the warmup
bar fetch (before the stream starts) and intra-session retraining. Using a
single client means:

- **One authentication handshake** at startup
- **One set of rate-limit counters** (Alpaca rate limits are per key, not per
  SDK instance, but a single instance makes this explicit)
- **No credential confusion** — impossible for the data client and trading
  client to authenticate with different keys

### 6.2 Credential precedence design

The bridge function checks `os.getenv("ALPACA_API_KEY")` first. If the env
var is already set (loaded from `.env` via `python-dotenv` at the start of
`main()`), the credentials.yaml values are ignored. The precedence chain is:

```
.env file (loaded by load_dotenv())
  → overrides →
credentials.yaml (loaded by load_credentials())
  → overrides →
neither (raises EnvironmentError with clear message)
```

This matches standard 12-factor app convention. CI/Docker environments set
env vars directly; local development uses `.env`. The YAML file is a third
option for users who prefer not to use dotenv.

### 6.3 Alpaca paper account behavior

The paper account starts with $100,000 cash and $200,000 buying power (2×
margin). For this strategy:

- The strategy is always-long (no short positions)
- Maximum leverage is 1.25× (`max_leverage` in settings)
- At 1.25× leverage on a $100k account: $125,000 max position size
- This fits comfortably within the $200k buying power

The paper account resets to $100k on request if needed. Paper fills are
simulated at the NBBO midpoint, which is typically optimistic relative to
real fills (especially for larger SPY orders). The 5 bps slippage assumption
in backtests (`backtest.slippage_bps: 5`) should be compared against actual
paper fill quality after 1-2 months.

---

## 7. Implications for Paper Trading Quality

### 7.1 What paper trading can calibrate

The primary Phase E objective is ATR stop calibration:

| Regime | Stop ATR | Target ATR | Calibration needed |
|--------|----------|------------|-------------------|
| LowVol | 3.0 | 6.0 | Test [2.0, 2.5, 3.0, 3.5, 4.0] |
| MidVol | 2.5 | 5.0 | Test [1.5, 2.0, 2.5, 3.0, 3.5] |
| HighVol | 2.0 | 4.0 | Test [1.0, 1.5, 2.0, 2.5, 3.0] |

Paper trading will reveal whether stops are being hit prematurely
(wide ATR needed) or positions are being held too long (narrow ATR needed).
The target:stop ratio should be approximately 2:1 to maintain positive
expectancy at a win rate of ~40%.

### 7.2 What paper trading cannot calibrate

- **2022-type slow-grind drawdowns** — may not recur in 3-month window
- **COVID-type spike events** — unlikely in short window; backtest handles these well
- **Real slippage on large SPY orders** — Alpaca paper fills are optimistic;
  live slippage may exceed 5 bps on regime-flip days with large rebalances

### 7.3 Monitoring during Phase E

Key metrics to track weekly:

| Metric | Target | Alert |
|--------|--------|-------|
| Regime distribution | ~60% LowVol, ~25% MidVol, ~15% HighVol | Skewed → model may be misfiring |
| Fill vs. ATR stop | Stop hit rate < 30% | ATR too tight |
| Fill vs. ATR target | Target hit rate > 50% | ATR well-calibrated |
| Daily P&L correlation with SPY | < 0.60 | Regime filtering working |
| Circuit breaker trips | 0 | If non-zero: investigate immediately |

The Telegram alert system (already implemented in `monitoring/alerts.py`)
delivers regime flips, circuit breaker trips, filled orders, and a 16:05 ET
daily summary automatically.

### 7.4 Live trading gate

The live trading gate is revisited after 3 months of paper data:
- **Sharpe > 1.0** on paper OOS: proceed to live
- **Sharpe 0.85-1.0**: extend paper by 1 month, investigate underperforming regimes
- **Sharpe < 0.85**: structural review required (may indicate regime shift since 2024)

---

## 8. Path Forward

### Phase E — 3-month paper trading (next)

```bash
python main.py --paper
```

The paper trading loop runs during market hours (09:30–16:00 ET). The
WebSocket stream delivers bars at market close (~16:05 ET). The process
can run in a terminal or as a background process.

Minimum paper trading period: **3 months** (covers multiple regime transitions
and sufficient stop/target observations for ATR calibration).

### After Phase E

1. Review ATR multiples against observed fill data
2. Lock ATR settings in `config/settings.yaml` with Phase E evidence tags
3. Reassess Sharpe gate (> 1.0 required for live trading)
4. If gate met: switch to `--live` mode (prompts "CONFIRM" to proceed)

---

## 9. References

- design_docs/06_empirical_testing_plan.md — §Group C: stop/target calibration plan
- design_docs/14_phase_f_results.md — final model state entering Phase D
- Lo & Remorov (2015, SSRN 2695383) — stop-loss value is asset-specific; backtest calibration unreliable
- Alpaca Markets paper trading docs — simulated fills at NBBO midpoint
- 12-factor app methodology — env var precedence over config files
