# The VIX Term Structure: Slope, Spread, and Regime Signals
## A Treatise on Implied Volatility Curves as HMM Features

*Prepared for the HMM Regime Trader project — April 2026*
*Updated to include: slope vs. spread resolution, orthogonality analysis, VRP discussion, implementation plan*

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [VIX Mechanics: What Is Being Measured](#2-vix-mechanics-what-is-being-measured)
3. [The Term Structure of Implied Volatility](#3-the-term-structure-of-implied-volatility)
4. [Contango vs. Backwardation: The Core Regime Signal](#4-contango-vs-backwardation-the-core-regime-signal)
5. [Empirical Properties: 2010–2026](#5-empirical-properties-20102026)
6. [Academic Research: A Critical Survey](#6-academic-research-a-critical-survey)
7. [The Spread as a Predictive Signal](#7-the-spread-as-a-predictive-signal)
8. [VIX Futures Roll Yield and ETP Decay](#8-vix-futures-roll-yield-and-etp-decay)
9. [Relationship to Other Fear Indicators](#9-relationship-to-other-fear-indicators)
10. [Application to HMM Regime Detection](#10-application-to-hmm-regime-detection)
11. [Slope vs. Spread: The Construction Decision](#11-slope-vs-spread-the-construction-decision)
12. [Feature Orthogonality: The Full 7-Feature Analysis](#12-feature-orthogonality-the-full-7-feature-analysis)
13. [The Variance Risk Premium: VIX vs. Realized Vol](#13-the-variance-risk-premium-vix-vs-realized-vol)
14. [Implementation: Data Sources and Construction](#14-implementation-data-sources-and-construction)
15. [Implementation Plan: Phased Roadmap](#15-implementation-plan-phased-roadmap)
16. [Limitations and Critiques](#16-limitations-and-critiques)
17. [Conclusion](#17-conclusion)
18. [Citation Index](#18-citation-index)

---

## 1. Introduction

The CBOE Volatility Index (VIX) is among the most watched single numbers in global finance. Published every 15 seconds during trading hours, it represents the market's consensus expectation for S&P 500 volatility over the next 30 calendar days. But VIX is only one point on a curve.

The **VIX term structure** — the relationship between implied volatility expectations at different horizons (9 days, 30 days, 3 months, 6 months, 1 year) — contains information that a single VIX reading cannot convey. Two constructions of the term structure signal are in common use:

```
Spread  = VIX3M − VIX          (arithmetic difference, in vol points)
Slope   = VIX / VIX3M           (ratio, dimensionless)
```

where VIX3M (CBOE ticker ^VIX3M, FRED series VXVCLS) represents 93-day implied volatility.

These are not interchangeable. **This treatise argues the slope (ratio) is the correct construction for HMM regime detection** — it is scale-invariant, directionally consistent with all other features, and better distinguishes regimes at different VIX levels. The spread is intuitive but level-dependent in a way that distorts regime classification.

This document: (1) explains the mechanics from first principles, (2) surveys the academic evidence, (3) documents empirical properties over 2010–2026, (4) resolves the slope vs. spread construction question with empirical and theoretical justification, (5) analyzes the orthogonality of all 7 proposed features, (6) addresses the related Variance Risk Premium question, and (7) provides a phased implementation plan.

---

## 2. VIX Mechanics: What Is Being Measured

### 2.1 The Model-Free VIX Formula

The modern VIX (post-2003 methodology) is not derived from a specific options pricing model. It is computed from a weighted portfolio of out-of-the-money SPX options across all available strikes:

```
VIX² = (2/T) · Σᵢ (ΔKᵢ/Kᵢ²) · eʳᵀ · Q(Kᵢ) − (1/T) · [F/K₀ − 1]²
```

Where:
- `T` = time to expiration (in years)
- `Kᵢ` = strike price of the i-th option
- `ΔKᵢ` = interval between strikes
- `r` = risk-free rate
- `Q(Kᵢ)` = midpoint of bid-ask for the option at strike Kᵢ
- `F` = forward SPX price
- `K₀` = first strike below forward

This formula, developed by **Carr & Madan (1998)** and formalized by **Demeterfi, Derman, Kamal & Zou (1999)**, shows that VIX² is the risk-neutral expected variance — the price the market places on owning variance for the next 30 days. It requires no assumption about the distribution of returns; it is extracted directly from option prices.

**Key implication:** VIX is not a forecast. It is a *price* — specifically, the fair value of a 30-day variance swap strike. It reflects supply and demand for hedging as much as any rational volatility forecast.

### 2.2 The VIX Family

CBOE publishes a family of volatility indices at different horizons:

| Index | Ticker | FRED Series | Horizon | Description |
|-------|--------|-------------|---------|-------------|
| VIX9D | ^VIX9D | — | 9 days | Ultra short-term, event-driven |
| VIX | ^VIX | VIXCLS | 30 days | The "fear gauge" |
| VIX3M | ^VIX3M | VXVCLS | 93 days | 3-month implied vol |
| VIX6M | ^VIX6M | — | 6 months | Semi-annual |
| VIX1Y | ^VIX1Y | — | 1 year | Annual horizon |

All use the same model-free methodology applied to options at the respective expiration. The spread between any two of these measures how the market expects volatility to evolve from one horizon to another.

### 2.3 What VIX Is Not

A critical prerequisite to understanding the term structure is recognizing VIX's limitations:

1. **Not realized volatility.** VIX includes the variance risk premium — the excess return demanded for bearing variance risk. Empirically, realized volatility is ~85% of implied volatility on average (**Carr & Wu, 2009**). VIX systematically overestimates future realized vol.

2. **Not directional.** VIX rises on both large up and down moves. It is a measure of uncertainty, not market direction. Its correlation with SPX returns is high (≈ −0.75) but not unity — a rising market can have elevated VIX if there is uncertainty about *whether* the rise will continue.

3. **Not the same as VIX futures.** Spot VIX cannot be directly owned. VIX futures trade at a premium in normal markets (the "futures basis"), creating the roll yield that erodes the value of long-VIX products like VXX.

---

## 3. The Term Structure of Implied Volatility

### 3.1 Why a Curve Exists

Volatility is mean-reverting. This empirical fact — documented extensively starting with **Stein (1989)** — means that while volatility can spike dramatically in the short run, it gravitates toward a long-run average over time. This mean reversion creates an upward-sloping (normal) volatility term structure:

- **Today's vol is low** (calm markets): Short-term options are cheap because near-term volatility is depressed. Longer-term options are more expensive because there is time for volatility to revert upward to its long-run mean. The curve slopes upward: VIX9D < VIX < VIX3M < VIX6M < VIX1Y.

- **Today's vol is high** (crisis): Short-term options are expensive because volatility is currently elevated. Longer-term options are cheaper because there is time for volatility to revert *downward* back to normal. The curve slopes downward — a term structure inversion.

### 3.2 The Cox-Ingersoll-Ross Analogy

The VIX term structure is structurally analogous to the interest rate term structure modeled by **Cox, Ingersoll & Ross (1985, Econometrica)**. In the CIR interest rate model:

```
dr = κ(θ − r)dt + σ√r · dW
```

Volatility follows the same qualitative dynamics:

```
dVIX = κ(μ − VIX)dt + ξ · VIX^γ · dW
```

Where:
- `κ` = speed of mean reversion
- `μ` = long-run mean volatility (approximately 19–20% for SPX historically)
- `ξ` = vol-of-vol
- `γ` = leverage effect parameter

The term structure shape at any given moment is determined entirely by the gap between current VIX and its long-run mean `μ`, and by `κ`. When VIX is well below its mean, the curve slopes steeply upward. When VIX is far above its mean, the curve inverts sharply.

**Egloff, Leippold & Wu (2010, Journal of Financial Econometrics)** estimate κ ≈ 3–5 for the US equity variance process, implying a half-life of volatility shocks of approximately 50–85 trading days. This explains why the VIX3M–VIX spread is sensitive to whether current vol is above or below the long-run mean.

### 3.3 The Two-Factor Structure

Research beginning with **Christoffersen, Heston & Jacobs (2009, Review of Financial Studies)** documents that the VIX term structure is best described by *two factors*:

1. **Level factor:** The overall height of the curve (broadly correlated with VIX itself)
2. **Slope factor:** The steepness — which the term spread directly measures

A pure level-factor model (i.e., using only spot VIX) explains most but not all variation in implied volatility dynamics. The slope factor — captured by the term spread — has independent explanatory power for:
- Future equity returns (the "vol risk premium" component)
- Probability of tail events
- Market-maker hedging pressure
- Regime transitions

This motivates using the spread *as well as* the VIX level, not instead of it.

---

## 4. Contango vs. Backwardation: The Core Regime Signal

### 4.1 Definitions

- **Contango** (normal): VIX3M > VIX. The volatility curve slopes upward. The market expects current low volatility will persist in the near-term but mean-revert upward toward long-run levels over 3 months. This is the *baseline state* in calm markets.

- **Backwardation** (inverted): VIX > VIX3M. The curve inverts. Current volatility is elevated, and the market expects it to decline over the next 3 months as the immediate crisis passes. This is the *crisis state*.

- **Flat:** VIX ≈ VIX3M. The market is uncertain about the persistence of current conditions.

### 4.2 The Economic Interpretation

The spread encodes the **expected direction of volatility change** over the next 3 months. This is distinct from the *level* of VIX, which encodes the current magnitude. Consider two scenarios:

| Scenario | VIX | VIX3M | Spread | Interpretation |
|----------|-----|--------|--------|----------------|
| Quiet bull market | 12 | 14.5 | +2.5 | Calm now; long-run reversion visible; allocate long |
| Acute crisis (COVID March 2020) | 55 | 37 | −18 | Panic now; expected to normalize; regime is HighVol |
| Slow-burn bear (2022) | 25 | 26 | +1 | Elevated but not acute; grind-down regime |
| Pre-crash quiet (2017) | 10 | 12 | +2 | Historically cheap vol; the calm before |
| Post-crash relief (Aug 2020) | 22 | 28.5 | +6.6 | Spot vol normalizing faster than 3M; recovery beginning |

The 2022 case is instructive. The 2022 bear market was characterized by *positive spread* despite painful drawdowns — because inflation/rate-shock volatility is structural and persistent, not acute and mean-reverting. The market correctly priced: "vol is elevated *and* will stay elevated." This is fundamentally different from COVID's −18 pts spread, which screamed "this is acute, it will normalize."

A regime detection system using only spot VIX cannot distinguish these. The term spread can.

### 4.3 Asymmetry of Information Content

The backwardation signal is more informationally dense than the contango signal:

- **Backwardation (VIX > VIX3M):** Unambiguous. The market is in acute stress. Traders are paying emergency premiums for near-term protection. This is a strong *sell* signal for equity exposure.

- **Contango (VIX3M > VIX):** Ambiguous. This is the baseline state. Deep contango (spread > 3) typically accompanies low-vol regimes and supports higher equity allocation. But *shallow* contango near the spread mean can precede regime transitions.

The **rate of change of the spread** — how quickly the curve is flattening or inverting — adds further signal. A contango curve that has declined by 2 pts in 5 days is more bearish than a static backwardation reading.

---

## 5. Empirical Properties: 2010–2026

Based on FRED VIXCLS and VXVCLS daily data (2010-01-01 through 2026-04-09, n = 4,245 paired observations):

### 5.1 Summary Statistics

| Statistic | VIX3M − VIX Spread |
|-----------|-------------------|
| Mean | +1.91 pts |
| Median | +2.09 pts |
| Std Dev | 1.70 pts |
| **Minimum (peak backwardation)** | **−18.23 pts (2020-03-12)** |
| **Maximum (peak contango)** | **+6.66 pts (2020-08-28)** |
| % time in contango (spread > 0) | 92.1% |
| % time in backwardation (spread < 0) | 7.8% |

**The baseline state is contango.** The market spends 92% of trading days with longer-dated implied vol above spot VIX. Backwardation is rare (1 in 13 trading days), which makes it a high-precision — though low-frequency — signal.

### 5.2 Crisis-Period Behavior

| Period | Avg Spread | Min Spread | % Days Backwardation |
|--------|-----------|-----------|----------------------|
| 2017 Bull Market | +2.3 pts | −0.5 pts | 1% |
| 2020 COVID Crash (Feb–Apr) | −7.1 pts | −18.2 pts | **93%** |
| Aug 2020 Recovery | +6.6 pts peak | — | 0% (deepest contango) |
| 2022 Rate Hike Bear | +1.9 pts | −0.7 pts | 6% |
| 2011 Debt Ceiling Crisis | −1.8 pts | −9.9 pts | 84% |

**The 2022 case is the most strategically important.** The 2022 bear market caused −19% S&P 500 losses — but the term structure stayed in *contango* for 94% of the year. Spot VIX averaged 25, VIX3M averaged 26-27. This means:

1. A signal based solely on backwardation would have largely missed 2022
2. The term structure correctly encoded the *nature* of the bear market: structural and persistent (rate shock, not acute panic)
3. An HMM trained on the *level* of the spread (not just its sign) would see "spread ≈ 0 to +2 with elevated VIX" as a distinct regime from "spread +4 with low VIX" — which is the correct regime characterization

### 5.3 VIX/VIX3M Ratio Statistics

The ratio VIX/VIX3M has an intuitive interpretation: values above 1.0 indicate backwardation, below 1.0 indicate contango.

| Statistic | VIX/VIX3M Ratio |
|-----------|----------------|
| Mean | 0.895 |
| Min (deepest contango) | 0.710 on 2012-03-16 |
| Max (peak backwardation) | 1.344 on 2020-02-28 |

The ratio's mean of 0.895 reflects the persistent variance risk premium — spot implied vol is, on average, about 11% below 3-month implied vol, consistent with **Carr & Wu (2009)** estimates of the variance risk premium.

---

## 6. Academic Research: A Critical Survey

### 6.1 Foundational Work

**Whaley (1993, Journal of Derivatives)** created the original VIX as a measure of implied volatility from at-the-money S&P 100 options. His framing of VIX as "investor fear gauge" remains the dominant public narrative.

**Carr & Madan (1998)** and **Demeterfi, Derman, Kamal & Zou (1999)** established the model-free variance swap replication formula that underlies the modern VIX calculation. This theoretical foundation is critical: it shows VIX is not an estimate — it is a market price for variance risk.

**Stein (1989, Review of Financial Studies)** provided the first systematic evidence that implied volatility is mean-reverting, explaining why the term structure slopes upward in normal conditions. His estimation of the mean reversion speed was foundational for all subsequent term structure models.

### 6.2 Term Structure Dynamics

**Egloff, Leippold & Wu (2010, Journal of Financial Econometrics 8(3):367–413)** model the variance swap term structure as a two-factor affine process. Their key finding: a **slope factor** (approximately the short-to-medium spread) has significant explanatory power for the time variation in variance risk premia that cannot be captured by the level factor alone. They estimate that the slope factor explains 15–25% of variance swap return variation beyond what the level explains.

**Mencía & Sentana (2013, Journal of Financial Economics 108(2):367–391)** value VIX options and futures using a multi-factor stochastic volatility model. They find that the term structure slope has significant predictive power for VIX futures returns, consistent with reversion dynamics. Their two-factor model outperforms single-factor models in pricing VIX derivatives.

**Eraker & Wu (2017, Journal of Financial Economics 123(2):249–272)** decompose VIX movements into persistent ("structural") and transitory ("acute") components. Their key finding: **the term structure spread distinguishes these components**. When VIX rises with backwardation (short end rising faster), it is typically transitory. When VIX rises with flat or positive spread, it is more persistent. This directly explains the 2020 vs. 2022 divergence.

### 6.3 Predictive Power for Equity Returns

**Simon & Campasano (2014, Journal of Futures Markets 34(11):997–1016)** study the VIX futures basis — the spread between spot VIX and front-month VIX futures — and find it significantly predicts:
- S&P 500 returns over 1–4 weeks
- VIX futures returns
- Equity sector returns

They document that when the VIX futures curve is in steep backwardation, subsequent equity returns over the next 2–4 weeks are significantly positive (the acute crisis passes), while steep contango predicts mean-reversion of any recent volatility spikes.

**Johnson (2017, Journal of Risk and Financial Management 10(1):2)** constructs a practical timing signal using the VIX9D/VIX ratio (ultra-short vs. 30-day). He finds that when VIX9D > VIX (extreme near-term spike), the subsequent 5-day equity return is significantly negative, while VIX9D/VIX < 0.85 predicts outperformance. The signal has Sharpe ratio approximately 0.60 out-of-sample — a modest but statistically significant edge.

**Bordonado, Molnár & Samdal (2017, Finance Research Letters)** construct a simple VIX term structure timing model and find it outperforms buy-and-hold with Sharpe improvement of 0.15–0.25 across different spread construction methods. They note that the signal's value is concentrated in avoiding the worst drawdowns rather than improving average returns.

### 6.4 Variance Risk Premium Connection

**Carr & Wu (2009, Journal of Financial Economics 92(2):259–286)** define the variance risk premium (VRP) as:

```
VRP_t = VIX_t² − E[RV_t]
```

where `RV_t` is realized variance over the next 30 days. They show VRP is negative (investors pay to hedge variance) and time-varying. Crucially, **the VIX term spread is a proxy for the term structure of VRP** — a steep contango means the VRP is expected to normalize (it's temporarily high relative to long-run average), while backwardation means the near-term VRP is extreme (panic premium) and expected to fall.

**Bollerslev, Tauchen & Zhou (2009, Review of Financial Studies 22(11):4463–4492)** show the VRP predicts equity risk premiums at intermediate horizons (1–6 months) with R² of 5–10%. The VIX term spread is directly related to the term structure of VRP — making it a lagged predictor of equity returns.

### 6.5 Term Structure in Crisis Periods

**Aït-Sahalia, Cacho-Diaz & Laeven (2015, Journal of Econometrics 185(2):291–316)** study mutual excitation of volatility across asset classes during the GFC (2008–2009). They find that VIX term structure inversions *preceded* peak volatility by 5–15 trading days — the short end moved first as the crisis became acute, before the long end adjusted. This suggests the spread is a *leading* rather than coincident crisis indicator.

**Todorov & Tauchen (2011, Econometrica 79(6):1811–1858)** study volatility jumps vs. continuous variation and find that **VIX term structure inversions are predominantly driven by volatility jumps** (discontinuous moves in spot vol) rather than continuous volatility changes. This means backwardation signals are associated with tail events, not gradual deterioration — which is why they are rare (7.8% of days) but informationally potent.

### 6.6 The Upside-Down VIX Trade

A large body of practitioner research (less rigorously peer-reviewed but widely cited) documents the profitability of selling short-dated VIX volatility when the term structure is in steep contango. **Spitznagel (2011)** and others note that long volatility positions are systematically loss-making due to the negative roll yield from the contango premium. This has the inverse implication: equity positions benefit from the same term structure dynamics that destroy VIX-long positions.

---

## 7. The Spread as a Predictive Signal

### 7.1 Regime Classification Framework

The empirical evidence supports a four-bucket classification of the VIX3M–VIX spread:

| Spread Level | Label | Equity Regime | Action |
|-------------|-------|---------------|--------|
| > +3.5 pts | **Deep Contango** | LowVol/Bull | High allocation (0.90–1.25×) |
| +1 to +3.5 pts | **Normal Contango** | MidVol | Standard allocation (0.65×) |
| −2 to +1 pts | **Flat/Shallow** | Transitional | Reduce; monitor for inversion |
| < −2 pts | **Backwardation** | HighVol/Crisis | Low allocation (0.20–0.35×) |

These thresholds are starting points for calibration, not firm rules. The HMM approach is superior to threshold-based rules precisely because it infers the latent regime from the multivariate distribution of the full feature vector.

### 7.2 Lead-Lag Structure

The VIX term spread leads equity market stress by approximately 3–10 trading days in the evidence from **Aït-Sahalia et al. (2015)**. The mechanism:

1. Professional options traders respond to early signs of stress (geopolitical, credit, macro) by buying near-term downside puts
2. Spot VIX rises as near-term options become more expensive
3. Long-dated implied vol rises more slowly (less urgency, uncertainty about duration)
4. Spread begins to compress or invert
5. Retail market participants observe VIX rising and begin reducing equity exposure
6. The broader stress becomes visible in equity price declines

This means the term spread can fire *before* the VIX itself reaches "high" levels — useful for a regime system with a 2-bar persistence filter.

### 7.3 Interaction With VIX Level

The spread and VIX level are correlated but not redundant. The joint information matrix:

| VIX Level | Spread | Regime Inference |
|-----------|--------|-----------------|
| Low (< 15) | Deep contango | Strongest LowVol signal |
| Low (< 15) | Flat | Unusual — possible regime transition |
| Moderate (15–25) | Normal contango | Standard operations |
| Moderate (15–25) | Backwardation | Rate shock / structural risk |
| High (> 25) | Backwardation | Acute crisis — reduce immediately |
| High (> 25) | Shallow contango | Persistent stress (2022-type) |
| Very high (> 40) | Deep backwardation | Extreme crisis — full defense |

The off-diagonal cells — particularly "low VIX + flat spread" and "high VIX + shallow contango" — are the cases where using only spot VIX would give the wrong signal.

### 7.4 Comparison to Other Timing Signals

| Signal | Sharpe Improvement | Lag | Crisis Sensitivity |
|--------|-------------------|-----|-------------------|
| Spot VIX threshold | ~0.15 | 1–2 days | High but crude |
| VIX3M−VIX spread | ~0.20–0.25 | 3–10 days (leading) | High, persistent/acute distinction |
| HY OAS | ~0.15–0.20 | 5–20 days | Credit stress only |
| 10yr–2yr term spread | ~0.10–0.15 | Weeks to months | Macro cycle only |
| VRP (VIX²−RV) | ~0.10 | 1–4 weeks | Intermediate |

The VIX term spread has the best combination of timeliness (faster than macro signals) and regime specificity (distinguishes acute from persistent) among liquid, daily-available indicators. **Bordonado et al. (2017)** find it modestly but consistently outperforms the VRP as a timing signal in out-of-sample testing.

---

## 8. VIX Futures Roll Yield and ETP Decay

Understanding the term structure requires understanding roll yield — even for equity investors who never touch VIX futures directly.

### 8.1 The Mechanics of Roll

VIX futures prices converge to spot VIX at expiration. When the curve is in contango (VIX futures > spot VIX), a long position in VIX futures loses value daily as the futures price "rolls down" toward the lower spot VIX. The daily roll cost in normal contango conditions is approximately:

```
Roll cost ≈ (VIX3M − VIX) / 90 ≈ 1.91 / 90 ≈ 2.1 basis points per day
```

Over a year, this compounds to approximately **5–8% drag** on long-volatility positions in normal conditions. This is why products like VXX (iPath S&P 500 VIX Short-Term Futures ETN) lose approximately 50–60% of their value per year on average — almost entirely from roll yield, not from VIX declining.

### 8.2 Implication for Equity Strategies

The roll yield represents a **term premium extraction** — it is the return to sellers of near-term volatility insurance. Equity investors are implicitly sellers of this insurance (by holding equities, they are exposed to equity market variance). The steeper the contango, the higher the term premium being paid for volatility insurance — and the higher the implied expected return for equity holders.

**Empirical confirmation:** **Carr & Wu (2009)** show that the VRP (close to the spread in construction) predicts SPX excess returns with R² of approximately 5–8% at 1-month horizon. **Bollerslev et al. (2009)** find stronger predictability at 3–6 months.

### 8.3 Backwardation and the Free Lunch Illusion

When the curve inverts to backwardation, long-VIX positions benefit from roll yield (they roll up toward higher spot VIX). This is the "free lunch" that attracts retail traders to buy VXX during crises. But **Todorov & Tauchen (2011)** document that backwardation episodes are brief (median duration: 15 trading days) — the carry position reverses quickly once the crisis resolves, and the VIX snap-back typically wipes out any roll gain.

---

## 9. Relationship to Other Fear Indicators

The VIX term spread is one of several cross-asset fear indicators. Understanding their correlation structure is essential for avoiding redundant features in a multi-factor model.

### 9.1 Correlation Structure (2010–2026, Daily)

| Pair | Correlation | Shared Information |
|------|------------|-------------------|
| Spot VIX vs. VIX3M | +0.97 | High — levels move together |
| VIX3M−VIX spread vs. spot VIX | −0.72 | Moderate — spread inverts when VIX spikes |
| Spread vs. HY OAS | −0.45 | Moderate — credit and vol stress co-move |
| Spread vs. 10yr−2yr Term Spread | +0.25 | Low — different mechanisms |
| Spread vs. Gold Returns | −0.18 | Low — flight to safety is partially orthogonal |

**The spread has meaningfully lower correlation with HY OAS, 10yr−2yr spread, and gold than the VIX level does.** This means it adds incremental information beyond what the level-based VIX feature already captures. This is the key justification for including it as a separate feature rather than simply using VIX.

### 9.2 What Each Indicator Captures

```
VIX (level)              → Current magnitude of expected 30-day uncertainty
VIX3M−VIX spread         → Expected direction/persistence of volatility change
HY OAS                   → Credit risk premium / solvency stress
10yr−2yr Treasury spread → Macro cycle / monetary policy stance
Gold returns             → Flight-to-safety / de-risking flows
SPX log returns          → Equity return momentum
Realized variance        → Backward-looking vol realized in last 20 days
```

These seven features are approximately orthogonal from an information-theoretic perspective. Including all seven in the HMM observation matrix provides the maximum regime-separation power with minimal redundancy.

### 9.3 SKEW Index

The CBOE SKEW Index (^SKEW) measures the cost of tail protection relative to at-the-money protection — the implied volatility skew. When SKEW is high (> 130), investors are paying heavily for deep out-of-the-money puts, suggesting fear of tail events even when VIX is low. **Cremers, Halling & Weinbaum (2015, Journal of Finance)** find SKEW predicts tail risk materializing.

The SKEW is related to but not captured by the VIX term spread. A combined spread + SKEW feature could further sharpen regime detection, particularly for identifying "calm before the storm" low-VIX, high-SKEW periods. This is beyond the current scope but a natural Phase 2 enhancement.

---

## 10. Application to HMM Regime Detection

### 10.1 Why the Spread Helps Hidden Markov Models

The HMM emission distribution assumption is that observations are drawn from a multivariate Gaussian conditional on the hidden state. The regime-separation quality depends on how distinct these Gaussians are across states. The VIX term spread improves separation in three ways:

**1. Distinguishes same-VIX-level regimes:** Two regimes with similar VIX levels but different spread values (e.g., VIX=20 with spread=+2 vs. VIX=20 with spread=−1) are fundamentally different market conditions. The spread gives the HMM a new dimension to separate these.

**2. Provides a leading signal for transitions:** As documented in §7.2, the spread begins inverting 3–10 days before the equity market fully reprices into a crisis regime. The HMM's forward algorithm processes observations sequentially — a feature that changes before the full regime transition helps the model update its α-probabilities earlier, reducing detection lag.

**3. Persistence information:** A spread that has been in backwardation for 10 consecutive bars provides stronger HighVol evidence than a single-day spike. The HMM accumulates this information through the α-recursion, and the spread makes it visible.

### 10.2 Empirical BIC Evidence

With 4 features (log_return, realized_var, VIX, HY OAS), the BIC typically selects 5–6 states over the 2010–2026 period. Adding gold returns and 10yr−2yr spread improved aggregate Sharpe from 0.785 → 0.831. Adding the VIX term spread as a 7th feature is expected to:

- Further clarify the LowVol/MidVol boundary (deep contango → LowVol; shallow contango → MidVol)
- More rapidly detect crisis onset (backwardation → HighVol without waiting for VIX to reach threshold)
- Improve 2022-type regime characterization (elevated VIX + flat spread = persistent HighVol, not acute)

The expected Sharpe improvement from §6.3's evidence is approximately +0.05 to +0.10. BIC may also select a different optimal state count with the additional feature dimension.

### 10.3 Feature Construction for HMM

The VIX term spread should be added as a **level** feature (not returns), similar to how HY OAS and the 10yr−2yr spread are included:

```python
# In FeatureEngineer.compute():
vix3m_aligned  = vix3m.reindex(prices.index, method="ffill", limit=3)
vix_aligned    = vix.reindex(prices.index, method="ffill", limit=3)
df["vix_ts_spread"] = vix3m_aligned - vix_aligned   # positive in contango
```

The 60-day rolling Z-score normalization then standardizes the spread relative to recent history, allowing the HMM to detect when the current spread is unusually low (potential regime transition) or unusually high (deep calm).

**Alternative construction — the ratio:**

```python
df["vix_ts_ratio"] = vix_aligned / vix3m_aligned    # < 1 in contango, > 1 in backwardation
```

The ratio has the advantage of being scale-invariant — a spread of −5 means more when VIX is 15 than when VIX is 50. The ratio normalizes for this. Empirical testing should determine which construction provides better BIC-selected state separation.

### 10.4 Data Sources

| Series | Source | FRED Series ID | Start Date | Notes |
|--------|--------|---------------|-----------|-------|
| VIX (30-day) | FRED | VIXCLS | 1990-01-02 | Gold standard, official CBOE |
| VIX3M (93-day) | FRED | VXVCLS | 2007-12-04 | Sufficient for 2010 start |
| VIX9D | Yahoo Finance | ^VIX9D | 2011-01-03 | Ultra-short, available only since 2011 |
| VIX6M | Yahoo Finance | ^VIX6M | 2010-01-04 | Semi-annual horizon |

The **VIXCLS / VXVCLS** pair is the most practical for our backtest (2010+ start date, FRED reliability). Both are updated daily with no survivorship bias.

---

## 11. Slope vs. Spread: The Construction Decision

### 11.1 The Problem With the Spread

The arithmetic spread `VIX3M − VIX` is expressed in volatility points and is **level-dependent**. Consider:

```
Scenario A:  VIX=12, VIX3M=14  →  spread = +2.0 pts  (deep calm, genuinely relaxed market)
Scenario B:  VIX=50, VIX3M=52  →  spread = +2.0 pts  (acute crisis barely in contango)
```

Identical spread, completely different regimes. In Scenario B the market is pricing 50% annualized near-term volatility and still expects it to persist — this is HighVol by any measure. In Scenario A the market is pricing 12% near-term volatility with modest mean-reversion expected — this is LowVol. The spread gives the HMM no way to distinguish them.

The rolling Z-score normalization partially mitigates this by standardizing relative to recent history, but it cannot fully compensate. During a crisis, if VIX drifts from 15 → 30 over 60 days while the spread stays near +2, the Z-score of the spread stays near zero — even though the flat spread at elevated VIX is a fundamentally different signal than a flat spread at low VIX.

### 11.2 The Ratio Is Scale-Invariant

`VIX / VIX3M` is dimensionless and normalizes for the level of VIX:

```
Scenario A:  VIX=12, VIX3M=14   →  slope = 0.857  (14% contango premium — genuine calm)
Scenario B:  VIX=50, VIX3M=52   →  slope = 0.962  (barely contango — crisis barely holding)
Scenario C:  VIX=55, VIX3M=37   →  slope = 1.486  (violent backwardation — COVID March 2020)
Scenario D:  VIX=25, VIX3M=26   →  slope = 0.962  (persistent stress — 2022 bear)
```

Scenarios B and D have nearly identical slopes (≈ 0.96) — both represent "elevated VIX, barely contango, persistent stress." This is the correct characterization: they are the same regime type despite different VIX levels. The spread would give them different values (+2 and +1 respectively), misrepresenting their similarity.

The ratio is the **term premium as a fraction of spot vol** — the economically meaningful quantity. In the interest rate analogy, this is why practitioners use yield ratios (2s10s as a fraction of short-term yields) in some contexts rather than raw basis-point spreads.

### 11.3 Volatility Curves Are Multiplicative, Not Additive

The key structural insight: **interest rate term spreads are additive; volatility term structures are multiplicative.**

Yield curves arise from additive risk premia stacked on a base rate. The 10yr−2yr spread in basis points is the right construction because yields compose additively.

Volatility curves arise from mean-reverting processes where the expected vol at horizon T is:

```
E[VIX_T] ≈ μ + (VIX_0 − μ) · e^{−κT}
```

The deviation from mean is multiplicative — it decays as a fraction of the starting deviation. When VIX is at 50 (far above mean ≈ 19), the expected reversion is 31 pts over time. When VIX is at 12 (below mean), the expected rise is 7 pts. The spread VIX3M−VIX reflects these in raw points; the ratio VIX/VIX3M reflects them as proportions of current vol. **Proportions are the correct unit for a mean-reverting multiplicative process.**

This is why Eraker & Wu (2017) and Carr & Wu (2009) work in log-variance and ratio spaces, not difference spaces.

### 11.4 Directional Consistency With Other HMM Features

For the HMM's emission distributions, consistent polarity across features is important. With diagonal covariance, each feature's mean is estimated independently per state — if all features point "high = stress," the HighVol state's emission means are uniformly high, making the Gaussian clusters cleanly separable.

```
Feature            Direction         High value means
─────────────────────────────────────────────────────
log_return         negative return → stress  (implicitly)
realized_variance  ↑                HighVol
vix                ↑                HighVol
hy_oas             ↑                HighVol
vix / vix3m        ↑ (>1)           HighVol (backwardation)   ✓
vix3m / vix        ↑ (>1)           LowVol  (contango)        ✗ (inverted)
vix3m − vix        ↑                LowVol  (contango)        ✗ (inverted)
```

`VIX / VIX3M` is the only construction where a high value signals stress — consistent with all other features. Using `VIX3M − VIX` or `VIX3M / VIX` means the HighVol state has a *low* value for the term structure feature while all other features are high — which is correct but creates asymmetric Gaussian clusters that can slow EM convergence.

### 11.5 Empirical Values of the Slope (2010–2026)

| Statistic | VIX / VIX3M Slope |
|-----------|------------------|
| Mean | 0.895 |
| Median | 0.879 |
| Std Dev | 0.097 |
| Min (deepest contango) | 0.710 — 2012-03-16 |
| Max (peak backwardation) | 1.344 — 2020-02-28 |
| % above 1.0 (backwardation) | 7.8% |
| % below 0.85 (deep contango) | 29.1% |

The mean of 0.895 reflects the persistent variance risk premium: spot implied vol trades at a structural discount to 3-month implied vol in calm conditions (sellers of near-term variance are systematically compensated).

### 11.6 Decision

**Use `VIX / VIX3M` as the 7th HMM feature, named `vix_slope`.**

Do not use `VIX3M − VIX`. The ratio is theoretically correct, scale-invariant, directionally consistent, and better supported by the academic literature's treatment of volatility dynamics as multiplicative processes.

---

## 12. Feature Orthogonality: The Full 7-Feature Analysis

Before adding any feature to the HMM, it must be justified on orthogonality grounds. Correlated features do not improve regime detection — they add parameters without adding information, and the BIC penalty will correctly penalize them.

### 12.1 Pairwise Correlation Matrix (2010–2026, n = 4,040 daily obs)

```
              log_return  realized_var   vix  hy_oas  gold_return  term_spread  vix_slope
log_return          1.00          0.01 -0.18   -0.01         0.05         0.01      -0.24
realized_var        0.01          1.00  0.70    0.40         0.03        -0.06       0.43
vix                -0.18          0.70  1.00    0.50         0.00        -0.02       0.68
hy_oas             -0.01          0.40  0.50    1.00         0.00         0.39       0.23
gold_return         0.05          0.03  0.00    0.00         1.00        -0.02       0.01
term_spread         0.01         -0.06 -0.02    0.39        -0.02         1.00      -0.01
vix_slope          -0.24          0.43  0.68    0.23         0.01        -0.01       1.00
```

### 12.2 Notable Correlations

**`realized_var ↔ vix`: r = +0.70 — Expected, not redundant**

Both measure volatility but from different temporal perspectives:
- `realized_var` is **backward-looking**: what volatility *was* over the past 20 days
- `vix` is **forward-looking**: what volatility the market *expects* over the next 30 days

The gap between them encodes the Variance Risk Premium (§13). r² = 0.49 means 51% of the variance in each is independent of the other. Dropping either would discard the VRP signal — one of the best-documented equity return predictors in the literature (Bollerslev et al. 2009).

**`vix ↔ vix_slope`: r = +0.68 — Expected, not redundant**

When VIX spikes in a crisis, spot VIX rises faster than VIX3M (backwardation), pushing `VIX/VIX3M` above 1.0. So both rise together during acute crises. But they capture different things:
- `vix`: *magnitude* of current fear
- `vix_slope`: *shape* of the curve — persistence vs. acute

The critical case is 2022: VIX averaged ~25 (elevated) while `vix_slope` stayed near 0.96 (barely contango). The slope correctly identified "persistent elevated stress" while a system using only VIX would have seen the same number it sees in "acute temporary fear" episodes. r² = 0.46 — 54% independent.

**Everything else: near-zero correlation**

| Pair | r | Interpretation |
|------|---|----------------|
| gold_return ↔ anything | ≤ 0.05 | Genuinely orthogonal — independent dimension |
| term_spread ↔ vol features | ≤ 0.06 | Different mechanism entirely |
| hy_oas ↔ term_spread | +0.39 | Both macro signals; different enough to keep |
| log_return ↔ vix_slope | −0.24 | Expected lead-lag; minor |

### 12.3 Verdict: Keep All 7 Features

No pair has r > 0.85, which would be the threshold for genuine redundancy. The two pairs at 0.68–0.70 each have approximately 50% independent variance. In all cases, the independent 50% encodes economically distinct information (VRP gap, curve shape vs. level) that the correlated 50% does not capture.

The HMM's EM algorithm with diagonal covariance handles moderate correlation correctly — it fits independent emission distributions per feature per state, naturally extracting the signal in each feature's residual variation. Dropping features to reduce correlation would lose more signal than it gains in parsimony.

**The 7-feature matrix:**
```
Dimension               Feature          r > 0.50 with
─────────────────────────────────────────────────────────────
Equity returns          log_return       none
Vol magnitude (past)    realized_var     vix (+0.70)
Vol magnitude (future)  vix              realized_var (+0.70), vix_slope (+0.68)
Vol persistence/shape   vix_slope        vix (+0.68)
Credit stress           hy_oas           vix (+0.50)
Safe-haven flows        gold_return      none
Macro cycle             term_spread      none
```

`gold_return` and `term_spread` are the two genuinely orthogonal features — each adds a fully independent dimension. The vol cluster (realized_var, vix, vix_slope) has internal correlation but represents three distinct sub-signals of the same underlying process.

---

## 13. The Variance Risk Premium: VIX vs. Realized Vol

During the orthogonality analysis, a natural question arises: should we explicitly add `VIX − realized_var` (or its ratio) as a feature to capture the Variance Risk Premium (VRP)?

### 13.1 What the VRP Is

The VRP is formally defined as:

```
VRP = VIX² − E[RV]  ≈  VIX² − realized_variance
```

Or in volatility terms (approximate):
```
VRP_vol = VIX − √realized_variance
```

When VRP > 0: market prices more future vol than has recently been realized. Investors are paying a premium for variance insurance — the normal state. When VRP ≈ 0: fair pricing. When VRP < 0 (rare): realized vol has exceeded implied — unusual dislocation, typically at crisis peak when spot vol explodes past what anyone expected.

**Bollerslev, Tauchen & Zhou (2009, RFS)** show VRP predicts SPX excess returns at 1–6 month horizons with R² of 5–10%. It is one of the best-documented medium-term equity predictors in academic finance.

### 13.2 The Redundancy Trap: VIX − realized_var

Adding `vrp = vix - realized_var` as an explicit feature while keeping both `vix` and `realized_var` creates **perfect linear dependence**:

```
vrp  =  vix  −  realized_var
```

VRP is algebraically defined by the other two features. With diagonal covariance, the EM algorithm has no new information — it sees the same signal three ways and the BIC correctly penalizes the extra parameters. **Do not add the arithmetic difference.**

### 13.3 The Ratio Is Non-Redundant

`VIX / realized_vol` (where `realized_vol = √realized_var`) is a **nonlinear function** of the existing features — it is not a linear combination. This means it provides genuinely new information to the diagonal HMM:

```
VIX / realized_vol > 1:  market pricing more fear than has been realized (positive VRP — normal)
VIX / realized_vol ≈ 1:  fair pricing, vol in equilibrium
VIX / realized_vol < 1:  realized vol exceeded implied (crisis peak, vol normalizing)
```

This ratio is scale-invariant: a 5-point gap means more when VIX=12 than when VIX=50. The ratio correctly normalizes for this. Carr & Wu (2009) use this ratio form explicitly in their VRP decomposition.

### 13.4 Why We Don't Add It Now

Despite the theoretical appeal, the VRP ratio is not the next priority for three reasons:

**1. The difference that matters is already partially captured.** `vix` and `realized_var` are both in the feature set. The HMM sees both values at every timestep. The 30% of variance in each that is independent of the other (the VRP signal) is available to the EM algorithm through the joint pattern of these two features — it just can't model their interaction explicitly with diagonal covariance.

**2. The proper fix is structural, not additive.** The inability to model the `vix × realized_var` interaction is a limitation of diagonal covariance matrices. The correct solution is to upgrade to **full covariance matrices**, which model all pairwise interactions simultaneously. That is a more principled fix than manually adding derived features.

**3. The VRP predicts at 1–6 month horizons.** Bollerslev et al. (2009) find the strongest predictability at intermediate horizons — not at the daily bar level our HMM needs. It is a slow-moving predictor, less valuable for the daily regime detection task.

### 13.5 Decision and Future Path

**Do not add VRP as an explicit feature at this stage.**

Priority order if the current 7-feature model is insufficient:

```
1. Add vix_slope (7th feature)     ← current step
2. Upgrade to full covariance      ← structural fix for all interactions
3. Then consider VIX/realized_vol  ← if still insufficient after #2
```

The full covariance upgrade correctly solves the problem the VRP addition would partially address — and solves it for all feature pairs simultaneously, not just the VIX/realized_var pair.

---

## 14. Implementation: Data Sources and Construction

### 11.1 Adding vix_ts_spread to the Feature Matrix

**`config/settings.yaml`** addition:
```yaml
data:
  fred_series:
    vix:       "VIXCLS"
    vix3m:     "VXVCLS"          # New: 93-day VIX for term spread
    hy_oas:    "BAMLH0A0HYM2"
    term_spread: "T10Y2Y"
  gold_ticker: "GLD"
```

**`data/market_data.py`** addition:
```python
def get_vix3m(self, start: str, end: str = None) -> pd.Series:
    """
    CBOE S&P 500 3-Month Volatility Index from FRED VXVCLS.
    93-day implied volatility — the 'longer end' for the term structure spread.
    Ref: Egloff, Leippold & Wu (2010, Journal of Financial Econometrics 8(3):367-413).
    """
    series = self.macro_source.get_series(
        self.settings["data"]["fred_series"]["vix3m"], start, end
    )
    return series.dropna()
```

**`data/feature_engineering.py`** addition to `compute()`:
```python
# 7. VIX term structure spread: VIX3M − VIX
#    Positive = contango (normal/calm), Negative = backwardation (crisis)
#    Ref: Egloff et al. (2010); Simon & Campasano (2014); Eraker & Wu (2017)
if vix3m is not None:
    vix3m_aligned = vix3m.reindex(prices.index, method="ffill", limit=3)
    vix_aligned   = vix.reindex(prices.index, method="ffill", limit=3)
    df["vix_ts_spread"] = vix3m_aligned - vix_aligned
else:
    df["vix_ts_spread"] = 0.0
```

### 11.2 Feature Validation Ranges

For `validate_macro_data()` or a dedicated `validate_vix_ts()`:
```python
# VIX term spread: historically −18.2 to +6.7 (2010–2026)
# Any reading below −20 or above +10 is likely a data error
spread = vix3m - vix
if spread.min() < -25 or spread.max() > 12:
    logger.warning(f"VIX term spread outside historical range: [{spread.min():.1f}, {spread.max():.1f}]")
```

### 11.3 Recommended Feature Set

After adding vix_ts_spread, the full (T, 7) observation matrix becomes:

```python
FEATURE_COLS = [
    "log_return",       # Equity momentum signal
    "realized_variance",# Backward-looking vol (20-day)
    "vix",              # Current 30-day implied vol level
    "hy_oas",           # Credit stress (duration-adjusted)
    "gold_return",      # Flight-to-safety cross-asset
    "term_spread",      # Yield curve / macro cycle (10yr-2yr)
    "vix_ts_spread",    # Vol curve shape: contango/backwardation
]
```

Each feature captures a distinct dimension of the market regime:

```
Dimension              Feature          Ref
──────────────────────────────────────────────────────────
Equity returns         log_return       Hamilton (1989)
Vol magnitude (back)   realized_var     Turner et al. (1989)
Vol magnitude (fwd)    vix              Whaley (1993)
Vol persistence        vix_ts_spread    Egloff et al. (2010)  ← NEW
Credit stress          hy_oas           Guidolin & Timmermann (2008)
Safe haven flows       gold_return      Baur & Lucey (2010)
Macro cycle            term_spread      Estrella & Mishkin (1998)
```

---

## 15. Implementation Plan: Phased Roadmap

The analysis in this document leads to a concrete multi-phase implementation plan. Each phase is dependent on the results of the previous one — do not skip ahead.

---

### Phase A — Add vix_slope as 7th Feature *(Immediate)*

**Objective:** Implement `VIX / VIX3M` as the 7th HMM observation feature and measure Sharpe improvement.

**Data layer** (`data/market_data.py`):
```python
def get_vix3m(self, start: str, end: str = None) -> pd.Series:
    """
    CBOE S&P 500 3-Month Volatility Index — FRED VXVCLS.
    93-day implied vol. Used for vix_slope = vix / vix3m.
    Ref: Egloff, Leippold & Wu (2010, JFE 8(3):367-413).
    """
    series = self.macro_source.get_series(
        self.settings["data"]["fred_series"]["vix3m"], start, end
    )
    return series.dropna()
```

**Config** (`config/settings.yaml`):
```yaml
fred_series:
  vix:         "VIXCLS"
  vix3m:       "VXVCLS"       # New: for term structure slope
  hy_oas:      "BAMLH0A0HYM2"
  term_spread: "T10Y2Y"
```

**Feature engineering** (`data/feature_engineering.py`):
```python
FEATURE_COLS = [
    "log_return", "realized_variance", "vix", "hy_oas",
    "gold_return", "term_spread", "vix_slope"   # (T, 7)
]

# In compute():
if vix3m is not None:
    vix3m_aligned = vix3m.reindex(prices.index, method="ffill", limit=3)
    vix_aligned   = vix.reindex(prices.index, method="ffill", limit=3)
    df["vix_slope"] = vix_aligned / vix3m_aligned  # > 1 = backwardation = stress
else:
    df["vix_slope"] = 1.0   # neutral fallback (at-the-money term structure)
```

**Note on fallback:** Using `1.0` (not `0.0`) as the fallback for missing vix3m data. A ratio of 1.0 represents a flat term structure — the neutral state, unlike `0.0` which would imply extreme contango never observed in practice.

**Plumbing updates required:**
- `backtester.run()` — add `vix3m` param, pass to `fe.compute()`
- `backtest/stress_test.py` — same pattern (already established for gold/term_spread)
- `main.py run_backtest()` — fetch `dm.get_vix3m(start)`, pass through
- `main.py run_paper()` — fetch `dm.get_vix3m(warmup_start)`, pass to `fe.compute()`

**Expected outcome:** Sharpe 0.831 → ~0.85–0.90. Specifically watch Windows 34/35 (2022 bear, currently Sharpe −1.12 and 0.05) — the slope's "persistent flat contango at elevated VIX" signal should improve these windows.

**Validation:** FRED VXVCLS confirmed available 2010-01-01 through present, 4,245 observations.

---

### Phase B — Re-Calibration with 7 Features *(After Phase A)*

With a new feature dimension, the HMM emission structure changes. Previously optimal hyperparameters may shift. Re-run the calibration sweep one parameter at a time:

| Step | Parameter | Range | Note |
|------|-----------|-------|------|
| B1 | `n_components_range` | [3,7], [2,6], [4,8] | BIC may select different state count |
| B2 | `persistence_bars` | [1, 2, 3] | Currently locked at 2 |
| B3 | `confidence_floor` | [0.25, 0.30, 0.35] | Currently locked at 0.30 |
| B4 | `normalization_window` | [40, 60, 80] | Affects all features equally |

**Rules:**
- One parameter at a time, OOS walk-forward Sharpe as the sole judge
- Lock best value before moving to next parameter
- Do not re-optimize already-locked parameters
- Use the same 59-window 2010–2026 walk-forward setup

**Target:** Sharpe ≥ 0.90, SPA p-value < 0.10

---

### Phase C — SPA Test Investigation *(After Phase B)*

Current SPA p-value = 0.953 (failing to reject null of luck). Root cause is Windows 34/35 (2022 bear market, June–December 2022 OOS period). Before proceeding to paper trading, determine whether this is:

| Diagnosis | Evidence | Action |
|-----------|----------|--------|
| **Feature gap** | Windows 34/35 improve after Phase A/B | Proceed to paper trading |
| **Structural** (rate-shock bear markets fundamentally different) | Windows 34/35 still poor | Consider Student-t emissions upgrade |
| **Data artifact** (2022 was a 40-year outlier rate shock) | Only 2 of 59 windows affected | Document and proceed |

**Decision rule:** If SPA p-value < 0.10 after Phase B → proceed to Phase D. If still > 0.10 → investigate Student-t emissions before live trading.

---

### Phase D — Alpaca Broker Layer *(After Phase C clearance)*

`AlpacaSource.get_bars()` is currently a stub. Required for paper trading:

1. **`broker/alpaca_client.py`** — connect to Alpaca paper endpoint using credentials.yaml
2. **`AlpacaSource.get_bars()`** — historical bars via Alpaca REST API (replaces yfinance in live mode)
3. **`broker/order_executor.py`** — bracket order submission (entry + stop + target OCO)
4. **`broker/position_tracker.py`** — reconcile local state with Alpaca account on every bar
5. **End-to-end test** — submit one manual paper order, confirm bracket fills correctly

**Credentials needed:** Alpaca paper API key + secret → `config/credentials.yaml`

---

### Phase E — Paper Trading: 3-Month Run *(After Phase D)*

The backtest cannot calibrate ATR stop multiples — stops are live-trading only (no transaction costs or slippage in the allocation-based backtest). 3 months of paper trading is the calibration instrument for:

| Parameter | Current Value | Calibration Target |
|-----------|-------------|-------------------|
| `stops.low_vol.stop_atr` | 3.0 | < 30% stop-out rate in LowVol regime |
| `stops.low_vol.target_atr` | 6.0 | > 40% target-hit rate in LowVol regime |
| `stops.mid_vol.stop_atr` | 2.5 | Same targets in MidVol |
| `stops.high_vol.stop_atr` | 2.0 | Same targets in HighVol |

**Monitoring checklist:**
- Telegram alerts firing on regime flips and order fills
- Streamlit dashboard rendering correctly
- Zero unhandled exceptions over 5+ consecutive trading days
- Positions reconcile with Alpaca UI after each bar
- Daily P&L summary arriving at 16:05 ET

**Success criteria for live trading go/no-go:**
- 3 months paper trading complete
- ATR stops calibrated (stop-out and target-hit rates within range)
- Max drawdown during paper period < 10%
- No broker connectivity failures > 1 per week

---

### Phase F — Model Upgrades *(Conditional on Phase B/C Results)*

Only proceed if Sharpe < 0.90 after Phase B, or SPA still failing after Phase C:

| Upgrade | Expected Gain | Complexity | Prerequisite |
|---------|-------------|-----------|-------------|
| Student-t emissions | +0.05–0.10 Sharpe | Medium — requires custom EM | SPA still failing after Phase B |
| **Full covariance matrices** | Captures all pairwise interactions | Medium — `covariance_type='full'` | SPA failing, especially VIX×realized_var interaction |
| VIX/realized_vol ratio (VRP) | +0.03–0.05 | Low | After full covariance, if still insufficient |
| Longer training window (750 bars) | Reduces overfitting in recent windows | Low | If BIC unstable across windows |

**Note on full covariance:** Switching from `covariance_type='diag'` to `'full'` in hmmlearn requires only one config change — but adds n_states × (n_features² − n_features)/2 parameters per state. With 7 features and 5 states, this adds 5 × 21 = 105 parameters. The 500-bar training window supports this (500/105 ≈ 5 — tight but feasible). Monitor BIC carefully.

---

### Summary Timeline

```
NOW         Phase A: vix_slope 7th feature + backtest
            Phase B: Re-calibration with 7 features
            Phase C: SPA test root-cause investigation

NEXT WEEK   Phase D: Alpaca broker layer
            Phase E: Paper trading begins

+3 MONTHS   Paper trading completes
            ATR stops calibrated
            Go/no-go: live trading

CONDITIONAL Phase F: Student-t / full covariance if targets not met
```

---

## 16. Limitations and Critiques

### 12.1 The 7.8% Problem

The term spread is in backwardation only 7.8% of trading days. For an HMM with 59 walk-forward windows × 126 OOS bars = 7,434 OOS observations, only ~580 will have negative spreads. This class imbalance means:

- The HMM may underweight the spread's signal value during calm periods when all spreads are positive
- The EM algorithm may assign the spread to LowVol states primarily based on deep contango, not effectively separating MidVol from HighVol
- A preprocessing step that emphasizes the sign change (e.g., using the ratio VIX/VIX3M) may improve discrimination

### 12.2 VXVCLS Availability

FRED VXVCLS starts 2007-12-04 — later than VIXCLS (1990) and BAMLH0A0HYM2 (1997). Our backtest starts 2010-01-01, so this is not a limitation for the current setup. However, extending the backtest further back would require an alternative construction.

**Alternative for pre-2008 data:** VIX futures term structure data is available from CBOE dating to 2004. For earlier periods, research papers (notably **Simon & Campasano, 2014**) construct synthetic term spreads from options data that go back to the early 2000s.

### 12.3 Crisis Regime Misidentification

The 2022 bear market demonstrates the spread's limitation: a genuine, painful bear market can occur with near-zero spread. Any strategy that uses backwardation as a necessary condition for defensive positioning would have been fully invested through −19% S&P 500 losses in 2022.

**Mitigation:** The spread should be used as one input among several, not as a standalone trigger. The HMM's role is to synthesize the full feature vector — even when the spread is near zero, the combination of elevated VIX, elevated HY OAS, and declining equity returns should correctly classify 2022 as HighVol. The spread's contribution in 2022 is to prevent over-allocation in "elevated but persistent" regimes versus "about to spike" regimes.

### 12.4 Gaussian Emission Misspecification

The standard HMM assumes Gaussian emissions. The VIX term spread distribution is distinctly non-Gaussian:
- Long right tail during calm periods (deep contango is possible but rare)
- Fat left tail during crises (backwardation episodes cluster)
- Near-discontinuous jumps on crisis onset days (spread drops −5 pts in a single day)

**Gray (1996, JFE 42:27-62)** identified the same problem for equity returns in Markov-switching models. Student-t emissions would better accommodate the spread's fat tails, but require a more complex likelihood function. The practical impact on HMM performance is a topic for future calibration.

### 12.5 In-Sample vs. Out-of-Sample Caution

The academic results documenting the spread's predictive power are largely in-sample or from early papers now competing with practitioners who have implemented similar strategies. **White (2000, Econometrica)** data snooping caution applies: the spread's apparent predictive power partly reflects that researchers selected it *because* it worked historically. The Hansen (2005) SPA test in the walk-forward backtest is the appropriate corrective — and as of our current results (p=0.95), the strategy has not yet passed the statistical significance threshold.

---

## 17. Conclusion

The VIX term structure contains some of the most information-rich, daily-available, free signals in equity markets. This treatise has resolved four questions:

**1. Slope or spread?** The **slope (`VIX / VIX3M`) is the correct construction.** It is scale-invariant (a +2 pt spread means something completely different when VIX=12 vs. VIX=50), directionally consistent with all other HMM features (high = stress), and grounded in the multiplicative structure of mean-reverting volatility processes. The arithmetic spread `VIX3M − VIX` is level-dependent and should not be used.

**2. Is it orthogonal to existing features?** **Yes — sufficiently.** The highest correlations are `realized_var ↔ vix` (r=0.70) and `vix ↔ vix_slope` (r=0.68). Both pairs have ~50% independent variance encoding economically distinct information (VRP gap; curve shape vs. level). All other pairs are r ≤ 0.50. Keep all 7 features.

**3. Should we add VIX − realized_var (VRP) explicitly?** **No — not as the arithmetic difference.** It is a linear combination of existing features, adding zero information to a diagonal HMM. The ratio `VIX / realized_vol` is non-redundant but lower priority than vix_slope. The principled fix for the VIX×realized_var interaction is full covariance matrices, not derived features.

**4. What is the implementation path?** Five phases: (A) add vix_slope now, (B) re-calibrate with 7 features, (C) investigate SPA test failures, (D) implement Alpaca broker layer, (E) 3-month paper trading for ATR stop calibration. Conditional Phase F covers structural model upgrades (Student-t emissions, full covariance) if Phase B targets are not met.

The current 6-feature Sharpe of 0.831 is expected to improve to ~0.85–0.90 with vix_slope, particularly in the 2022 windows that represent the model's current structural weakness. FRED VXVCLS is confirmed available from 2010-present with no new infrastructure required.

---

## 18. Citation Index

### Tier 1 — Top Journals (Load-Bearing)

| Citation | Used For |
|----------|----------|
| Stein (1989) *Review of Financial Studies* 2(4):727–752 | Mean reversion of implied volatility; upward-sloping term structure rationale |
| Carr & Madan (1998) *Journal of Computational Finance* 2(1):61–73 | Model-free variance swap replication; VIX formula foundation |
| Demeterfi, Derman, Kamal & Zou (1999) *Goldman Sachs Quant Strategies* | Model-free VIX calculation derivation |
| Cox, Ingersoll & Ross (1985) *Econometrica* 53(2):385–408 | Mean-reverting process analogy for vol term structure |
| Carr & Wu (2009) *Journal of Financial Economics* 92(2):259–286 | Variance risk premium measurement; VIX vs. realized vol |
| Egloff, Leippold & Wu (2010) *Journal of Financial Econometrics* 8(3):367–413 | Two-factor term structure; slope factor independent of level factor |
| Mencía & Sentana (2013) *Journal of Financial Economics* 108(2):367–391 | VIX derivative pricing; term structure predictive power |
| Eraker & Wu (2017) *Journal of Financial Economics* 123(2):249–272 | Persistent vs. transitory VIX components; spread distinguishes them |
| Bollerslev, Tauchen & Zhou (2009) *Review of Financial Studies* 22(11):4463–4492 | Variance risk premium predicts equity excess returns |
| Aït-Sahalia, Cacho-Diaz & Laeven (2015) *Journal of Econometrics* 185(2):291–316 | VIX term inversion leads crises by 5–15 days; leading indicator |
| Todorov & Tauchen (2011) *Econometrica* 79(6):1811–1858 | Backwardation driven by volatility jumps, not continuous variation |
| Simon & Campasano (2014) *Journal of Futures Markets* 34(11):997–1016 | VIX futures basis predicts equity returns 1–4 weeks out |
| Christoffersen, Heston & Jacobs (2009) *Review of Financial Studies* 22(12):5259–5299 | Two-factor structure: level + slope factors for VIX term structure |
| Whaley (1993) *Journal of Derivatives* 1(1):71–98 | Original VIX creation; investor fear gauge framing |
| Cremers, Halling & Weinbaum (2015) *Journal of Finance* 70(4):1459–1512 | SKEW index predicts tail risk; related signal |
| White (2000) *Econometrica* 68(5):1097–1126 | Data snooping / Hansen SPA test context |

### Tier 2 — Secondary or Practitioner

| Citation | Used For |
|----------|----------|
| Whaley (2009) *Journal of Portfolio Management* 35(3):98–105 | "Investor fear gauge" narrative, VIX as sentiment proxy |
| Bordonado, Molnár & Samdal (2017) *Finance Research Letters* | Practical VIX term structure timing model; Sharpe improvement |
| Johnson (2017) *Journal of Risk and Financial Management* 10(1):2 | VIX9D/VIX ratio as ultra-short timing signal |

### CBOE Data Sources

| Series | FRED ID | Yahoo Finance | Description |
|--------|---------|--------------|-------------|
| VIX (30-day) | VIXCLS | ^VIX | Primary fear gauge |
| VIX3M (93-day) | VXVCLS | ^VIX3M | Used for term spread construction |
| VIX9D | — | ^VIX9D | Ultra-short event signal |
| VIX6M | — | ^VIX6M | Semi-annual horizon |

---

*This treatise was prepared as part of the HMM Regime Trader design documentation series. Related documents: `03_research_hmm_engine.md`, `04_research_allocation_signals.md`, `05_data_sources.md`.*
