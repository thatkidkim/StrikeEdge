
# StrikeEdge — Kalshi BTC Hourly Market Edge Finder

A quantitative Monte Carlo engine for estimating fair probabilities on Kalshi BTC Hourly Price Markets using jump diffusion, vol-scaled GBM, and regime detection.

The model simulates thousands of BTC price paths from the current time to a selected Kalshi hourly expiry, reproduces the CF Benchmarks BRTI settlement methodology, and identifies pricing discrepancies between the model and the live market.

---

![StrikeEdge Dashboard](examples/btc_monte_carlo_output.png)

## Features

- Live BTC/USD data from Coinbase Advanced API (Kraken fallback)
- Realized volatility calibration from recent hourly candles
- Vol-scaled Geometric Brownian Motion with regime adjustment
- Merton jump diffusion for sudden market events
- Bull / Bear / Neutral regime detection
- Simulation from current time → expiry minute
- BRTI settlement approximation (average of final 60 simulated second-by-second prices)
- Full strike ladder with P(YES) probabilities
- Edge detection for both OVER and UNDER contracts
- Half-Kelly bet sizing
- Sanity check: suppresses edge signals when simulated vol diverges from theory
- Dashboard visualization with price history, settlement distribution, and strike ladder

---


## Model

Each simulation path combines multiple stochastic processes.

### Geometric Brownian Motion

Models the continuous movement of BTC prices.

\[
dS=\mu Sdt+\sigma SdW
\]

---

### Heston Stochastic Volatility

Volatility is itself random and mean-reverting.

Geometric Brownian Motion (Vol-Scaled)

Models the continuous movement of BTC prices. Volatility is scaled by regime:

𝑑
𝑆
=
𝜇
𝑆
 
𝑑
𝑡
+
𝜎
eff
𝑆
 
𝑑
𝑊
dS=μSdt+σ
eff
	​

SdW

where 
𝜎
eff
=
𝜎
realized
×
regime_scale
σ
eff
	​

=σ
realized
	​

×regime_scale.

Note: Heston stochastic volatility was evaluated and excluded. At the sub-hour step sizes used here, Heston produces numerical instability. Vol-scaled GBM with regime adjustment achieves equivalent practical effect.


### Merton Jump Diffusion

Models sudden market events such as:

- ETF announcements
- CPI releases
- FOMC meetings
- Liquidations
- Whale trades (Large block trades)

Jump arrivals follow a Poisson process while jump sizes are normally distributed.

---

### Regime Detection

Recent price action is classified into one of three regimes:

Regime	Condition	Vol Scale
Bull	48h avg > prior 48h avg by >1%, positive drift	1.05×
Bear	48h avg < prior 48h avg by >1%, negative drift	1.10×
Neutral	Neither	1.00×

The detected regime also applies a small drift adjustment during simulation.

---

## Settlement Logic

Kalshi hourly BTC markets resolve using the **CF Benchmarks BRTI**.

Resolution is based on:

> the simple average of the final 60 BRTI observations immediately before expiration.

This project approximates that by simulating 60 one-second prices during the final minute and averaging them.

For a strike \(K\):

```
YES if

Settlement >= K
```

The model computes

```
P(Settlement >= Strike)
```

for every strike.

---


## Sanity Check

After each simulation, the model compares its simulated standard deviation against the theoretical GBM prediction:

ratio = sim_std / (S0 × σ × √T)

If |ratio - 1| > 15%, edge signals are suppressed and the dashboard flags [⚠ vol inflated]. This prevents acting on a miscalibrated simulation.


## Edge Finder

The model compares its estimated probabilities against current Kalshi market prices.

For each strike:

```
Model Probability
−
Market Probability
=
Edge
```

Positive edge indicates potential value.

Both sides are evaluated independently.

```
OVER

Model: 61%

Market: 89%

Edge: -28%

--------------------

UNDER

Model: 39%

Market: 12%

Edge: +27%
```

Kelly sizing is also reported for position sizing.

---


## Phase 1 Backtesting

//image to be inserted

btc_backtest.py validates model calibration against 30 days of historical BTC closes — no Kalshi data required.

For each past hourly expiry it:

Computes vol and regime using only data available before that expiry (no lookahead)
Runs the full Monte Carlo simulation
Evaluates P(settlement >= strike) at 8 fixed offsets: ±$100, ±$200, ±$300, ±$500
Checks what BTC actually did
Calibration Results (30-day window, 553 simulations)
Offset	ECE	Skill vs 50/50	Verdict
±$500	0.008–0.011	+95–96%	✓ GOOD
±$300	0.014–0.017	+81–82%	✓ GOOD
±$200	0.005–0.028	+63%	✓ GOOD
±$100	0.040–0.045	+30%	✓ GOOD

Takeaway: The model is well-calibrated at strikes $200+ from spot. Focus on those levels for real edge signals. The ±$100 zone is weaker — only act there if the edge is large (>15%).

ECE = Expected Calibration Error (0 = perfect). Skill = improvement over naive 50/50 baseline.


## Installation

Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/btc-monte-carlo-kalshi.git
cd btc-monte-carlo-kalshi
```

Install dependencies

```bash
pip install -r requirements.txt
```

or

```bash
pip install numpy matplotlib requests pytz
```

---

## Running

Run

```bash
python btc_monte_carlo.py
```

Inside the script configure:

```python
EXPIRY_HOUR_ET = 17

target_strike = 64000

market_prob_over = 0.89
market_prob_under = 0.12
```

The script automatically:

- downloads BTC data
- calibrates volatility
- detects market regime
- runs Monte Carlo simulations
- computes strike probabilities
- calculates market edge
- generates the dashboard

---

## Repository Structure

```
btc-monte-carlo-kalshi/
│
├── btc_monte_carlo.py
├── README.md
├── requirements.txt
├── LICENSE
└── examples/
    └── dashboard.png
```

---

## Current Assumptions

- Uses hourly historical candles
- Realized volatility estimated from recent history
- Constant jump parameters
- Simplified regime detection
- Simulated BRTI rather than official index data

---

## Planned Improvements

- Historical backtesting against actual Kalshi markets
- Automatic Kalshi API integration
- Live order book pricing
- Parameter calibration from historical BTC data
- Bayesian volatility updating
- GPU acceleration
- GARCH/EGARCH volatility models
- Real BRTI data integration
- Option-implied volatility calibration

---

## Disclaimer

This project is intended for research and educational purposes.

The probabilities generated by the model are statistical estimates and should not be interpreted as guarantees or financial advice. Cryptocurrency markets are highly volatile, and real-world market behavior may differ significantly from the assumptions of the model.

---

## License

MIT License.
