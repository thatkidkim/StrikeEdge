
"""
BTC Monte Carlo — Kalshi Hourly Price Market Edge Finder
=========================================================
Targets the Kalshi "BTC price" directional market:
  Resolves YES if the 60-second BRTI average BEFORE expiry hour
  is AT OR ABOVE the strike price (e.g. $64,000 or above).

Key rules baked in:
  • Settlement = simple average of 60 CF Benchmarks BRTI ticks
    in the final minute before the hour mark.
  • Threshold is a "≥" check:  P(BRTI_avg >= strike)
  • Markets are NOT mutually exclusive.
  • Strike increments: $100 (Kalshi lists every $100 level).
  • Script auto-generates all $100 strikes within ±$500 of spot.
  • Simulation runs from NOW to the chosen expiry hour.

Simulation approach (v4 — numerically stable):
  Heston (Euler-Maruyama) was replaced with vol-scaling GBM.
  Euler-Maruyama is unstable at minute-step sizes: the variance
  noise term can overpower mean reversion, causing absorption bias
  that inflates std dev 3–6x regardless of kappa scaling.

  Instead we use:
    σ_eff = realized_vol × regime_scale
    dS    = S × exp((μ - ½σ²)dt + σ√dt × Z)
  with Merton jump diffusion added on top (jumps sampled
  independently — one draw per jump, summed).

  This produces std dev within ~5% of the theoretical GBM
  prediction S0 × σ_hourly × √T, which is what the sanity
  check verifies before showing any edge signal.

Requirements:
  pip install requests numpy matplotlib pytz
  there is a requirements.txt file included for convenience.

  also requires Python 3.9+ (for type hinting).
  If you don't have matplotlib, you can still run the sim 
  and print results, but no dashboard will be shown.

Run:
  python btc_monte_carlo.py
"""

from __future__ import annotations
import time
import sys
import requests
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime, timezone, timedelta
import warnings
warnings.filterwarnings("ignore")

try:
    import pytz
    EASTERN = pytz.timezone("America/New_York")
except ImportError:
    print("ERROR: pytz is required.  Run:  pip install pytz")
    sys.exit(1)


# ═══════════════════════════════════════════════════════
#  CONFIG  — edit to tune
# ═══════════════════════════════════════════════════════
N_PATHS           = 50_000  # Monte Carlo paths
STEPS_PER_HOUR    = 60      # Steps per hour (1 per minute)
BRTI_SUB_STEPS    = 60      # Per-second ticks in the final minute (BRTI window)
VOL_LOOKBACK_D    = 7       # Days of hourly candles for realized vol

# Vol-scaling GBM (replaces Euler-Maruyama Heston which is
# numerically unstable at minute step sizes)
VOL_SCALE_BULL    = 1.05    # Slight vol premium in bull regime
VOL_SCALE_BEAR    = 1.10    # Fear premium in bear regime
VOL_SCALE_NEUTRAL = 1.00    # No adjustment in neutral

# Jump diffusion (Merton — independent sampling)
LAMBDA_JUMPS      = 0.5    # was 2.0  (fewer jumps per day)
JUMP_MEAN         = 0.0     # Mean jump log-size
JUMP_SIGMA        = 0.003  # was 0.006 (smaller jump sizes)

# Regime switching drift adjustment
REGIME_DRIFT_ADJ  = 0.05    # Fraction of hourly vol added to drift per regime

# Edge filter
MIN_EDGE          = 0.05    # Minimum model-vs-market gap to flag a bet (5%)
BANKROLL          = 1_000.0 # $ used for Kelly sizing example

# Kalshi strike ladder
STRIKE_STEP       = 100     # $100 increments
STRIKE_RANGE      = 500     # Show strikes from spot-500 to spot+500

# Sanity check: warn if sim std deviates from theory by more than this
SANITY_TOLERANCE  = 0.15    # 15%


# ═══════════════════════════════════════════════════════
#  EXPIRY TIME HELPERS
# ═══════════════════════════════════════════════════════
def parse_expiry(hour_et: int) -> tuple[datetime, float]:
    """
    Given an expiry hour in Eastern time (0-23), returns:
      - expiry_utc   : next occurrence of that hour as UTC datetime
      - minutes_left : exact minutes from now until expiry
    Automatically targets tomorrow if the hour has passed today.
    """
    now_utc     = datetime.now(timezone.utc)
    now_eastern = now_utc.astimezone(EASTERN)
    expiry_eastern = now_eastern.replace(
        hour=hour_et, minute=0, second=0, microsecond=0
    )
    if (expiry_eastern - now_eastern).total_seconds() < 60:
        expiry_eastern += timedelta(days=1)
    expiry_utc   = expiry_eastern.astimezone(timezone.utc)
    minutes_left = (expiry_utc - now_utc).total_seconds() / 60.0
    return expiry_utc, minutes_left


# ═══════════════════════════════════════════════════════
#  STEP 1 — FETCH LIVE BTC DATA
#  Primary:  Coinbase Advanced API  (US-friendly, no key)
#  Fallback: Kraken REST API        (US-friendly, no key)
# ═══════════════════════════════════════════════════════
def _fetch_coinbase(lookback_days: int) -> list[dict]:
    end_ts   = int(time.time())
    start_ts = end_ts - lookback_days * 24 * 3600
    url = "https://api.coinbase.com/api/v3/brokerage/market/products/BTC-USD/candles"
    params = {
        "start":       str(start_ts),
        "end":         str(end_ts),
        "granularity": "ONE_HOUR",
        "limit":       min(lookback_days * 24 + 2, 350),
    }

    r = requests.get(url, params=params,
                     headers={"Content-Type": "application/json"}, timeout=10)
    r.raise_for_status()
    raw = r.json().get("candles", [])
    if not raw:
        raise ValueError("Coinbase returned empty candles")
    candles = [
        {
            "time":  datetime.fromtimestamp(int(k["start"]), tz=timezone.utc),
            "open":  float(k["open"]),  "high": float(k["high"]),
            "low":   float(k["low"]),   "close": float(k["close"]),
            "vol":   float(k["volume"]),
        }

        for k in raw
    ]

    candles.sort(key=lambda c: c["time"])
    return candles


def _fetch_kraken(lookback_days: int) -> list[dict]:
    since = int(time.time()) - lookback_days * 24 * 3600
    r = requests.get("https://api.kraken.com/0/public/OHLC",
                     params={"pair": "XBTUSD", "interval": 60, "since": since},
                     timeout=10)
    r.raise_for_status()
    data = r.json()
    if data.get("error"):
        raise ValueError(f"Kraken error: {data['error']}")
    pair_key = next(k for k in data["result"] if k != "last")
    candles = [
        {
            "time":  datetime.fromtimestamp(int(k[0]), tz=timezone.utc),
            "open":  float(k[1]), "high": float(k[2]),
            "low":   float(k[3]), "close": float(k[4]),
            "vol":   float(k[6]),
        }

        for k in data["result"][pair_key]
    ]

    candles.sort(key=lambda c: c["time"])
    return candles


def fetch_btc_candles(lookback_days: int = 7) -> list[dict]:
    """Try Coinbase first, fall back to Kraken. Both US-accessible, no API key."""
    print(f"[1/5] Fetching {lookback_days}d of hourly BTC/USD candles...")
    for name, fn in [("Coinbase", _fetch_coinbase), ("Kraken", _fetch_kraken)]:
        try:
            print(f"      Trying {name}...", end=" ", flush=True)
            candles = fn(lookback_days)
            print(f"OK  ({len(candles)} candles)")
            print(f"      Latest close : ${candles[-1]['close']:,.2f}  "
                  f"({candles[-1]['time'].strftime('%Y-%m-%d %H:%M UTC')})")
            return candles
        except Exception as e:
            print(f"FAILED — {e}")
    raise RuntimeError("Both Coinbase and Kraken failed. Check your internet connection.")


# ═══════════════════════════════════════════════════════
#  STEP 2 — REALIZED VOLATILITY
# ═══════════════════════════════════════════════════════
def calc_realized_vol(candles: list[dict]) -> dict:
    closes       = np.array([c["close"] for c in candles])
    log_returns  = np.diff(np.log(closes))
    hourly_vol   = np.std(log_returns, ddof=1)
    annual_vol   = hourly_vol * np.sqrt(8760)
    hourly_drift = np.mean(log_returns)
    print(f"[2/5] Volatility  →  Hourly: {hourly_vol*100:.4f}%  "
          f"| Annualized: {annual_vol*100:.1f}%  "
          f"| Drift: {hourly_drift*10000:.2f} bps/hr")
    return {"hourly_vol": hourly_vol, "annual_vol": annual_vol,
            "hourly_drift": hourly_drift, "log_returns": log_returns}


# ═══════════════════════════════════════════════════════
#  STEP 3 — REGIME DETECTION
# ═══════════════════════════════════════════════════════
def detect_regime(candles: list[dict]) -> str:
    n      = len(candles)
    recent = [c["close"] for c in candles[-min(48, n):]]
    older  = [c["close"] for c in candles[-min(96, n):-min(48, n)]]
    if len(older) < 5:
        return "neutral"
    r_avg = np.mean(recent)
    o_avg = np.mean(older)
    r_ret = np.mean(np.diff(np.log(recent))) if len(recent) > 1 else 0
    if   r_avg > o_avg * 1.01 and r_ret > 0:  regime = "bull"
    elif r_avg < o_avg * 0.99 and r_ret < 0:  regime = "bear"
    else:                                       regime = "neutral"
    print(f"[3/5] Regime     →  {regime.upper()}  "
          f"(recent avg ${r_avg:,.0f}  vs  older avg ${o_avg:,.0f})")
    return regime


# ═══════════════════════════════════════════════════════
#  STEP 4 — MONTE CARLO SIMULATION
#
#  Uses vol-scaling GBM + Merton jumps (no Heston).
#
#  Why not Heston?
#    Euler-Maruyama discretisation of Heston is numerically
#    unstable at minute-step sizes (dt = 1/60). The variance
#    noise term XI*sqrt(V*dt) routinely exceeds the mean-
#    reversion pull, creating absorption bias that inflates
#    the price std dev 3–6x regardless of kappa scaling.
#    The QE or Broadie-Kaya exact schemes fix this but are
#    significantly more complex.
#
#  What we use instead:
#    σ_eff = hourly_realized_vol × regime_vol_scale
#    Each step: S *= exp((μ - ½σ²)dt + σ√dt × Z)
#    Plus independent Merton jumps summed per step.
#
#  This gives std dev within ~5% of S0 × σ × √T (GBM theory)
#  while still capturing regime-dependent vol and jump risk.
# ═══════════════════════════════════════════════════════
def run_monte_carlo(S0: float, drift: float, hourly_vol: float,
                    regime: str, minutes_to_expiry: float) -> np.ndarray:
    """
    Returns array of shape (N_PATHS,) — each value is the
    simulated BRTI-average settlement price for that path.
    """
    hours_away  = minutes_to_expiry / 60.0
    total_steps = max(int(round(minutes_to_expiry)), 1)
    dt          = 1.0 / STEPS_PER_HOUR   # 1 minute as fraction of 1 hour

    # Regime-scaled effective vol
    scale = {"bull": VOL_SCALE_BULL,
             "bear": VOL_SCALE_BEAR,
             "neutral": VOL_SCALE_NEUTRAL}[regime]
    sigma_eff = hourly_vol * scale

    # Drift with regime adjustment
    adj = {"bull":  REGIME_DRIFT_ADJ,
           "bear": -REGIME_DRIFT_ADJ,
           "neutral": 0.0}
    mu = drift + adj[regime] * hourly_vol

    # Jump rate per minute-step
    lambda_ps = LAMBDA_JUMPS / (24 * STEPS_PER_HOUR)

    print(f"[4/5] Simulating {N_PATHS:,} paths × {total_steps} steps "
          f"({hours_away:.1f} hrs to expiry) + {BRTI_SUB_STEPS} BRTI ticks  "
          f"[regime={regime}, σ_eff={sigma_eff*100:.3f}%/hr]...")

    rng = np.random.default_rng()
    S   = np.full(N_PATHS, S0, dtype=np.float64)

    # ── Walk from now to start of the final minute ──
    main_steps = max(total_steps - 1, 0)
    var_dt     = sigma_eff ** 2 * dt    # precompute
    sqrt_dt    = sigma_eff * np.sqrt(dt)

    for _ in range(main_steps):
        Z = rng.standard_normal(N_PATHS)

        # Merton jumps — independent draws per jump
        jumps     = rng.poisson(lambda_ps, N_PATHS)
        max_j     = int(jumps.max()) if jumps.max() > 0 else 0
        if max_j > 0:
            draws      = rng.normal(JUMP_MEAN, JUMP_SIGMA, (N_PATHS, max_j))
            mask       = np.arange(max_j)[None, :] < jumps[:, None]
            jump_sizes = (draws * mask).sum(axis=1)
        else:
            jump_sizes = np.zeros(N_PATHS)

        S = S * np.exp((mu - 0.5 * var_dt / dt) * dt + sqrt_dt * Z + jump_sizes)

    # ── Final minute: 60 per-second BRTI ticks, averaged for settlement ──
    # Use second-level dt; no jumps in this 60-second window
    dt_sec   = 1.0 / 3600
    var_sec  = sigma_eff ** 2 * dt_sec
    sqrt_sec = sigma_eff * np.sqrt(dt_sec)

    brti_ticks = np.zeros((N_PATHS, BRTI_SUB_STEPS), dtype=np.float64)
    S_tick     = S.copy()

    for tick in range(BRTI_SUB_STEPS):
        Z      = rng.standard_normal(N_PATHS)
        S_tick = S_tick * np.exp((mu - 0.5 * var_sec / dt_sec) * dt_sec
                                 + sqrt_sec * Z)
        brti_ticks[:, tick] = S_tick

    # Settlement = simple average of 60 ticks (Kalshi rule)
    return brti_ticks.mean(axis=1)


# ═══════════════════════════════════════════════════════
#  SANITY CHECK
#  Compares simulated std dev against theoretical GBM:
#    theory_std = S0 × σ_hourly × √(hours_away)
#  Suppresses edge analysis if ratio falls outside tolerance.
# ═══════════════════════════════════════════════════════
def sanity_check(settlement: np.ndarray, S0: float,
                 hourly_vol: float, minutes_to_expiry: float) -> dict:
    hours_away  = minutes_to_expiry / 60.0
    sim_std     = float(np.std(settlement))
    theory_std  = S0 * hourly_vol * np.sqrt(hours_away)
    ratio       = sim_std / theory_std
    deviation   = abs(ratio - 1.0)
    ok          = deviation <= SANITY_TOLERANCE

    status = "OK" if ok else \
             "WARNING — model vol is inflated, edge signals suppressed"
    print(f"\n  ┌─ VOLATILITY SANITY CHECK {'─'*34}")
    print(f"  │  Theoretical GBM std  : ${theory_std:>10,.2f}  "
          f"(S0 × σ × √{hours_away:.2f}h)")
    print(f"  │  Simulated std dev    : ${sim_std:>10,.2f}")
    print(f"  │  Ratio sim/theory     :  {ratio:.3f}  "
          f"(tolerance ±{SANITY_TOLERANCE*100:.0f}%)")
    print(f"  │  Status               :  {status}")
    print(f"  └{'─'*56}\n")

    return {"sim_std": sim_std, "theory_std": theory_std,
            "ratio": ratio, "ok": ok}


# ═══════════════════════════════════════════════════════
#  STEP 5 — STATISTICS
#  Kalshi resolves YES if settlement >= strike
# ═══════════════════════════════════════════════════════
def compute_stats(settlement: np.ndarray, strike: float | None = None) -> dict:
    n      = len(settlement)
    mean   = float(np.mean(settlement))
    median = float(np.median(settlement))
    std    = float(np.std(settlement))
    pcts   = {p: float(np.percentile(settlement, p))
              for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]}
    result = {"n": n, "mean": mean, "median": median, "std": std,
              "percentiles": pcts,
              "prob_yes": None, "ci_low": None, "ci_high": None}
    if strike is not None:
        count_yes = int(np.sum(settlement >= strike))
        prob_yes  = count_yes / n
        se        = np.sqrt(prob_yes * (1 - prob_yes) / n)
        result.update({"prob_yes": prob_yes,
                        "ci_low":  max(prob_yes - 1.96 * se, 0),
                        "ci_high": min(prob_yes + 1.96 * se, 1)})
    return result


# ═══════════════════════════════════════════════════════
#  KALSHI STRIKE LADDER
# ═══════════════════════════════════════════════════════
def build_strike_ladder(settlement: np.ndarray, spot: float) -> list[dict]:
    base    = round(spot / STRIKE_STEP) * STRIKE_STEP
    strikes = range(int(base - STRIKE_RANGE),
                    int(base + STRIKE_RANGE + STRIKE_STEP),
                    STRIKE_STEP)
    n = len(settlement)
    return [{"strike": s, "prob_yes": float(np.sum(settlement >= s)) / n}
            for s in strikes]


# ═══════════════════════════════════════════════════════
#  EDGE FINDER  (Kelly sizing)
# ═══════════════════════════════════════════════════════
def find_edge(model_prob: float, market_prob: float) -> dict:
    edge = model_prob - market_prob
    if edge > MIN_EDGE:
        direction = "YES"
        p = model_prob
        b = (1 - market_prob) / market_prob
    elif edge < -MIN_EDGE:
        direction = "NO"
        p = 1 - model_prob
        b = market_prob / (1 - market_prob)
    else:
        direction = "SKIP"
        p, b = model_prob, 1.0
    kelly      = max((b * p - (1 - p)) / b, 0)
    half_kelly = kelly * 0.5
    return {"direction": direction, "edge": edge,
            "kelly": kelly, "half_kelly": half_kelly,
            "bet_amount": half_kelly * BANKROLL,
            "model_prob": model_prob, "market_prob": market_prob}


# ═══════════════════════════════════════════════════════
#  PRINT RESULTS
# ═══════════════════════════════════════════════════════
def print_results(S0: float, stats: dict, sanity: dict,
                  regime: str, vol: dict, ladder: list[dict],
                  target_strike: float | None,
                  market_prob_over: float | None,
                  market_prob_under: float | None,
                  expiry_utc: datetime, minutes_left: float):

    expiry_eastern = expiry_utc.astimezone(EASTERN)
    tz_label   = expiry_eastern.strftime("%Z")
    hours_left = minutes_left / 60

    print("\n" + "═" * 66)
    print("  BTC MONTE CARLO  •  KALSHI HOURLY PRICE MARKET")
    print("═" * 66)
    print(f"  Spot price      : ${S0:>12,.2f}")
    print(f"  Current time    : "
          f"{datetime.now(EASTERN).strftime('%Y-%m-%d %H:%M:%S')} {tz_label}")
    print(f"  Target expiry   : "
          f"{expiry_eastern.strftime('%Y-%m-%d %H:%M')} {tz_label}  "
          f"({hours_left:.1f} hrs  /  {minutes_left:.0f} min away)")
    print(f"  Regime          : {regime.upper()}")
    print(f"  Realized vol    : {vol['annual_vol']*100:.1f}% annualized")
    print(f"  Hourly drift    : {vol['hourly_drift']*10000:.2f} bps")
    print(f"  Paths           : {stats['n']:,}")
    print(f"  Settlement      : BRTI 60-tick average (final minute before expiry)")
    print("─" * 66)
    print(f"  Sim mean        : ${stats['mean']:>12,.2f}")
    print(f"  Sim median      : ${stats['median']:>12,.2f}")
    print(f"  Sim std dev     : ${stats['std']:>12,.2f}  "
          f"(theory: ${sanity['theory_std']:,.2f}  "
          f"ratio: {sanity['ratio']:.3f}"
          f"{'  ✓' if sanity['ok'] else '  ⚠ INFLATED'})")

    print("─" * 66)
    print("  SETTLEMENT PERCENTILES")
    for p, v in stats["percentiles"].items():
        bar = "█" * int((p / 99) * 22)
        print(f"    {p:>3}th : ${v:>12,.2f}  {bar}")

    print("─" * 66)
    print(f"  KALSHI STRIKE LADDER  (P(YES) = P(settlement ≥ strike))")
    print(f"  {'Strike':>10}   {'Model P(YES)':>13}   bar")
    for row in ladder:
        pct    = row["prob_yes"] * 100
        bar    = "█" * int(pct / 5)
        marker = "  ◄ SPOT" if abs(row["strike"] - S0) < STRIKE_STEP / 2 else ""
        print(f"  ${row['strike']:>9,.0f}   {pct:>12.1f}%   {bar}{marker}")

    if target_strike is not None and stats["prob_yes"] is not None:
        if not sanity["ok"]:
            print("─" * 66)
            print("  ⚠  EDGE ANALYSIS SUPPRESSED — simulated vol is inflated.")
            print("     Adjust VOL_SCALE / JUMP_SIGMA parameters and re-run.")
            print("═" * 66 + "\n")
            return

        print("─" * 66)
        print(f"  EDGE ANALYSIS  —  Strike ${target_strike:,.0f}  "
              f"@ expiry {expiry_eastern.strftime('%H:%M')} {tz_label}")
        print(f"    Model P(OVER)  : {stats['prob_yes']*100:.2f}%")
        print(f"    Model P(UNDER) : {(1-stats['prob_yes'])*100:.2f}%")

        if market_prob_over is not None:
            e = find_edge(stats["prob_yes"], market_prob_over)
            print(f"    Market OVER    : {e['market_prob']*100:.0f}¢  "
                  f"(implied {e['market_prob']*100:.1f}%)")
            print(f"    Edge OVER      : {e['edge']*100:+.2f}%")
            print(f"    Decision OVER  : *** {e['direction']} ***")
            if e["direction"] != "SKIP":
                print(f"    Full Kelly OVR : {e['kelly']*100:.1f}% of bankroll")
                print(f"    Half-Kelly OVR : {e['half_kelly']*100:.1f}% of bankroll")
                print(f"    Bet OVR (${BANKROLL:,.0f}): ${e['bet_amount']:.2f}")

        if market_prob_under is not None:
            e = find_edge(1 - stats["prob_yes"], market_prob_under)
            print(f"    Market UNDER   : {e['market_prob']*100:.0f}¢  "
                  f"(implied {e['market_prob']*100:.1f}%)")
            print(f"    Edge UNDER     : {e['edge']*100:+.2f}%")
            print(f"    Decision UNDER : *** {e['direction']} ***")
            if e["direction"] != "SKIP":
                print(f"    Full Kelly UND : {e['kelly']*100:.1f}% of bankroll")
                print(f"    Half-Kelly UND : {e['half_kelly']*100:.1f}% of bankroll")
                print(f"    Bet UND (${BANKROLL:,.0f}): ${e['bet_amount']:.2f}")

        print(f"    95% CI         : [{stats['ci_low']*100:.2f}%, "
              f"{stats['ci_high']*100:.2f}%]")
    print("═" * 66 + "\n")


# ═══════════════════════════════════════════════════════
#  PLOT DASHBOARD
# ═══════════════════════════════════════════════════════
def plot_dashboard(candles: list[dict], settlement: np.ndarray,
                   stats: dict, sanity: dict, S0: float,
                   ladder: list[dict],
                   target_strike: float | None,
                   market_prob_over: float | None,
                   market_prob_under: float | None,
                   regime: str, vol: dict,
                   expiry_utc: datetime, minutes_left: float):

    expiry_eastern = expiry_utc.astimezone(EASTERN)
    tz_label   = expiry_eastern.strftime("%Z")
    hours_left = minutes_left / 60

    fig = plt.figure(figsize=(15, 9), facecolor="#0f0f0f")
    sanity_tag  = "✓ vol ok" if sanity["ok"] else "⚠ vol inflated"
    title_color = "white" if sanity["ok"] else "#f97316"
    fig.suptitle(
        f"BTC Monte Carlo  •  Kalshi Expiry "
        f"{expiry_eastern.strftime('%Y-%m-%d %H:%M')} {tz_label}  "
        f"({hours_left:.1f} hrs away)  [{sanity_tag}]",
        color=title_color, fontsize=13, fontweight="bold", y=0.98
    )

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.38,
                           left=0.06, right=0.97, top=0.93, bottom=0.08)
    ax_price  = fig.add_subplot(gs[0, :2])
    ax_dist   = fig.add_subplot(gs[1, :2])
    ax_info   = fig.add_subplot(gs[0, 2])
    ax_ladder = fig.add_subplot(gs[1, 2])

    bg = "#1a1a1a"
    for ax in [ax_price, ax_dist, ax_info, ax_ladder]:
        ax.set_facecolor(bg)
        ax.tick_params(colors="#888", labelsize=9)
        for sp in ax.spines.values():
            sp.set_color("#333")

    # ── price history ──
    times  = [c["time"] for c in candles]
    closes = [c["close"] for c in candles]
    clr    = "#22c55e" if regime == "bull" else \
             "#ef4444" if regime == "bear" else "#60a5fa"
    ax_price.plot(times, closes, color=clr, linewidth=1.2)
    ax_price.fill_between(times, closes, min(closes), alpha=0.12, color=clr)
    ax_price.axhline(S0, color="#facc15", lw=0.9, ls="--",
                     alpha=0.8, label=f"Spot ${S0:,.0f}")
    if target_strike:
        ax_price.axhline(target_strike, color="#f97316", lw=0.9,
                         ls="--", label=f"Strike ${target_strike:,.0f}")
    ax_price.legend(fontsize=8, facecolor=bg, labelcolor="white", framealpha=0.4)
    ax_price.set_title(
        f"BTC Price History  |  Regime: {regime.upper()}  "
        f"|  Vol: {vol['annual_vol']*100:.1f}% ann.",
        color="white", fontsize=10, pad=6
    )
    ax_price.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax_price.tick_params(axis="x", rotation=20)

    # ── settlement distribution ──
    bins  = np.linspace(settlement.min(), settlement.max(), 80)
    below = settlement[settlement < S0]
    above = settlement[settlement >= S0]
    ax_dist.hist(below, bins=bins, color="#ef4444", alpha=0.7, label="Below spot")
    ax_dist.hist(above, bins=bins, color="#22c55e", alpha=0.7, label="At/above spot")
    ax_dist.axvline(S0,            color="#facc15", lw=1.2, ls="--",
                    label=f"Spot ${S0:,.0f}")
    ax_dist.axvline(stats["mean"], color="#60a5fa", lw=1.0, ls=":",
                    label=f"Mean ${stats['mean']:,.0f}")
    if target_strike:
        ax_dist.axvline(target_strike, color="#f97316", lw=1.2, ls="--",
                        label=f"Strike ${target_strike:,.0f}")
    ax_dist.axvline(S0 - sanity["theory_std"], color="#888", lw=0.7,
                    ls=":", alpha=0.6,
                    label=f"±1σ theory (${sanity['theory_std']:,.0f})")
    ax_dist.axvline(S0 + sanity["theory_std"], color="#888", lw=0.7,
                    ls=":", alpha=0.6)
    ax_dist.set_title(
        f"Simulated BRTI Settlement  |  {stats['n']:,} paths  "
        f"|  {hours_left:.1f} hr horizon  "
        f"|  Std ${stats['std']:,.0f} vs theory ${sanity['theory_std']:,.0f}",
        color=title_color, fontsize=9, pad=6
    )
    
    ax_dist.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax_dist.legend(fontsize=7.5, facecolor=bg, labelcolor="white", framealpha=0.4)
    ax_dist.set_ylabel("Paths", color="#888", fontsize=9)

    # ── info panel ──
    ax_info.axis("off")
    ax_info.add_patch(plt.Rectangle((0, 0), 1, 1,
                                    transform=ax_info.transAxes,
                                    facecolor=bg, zorder=-1))
    info_lines: list[tuple[str, str]] = [
        ("Spot price",    f"${S0:,.2f}"),
        ("Expiry",        expiry_eastern.strftime("%H:%M") + f" {tz_label}"),
        ("Time away",     f"{hours_left:.1f} hrs  ({minutes_left:.0f} min)"),
        ("Regime",        regime.upper()),
        ("Realized vol",  f"{vol['annual_vol']*100:.1f}%"),
        ("Drift",         f"{vol['hourly_drift']*10000:.2f} bps/hr"),
        ("",              ""),
        ("Sim mean",      f"${stats['mean']:,.0f}"),
        ("Sim median",    f"${stats['median']:,.0f}"),
        ("Sim std",       f"${stats['std']:,.0f}"),
        ("Theory std",    f"${sanity['theory_std']:,.0f}"),
        ("Std ratio",     f"{sanity['ratio']:.3f} "
                          f"{'✓' if sanity['ok'] else '⚠ INFLATED'}"),
        ("",              ""),
        ("Settlement",    "60-tick BRTI avg"),
        ("Resolves YES",  "if avg ≥ strike"),
    ]
    if target_strike and stats["prob_yes"] is not None:
        info_lines += [
            ("",                          ""),
            (f"P(OVER) @${target_strike:,.0f}",
             f"{stats['prob_yes']*100:.2f}%"),
            (f"P(UNDER) @${target_strike:,.0f}",
             f"{(1-stats['prob_yes'])*100:.2f}%"),
            ("95% CI",
             f"[{stats['ci_low']*100:.1f}%, {stats['ci_high']*100:.1f}%]"),
        ]
        if sanity["ok"]:
            if market_prob_over is not None:
                e = find_edge(stats["prob_yes"], market_prob_over)
                info_lines += [
                    ("Market OVER",   f"{market_prob_over*100:.0f}¢"),
                    ("Edge OVER",     f"{e['edge']*100:+.1f}%"),
                    ("Decision OVR",  f"*** {e['direction']} ***"),
                ]
            if market_prob_under is not None:
                e = find_edge(1 - stats["prob_yes"], market_prob_under)
                info_lines += [
                    ("Market UNDER",  f"{market_prob_under*100:.0f}¢"),
                    ("Edge UNDER",    f"{e['edge']*100:+.1f}%"),
                    ("Decision UND",  f"*** {e['direction']} ***"),
                ]
        else:
            info_lines += [("⚠ Edge suppressed", "vol inflated")]

    y = 0.97
    for lbl, val in info_lines:
        if lbl == "":
            y -= 0.025; continue
        ax_info.text(0.04, y, lbl, color="#888", fontsize=8,
                     transform=ax_info.transAxes, va="top")
        color = "#f97316" if ("⚠" in val or "INFLATED" in val) else "white"
        ax_info.text(0.97, y, val, color=color, fontsize=8, fontweight="bold",
                     transform=ax_info.transAxes, va="top", ha="right")
        y -= 0.057
    ax_info.set_title("Summary", color="white", fontsize=10, pad=6)

    # ── strike ladder ──
    strikes  = [f"${r['strike']:,.0f}" for r in ladder]
    probs    = [r["prob_yes"] * 100 for r in ladder]
    bar_clrs = ["#22c55e" if p >= 50 else "#ef4444" for p in probs]
    bars = ax_ladder.barh(strikes, probs, color=bar_clrs, alpha=0.8)
    ax_ladder.axvline(50, color="#facc15", lw=0.8, ls="--", alpha=0.6)
    if target_strike:
        lbl = f"${target_strike:,.0f}"
        if lbl in strikes:
            idx = strikes.index(lbl)
            bars[idx].set_edgecolor("#f97316")
            bars[idx].set_linewidth(2)
    ax_ladder.set_xlim(0, 105)
    ax_ladder.set_title("Strike Ladder — P(YES)",
                        color="white", fontsize=10, pad=6)
    ax_ladder.set_xlabel("Model P(YES) %", color="#888", fontsize=9)
    for bar, p in zip(bars, probs):
        ax_ladder.text(min(p + 1.5, 100),
                       bar.get_y() + bar.get_height() / 2,
                       f"{p:.0f}%", color="white", va="center", fontsize=7.5)

    plt.savefig("btc_monte_carlo_output.png", dpi=150,
                bbox_inches="tight", facecolor="#0f0f0f")
    print("  Chart saved → btc_monte_carlo_output.png")
    plt.show()


# ═══════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════
def main():
    print("\n" + "═" * 66)
    print("  BTC Monte Carlo  •  Kalshi Hourly Price Market")
    print("═" * 66)

    # ────────────────────────────────────────────────────
    #  ① EXPIRY HOUR  (Eastern time, 24-hour clock)
    #
    #     Examples:
    #       EXPIRY_HOUR_ET = 7    →  next 7:00 AM  EDT/EST
    #       EXPIRY_HOUR_ET = 19   →  next 7:00 PM  EDT/EST
    #       EXPIRY_HOUR_ET = 0    →  next midnight  EDT/EST
    # ────────────────────────────────────────────────────
    EXPIRY_HOUR_ET = 17          # ← change this

    # ────────────────────────────────────────────────────
    #  ② STRIKE & MARKET ODDS
    #
    #     target_strike     = Kalshi price level, e.g. 64_000
    #     market_prob_over  = YES price in cents → decimal
    #                         e.g. Yes 77¢  →  0.77
    #     market_prob_under = NO price in cents → decimal
    #                         e.g. No  25¢  →  0.25
    #
    #     Leave as None to skip that side / skip edge analysis.
    # ────────────────────────────────────────────────────
    target_strike     = 63_750    # e.g. 64_000
    market_prob_over  = 0.94    # e.g. 0.77
    market_prob_under = 0.07    # e.g. 0.25

    # ── resolve expiry ──
    expiry_utc, minutes_left = parse_expiry(EXPIRY_HOUR_ET)
    expiry_eastern = expiry_utc.astimezone(EASTERN)
    tz_label = expiry_eastern.strftime("%Z")
    print(f"  Target expiry : "
          f"{expiry_eastern.strftime('%Y-%m-%d %H:%M')} {tz_label}  "
          f"({minutes_left/60:.1f} hrs  /  {minutes_left:.0f} min away)")

    # ── fetch & calibrate ──
    candles = fetch_btc_candles(VOL_LOOKBACK_D)
    S0      = candles[-1]["close"]
    vol     = calc_realized_vol(candles)
    regime  = detect_regime(candles)

    # ── simulate ──
    settlement = run_monte_carlo(
        S0=S0,
        drift=vol["hourly_drift"],
        hourly_vol=vol["hourly_vol"],
        regime=regime,
        minutes_to_expiry=minutes_left,
    )

    # ── sanity check ──
    sanity = sanity_check(settlement, S0, vol["hourly_vol"], minutes_left)

    # ── stats ──
    stats  = compute_stats(settlement, target_strike)
    ladder = build_strike_ladder(settlement, S0)
    print(f"[5/5] Stats      →  Mean ${stats['mean']:,.2f} | "
          f"Median ${stats['median']:,.2f} | Std ±${stats['std']:,.2f}")

    # ── output ──
    print_results(S0, stats, sanity, regime, vol, ladder,
                  target_strike, market_prob_over, market_prob_under,
                  expiry_utc, minutes_left)
    plot_dashboard(candles, settlement, stats, sanity, S0, ladder,
                   target_strike, market_prob_over, market_prob_under,
                   regime, vol, expiry_utc, minutes_left)


if __name__ == "__main__":
    main()
