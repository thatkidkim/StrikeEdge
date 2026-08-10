"""
BTC Monte Carlo — Phase 1 Backtester  (v2 — off-the-money strikes)
====================================================================
Tests whether the model's probability estimates are well-calibrated
against real historical BTC price data. No Kalshi data required.

v2 changes vs v1:
  Instead of testing P(OVER spot) at-the-money (which is always ~50%
  and proves nothing), this version tests the model at FIXED DOLLAR
  OFFSETS from spot: ±$200, ±$300, ±$500.

  This is where the model should show real skill — when the strike is
  far enough from spot that the model can meaningfully differentiate
  between "very likely YES" and "very likely NO".

  For each historical hour the script:
    1. Computes vol + regime using only pre-expiry data (no lookahead)
    2. Runs ONE Monte Carlo simulation
    3. Evaluates P(settlement >= strike) for each offset level
    4. Checks actual outcome: did BTC close above each strike?
    5. Logs a separate result row per offset

  Calibration is then shown PER OFFSET — you can see at which distance
  from spot the model's probabilities become reliable and useful.

Calibration metrics:
  Brier Score  — mean squared error between predicted prob and outcome
  ECE          — Expected Calibration Error (0 = perfectly calibrated)
  Log Loss     — how surprised the model was by real outcomes
  Skill Score  — improvement over naive 50/50 baseline

Requirements:
  pip install requests numpy matplotlib pytz

Run:
  python btc_backtest.py
"""

from __future__ import annotations
import time
import json
import os
import requests
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime, timezone, timedelta
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

try:
    import pytz
    EASTERN = pytz.timezone("America/New_York")
except ImportError:
    import sys
    print("ERROR: pytz is required.  Run:  pip install pytz")
    sys.exit(1)


# ═══════════════════════════════════════════════════════
#  CONFIG — edit to tune
# ═══════════════════════════════════════════════════════

BACKTEST_DAYS       = 30       # Days of history to backtest over
MINUTES_BEFORE      = 38       # Minutes before each close to simulate from
VOL_LOOKBACK_HOURS  = 7 * 24  # Candles used for vol/regime (168h = 7d)
N_PATHS             = 10_000  # Paths per simulation (10k=fast, 50k=accurate)
STEPS_PER_HOUR      = 60
BRTI_SUB_STEPS      = 60

# Strike offsets to test (dollars above/below spot)
# Negative = below spot (UNDER bets), Positive = above spot (OVER bets)
STRIKE_OFFSETS      = [-500, -300, -200, -100, 100, 200, 300, 500]

# Vol / jump params — must match btc_monte_carlo.py
VOL_SCALE_BULL      = 1.05
VOL_SCALE_BEAR      = 1.10
VOL_SCALE_NEUTRAL   = 1.00
LAMBDA_JUMPS        = 0.5
JUMP_MEAN           = 0.0
JUMP_SIGMA          = 0.003
REGIME_DRIFT_ADJ    = 0.05
SANITY_TOLERANCE    = 0.15

N_BUCKETS           = 8        # Calibration curve buckets per offset
USE_CACHE           = True
CACHE_FILE          = "btc_backtest_cache.json"
OUTPUT_FILE         = "btc_backtest_output.png"


# ═══════════════════════════════════════════════════════
#  DATA FETCHING
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
            "time":  datetime.fromtimestamp(int(k["start"]), tz=timezone.utc).isoformat(),
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
            "time":  datetime.fromtimestamp(int(k[0]), tz=timezone.utc).isoformat(),
            "open":  float(k[1]), "high": float(k[2]),
            "low":   float(k[3]), "close": float(k[4]),
            "vol":   float(k[6]),
        }
        for k in data["result"][pair_key]
    ]
    candles.sort(key=lambda c: c["time"])
    return candles


def fetch_all_candles() -> list[dict]:
    days_needed = BACKTEST_DAYS + (VOL_LOOKBACK_HOURS // 24) + 2

    if USE_CACHE and os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE) as f:
                cached = json.load(f)
            if cached.get("fetched_at", 0) > time.time() - 3600:
                c = cached["candles"]
                age = int((time.time() - cached["fetched_at"]) / 60)
                print(f"  Loaded {len(c)} candles from cache  ({age}min old)")
                return c
        except Exception:
            pass

    print(f"  Fetching {days_needed}d of hourly candles...")
    for name, fn in [("Coinbase", _fetch_coinbase), ("Kraken", _fetch_kraken)]:
        try:
            print(f"  Trying {name}...", end=" ", flush=True)
            candles = fn(days_needed)
            print(f"OK  ({len(candles)} candles)")
            if USE_CACHE:
                with open(CACHE_FILE, "w") as f:
                    json.dump({"fetched_at": time.time(), "candles": candles}, f)
                print(f"  Cached → {CACHE_FILE}")
            return candles
        except Exception as e:
            print(f"FAILED — {e}")
    raise RuntimeError("Both Coinbase and Kraken failed.")


# ═══════════════════════════════════════════════════════
#  MODEL FUNCTIONS  (silent — no print output)
# ═══════════════════════════════════════════════════════

def calc_vol_silent(candles: list[dict]) -> dict:
    closes       = np.array([c["close"] for c in candles])
    log_returns  = np.diff(np.log(closes))
    hourly_vol   = np.std(log_returns, ddof=1)
    annual_vol   = hourly_vol * np.sqrt(8760)
    hourly_drift = np.mean(log_returns)
    return {"hourly_vol": hourly_vol, "annual_vol": annual_vol,
            "hourly_drift": hourly_drift}


def detect_regime_silent(candles: list[dict]) -> str:
    n      = len(candles)
    recent = [c["close"] for c in candles[-min(48, n):]]
    older  = [c["close"] for c in candles[-min(96, n):-min(48, n)]]
    if len(older) < 5:
        return "neutral"
    r_avg = np.mean(recent)
    o_avg = np.mean(older)
    r_ret = np.mean(np.diff(np.log(recent))) if len(recent) > 1 else 0
    if   r_avg > o_avg * 1.01 and r_ret > 0:  return "bull"
    elif r_avg < o_avg * 0.99 and r_ret < 0:  return "bear"
    else:                                       return "neutral"


def run_monte_carlo_silent(S0: float, drift: float, hourly_vol: float,
                           regime: str, minutes_to_expiry: float,
                           rng: np.random.Generator) -> np.ndarray:
    total_steps = max(int(round(minutes_to_expiry)), 1)
    dt          = 1.0 / STEPS_PER_HOUR
    scale       = {"bull": VOL_SCALE_BULL,
                   "bear": VOL_SCALE_BEAR,
                   "neutral": VOL_SCALE_NEUTRAL}[regime]
    sigma_eff   = hourly_vol * scale
    adj         = {"bull": REGIME_DRIFT_ADJ, "bear": -REGIME_DRIFT_ADJ, "neutral": 0.0}
    mu          = drift + adj[regime] * hourly_vol
    lambda_ps   = LAMBDA_JUMPS / (24 * STEPS_PER_HOUR)

    S          = np.full(N_PATHS, S0, dtype=np.float64)
    main_steps = max(total_steps - 1, 0)
    var_dt     = sigma_eff ** 2 * dt
    sqrt_dt    = sigma_eff * np.sqrt(dt)

    for _ in range(main_steps):
        Z     = rng.standard_normal(N_PATHS)
        jumps = rng.poisson(lambda_ps, N_PATHS)
        max_j = int(jumps.max()) if jumps.max() > 0 else 0
        if max_j > 0:
            draws      = rng.normal(JUMP_MEAN, JUMP_SIGMA, (N_PATHS, max_j))
            mask       = np.arange(max_j)[None, :] < jumps[:, None]
            jump_sizes = (draws * mask).sum(axis=1)
        else:
            jump_sizes = np.zeros(N_PATHS)
        S = S * np.exp((mu - 0.5 * var_dt / dt) * dt + sqrt_dt * Z + jump_sizes)

    dt_sec   = 1.0 / 3600
    sqrt_sec = sigma_eff * np.sqrt(dt_sec)
    var_sec  = sigma_eff ** 2 * dt_sec
    brti     = np.zeros((N_PATHS, BRTI_SUB_STEPS), dtype=np.float64)
    S_tick   = S.copy()
    for tick in range(BRTI_SUB_STEPS):
        Z      = rng.standard_normal(N_PATHS)
        S_tick = S_tick * np.exp((mu - 0.5 * var_sec / dt_sec) * dt_sec + sqrt_sec * Z)
        brti[:, tick] = S_tick
    return brti.mean(axis=1)


def sanity_ok(settlement: np.ndarray, S0: float,
              hourly_vol: float, minutes: float) -> bool:
    hours      = minutes / 60.0
    sim_std    = float(np.std(settlement))
    theory_std = S0 * hourly_vol * np.sqrt(hours)
    if theory_std == 0:
        return False
    return abs(sim_std / theory_std - 1.0) <= SANITY_TOLERANCE


# ═══════════════════════════════════════════════════════
#  CALIBRATION METRICS
# ═══════════════════════════════════════════════════════

def brier_score(probs: np.ndarray, outcomes: np.ndarray) -> float:
    return float(np.mean((probs - outcomes) ** 2))

def log_loss(probs: np.ndarray, outcomes: np.ndarray, eps: float = 1e-7) -> float:
    p = np.clip(probs, eps, 1 - eps)
    return float(-np.mean(outcomes * np.log(p) + (1 - outcomes) * np.log(1 - p)))

def ece_score(probs: np.ndarray, outcomes: np.ndarray,
              n_buckets: int = N_BUCKETS) -> float:
    n    = len(probs)
    ece  = 0.0
    edges = np.linspace(0, 1, n_buckets + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / n) * abs(probs[mask].mean() - outcomes[mask].mean())
    return float(ece)

def skill_score(probs: np.ndarray, outcomes: np.ndarray) -> float:
    bs       = brier_score(probs, outcomes)
    baseline = brier_score(np.full_like(probs, 0.5), outcomes)
    return float(1.0 - bs / baseline) if baseline > 0 else 0.0

def calibration_buckets(probs: np.ndarray, outcomes: np.ndarray,
                         n_buckets: int = N_BUCKETS) -> list[dict]:
    edges = np.linspace(0, 1, n_buckets + 1)
    rows  = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask  = (probs >= lo) & (probs < hi)
        count = int(mask.sum())
        rows.append({
            "mid":        (lo + hi) / 2,
            "avg_pred":   float(probs[mask].mean())   if count > 0 else None,
            "avg_actual": float(outcomes[mask].mean()) if count > 0 else None,
            "count":      count,
        })
    return rows


# ═══════════════════════════════════════════════════════
#  MAIN BACKTEST LOOP
# ═══════════════════════════════════════════════════════

def run_backtest(all_candles: list[dict]) -> dict[int, list[dict]]:
    """
    Returns a dict keyed by offset (e.g. -500, -300, …, 500).
    Each value is a list of result dicts for that offset.
    """
    # Parse ISO strings back to datetimes
    for c in all_candles:
        if isinstance(c["time"], str):
            c["time"] = datetime.fromisoformat(c["time"])

    now      = datetime.now(timezone.utc)
    cutoff   = now - timedelta(hours=1)
    start_bt = now - timedelta(days=BACKTEST_DAYS)

    test_candles = [c for c in all_candles
                    if start_bt <= c["time"] <= cutoff]

    if len(test_candles) < 2:
        raise RuntimeError(
            f"Not enough candles in backtest window ({len(test_candles)}). "
            f"Try increasing BACKTEST_DAYS."
        )

    print(f"  Window  : {test_candles[0]['time'].strftime('%Y-%m-%d %H:%M')} UTC "
          f"→ {test_candles[-1]['time'].strftime('%Y-%m-%d %H:%M')} UTC")
    print(f"  Hours   : {len(test_candles)}  |  "
          f"Offsets : {STRIKE_OFFSETS}  |  "
          f"Paths/sim: {N_PATHS:,}")
    print()

    # One results list per offset
    results: dict[int, list[dict]] = {off: [] for off in STRIKE_OFFSETS}
    skipped_vol  = 0
    skipped_data = 0
    regime_counts = defaultdict(int)

    rng = np.random.default_rng(42)

    for i, target in enumerate(test_candles):
        if (i + 1) % 10 == 0 or i == 0:
            pct = (i + 1) / len(test_candles) * 100
            print(f"  [{i+1:>4}/{len(test_candles)}]  {pct:5.1f}%  "
                  f"{target['time'].strftime('%Y-%m-%d %H:%M')} UTC  "
                  f"spot=${target['close']:,.0f}", flush=True)

        # Build the lookback slice available BEFORE this candle closes
        lookback = [c for c in all_candles if c["time"] < target["time"]]
        if len(lookback) < max(VOL_LOOKBACK_HOURS, 100):
            skipped_data += 1
            continue

        lookback_slice = lookback[-VOL_LOOKBACK_HOURS:]
        S0     = lookback_slice[-1]["close"]
        vol    = calc_vol_silent(lookback_slice)
        regime = detect_regime_silent(lookback_slice)
        regime_counts[regime] += 1

        # Run ONE simulation — reuse settlement for all offsets
        settlement = run_monte_carlo_silent(
            S0=S0,
            drift=vol["hourly_drift"],
            hourly_vol=vol["hourly_vol"],
            regime=regime,
            minutes_to_expiry=float(MINUTES_BEFORE),
            rng=rng,
        )

        if not sanity_ok(settlement, S0, vol["hourly_vol"], float(MINUTES_BEFORE)):
            skipped_vol += 1
            continue

        actual_close = target["close"]

        # Evaluate each offset independently
        for offset in STRIKE_OFFSETS:
            strike       = S0 + offset
            prob_yes     = float(np.mean(settlement >= strike))
            actual_yes   = int(actual_close >= strike)
            results[offset].append({
                "time":         target["time"].isoformat(),
                "S0":           S0,
                "strike":       strike,
                "offset":       offset,
                "actual_close": actual_close,
                "actual_yes":   actual_yes,
                "prob_yes":     prob_yes,
                "regime":       regime,
                "hourly_vol":   vol["hourly_vol"],
            })

    print()
    total_sims = sum(len(v) for v in results.values()) // len(STRIKE_OFFSETS)
    print(f"  Valid   : {total_sims} hourly sims  "
          f"(skipped: {skipped_data} data, {skipped_vol} vol inflated)")
    print(f"  Regimes : BULL {regime_counts['bull']}  "
          f"BEAR {regime_counts['bear']}  NEUTRAL {regime_counts['neutral']}")
    return results


# ═══════════════════════════════════════════════════════
#  PRINT SUMMARY
# ═══════════════════════════════════════════════════════

def print_summary(results: dict[int, list[dict]]) -> dict[int, dict]:
    metrics: dict[int, dict] = {}

    print("\n" + "═" * 72)
    print("  PHASE 1 BACKTEST — OFF-THE-MONEY CALIBRATION")
    print("═" * 72)
    print(f"  {'Offset':>8}  {'N':>5}  {'AvgPred':>8}  {'AvgAct':>8}  "
          f"{'Brier':>7}  {'Skill':>7}  {'ECE':>7}  Verdict")
    print("─" * 72)

    for offset in STRIKE_OFFSETS:
        rows = results[offset]
        if len(rows) < 5:
            print(f"  {offset:>+8}  {'—':>5}  (not enough data)")
            continue

        probs    = np.array([r["prob_yes"]   for r in rows])
        outcomes = np.array([r["actual_yes"] for r in rows], dtype=float)

        bs   = brier_score(probs, outcomes)
        ll   = log_loss(probs, outcomes)
        ece  = ece_score(probs, outcomes)
        sk   = skill_score(probs, outcomes)
        ap   = float(probs.mean())
        aa   = float(outcomes.mean())

        # Calibration verdict
        if ece < 0.05 and sk > 0.05:
            verdict = "✓ GOOD"
        elif ece < 0.10:
            verdict = "~ OK"
        elif ece < 0.20:
            verdict = "⚠ MODERATE"
        else:
            verdict = "✗ POOR"

        metrics[offset] = {"brier": bs, "log_loss": ll, "ece": ece,
                           "skill": sk, "avg_pred": ap, "avg_actual": aa,
                           "n": len(rows), "probs": probs, "outcomes": outcomes,
                           "verdict": verdict}

        direction = "OVER" if offset > 0 else "UNDER"
        print(f"  {offset:>+8}  {len(rows):>5}  "
              f"{ap*100:>7.1f}%  {aa*100:>7.1f}%  "
              f"{bs:>7.4f}  {sk*100:>+6.1f}%  {ece:>7.4f}  "
              f"{verdict}  [{direction}]")

    print("═" * 72)
    print("  Brier: 0=perfect  0.25=coin flip  |  "
          "Skill: +%=beats baseline  |  ECE: 0=best")
    print()
    return metrics


# ═══════════════════════════════════════════════════════
#  PLOT DASHBOARD
# ═══════════════════════════════════════════════════════

def plot_results(results: dict[int, list[dict]],
                 metrics: dict[int, dict]) -> None:

    valid_offsets = [o for o in STRIKE_OFFSETS if o in metrics]
    n_offsets     = len(valid_offsets)
    if n_offsets == 0:
        print("  Nothing to plot.")
        return

    bg       = "#0f0f0f"
    panel_bg = "#1a1a1a"

    # Layout: top row = calibration curves (one per offset, 4 per row)
    #         bottom row = summary bar charts
    cols      = 4
    cal_rows  = (n_offsets + cols - 1) // cols
    total_rows = cal_rows + 1  # +1 for summary row

    fig = plt.figure(figsize=(18, 5 * total_rows + 1), facecolor=bg)
    fig.suptitle(
        f"BTC Monte Carlo — Phase 1 Off-the-Money Calibration  |  "
        f"{BACKTEST_DAYS}d window  |  {MINUTES_BEFORE}min before expiry  |  "
        f"{N_PATHS:,} paths/sim",
        color="white", fontsize=13, fontweight="bold", y=0.99
    )

    gs = gridspec.GridSpec(
        total_rows, cols, figure=fig,
        hspace=0.55, wspace=0.35,
        left=0.05, right=0.97,
        top=0.95, bottom=0.05
    )

    # ── Per-offset calibration curves ──
    for idx, offset in enumerate(valid_offsets):
        row = idx // cols
        col = idx % cols
        ax  = fig.add_subplot(gs[row, col])
        ax.set_facecolor(panel_bg)
        ax.tick_params(colors="#888", labelsize=8)
        for sp in ax.spines.values():
            sp.set_color("#333")

        m        = metrics[offset]
        probs    = m["probs"]
        outcomes = m["outcomes"]
        buckets  = calibration_buckets(probs, outcomes)

        # Perfect calibration line
        ax.plot([0, 1], [0, 1], color="#555", lw=1.0, ls="--", zorder=1)

        # Bucket dots
        bx = [b["avg_pred"]   for b in buckets if b["avg_pred"]   is not None]
        by = [b["avg_actual"] for b in buckets if b["avg_actual"] is not None]
        bn = [b["count"]      for b in buckets if b["avg_pred"]   is not None]

        if bx:
            color = "#22c55e" if m["verdict"].startswith("✓") else \
                    "#facc15" if m["verdict"].startswith("~") else "#ef4444"
            sizes = [max(15, min(250, n * 4)) for n in bn]
            ax.scatter(bx, by, s=sizes, color=color, zorder=3, alpha=0.9)
            ax.plot(bx, by, color=color, lw=1.2, alpha=0.5, zorder=2)
            for x, y, n in zip(bx, by, bn):
                ax.annotate(f"{n}", (x, y),
                            textcoords="offset points", xytext=(4, 4),
                            color="#aaa", fontsize=6.5)

        direction = "OVER spot" if offset > 0 else "UNDER spot"
        sign      = "+" if offset > 0 else ""
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_title(
            f"Strike {sign}${abs(offset):,}  [{direction}]\n"
            f"ECE={m['ece']:.3f}  Skill={m['skill']*100:+.1f}%  n={m['n']}  {m['verdict']}",
            color="white", fontsize=8.5, pad=4
        )
        ax.set_xlabel("Predicted P(YES)", color="#888", fontsize=7.5)
        ax.set_ylabel("Actual freq", color="#888", fontsize=7.5)

        # Avg predicted vs actual annotation
        ax.axvline(m["avg_pred"], color="#60a5fa", lw=0.8, ls=":",
                   alpha=0.7, label=f"Avg pred {m['avg_pred']*100:.0f}%")
        ax.axhline(m["avg_actual"], color="#f97316", lw=0.8, ls=":",
                   alpha=0.7, label=f"Avg actual {m['avg_actual']*100:.0f}%")
        ax.legend(fontsize=6.5, facecolor=panel_bg,
                  labelcolor="white", framealpha=0.3)

    # ── Summary bar charts (bottom row spans full width) ──
    # Split into 3 panels: ECE, Skill Score, Avg Pred vs Actual
    ax_ece   = fig.add_subplot(gs[cal_rows, 0])
    ax_skill = fig.add_subplot(gs[cal_rows, 1])
    ax_pred  = fig.add_subplot(gs[cal_rows, 2])
    ax_brier = fig.add_subplot(gs[cal_rows, 3])

    for ax in [ax_ece, ax_skill, ax_pred, ax_brier]:
        ax.set_facecolor(panel_bg)
        ax.tick_params(colors="#888", labelsize=8)
        for sp in ax.spines.values():
            sp.set_color("#333")

    labels = [f"{'+' if o>0 else ''}{o}" for o in valid_offsets]
    eces   = [metrics[o]["ece"]            for o in valid_offsets]
    skills = [metrics[o]["skill"] * 100    for o in valid_offsets]
    preds  = [metrics[o]["avg_pred"] * 100 for o in valid_offsets]
    acts   = [metrics[o]["avg_actual"] * 100 for o in valid_offsets]
    briers = [metrics[o]["brier"]          for o in valid_offsets]
    x      = np.arange(len(valid_offsets))
    bar_w  = 0.6

    # ECE
    ece_colors = ["#22c55e" if e < 0.05 else
                  "#facc15" if e < 0.10 else "#ef4444" for e in eces]
    ax_ece.bar(x, eces, width=bar_w, color=ece_colors, alpha=0.8)
    ax_ece.axhline(0.05, color="#22c55e", lw=0.8, ls="--", alpha=0.6,
                   label="Good (0.05)")
    ax_ece.axhline(0.10, color="#facc15", lw=0.8, ls="--", alpha=0.6,
                   label="OK (0.10)")
    ax_ece.set_xticks(x); ax_ece.set_xticklabels(labels, fontsize=8)
    ax_ece.set_title("ECE per Offset  (lower=better)",
                     color="white", fontsize=9, pad=4)
    ax_ece.set_ylabel("ECE", color="#888", fontsize=8)
    ax_ece.legend(fontsize=7, facecolor=panel_bg,
                  labelcolor="white", framealpha=0.3)
    for xi, v in zip(x, eces):
        ax_ece.text(xi, v + 0.002, f"{v:.3f}", color="white",
                    ha="center", fontsize=7)

    # Skill Score
    skill_colors = ["#22c55e" if s > 5 else
                    "#facc15" if s > -5 else "#ef4444" for s in skills]
    ax_skill.bar(x, skills, width=bar_w, color=skill_colors, alpha=0.8)
    ax_skill.axhline(0, color="#888", lw=0.8, ls="--", alpha=0.6)
    ax_skill.set_xticks(x); ax_skill.set_xticklabels(labels, fontsize=8)
    ax_skill.set_title("Skill Score vs 50/50  (higher=better)",
                       color="white", fontsize=9, pad=4)
    ax_skill.set_ylabel("Skill %", color="#888", fontsize=8)
    for xi, v in zip(x, skills):
        ax_skill.text(xi, v + (0.3 if v >= 0 else -1.5),
                      f"{v:+.1f}%", color="white", ha="center", fontsize=7)

    # Avg Pred vs Actual
    w2 = bar_w / 2 - 0.04
    ax_pred.bar(x - w2/2, preds, width=w2, color="#60a5fa",
                alpha=0.8, label="Avg predicted")
    ax_pred.bar(x + w2/2, acts,  width=w2, color="#f97316",
                alpha=0.8, label="Avg actual")
    ax_pred.axhline(50, color="#888", lw=0.8, ls="--", alpha=0.5)
    ax_pred.set_xticks(x); ax_pred.set_xticklabels(labels, fontsize=8)
    ax_pred.set_title("Avg Predicted vs Actual %",
                      color="white", fontsize=9, pad=4)
    ax_pred.set_ylabel("%", color="#888", fontsize=8)
    ax_pred.legend(fontsize=7.5, facecolor=panel_bg,
                   labelcolor="white", framealpha=0.3)

    # Brier Score
    brier_colors = ["#22c55e" if b < 0.20 else
                    "#facc15" if b < 0.24 else "#ef4444" for b in briers]
    ax_brier.bar(x, briers, width=bar_w, color=brier_colors, alpha=0.8)
    ax_brier.axhline(0.25, color="#ef4444", lw=0.8, ls="--",
                     alpha=0.6, label="Coin flip (0.25)")
    ax_brier.set_xticks(x); ax_brier.set_xticklabels(labels, fontsize=8)
    ax_brier.set_title("Brier Score  (lower=better)",
                       color="white", fontsize=9, pad=4)
    ax_brier.set_ylabel("Brier", color="#888", fontsize=8)
    ax_brier.legend(fontsize=7, facecolor=panel_bg,
                    labelcolor="white", framealpha=0.3)
    for xi, v in zip(x, briers):
        ax_brier.text(xi, v + 0.002, f"{v:.3f}", color="white",
                      ha="center", fontsize=7)

    plt.savefig(OUTPUT_FILE, dpi=150, bbox_inches="tight", facecolor=bg)
    print(f"  Chart saved → {OUTPUT_FILE}")
    plt.show()


# ═══════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════

def main() -> None:
    print("\n" + "═" * 66)
    print("  BTC Monte Carlo — Phase 1 Off-the-Money Backtest")
    print("═" * 66)
    print(f"  Config : {BACKTEST_DAYS}d window  |  "
          f"{MINUTES_BEFORE}min before expiry  |  "
          f"{N_PATHS:,} paths/sim")
    print(f"  Offsets: {STRIKE_OFFSETS}")
    print()

    print("─ Step 1: Fetching candles " + "─" * 39)
    all_candles = fetch_all_candles()

    print("\n─ Step 2: Running backtest " + "─" * 38)
    results = run_backtest(all_candles)

    print("─ Step 3: Metrics " + "─" * 47)
    metrics = print_summary(results)

    print("─ Step 4: Plotting " + "─" * 46)
    plot_results(results, metrics)


if __name__ == "__main__":
    main()
