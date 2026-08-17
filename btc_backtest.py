
"""
BTC Monte Carlo — Phase 1 Backtester  (v3 — multi-horizon)
===========================================================
Tests model calibration at multiple time horizons before expiry:
  60 min, 45 min, 30 min, 15 min

This shows how the model's accuracy sharpens as expiry approaches —
useful for knowing WHEN to place bets, not just whether to.

For each historical hour and each horizon the script:
  1. Computes vol + regime using only pre-expiry data (no lookahead)
  2. Runs the Monte Carlo simulation at that horizon
  3. Evaluates P(settlement >= strike) for each offset level
  4. Checks actual outcome: did BTC close above each strike?

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

# Time horizons to test (minutes before each hourly close)
# Shows how model accuracy sharpens as expiry approaches
MINUTES_BEFORE      = [60, 45, 30, 15]

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

N_BUCKETS           = 8
USE_CACHE           = True
CACHE_FILE          = "btc_backtest_cache.json"
OUTPUT_FILE         = "examples/btc_backtest_output.png"


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
                c   = cached["candles"]
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
    adj         = {"bull": REGIME_DRIFT_ADJ, "bear": -REGIME_DRIFT_ADJ,
                   "neutral": 0.0}
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
        S_tick = S_tick * np.exp(
            (mu - 0.5 * var_sec / dt_sec) * dt_sec + sqrt_sec * Z)
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

def log_loss(probs: np.ndarray, outcomes: np.ndarray,
             eps: float = 1e-7) -> float:
    p = np.clip(probs, eps, 1 - eps)
    return float(-np.mean(outcomes * np.log(p) + (1 - outcomes) * np.log(1 - p)))

def ece_score(probs: np.ndarray, outcomes: np.ndarray,
              n_buckets: int = N_BUCKETS) -> float:
    n     = len(probs)
    ece   = 0.0
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
            "avg_pred":   float(probs[mask].mean())    if count > 0 else None,
            "avg_actual": float(outcomes[mask].mean()) if count > 0 else None,
            "count":      count,
        })
    return rows


# ═══════════════════════════════════════════════════════
#  MAIN BACKTEST LOOP
# ═══════════════════════════════════════════════════════

# Results keyed by (horizon_minutes, offset)
ResultsDict = dict[tuple[int, int], list[dict]]

def run_backtest(all_candles: list[dict]) -> ResultsDict:
    """
    For each candle in the backtest window and each horizon in
    MINUTES_BEFORE, runs a separate Monte Carlo simulation and
    evaluates P(YES) at every STRIKE_OFFSETS level.

    Returns a dict keyed by (horizon_minutes, offset).
    """
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

    horizons = MINUTES_BEFORE if isinstance(MINUTES_BEFORE, list) \
               else [MINUTES_BEFORE]

    print(f"  Window   : {test_candles[0]['time'].strftime('%Y-%m-%d %H:%M')} UTC "
          f"→ {test_candles[-1]['time'].strftime('%Y-%m-%d %H:%M')} UTC")
    print(f"  Hours    : {len(test_candles)}  |  "
          f"Horizons : {horizons}min  |  "
          f"Offsets  : {STRIKE_OFFSETS}")
    print(f"  Paths/sim: {N_PATHS:,}  "
          f"(total sims ≈ {len(test_candles) * len(horizons):,})")
    print()

    # Initialise results dict for every (horizon, offset) combination
    results: ResultsDict = {
        (h, off): [] for h in horizons for off in STRIKE_OFFSETS
    }

    skipped_data = 0
    regime_counts = defaultdict(int)
    rng = np.random.default_rng(42)

    for i, target in enumerate(test_candles):
        if (i + 1) % 10 == 0 or i == 0:
            pct = (i + 1) / len(test_candles) * 100
            print(f"  [{i+1:>4}/{len(test_candles)}]  {pct:5.1f}%  "
                  f"{target['time'].strftime('%Y-%m-%d %H:%M')} UTC  "
                  f"spot=${target['close']:,.0f}", flush=True)

        # Lookback slice: all candles strictly before the target
        lookback = [c for c in all_candles if c["time"] < target["time"]]
        if len(lookback) < max(VOL_LOOKBACK_HOURS, 100):
            skipped_data += 1
            continue

        lookback_slice = lookback[-VOL_LOOKBACK_HOURS:]
        S0     = lookback_slice[-1]["close"]
        vol    = calc_vol_silent(lookback_slice)
        regime = detect_regime_silent(lookback_slice)
        regime_counts[regime] += 1
        actual_close = target["close"]

        # Run a SEPARATE simulation for each horizon
        for horizon in horizons:
            settlement = run_monte_carlo_silent(
                S0=S0,
                drift=vol["hourly_drift"],
                hourly_vol=vol["hourly_vol"],
                regime=regime,
                minutes_to_expiry=float(horizon),
                rng=rng,
            )

            if not sanity_ok(settlement, S0, vol["hourly_vol"], float(horizon)):
                # Skip this horizon for this candle — don't pollute other horizons
                continue

            for offset in STRIKE_OFFSETS:
                strike     = S0 + offset
                prob_yes   = float(np.mean(settlement >= strike))
                actual_yes = int(actual_close >= strike)
                results[(horizon, offset)].append({
                    "time":         target["time"].isoformat(),
                    "S0":           S0,
                    "strike":       strike,
                    "offset":       offset,
                    "horizon":      horizon,
                    "actual_close": actual_close,
                    "actual_yes":   actual_yes,
                    "prob_yes":     prob_yes,
                    "regime":       regime,
                    "hourly_vol":   vol["hourly_vol"],
                })

    print()
    for h in horizons:
        n = len(results[(h, STRIKE_OFFSETS[0])])
        print(f"  {h:>3}min horizon : {n} valid sims")
    print(f"  Skipped (insufficient data): {skipped_data}")
    print(f"  Regimes : BULL {regime_counts['bull']}  "
          f"BEAR {regime_counts['bear']}  NEUTRAL {regime_counts['neutral']}")
    return results


# ═══════════════════════════════════════════════════════
#  COMPUTE METRICS PER (HORIZON, OFFSET)
# ═══════════════════════════════════════════════════════

def compute_metrics(results: ResultsDict) -> dict[tuple[int,int], dict]:
    horizons = MINUTES_BEFORE if isinstance(MINUTES_BEFORE, list) \
               else [MINUTES_BEFORE]
    metrics: dict[tuple[int,int], dict] = {}

    for h in horizons:
        for offset in STRIKE_OFFSETS:
            rows = results.get((h, offset), [])
            if len(rows) < 5:
                continue
            probs    = np.array([r["prob_yes"]   for r in rows])
            outcomes = np.array([r["actual_yes"] for r in rows], dtype=float)
            bs  = brier_score(probs, outcomes)
            ll  = log_loss(probs, outcomes)
            ece = ece_score(probs, outcomes)
            sk  = skill_score(probs, outcomes)
            ap  = float(probs.mean())
            aa  = float(outcomes.mean())
            if ece < 0.05 and sk > 0.05:
                verdict = "✓ GOOD"
            elif ece < 0.10:
                verdict = "~ OK"
            elif ece < 0.20:
                verdict = "⚠ MODERATE"
            else:
                verdict = "✗ POOR"
            metrics[(h, offset)] = {
                "brier": bs, "log_loss": ll, "ece": ece, "skill": sk,
                "avg_pred": ap, "avg_actual": aa,
                "n": len(rows), "probs": probs, "outcomes": outcomes,
                "verdict": verdict,
            }
    return metrics


# ═══════════════════════════════════════════════════════
#  PRINT SUMMARY
# ═══════════════════════════════════════════════════════

def print_summary(metrics: dict[tuple[int,int], dict]) -> None:
    horizons = MINUTES_BEFORE if isinstance(MINUTES_BEFORE, list) \
               else [MINUTES_BEFORE]

    for h in horizons:
        print("\n" + "═" * 76)
        print(f"  HORIZON: {h} minutes before expiry")
        print("═" * 76)
        print(f"  {'Offset':>8}  {'N':>5}  {'AvgPred':>8}  {'AvgAct':>8}  "
              f"{'Brier':>7}  {'Skill':>7}  {'ECE':>7}  Verdict")
        print("─" * 76)
        for offset in STRIKE_OFFSETS:
            key = (h, offset)
            if key not in metrics:
                print(f"  {offset:>+8}  {'—':>5}  (not enough data)")
                continue
            m   = metrics[key]
            dir = "OVER" if offset > 0 else "UNDER"
            print(f"  {offset:>+8}  {m['n']:>5}  "
                  f"{m['avg_pred']*100:>7.1f}%  {m['avg_actual']*100:>7.1f}%  "
                  f"{m['brier']:>7.4f}  {m['skill']*100:>+6.1f}%  "
                  f"{m['ece']:>7.4f}  {m['verdict']}  [{dir}]")
        print("─" * 76)
        print("  Brier: 0=perfect  0.25=coin flip  |  "
              "Skill: +%=beats baseline  |  ECE: 0=best")
    print()


# ═══════════════════════════════════════════════════════
#  PLOT DASHBOARD
# ═══════════════════════════════════════════════════════

def plot_results(metrics: dict[tuple[int,int], dict]) -> None:
    horizons = MINUTES_BEFORE if isinstance(MINUTES_BEFORE, list) \
               else [MINUTES_BEFORE]
    n_h      = len(horizons)
    n_off    = len(STRIKE_OFFSETS)

    if n_h == 0 or n_off == 0:
        print("  Nothing to plot.")
        return

    bg       = "#0f0f0f"
    panel_bg = "#1a1a1a"

    # Layout:
    #   rows 0..n_h-1  : one row per horizon, n_off columns (calibration curves)
    #   row  n_h       : summary comparison row (skill score per horizon per offset)
    total_rows = n_h + 1
    cols       = n_off  # one column per offset

    fig = plt.figure(figsize=(max(18, cols * 2.4), 4.5 * total_rows + 2),
                     facecolor=bg)
    horizons_str = ", ".join(f"{h}min" for h in horizons)
    fig.suptitle(
        f"BTC Monte Carlo — Phase 1 Multi-Horizon Calibration  |  "
        f"{BACKTEST_DAYS}d window  |  Horizons: {horizons_str}  |  "
        f"{N_PATHS:,} paths/sim",
        color="white", fontsize=12, fontweight="bold", y=0.995
    )

    gs = gridspec.GridSpec(
        total_rows, cols, figure=fig,
        hspace=0.85, wspace=0.38,
        left=0.04, right=0.97,
        top=0.93, bottom=0.04
    )

    # ── Calibration curves: one row per horizon, one column per offset ──
    for h_idx, h in enumerate(horizons):
        for o_idx, offset in enumerate(STRIKE_OFFSETS):
            ax  = fig.add_subplot(gs[h_idx, o_idx])
            ax.set_facecolor(panel_bg)
            ax.tick_params(colors="#888", labelsize=7)
            for sp in ax.spines.values():
                sp.set_color("#333")

            key = (h, offset)
            if key not in metrics:
                ax.text(0.5, 0.5, "no data", color="#888",
                        ha="center", va="center", transform=ax.transAxes)
                continue

            m        = metrics[key]
            probs    = m["probs"]
            outcomes = m["outcomes"]
            buckets  = calibration_buckets(probs, outcomes)

            ax.plot([0, 1], [0, 1], color="#555", lw=0.8, ls="--", zorder=1)

            bx = [b["avg_pred"]   for b in buckets if b["avg_pred"]   is not None]
            by = [b["avg_actual"] for b in buckets if b["avg_actual"] is not None]
            bn = [b["count"]      for b in buckets if b["avg_pred"]   is not None]

            if bx:
                color = "#22c55e" if m["verdict"].startswith("✓") else \
                        "#facc15" if m["verdict"].startswith("~") else "#ef4444"
                sizes = [max(10, min(200, n * 4)) for n in bn]
                ax.scatter(bx, by, s=sizes, color=color, zorder=3, alpha=0.9)
                ax.plot(bx, by, color=color, lw=1.0, alpha=0.5, zorder=2)

            direction = "OVER" if offset > 0 else "UNDER"
            sign      = "+" if offset > 0 else ""
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)

            # Row label (horizon) on the leftmost column only
            if o_idx == 0:
                ax.set_ylabel(f"{h}min\nActual freq",
                              color="#aaa", fontsize=7.5)
            else:
                ax.set_ylabel("")

            ax.set_title(
                f"{sign}${abs(offset):,} [{direction}]\n"
                f"ECE={m['ece']:.3f}  Sk={m['skill']*100:+.0f}%  {m['verdict']}",
                color="white", fontsize=7, pad=5
            )
            ax.set_xlabel("Pred P(YES)", color="#888", fontsize=7)

    # ── Summary row: Skill Score per offset, grouped by horizon ──
    ax_skill = fig.add_subplot(gs[n_h, :n_off // 2])
    ax_ece   = fig.add_subplot(gs[n_h, n_off // 2:])

    for ax in [ax_skill, ax_ece]:
        ax.set_facecolor(panel_bg)
        ax.tick_params(colors="#888", labelsize=8)
        for sp in ax.spines.values():
            sp.set_color("#333")

    x        = np.arange(n_off)
    bar_w    = 0.8 / n_h
    colors_h = ["#60a5fa", "#34d399", "#f97316", "#f472b6"]  # one per horizon
    offset_labels = [f"{'+' if o>0 else ''}{o}" for o in STRIKE_OFFSETS]

    for h_idx, h in enumerate(horizons):
        skills = []
        eces   = []
        for offset in STRIKE_OFFSETS:
            key = (h, offset)
            if key in metrics:
                skills.append(metrics[key]["skill"] * 100)
                eces.append(metrics[key]["ece"])
            else:
                skills.append(0.0)
                eces.append(0.0)

        x_pos = x + (h_idx - n_h / 2 + 0.5) * bar_w
        color = colors_h[h_idx % len(colors_h)]
        ax_skill.bar(x_pos, skills, width=bar_w * 0.9,
                     color=color, alpha=0.8, label=f"{h}min")
        ax_ece.bar(x_pos, eces, width=bar_w * 0.9,
                   color=color, alpha=0.8, label=f"{h}min")

    ax_skill.axhline(0, color="#888", lw=0.8, ls="--", alpha=0.6)
    ax_skill.set_xticks(x)
    ax_skill.set_xticklabels(offset_labels, fontsize=8)
    ax_skill.set_title(
        "Skill Score by Horizon & Offset  (higher = better vs 50/50)",
        color="white", fontsize=9, pad=5)
    ax_skill.set_ylabel("Skill %", color="#888", fontsize=8)
    ax_skill.legend(fontsize=8, facecolor=panel_bg,
                    labelcolor="white", framealpha=0.3, title="Horizon",
                    title_fontsize=7)

    ax_ece.axhline(0.05, color="#22c55e", lw=0.8, ls="--",
                   alpha=0.6, label="Good (0.05)")
    ax_ece.axhline(0.10, color="#facc15", lw=0.8, ls="--",
                   alpha=0.6, label="OK (0.10)")
    ax_ece.set_xticks(x)
    ax_ece.set_xticklabels(offset_labels, fontsize=8)
    ax_ece.set_title(
        "ECE by Horizon & Offset  (lower = better calibration)",
        color="white", fontsize=9, pad=5)
    ax_ece.set_ylabel("ECE", color="#888", fontsize=8)
    ax_ece.legend(fontsize=8, facecolor=panel_bg,
                  labelcolor="white", framealpha=0.3)

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    plt.savefig(OUTPUT_FILE, dpi=150, bbox_inches="tight", facecolor=bg)
    print(f"  Chart saved → {OUTPUT_FILE}")
    plt.show()


# ═══════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════

def main() -> None:
    horizons = MINUTES_BEFORE if isinstance(MINUTES_BEFORE, list) \
               else [MINUTES_BEFORE]

    print("\n" + "═" * 66)
    print("  BTC Monte Carlo — Phase 1 Multi-Horizon Backtest")
    print("═" * 66)
    print(f"  Config  : {BACKTEST_DAYS}d window  |  "
          f"Horizons: {horizons}min  |  {N_PATHS:,} paths/sim")
    print(f"  Offsets : {STRIKE_OFFSETS}")
    print()

    print("─ Step 1: Fetching candles " + "─" * 39)
    all_candles = fetch_all_candles()

    print("\n─ Step 2: Running backtest " + "─" * 38)
    results = run_backtest(all_candles)

    print("\n─ Step 3: Computing metrics " + "─" * 37)
    metrics = compute_metrics(results)

    print("─ Step 4: Summary " + "─" * 47)
    print_summary(metrics)

    print("─ Step 5: Plotting " + "─" * 46)
    plot_results(metrics)


if __name__ == "__main__":
    main()
