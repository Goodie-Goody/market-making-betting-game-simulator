"""
Headless long-run experiment driver.

Runs OUTSIDE the notebook (no plots, flat RAM) for a bounded wall-clock time,
continuously sampling market regimes and streaming per-regime results to
Snappy-compressed Parquet part-files. The notebook then loads the Parquet dataset
purely as a monitor/analysis layer.

Why a sweep instead of one config repeated: this model is stationary, so running a
single regime longer only tightens an estimate we already have. Sampling across the
parameter space instead means longer runtime buys *coverage* -- a robustness map of
where each quoting policy wins.

Usage
-----
    python run.py --hours 24                 # run for 24 wall-clock hours
    python run.py --minutes 30               # short run
    python run.py --hours 24 --out sweep_results --episodes-per-config 300

Output
------
    <out>/part-000001.parquet, part-000002.parquet, ...   (one per flush)
    <out>/manifest.json                                    (run metadata)

Each row = one (regime, strategy) evaluation with its metrics and the exact params
+ seed needed to replay it.
"""

from __future__ import annotations
import argparse, json, os, time, platform
from datetime import datetime, timezone

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from engine import MarketConfig, STRATEGIES, simulate_episode, mark_to_market_pnl

# Ranges the sweep samples from. Widen/narrow freely.
PARAM_RANGES = {
    "sigma":       (0.15, 0.65),   # per-step mid volatility  -> inventory risk
    "toxic_frac":  (0.00, 0.35),   # fraction of informed flow -> adverse selection
    "gamma":       (0.05, 0.30),   # A-S risk aversion
    "k":           (0.80, 2.50),   # fill-intensity decay      -> execution
}


def sample_config(rng: np.random.Generator) -> MarketConfig:
    p = {name: float(rng.uniform(lo, hi)) for name, (lo, hi) in PARAM_RANGES.items()}
    return MarketConfig(**p)


def evaluate_config(cfg: MarketConfig, n_episodes: int, seed: int) -> list[dict]:
    """Run n_episodes per strategy under one regime; return one summary row each."""
    rows = []
    for name, strat in STRATEGIES.items():
        rng = np.random.default_rng(seed)          # deterministic per (config, strat)
        pnls = np.empty(n_episodes)
        abs_term_inv = np.empty(n_episodes)
        max_inv = np.empty(n_episodes)
        adverse = np.empty(n_episodes)
        for i in range(n_episodes):
            r = simulate_episode(strat, cfg, rng, keep_path=True)
            pnls[i] = r.pnl
            abs_term_inv[i] = abs(r.terminal_inventory)
            max_inv[i] = r.max_abs_inventory
            adverse[i] = r.adverse_cost
        std = float(pnls.std(ddof=0))
        rows.append({
            "strategy": name,
            "sigma": cfg.sigma, "toxic_frac": cfg.toxic_frac,
            "gamma": cfg.gamma, "k": cfg.k,
            "n_episodes": n_episodes, "seed": seed,
            "mean_pnl": float(pnls.mean()),
            "std_pnl": std,
            "ret_over_risk": float(pnls.mean() / (std + 1e-12)),
            "worst_pnl": float(pnls.min()),
            "mean_abs_terminal_inv": float(abs_term_inv.mean()),
            "p95_max_inv": float(np.quantile(max_inv, 0.95)),
            "mean_adverse_cost": float(adverse.mean()),
        })
    return rows


def flush(rows: list[dict], out_dir: str, part_idx: int) -> None:
    table = pa.Table.from_pylist(rows)
    path = os.path.join(out_dir, f"part-{part_idx:06d}.parquet")
    pq.write_table(table, path, compression="snappy")


def main() -> None:
    ap = argparse.ArgumentParser(description="Long-run market-making regime sweep.")
    ap.add_argument("--hours", type=float, default=0.0)
    ap.add_argument("--minutes", type=float, default=0.0)
    ap.add_argument("--out", type=str, default="sweep_results")
    ap.add_argument("--episodes-per-config", type=int, default=300)
    ap.add_argument("--flush-every", type=int, default=50, help="configs per Parquet part")
    ap.add_argument("--seed", type=int, default=2025)
    args = ap.parse_args()

    duration_s = args.hours * 3600 + args.minutes * 60
    if duration_s <= 0:
        duration_s = 60.0  # default: a 1-minute smoke run
    os.makedirs(args.out, exist_ok=True)

    master = np.random.default_rng(args.seed)
    start = time.time()
    buffer: list[dict] = []
    part_idx = n_configs = n_rows = 0
    last_report = start

    print(f"[start] target {duration_s/3600:.2f} h  |  {args.episodes_per_config} episodes/config"
          f"  |  writing to {args.out}/  |  master seed {args.seed}")

    try:
        while time.time() - start < duration_s:
            cfg = sample_config(master)
            cfg_seed = int(master.integers(0, 2**63 - 1))
            buffer.extend(evaluate_config(cfg, args.episodes_per_config, cfg_seed))
            n_configs += 1

            if n_configs % args.flush_every == 0:
                part_idx += 1
                n_rows += len(buffer)
                flush(buffer, args.out, part_idx)
                buffer = []

            now = time.time()
            if now - last_report >= 10:      # progress line every ~10s
                el = now - start
                rate = n_configs / el
                eta = (duration_s - el) / 3600
                print(f"  {el/3600:6.3f} h | {n_configs:>8,} regimes "
                      f"| {rate:6.1f} regimes/s | ETA {eta:5.2f} h", flush=True)
                last_report = now
    except KeyboardInterrupt:
        print("\n[interrupt] flushing remaining buffer...")

    if buffer:                               # final flush
        part_idx += 1
        n_rows += len(buffer)
        flush(buffer, args.out, part_idx)

    manifest = {
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "requested_seconds": duration_s,
        "elapsed_seconds": time.time() - start,
        "n_configs": n_configs,
        "n_rows": n_rows,
        "episodes_per_config": args.episodes_per_config,
        "master_seed": args.seed,
        "param_ranges": PARAM_RANGES,
        "strategies": list(STRATEGIES),
        "python": platform.python_version(),
    }
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    total_ep = n_configs * args.episodes_per_config * len(STRATEGIES)
    print(f"[done] {n_configs:,} regimes | {n_rows:,} rows | "
          f"{total_ep:,} total episodes | {part_idx} parts | "
          f"{manifest['elapsed_seconds']/3600:.3f} h")


if __name__ == "__main__":
    main()
    