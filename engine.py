"""
Market-making microstructure engine.

A single trading session ("episode") runs for T discrete steps. A reference
mid-price follows a random walk and settles at a terminal value V = m_T. We post
two-sided quotes each step. Counterparty flow is a mix of:

  * Uninformed (noise) traders whose fill probability decays with how far our
    quote sits from the mid:  P(fill | delta) = min(1, exp(-k * delta)).
    This is the Avellaneda-Stoikov Poisson-intensity execution model.

  * Informed (toxic) traders who know the terminal settlement V and only trade
    when it is profitable for THEM -- i.e. they lift our ask when ask < V and hit
    our bid when bid > V. This is the source of adverse selection.

We settle leftover inventory at V, so carrying inventory while the mid moves is
genuine risk. Three quoting strategies are compared under identical order flow.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple
import numpy as np


# --------------------------------------------------------------------------
# Reused primitives from the original Deep-ML scaffold (unchanged in spirit)
# --------------------------------------------------------------------------
def expected_value(values, probabilities) -> float:
    v = np.asarray(values, dtype=float)
    p = np.asarray(probabilities, dtype=float)
    return float(np.dot(v, p))


def mark_to_market_pnl(cash: float, inventory: float, settlement_value: float) -> float:
    return float(cash + inventory * settlement_value)


def summarize_pnls(pnls) -> Dict[str, float]:
    a = np.asarray(pnls, dtype=float)
    return {
        "mean": float(a.mean()),
        "std": float(a.std(ddof=0)),
        "worst": float(a.min()),
        "ret_over_risk": float(a.mean() / (a.std(ddof=0) + 1e-12)),
    }


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
@dataclass
class MarketConfig:
    S0: float = 100.0          # starting mid
    sigma: float = 0.30        # per-step std of the mid random walk
    T: int = 100               # steps per episode
    gamma: float = 0.10        # A-S risk aversion
    k: float = 1.50            # fill-intensity decay (higher = fills die off faster with spread)
    toxic_frac: float = 0.15   # fraction of arrivals that are informed
    base_half_spread: float = 0.90   # half-spread for naive / linear strategies
    skew_strength: float = 0.30      # inventory skew for the linear strategy


# --------------------------------------------------------------------------
# Quoting strategies: (mid, inventory q, step t, cfg) -> (bid, ask)
# --------------------------------------------------------------------------
def quote_naive(mid: float, q: float, t: int, cfg: MarketConfig) -> Tuple[float, float]:
    h = cfg.base_half_spread
    return mid - h, mid + h


def quote_linear_skew(mid: float, q: float, t: int, cfg: MarketConfig) -> Tuple[float, float]:
    center = mid - cfg.skew_strength * q
    h = cfg.base_half_spread
    return center - h, center + h


def quote_avellaneda_stoikov(mid: float, q: float, t: int, cfg: MarketConfig) -> Tuple[float, float]:
    tau = cfg.T - t                       # steps of variance remaining
    var_rem = (cfg.sigma ** 2) * tau
    reservation = mid - q * cfg.gamma * var_rem
    spread = cfg.gamma * var_rem + (2.0 / cfg.gamma) * np.log1p(cfg.gamma / cfg.k)
    half = spread / 2.0
    return reservation - half, reservation + half


STRATEGIES: Dict[str, Callable] = {
    "Naive symmetric": quote_naive,
    "Linear inventory skew": quote_linear_skew,
    "Avellaneda-Stoikov": quote_avellaneda_stoikov,
}


# --------------------------------------------------------------------------
# Episode simulation
# --------------------------------------------------------------------------
@dataclass
class EpisodeResult:
    pnl: float
    terminal_inventory: float
    max_abs_inventory: float
    n_fills: int
    n_toxic_fills: int
    adverse_cost: float               # cumulative loss to informed flow (vs settlement)
    inventory_path: np.ndarray = field(repr=False, default=None)


def simulate_episode(strategy: Callable, cfg: MarketConfig, rng: np.random.Generator,
                     keep_path: bool = False) -> EpisodeResult:
    # Pre-generate the mid path so informed traders can "know" the terminal V.
    shocks = rng.normal(0.0, cfg.sigma, size=cfg.T)
    mid = cfg.S0 + np.concatenate([[0.0], np.cumsum(shocks)])   # length T+1, mid[0]=S0
    V = float(mid[cfg.T])                                       # terminal settlement

    cash, q = 0.0, 0.0
    n_fills = n_toxic = 0
    adverse_cost = 0.0
    inv_path = np.empty(cfg.T) if keep_path else None

    for t in range(cfg.T):
        m = float(mid[t])
        bid, ask = strategy(m, q, t, cfg)

        informed = rng.random() < cfg.toxic_frac
        if informed:
            # Trades only when profitable for the counterparty -> bad for us.
            if ask < V:                      # they buy, we sell at ask below true value
                cash += ask; q -= 1.0
                n_fills += 1; n_toxic += 1
                adverse_cost += (V - ask)
            elif bid > V:                    # they sell, we buy at bid above true value
                cash -= bid; q += 1.0
                n_fills += 1; n_toxic += 1
                adverse_cost += (bid - V)
        else:
            # Noise trader: 50/50 side, fill probability decays with quote distance.
            if rng.random() < 0.5:           # incoming buy hits our ask
                delta = ask - m
                if rng.random() < min(1.0, np.exp(-cfg.k * delta)):
                    cash += ask; q -= 1.0; n_fills += 1
            else:                            # incoming sell hits our bid
                delta = m - bid
                if rng.random() < min(1.0, np.exp(-cfg.k * delta)):
                    cash -= bid; q += 1.0; n_fills += 1

        if keep_path:
            inv_path[t] = q

    pnl = mark_to_market_pnl(cash, q, V)
    return EpisodeResult(
        pnl=pnl,
        terminal_inventory=q,
        max_abs_inventory=float(np.max(np.abs(inv_path))) if keep_path else abs(q),
        n_fills=n_fills,
        n_toxic_fills=n_toxic,
        adverse_cost=adverse_cost,
        inventory_path=inv_path,
    )


def run_stress_test(cfg: MarketConfig, n_episodes: int, seed: int = 42) -> "pd.DataFrame":
    import pandas as pd
    rng = np.random.default_rng(seed)
    rows = []
    for name, strat in STRATEGIES.items():
        for ep in range(n_episodes):
            r = simulate_episode(strat, cfg, rng, keep_path=True)
            rows.append({
                "strategy": name, "episode": ep, "pnl": r.pnl,
                "terminal_inventory": r.terminal_inventory,
                "max_abs_inventory": r.max_abs_inventory,
                "n_fills": r.n_fills, "n_toxic_fills": r.n_toxic_fills,
                "adverse_cost": r.adverse_cost,
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import pandas as pd
    cfg = MarketConfig()
    df = run_stress_test(cfg, n_episodes=2000, seed=7)
    summary = df.groupby("strategy").agg(
        mean_pnl=("pnl", "mean"),
        std_pnl=("pnl", "std"),
        ret_over_risk=("pnl", lambda x: x.mean() / (x.std(ddof=0) + 1e-12)),
        mean_abs_term_inv=("terminal_inventory", lambda x: x.abs().mean()),
        p95_max_inv=("max_abs_inventory", lambda x: x.quantile(0.95)),
        mean_adverse=("adverse_cost", "mean"),
        mean_fills=("n_fills", "mean"),
    )
    pd.set_option("display.width", 160)
    print(summary.round(3))
