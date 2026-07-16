"""
Market-Making & Betting-Game Simulator scaffold.

Run this with: python scaffold.py
Uses functions defined in model.py.
"""

from model import *  # noqa: F401, F403 (pulls in your solution functions)

"""End-to-end demo of the market-making & betting-game simulator."""

import numpy as np


def _format_rule(rule):
    if isinstance(rule, str):
        return rule
    if isinstance(rule, dict):
        return "policy dict"
    try:
        return sorted(rule)
    except TypeError:
        return rule


def main():
    np.random.seed(0)

    # --- Betting-game warm-ups: expected value & dice/card puzzles ---
    ev_die = expected_value([1, 2, 3, 4, 5, 6], [1/6] * 6)
    print(f"Fair 6-sided die EV: {ev_die:.4f}")

    rr = one_reroll_die_value(6)
    reroll_val = rr['value']
    reroll_set = rr['reroll_faces']
    print(f"One-reroll 6-sided die value: {reroll_val:.4f}, reroll if in {_format_rule(reroll_set)}")

    pp = pay_per_reroll_die_game(sides=6, reroll_cost=0.5)
    pay_reroll_ev = pp['value']
    threshold = pp['threshold']
    print(f"Pay-per-reroll game: stop when >= {threshold}, EV = {pay_reroll_ev:.4f}")

    cc = red_black_card_game_value(num_red=3, num_black=3)
    card_ev = cc['value']
    card_rule = cc.get('stop_now', None)
    print(f"Red/black card game EV: {card_ev:.4f} (rule sample: {card_rule})")

    # --- Quoting primitives ---
    fair_value = 100.0
    spread_width = 2.0
    q = make_quotes(fair_value, spread_width)
    bid, ask = q['bid'], q['ask']
    print(f"\nSymmetric quotes around {fair_value}: bid={bid:.2f}, ask={ask:.2f}")

    wq = make_quotes(fair_value, uncertainty_spread(base_spread=1.0, uncertainty=2.5))
    wide_bid, wide_ask = wq['bid'], wq['ask']
    print(f"Uncertainty-widened quotes: bid={wide_bid:.2f}, ask={wide_ask:.2f}")

    sq = inventory_skewed_quotes(fair_value, spread_width, inventory=3, skew_strength=0.2)
    skew_bid, skew_ask = sq['bid'], sq['ask']
    print(f"Inventory-skewed quotes (inv=+3): bid={skew_bid:.2f}, ask={skew_ask:.2f}")

    # --- Single trade + P&L mechanics ---
    state = {"cash": 0.0, "inventory": 0}
    state = execute_trade(state, side="buy", bid=bid, ask=ask, size=1)  # counterparty buys at our ask
    print(f"\nAfter counterparty buy: cash={state['cash']:.2f}, inv={state['inventory']}")
    mtm = mark_to_market_pnl(state["cash"], state["inventory"], settlement_value=fair_value)
    print(f"Mark-to-market P&L at {fair_value}: {mtm:.4f}")

    # --- Adverse selection: informed counterparty scenario ---
    adv_loss = adverse_selection_loss(
        fair_value=100.0, bid=99.0, ask=101.0,
        informed_values=[98.0, 100.0, 102.0],
        informed_probabilities=[1/3, 1/3, 1/3],
    )
    print(f"Expected adverse-selection loss per trade: {adv_loss:.4f}")

    # --- Belief updating primitives ---
    updated_fv = update_fair_value_from_trade(fair_value, side="buy", bid=bid, ask=ask, adjustment=0.1)
    print(f"Fair value after informed buy: {updated_fv:.4f}")

    remaining_counts = {+1: 3, -1: 3}
    urc = update_remaining_card_value(remaining_counts, revealed_value=+1)
    new_ev = urc['expected_value'] if isinstance(urc, dict) else urc
    print(f"Remaining-card EV after revealing red: {new_ev:.4f}")

    # --- Full episode simulation ---
    config = {
        "spread_width": 2.0,
        "skew_strength": 0.15,
        "belief_adjustment": 0.1,
        "size": 1,
        "uncertainty": 1.0,
        "base_spread": 1.0,
    }
    true_value = 101.5
    counterparty_sides = ["buy", "sell", "buy", "buy", "sell", "buy", "sell"]
    episode = run_market_making_episode(
        true_value=true_value,
        counterparty_sides=counterparty_sides,
        initial_fair_value=100.0,
        config=config,
    )
    print(f"\nEpisode result: {episode}")

    # --- Many-episode Monte Carlo summary ---
    pnls = []
    for _ in range(200):
        n_trades = np.random.randint(5, 15)
        sides = list(np.random.choice(["buy", "sell"], size=n_trades))
        tv = float(np.random.normal(100.0, 2.0))
        ep = run_market_making_episode(
            true_value=tv,
            counterparty_sides=sides,
            initial_fair_value=100.0,
            config=config,
        )
        pnl = ep["pnl"] if isinstance(ep, dict) and "pnl" in ep else (
            ep.get("final_pnl") if isinstance(ep, dict) else ep[-1]
        )
        pnls.append(float(pnl))

    summary = summarize_episode_pnls(pnls)
    print(f"\nAcross 200 episodes: {summary}")


if __name__ == "__main__":
    main()
