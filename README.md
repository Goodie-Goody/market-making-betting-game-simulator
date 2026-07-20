# Market-Making Microstructure Lab

A simulation study of the three risks a market maker actually faces —
**inventory risk, adverse selection, and quote execution** — and which quoting
policy survives under which market conditions.

This started from a market-making & betting-game exercise (Deep-ML). I refactored
the quoting logic into a clean, class-based engine and added the ingredient the
original left out: **order flow that responds to your quotes.** Once fills depend on
where you quote, quoting decisions have consequences and the comparison below means
something.

## The question

> Given inventory risk, toxic (informed) flow, and spread-dependent fills, which
> quoting policy earns the best risk-adjusted return — and does the answer change
> with the market regime?

## The model

- **Mid price:** random walk over `T` steps, settled at terminal `V = m_T`, so held
  inventory is real directional risk.
- **Execution:** uninformed fills follow the Avellaneda–Stoikov Poisson intensity,
  `P(fill | δ) = min(1, e^(−k·δ))` — fills decay as you widen.
- **Adverse selection:** a fraction of arrivals are informed, know `V`, and trade
  only when it profits them (i.e. only when it costs you).

## Policies compared

| Policy | Idea |
|---|---|
| Naive symmetric | Fixed spread around the mid, no inventory awareness |
| Linear inventory skew | Fixed spread, mid shifted linearly against inventory |
| Avellaneda–Stoikov | Reservation price **and** optimal spread; both scale with volatility and time remaining |

## Headline result

Skewing beats not-skewing by a mile — the naive policy posts a **negative** mean
P&L under volatile, toxic flow. Between the two skewing policies the result is
conditional and honest: a linear skew wins raw P&L in calm markets, while
Avellaneda–Stoikov wins **risk-adjusted** return specifically in the hostile
(high-volatility, high-toxicity) regime it was designed for, because it widens
exactly when the market is most dangerous.

## Run it

```bash
pip install numpy pandas matplotlib jupyter
jupyter notebook Market_Making_Microstructure_Lab.ipynb
```

Everything runs off a single seeded RNG stream, so the notebook is fully
reproducible top-to-bottom.

## Limitations

Informed traders know the exact terminal value (a strong form of adverse selection);
the mid is a constant-volatility arithmetic random walk; one unit per fill, no queue
position, latency, or fees. These are deliberate simplifications, and each is a
natural next experiment rather than a rewrite.