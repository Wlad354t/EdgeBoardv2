"""Sport configurations: default model parameters and search grids.

Every number here is a *starting point*. `tune.py` re-fits the rating
parameters on training seasons only, so these defaults only matter when
you skip tuning (e.g. a small uploaded dataset).
"""
from dataclasses import dataclass, field


@dataclass
class SportConfig:
    name: str
    # Margin-Elo rating parameters (ratings are in points/runs/goals vs. league average)
    k: float                 # learning rate: share of the prediction error absorbed per game
    hfa: float               # home advantage in points
    regress: float           # fraction pulled back to league mean at season start
    mov_cap: float           # cap on |actual - predicted| per update (blowout dampening)
    # Probability model
    sigma_prior: float       # fallback SD of result around predicted margin
    use_key_numbers: bool    # learn integer-margin weights (NFL 3/7, etc.)
    # Betting defaults
    default_odds: int = -110
    min_edge: float = 0.03   # minimum expected value per $1 staked
    # Tuning grid
    grid: dict = field(default_factory=dict)


SPORTS = {
    "NFL": SportConfig(
        name="NFL", k=0.10, hfa=2.0, regress=0.33, mov_cap=14.0,
        sigma_prior=13.2, use_key_numbers=True,
        grid={"k": [0.06, 0.08, 0.10, 0.12, 0.15],
              "hfa": [1.0, 1.5, 2.0, 2.5],
              "regress": [0.2, 0.33, 0.5],
              "mov_cap": [10.0, 14.0, 21.0]},
    ),
    "NBA": SportConfig(
        name="NBA", k=0.05, hfa=2.5, regress=0.3, mov_cap=20.0,
        sigma_prior=12.0, use_key_numbers=False,
        grid={"k": [0.02, 0.035, 0.05, 0.07],
              "hfa": [1.5, 2.0, 2.5, 3.0],
              "regress": [0.2, 0.33, 0.5],
              "mov_cap": [15.0, 20.0, 30.0]},
    ),
    "MLB": SportConfig(
        name="MLB", k=0.015, hfa=0.15, regress=0.4, mov_cap=5.0,
        sigma_prior=4.4, use_key_numbers=True,
        grid={"k": [0.008, 0.012, 0.016, 0.022],
              "hfa": [0.0, 0.15, 0.3],
              "regress": [0.3, 0.45, 0.6],
              "mov_cap": [4.0, 6.0, 10.0]},
    ),
    "WNBA": SportConfig(
        name="WNBA", k=0.06, hfa=2.0, regress=0.35, mov_cap=20.0,
        sigma_prior=11.5, use_key_numbers=False,
        grid={"k": [0.03, 0.05, 0.07, 0.09],
              "hfa": [1.0, 2.0, 3.0],
              "regress": [0.2, 0.35, 0.5],
              "mov_cap": [15.0, 20.0, 30.0]},
    ),
}
