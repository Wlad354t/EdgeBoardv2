"""Turn a predicted margin into P(home covers), P(push), P(away covers).

The final margin is modelled as a discrete distribution over integers:

    P(margin = m)  is proportional to  Normal_pdf(m; mu, sigma) * w(|m|)

w(|m|) are "key-number" weights learned from training data: how much more
often a margin of |m| happens than a smooth curve would predict. In the NFL
this captures the spikes at 3, 7, 6, 10, 14; for smoother sports the weights
come out close to 1. Pushes on whole-number lines fall out exactly.
"""
from __future__ import annotations

import numpy as np

MAX_MARGIN = 80
MARGINS = np.arange(-MAX_MARGIN, MAX_MARGIN + 1)


def learn_key_weights(results: np.ndarray, bandwidth: float = 2.5,
                      min_games: int = 500, max_w: float = 3.0) -> np.ndarray:
    """Empirical frequency of |margin| divided by a kernel-smoothed version."""
    w = np.ones(MAX_MARGIN + 1)
    r = np.abs(results[~np.isnan(results)]).astype(int)
    r = r[r <= MAX_MARGIN]
    if len(r) < min_games:
        return w
    counts = np.bincount(r, minlength=MAX_MARGIN + 1).astype(float)
    # Mirror so the kernel at 0 isn't biased by the boundary.
    full = np.concatenate([counts[:0:-1], counts])
    x = np.arange(-int(4 * bandwidth), int(4 * bandwidth) + 1)
    kern = np.exp(-0.5 * (x / bandwidth) ** 2)
    kern /= kern.sum()
    smooth = np.convolve(full, kern, mode="same")[MAX_MARGIN:]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(smooth > 0, counts / smooth, 1.0)
    # Light shrinkage toward 1 for sparse tails.
    n_eff = smooth
    shrink = n_eff / (n_eff + 20.0)
    w = 1.0 + shrink * (ratio - 1.0)
    return np.clip(w, 1.0 / max_w, max_w)


def margin_pmf(mu: float, sigma: float, key_w: np.ndarray) -> np.ndarray:
    z = (MARGINS - mu) / sigma
    p = np.exp(-0.5 * z * z) * key_w[np.abs(MARGINS)]
    return p / p.sum()


def implied_mean(lines: np.ndarray, p_fair: np.ndarray, sigma: float, key_w: np.ndarray,
                 mu_grid: np.ndarray | None = None) -> np.ndarray:
    """Market-implied mean margin.

    A spread is set so each side is a coin flip after removing the vig, which
    makes the line a *median*, not a mean. Near key numbers those differ
    (home -2.5 wins by exactly 3 a lot). Without this correction, a
    Normal-plus-key-numbers model invents fake edges on every line next to
    3 or 7. We solve for the mean mu0 at which the model reproduces the
    market's no-vig probability, so the market alone produces zero edge and
    any edge must come from the model's own information.
    """
    if mu_grid is None:
        mu_grid = np.arange(-50.0, 50.0001, 0.05)
    z = (MARGINS[None, :] - mu_grid[:, None]) / sigma
    P = np.exp(-0.5 * z * z) * key_w[np.abs(MARGINS)][None, :]
    P /= P.sum(axis=1, keepdims=True)
    out = np.full(len(lines), np.nan)
    for L in np.unique(lines[~np.isnan(lines)]):
        sel = lines == L
        above = P[:, MARGINS > L].sum(axis=1)
        below = P[:, MARGINS < L].sum(axis=1)
        p_np = above / (above + below)           # increasing in mu
        out[sel] = np.interp(p_fair[sel], p_np, mu_grid)
    return out


def cover_probs(mu: float, sigma: float, line: float, key_w: np.ndarray) -> tuple[float, float, float]:
    """Return (p_home_cover, p_push, p_away_cover) for a home-perspective line."""
    p = margin_pmf(mu, sigma, key_w)
    home = p[MARGINS > line].sum()
    push = p[MARGINS == line].sum() if float(line).is_integer() else 0.0
    return float(home), float(push), float(1.0 - home - push)
