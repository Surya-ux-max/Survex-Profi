"""
Logistics cost generator.

Cost depends on:
  - Product category (base fixed cost)
  - Original price   (value-linked component — insurance/handling)
  - Quantity         (per-unit shipping cost)
  - Multiplicative log-normal noise (±20 %) for realism

Returns a positive float rounded to 2 decimal places.
"""

import numpy as np
from config import CATEGORY_LOGISTICS_PARAMS


def generate_logistics_costs(
    categories:      np.ndarray,
    original_prices: np.ndarray,
    quantities:      np.ndarray,
    rng:             np.random.Generator,
) -> np.ndarray:
    """
    Returns a float numpy array of logistics costs > 0.
    """
    n    = len(categories)
    cost = np.empty(n, dtype=float)

    for cat, params in CATEGORY_LOGISTICS_PARAMS.items():
        mask = categories == cat
        if not mask.any():
            continue

        base        = params["base"]
        val_factor  = params["value_factor"]
        qty_factor  = params["qty_factor"]

        prices_sub  = original_prices[mask]
        qty_sub     = quantities[mask]
        n_sub       = mask.sum()

        # Deterministic component
        det = base + val_factor * prices_sub + qty_factor * qty_sub

        # Multiplicative noise: log-normal with mean=1, std≈0.15
        noise = rng.lognormal(mean=0.0, sigma=0.15, size=n_sub)

        cost[mask] = np.round(det * noise, 2)

    # Ensure strictly positive (safety clip)
    cost = np.maximum(cost, 10.0)
    return cost
