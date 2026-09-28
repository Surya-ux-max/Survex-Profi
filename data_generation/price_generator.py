"""
Price generator — log-normal, category-specific, clipped to realistic ranges.
"""

import numpy as np
from config import CATEGORY_PRICE_PARAMS


def generate_prices(categories: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Returns an array of Original_Price values (float → rounded to int ₹).
    Each price is drawn from a category-specific log-normal distribution
    and clipped to the defined min/max for that category.
    """
    prices = np.empty(len(categories), dtype=float)

    for cat, params in CATEGORY_PRICE_PARAMS.items():
        mask = categories == cat
        n    = mask.sum()
        if n == 0:
            continue

        raw = rng.lognormal(
            mean  = params["mean_log"],
            sigma = params["std_log"],
            size  = n,
        )
        clipped = np.clip(raw, params["min"], params["max"])
        prices[mask] = np.round(clipped).astype(float)

    return prices
