"""
Discount generator.

Discount is driven by:
  - Product category (base probability & magnitude)
  - Promotion period  (boost to probability + magnitude)
  - Customer segment  (price-sensitive → more discounts; loyal → fewer)

Returns integer discount % in [0, 50].
"""

import numpy as np
from config import (
    CATEGORY_DISCOUNT_PARAMS,
    PROMOTION_DISCOUNT_BOOST,
)

# Per-segment discount probability multipliers (hidden behaviour)
SEGMENT_DISCOUNT_MULT = {
    "price_sensitive":    1.35,
    "regular":            1.00,
    "promotion_sensitive":1.20,
    "loyal":              0.75,
}

# During promotion: additional mean discount shift (percentage points)
PROMOTION_MEAN_BOOST = 6  # extra % added to mean when Promotion


def generate_discounts(
    categories:       np.ndarray,
    is_promotion:     np.ndarray,
    customer_segments: list,
    rng:              np.random.Generator,
) -> np.ndarray:
    """
    Returns an integer numpy array of discount percentages in [0, 50].
    """
    n        = len(categories)
    discounts = np.zeros(n, dtype=int)

    for cat, params in CATEGORY_DISCOUNT_PARAMS.items():
        mask = np.where(categories == cat)[0]
        if len(mask) == 0:
            continue

        base_prob    = params["base_prob"]
        base_mean    = params["mean"]
        base_std     = params["std"]

        promo_flags  = is_promotion[mask]
        seg_list     = [customer_segments[i] for i in mask]

        # ── Step 1: Compute per-row discount probability ──────────────────
        seg_mult  = np.array([SEGMENT_DISCOUNT_MULT[s] for s in seg_list])
        promo_mult= np.where(promo_flags, PROMOTION_DISCOUNT_BOOST, 1.0)
        row_prob  = np.clip(base_prob * seg_mult * promo_mult, 0.0, 0.92)

        # ── Step 2: Decide which rows get a discount ───────────────────────
        gets_discount = rng.random(len(mask)) < row_prob

        # ── Step 3: Draw magnitude for discounted rows ─────────────────────
        n_disc = gets_discount.sum()
        if n_disc > 0:
            # Promotion shifts the mean upward
            promo_shift = np.where(promo_flags[gets_discount], PROMOTION_MEAN_BOOST, 0)
            means = base_mean + promo_shift

            # Truncated normal: draw from normal, then clip [1, 50]
            raw = rng.normal(loc=means, scale=base_std, size=n_disc)
            raw = np.clip(np.round(raw).astype(int), 1, 50)

            discounted_vals = raw
        else:
            discounted_vals = np.array([], dtype=int)

        row_discounts = np.zeros(len(mask), dtype=int)
        row_discounts[gets_discount] = discounted_vals
        discounts[mask] = row_discounts

    return discounts
