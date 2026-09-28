"""
Quantity generator.

Quantity is driven by:
  - Discount level  → non-linear / diminishing-returns relationship
  - Product category → different base multipliers
  - Promotion period → slight positive nudge
  - Poisson-like noise for realism

Returns integer quantity in [1, 15].
"""

import numpy as np
from config import CATEGORY_QTY_BASE

# Discount → mean quantity mapping (piecewise linear, with diminishing returns)
# discount_band_upper → expected_mean_qty (relative, before category scaling)
DISCOUNT_QTY_MAP = [
    (0,  1.50),
    (10, 1.80),
    (20, 2.20),
    (30, 2.60),
    (40, 2.70),
    (50, 2.80),
]
_DISC_BREAK  = np.array([x[0] for x in DISCOUNT_QTY_MAP], dtype=float)
_QTY_BREAK   = np.array([x[1] for x in DISCOUNT_QTY_MAP], dtype=float)

# Category-level discount sensitivity (how strongly qty responds to discount)
CATEGORY_DISCOUNT_SENSITIVITY = {
    "Electronics":            0.6,   # weak  — people don't bulk-buy laptops
    "Clothing":               1.2,   # strong
    "Home & Kitchen":         1.0,   # moderate
    "Beauty & Personal Care": 1.3,   # strong — toiletries bought in bulk
    "Sports & Fitness":       0.8,   # moderate-weak
    "Books":                  1.1,   # moderate-strong
    "Grocery":                1.4,   # strongest — very price-elastic
    "Furniture":              0.5,   # very weak
    "Accessories":            1.1,   # moderate
    "Toys & Games":           1.0,   # moderate
}

PROMOTION_QTY_BOOST = 0.15   # adds 15 % to base mean during promotion


def _discount_to_mean_qty(discounts: np.ndarray) -> np.ndarray:
    """Piecewise-linear interpolation of discount → base mean quantity."""
    return np.interp(discounts.astype(float), _DISC_BREAK, _QTY_BREAK)


def generate_quantities(
    categories:   np.ndarray,
    discounts:    np.ndarray,
    is_promotion: np.ndarray,
    rng:          np.random.Generator,
) -> np.ndarray:
    """
    Returns an integer numpy array of quantities in [1, 15].
    """
    # Base mean from discount curve
    base_mean = _discount_to_mean_qty(discounts)

    # Category scaling: base qty × category_base
    cat_base = np.array([CATEGORY_QTY_BASE[c] for c in categories], dtype=float)

    # Category discount sensitivity: how much the discount curve stretches
    cat_sens  = np.array([CATEGORY_DISCOUNT_SENSITIVITY[c] for c in categories], dtype=float)

    # Effective mean: combine base qty, category scaling, sensitivity
    # Sensitivity stretches the discount-response above the zero-discount floor
    floor_mean = CATEGORY_QTY_BASE.copy()
    zero_qty   = np.array([CATEGORY_QTY_BASE[c] * 1.50 for c in categories], dtype=float)
    delta      = (base_mean - 1.50) * cat_sens          # how much above floor
    effective_mean = (zero_qty + delta) * (cat_base / 1.50)

    # Promotion boost
    effective_mean *= np.where(is_promotion, 1.0 + PROMOTION_QTY_BOOST, 1.0)

    # Draw from Poisson (produces integer, skewed, realistic)
    # Offset by 1 so minimum = 1
    lambda_param = np.clip(effective_mean - 1.0, 0.1, 13.0)
    qty = rng.poisson(lam=lambda_param) + 1

    # Clip to [1, 15]
    qty = np.clip(qty, 1, 15).astype(int)
    return qty
