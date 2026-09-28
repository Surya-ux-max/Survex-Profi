"""
Dataset validator.

Runs all 16 required checks plus Selling_Price consistency verification.
Prints a formatted report. Raises ValueError on any failure.
"""

import numpy as np
import pandas as pd


def validate(df: pd.DataFrame) -> None:
    """
    Validates all constraints defined in the dataset specification.
    Prints PASS/FAIL for every check, then raises on any failure.
    """
    errors  = []
    results = []

    def check(name: str, condition: bool, detail: str = ""):
        tag = "✅ PASS" if condition else "❌ FAIL"
        results.append(f"  {tag}  {name}" + (f" — {detail}" if detail else ""))
        if not condition:
            errors.append(name)

    # ── 1. Row count ─────────────────────────────────────────────────────────
    check("Row count = 50,000", len(df) == 50_000, f"got {len(df):,}")

    # ── 2. Column count ──────────────────────────────────────────────────────
    check("Column count = 9", len(df.columns) == 9, f"got {len(df.columns)}")

    # ── 3. No duplicate Order_ID ─────────────────────────────────────────────
    check("No duplicate Order_ID", df["Order_ID"].nunique() == len(df))

    # ── 4. No missing values ─────────────────────────────────────────────────
    check("No missing values", df.isnull().sum().sum() == 0)

    # ── 5. No negative Original_Price ────────────────────────────────────────
    check("Original_Price > 0", (df["Original_Price"] > 0).all())

    # ── 6. No negative Logistics_Cost ────────────────────────────────────────
    check("Logistics_Cost > 0", (df["Logistics_Cost"] > 0).all())

    # ── 7. Quantity integer in [1, 15] ────────────────────────────────────────
    qty_ok = (
        df["Quantity"].dtype in [np.int32, np.int64, int]
        and df["Quantity"].between(1, 15).all()
    )
    check("Quantity integer in [1, 15]", qty_ok,
          f"min={df['Quantity'].min()}, max={df['Quantity'].max()}")

    # ── 8. Discount in [0, 50] ────────────────────────────────────────────────
    check("Discount in [0%, 50%]",
          df["Discount"].between(0, 50).all(),
          f"min={df['Discount'].min()}, max={df['Discount'].max()}")

    # ── 9. Selling_Price <= Original_Price ───────────────────────────────────
    check("Selling_Price <= Original_Price",
          (df["Selling_Price"] <= df["Original_Price"] + 0.01).all())

    # ── 10. Selling_Price > 0 ─────────────────────────────────────────────────
    check("Selling_Price > 0", (df["Selling_Price"] > 0).all())

    # ── 11. Exactly 10 product categories ────────────────────────────────────
    n_cats = df["Product_Category"].nunique()
    check("Exactly 10 product categories", n_cats == 10, f"got {n_cats}")

    # ── 12. ~10,000 unique customers ──────────────────────────────────────────
    n_cust = df["Customer_ID"].nunique()
    check("~10,000 unique customers",
          9_500 <= n_cust <= 10_500, f"got {n_cust:,}")

    # ── 13. Both Promotion and Non-Promotion exist ───────────────────────────
    promo_vals = set(df["Promotion_Period"].unique())
    check("Both Promotion & Non-Promotion exist",
          {"Promotion", "Non-Promotion"}.issubset(promo_vals))

    # ── 14. Discounted and non-discounted orders both exist ───────────────────
    check("Discounted and non-discounted orders both exist",
          df["Discount"].eq(0).any() and df["Discount"].gt(0).any())

    # ── 15. Realistic order frequency ────────────────────────────────────────
    freq = df["Customer_ID"].value_counts()
    has_low    = (freq <= 3).any()
    has_medium = freq.between(4, 10).any()
    has_high   = (freq > 10).any()
    check("Customer order frequency: long-tail distribution",
          has_low and has_medium and has_high,
          f"1-3 orders: {(freq<=3).sum()} | 4-10: {freq.between(4,10).sum()} | 10+: {(freq>10).sum()}")

    # ── 16. Selling_Price ≈ Original_Price × (1 - Discount/100) ─────────────
    expected = (df["Original_Price"] * (1 - df["Discount"] / 100)).round(2)
    max_err   = (df["Selling_Price"] - expected).abs().max()
    check("Selling_Price formula consistency (max error < ₹1)",
          max_err < 1.0, f"max absolute error = ₹{max_err:.4f}")

    # ── Print report ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  DATASET VALIDATION REPORT")
    print("=" * 60)
    for r in results:
        print(r)
    print("=" * 60)

    if errors:
        raise ValueError(
            f"\n❌ Validation failed on {len(errors)} check(s):\n  "
            + "\n  ".join(errors)
        )
    else:
        print("  🎉  All 16 checks PASSED — dataset is clean and ready.\n")
