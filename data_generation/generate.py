"""
Main dataset generation script.
Orchestrates all generators and saves the output CSV to ../data/

Run:
    python data_generation/generate.py   (from project root)
    python generate.py                   (from data_generation/)
"""

import sys
import os

# ── Ensure this module's own directory is always on the path ──────────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import numpy as np
import pandas as pd

from config import (
    RANDOM_SEED, NUM_ROWS, NUM_CUSTOMERS,
    CATEGORIES, CATEGORY_WEIGHTS,
    PROMOTION_WEIGHT, OUTPUT_FILE,
)
from customers          import build_customer_pool, build_customer_order_weights
from price_generator    import generate_prices
from discount_generator import generate_discounts
from quantity_generator import generate_quantities
from logistics_generator import generate_logistics_costs
from validator          import validate


# ─── Output directory ─────────────────────────────────────────────────────────
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_PATH = os.path.join(OUTPUT_DIR, OUTPUT_FILE)


def generate_dataset() -> pd.DataFrame:
    print("🔧  Initialising random generator (seed={})...".format(RANDOM_SEED))
    rng = np.random.default_rng(RANDOM_SEED)

    # ── 1. Build customer pool ────────────────────────────────────────────────
    print("👥  Building customer pool ({:,} customers)...".format(NUM_CUSTOMERS))
    customer_pool   = build_customer_pool(rng)
    customer_ids    = list(customer_pool.keys())
    customer_weights = build_customer_order_weights(rng, customer_pool, NUM_ROWS)

    # ── 2. Sample customers for each order (long-tail) ────────────────────────
    print("🛒  Sampling customer IDs for {:,} orders...".format(NUM_ROWS))
    order_customer_ids = rng.choice(customer_ids, size=NUM_ROWS, p=customer_weights)
    order_segments     = [customer_pool[cid] for cid in order_customer_ids]

    # ── 3. Assign product categories (realistic weights) ──────────────────────
    print("📦  Assigning product categories...")
    categories = rng.choice(CATEGORIES, size=NUM_ROWS, p=CATEGORY_WEIGHTS)

    # ── 4. Assign promotion period ────────────────────────────────────────────
    print("🏷️   Assigning promotion periods...")
    is_promotion  = rng.random(NUM_ROWS) < PROMOTION_WEIGHT
    promo_labels  = np.where(is_promotion, "Promotion", "Non-Promotion")

    # ── 5. Generate Original_Price ────────────────────────────────────────────
    print("💰  Generating original prices...")
    original_prices = generate_prices(categories, rng)

    # ── 6. Generate Discount ──────────────────────────────────────────────────
    print("🏷️   Generating discounts...")
    discounts = generate_discounts(categories, is_promotion, order_segments, rng)

    # ── 7. Compute Selling_Price ──────────────────────────────────────────────
    print("💲  Computing selling prices...")
    selling_prices = np.round(original_prices * (1 - discounts / 100), 2)

    # ── 8. Generate Quantity ──────────────────────────────────────────────────
    print("📊  Generating quantities...")
    quantities = generate_quantities(categories, discounts, is_promotion, rng)

    # ── 9. Generate Logistics_Cost ────────────────────────────────────────────
    print("🚚  Generating logistics costs...")
    logistics_costs = generate_logistics_costs(
        categories, original_prices, quantities, rng
    )

    # ── 10. Assemble DataFrame ────────────────────────────────────────────────
    print("🗂️   Assembling DataFrame...")
    order_ids = [f"ORD{str(i+1).zfill(6)}" for i in range(NUM_ROWS)]

    df = pd.DataFrame({
        "Order_ID":         order_ids,
        "Product_Category": categories,
        "Original_Price":   original_prices.astype(int),
        "Discount":         discounts,
        "Selling_Price":    selling_prices,
        "Quantity":         quantities,
        "Logistics_Cost":   logistics_costs,
        "Customer_ID":      order_customer_ids,
        "Promotion_Period": promo_labels,
    })

    return df


def print_summary(df: pd.DataFrame) -> None:
    discounted     = df["Discount"].gt(0)
    print("\n" + "=" * 60)
    print("  DATASET SUMMARY REPORT")
    print("=" * 60)
    print(f"  Rows                   : {len(df):,}")
    print(f"  Columns                : {len(df.columns)}")
    print(f"  Unique Customers       : {df['Customer_ID'].nunique():,}")
    print(f"  Unique Categories      : {df['Product_Category'].nunique()}")
    print(f"  Discounted Orders      : {discounted.sum():,}  ({discounted.mean()*100:.1f}%)")
    print(f"  Non-Discounted Orders  : {(~discounted).sum():,}  ({(~discounted).mean()*100:.1f}%)")
    print(f"  Promotion Orders       : {(df['Promotion_Period']=='Promotion').sum():,}  "
          f"({(df['Promotion_Period']=='Promotion').mean()*100:.1f}%)")
    print(f"  Avg Original Price     : ₹{df['Original_Price'].mean():,.2f}")
    print(f"  Avg Discount           : {df['Discount'].mean():.2f}%")
    print(f"  Avg Selling Price      : ₹{df['Selling_Price'].mean():,.2f}")
    print(f"  Avg Quantity           : {df['Quantity'].mean():.2f}")
    print(f"  Avg Logistics Cost     : ₹{df['Logistics_Cost'].mean():,.2f}")
    print("=" * 60)

    print("\n  Category Distribution:")
    cat_dist = df["Product_Category"].value_counts()
    for cat, cnt in cat_dist.items():
        print(f"    {cat:<28} {cnt:>6,}  ({cnt/len(df)*100:.1f}%)")

    print("\n  Avg Discount by Category:")
    cat_disc = df.groupby("Product_Category")["Discount"].mean().sort_values(ascending=False)
    for cat, avg in cat_disc.items():
        print(f"    {cat:<28} {avg:.2f}%")

    print("\n  Avg Quantity by Discount Band:")
    bins   = [-1, 0, 10, 20, 30, 50]
    labels = ["0%", "1-10%", "11-20%", "21-30%", "31-50%"]
    df["_disc_band"] = pd.cut(df["Discount"], bins=bins, labels=labels)
    band_qty = df.groupby("_disc_band", observed=True)["Quantity"].mean()
    for band, qty in band_qty.items():
        print(f"    {str(band):<10} avg qty = {qty:.2f}")
    df.drop(columns=["_disc_band"], inplace=True)
    print()


def print_data_dictionary() -> None:
    print("=" * 60)
    print("  DATA DICTIONARY")
    print("=" * 60)
    cols = [
        ("Order_ID",         "string",  "Unique order identifier. Format: ORD000001."),
        ("Product_Category", "string",  "One of 10 product categories."),
        ("Original_Price",   "integer", "Listed price before discount (₹). Range varies by category."),
        ("Discount",         "integer", "Discount percentage applied [0–50]."),
        ("Selling_Price",    "float",   "Final price paid = Original_Price × (1 − Discount/100). 2 d.p."),
        ("Quantity",         "integer", "Number of units ordered [1–15]."),
        ("Logistics_Cost",   "float",   "Shipping & handling cost (₹). Depends on category, price, qty."),
        ("Customer_ID",      "string",  "Customer identifier. Format: C00001. ~10,000 unique customers."),
        ("Promotion_Period", "string",  "'Promotion' or 'Non-Promotion'."),
    ]
    for col, dtype, desc in cols:
        print(f"  {col:<28} [{dtype:<8}]  {desc}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    df = generate_dataset()

    print("\n🔍  Running validation checks...")
    validate(df)

    print_summary(df)
    print_data_dictionary()

    print(f"💾  Saving dataset to: {OUTPUT_PATH}")
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"✅  Done! {len(df):,} rows × {len(df.columns)} columns saved.\n")
