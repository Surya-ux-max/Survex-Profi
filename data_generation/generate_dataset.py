"""
Self-contained dataset generation script.
All logic is in one file to avoid module resolution issues.

Run from any directory:
    python d:/DS-Hack-Day1/data_generation/generate_dataset.py

Output: d:/DS-Hack-Day1/data/ecommerce_discount_profitability_50000.csv
"""

import os
import sys
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
RANDOM_SEED   = 42
NUM_ROWS      = 50_000
NUM_CUSTOMERS = 10_000
OUTPUT_FILE   = "ecommerce_discount_profitability_50000.csv"

CATEGORIES = [
    "Electronics", "Clothing", "Home & Kitchen",
    "Beauty & Personal Care", "Sports & Fitness", "Books",
    "Grocery", "Furniture", "Accessories", "Toys & Games",
]
CATEGORY_WEIGHTS = [0.15, 0.18, 0.12, 0.10, 0.08, 0.07, 0.10, 0.05, 0.08, 0.07]

# Log-normal price params per category
CATEGORY_PRICE_PARAMS = {
    "Electronics":            {"mean_log": 9.5,  "std_log": 1.0,  "min": 1_000,  "max": 100_000},
    "Clothing":               {"mean_log": 7.5,  "std_log": 0.8,  "min": 300,    "max": 10_000},
    "Home & Kitchen":         {"mean_log": 8.2,  "std_log": 0.9,  "min": 500,    "max": 30_000},
    "Beauty & Personal Care": {"mean_log": 7.0,  "std_log": 0.8,  "min": 200,    "max": 8_000},
    "Sports & Fitness":       {"mean_log": 8.0,  "std_log": 0.85, "min": 500,    "max": 25_000},
    "Books":                  {"mean_log": 6.0,  "std_log": 0.6,  "min": 150,    "max": 3_000},
    "Grocery":                {"mean_log": 5.5,  "std_log": 0.7,  "min": 50,     "max": 5_000},
    "Furniture":              {"mean_log": 9.2,  "std_log": 1.0,  "min": 2_000,  "max": 100_000},
    "Accessories":            {"mean_log": 7.2,  "std_log": 0.85, "min": 100,    "max": 15_000},
    "Toys & Games":           {"mean_log": 7.0,  "std_log": 0.8,  "min": 200,    "max": 15_000},
}

# Discount behaviour per category
CATEGORY_DISCOUNT_PARAMS = {
    "Electronics":            {"base_prob": 0.55, "mean": 15, "std": 10},
    "Clothing":               {"base_prob": 0.65, "mean": 20, "std": 12},
    "Home & Kitchen":         {"base_prob": 0.55, "mean": 18, "std": 11},
    "Beauty & Personal Care": {"base_prob": 0.60, "mean": 22, "std": 12},
    "Sports & Fitness":       {"base_prob": 0.50, "mean": 15, "std": 10},
    "Books":                  {"base_prob": 0.40, "mean": 12, "std": 8},
    "Grocery":                {"base_prob": 0.45, "mean": 10, "std": 7},
    "Furniture":              {"base_prob": 0.50, "mean": 18, "std": 11},
    "Accessories":            {"base_prob": 0.60, "mean": 20, "std": 12},
    "Toys & Games":           {"base_prob": 0.58, "mean": 20, "std": 12},
}

# Category base quantity multipliers
CATEGORY_QTY_BASE = {
    "Electronics": 1.2, "Clothing": 1.5, "Home & Kitchen": 1.4,
    "Beauty & Personal Care": 1.8, "Sports & Fitness": 1.3, "Books": 2.0,
    "Grocery": 2.5, "Furniture": 1.1, "Accessories": 2.0, "Toys & Games": 1.6,
}

# Category discount quantity sensitivity
CATEGORY_DISCOUNT_SENSITIVITY = {
    "Electronics": 0.6, "Clothing": 1.2, "Home & Kitchen": 1.0,
    "Beauty & Personal Care": 1.3, "Sports & Fitness": 0.8, "Books": 1.1,
    "Grocery": 1.4, "Furniture": 0.5, "Accessories": 1.1, "Toys & Games": 1.0,
}

# Logistics cost params per category
CATEGORY_LOGISTICS_PARAMS = {
    "Electronics":            {"base": 150, "value_factor": 0.015, "qty_factor": 30},
    "Clothing":               {"base":  60, "value_factor": 0.008, "qty_factor": 15},
    "Home & Kitchen":         {"base": 100, "value_factor": 0.012, "qty_factor": 20},
    "Beauty & Personal Care": {"base":  50, "value_factor": 0.008, "qty_factor": 10},
    "Sports & Fitness":       {"base": 100, "value_factor": 0.012, "qty_factor": 20},
    "Books":                  {"base":  30, "value_factor": 0.005, "qty_factor":  8},
    "Grocery":                {"base":  70, "value_factor": 0.010, "qty_factor": 12},
    "Furniture":              {"base": 600, "value_factor": 0.025, "qty_factor": 80},
    "Accessories":            {"base":  50, "value_factor": 0.008, "qty_factor": 12},
    "Toys & Games":           {"base":  60, "value_factor": 0.008, "qty_factor": 15},
}

# Customer segments
CUSTOMER_SEGMENTS = {
    "price_sensitive":     0.30,
    "regular":             0.40,
    "promotion_sensitive": 0.20,
    "loyal":               0.10,
}

SEGMENT_DISCOUNT_MULT = {
    "price_sensitive": 1.35, "regular": 1.00,
    "promotion_sensitive": 1.20, "loyal": 0.75,
}

PROMOTION_WEIGHT         = 0.40
PROMOTION_DISCOUNT_BOOST = 1.40
PROMOTION_MEAN_BOOST     = 6       # extra discount % during promotion
PROMOTION_QTY_BOOST      = 0.15   # 15% quantity boost during promotion

# Piecewise discount → mean quantity table
DISCOUNT_QTY_MAP = [(0, 1.50), (10, 1.80), (20, 2.20), (30, 2.60), (40, 2.70), (50, 2.80)]
_DISC_BREAK = np.array([x[0] for x in DISCOUNT_QTY_MAP], dtype=float)
_QTY_BREAK  = np.array([x[1] for x in DISCOUNT_QTY_MAP], dtype=float)


# ─────────────────────────────────────────────────────────────────────────────
# GENERATORS
# ─────────────────────────────────────────────────────────────────────────────

def build_customer_pool(rng):
    segments = list(CUSTOMER_SEGMENTS.keys())
    weights  = list(CUSTOMER_SEGMENTS.values())
    assigned = rng.choice(segments, size=NUM_CUSTOMERS, p=weights)
    customer_ids = [f"C{str(i+1).zfill(5)}" for i in range(NUM_CUSTOMERS)]
    return dict(zip(customer_ids, assigned))


def build_customer_weights(rng, customer_pool):
    n = len(customer_pool)
    base = rng.exponential(scale=1.0, size=n)
    boost = np.array([
        1.8 if s == "loyal" else 1.4 if s == "promotion_sensitive" else 1.0
        for s in customer_pool.values()
    ])
    raw = base * boost
    return raw / raw.sum()


def sample_customers(rng, pool, n_rows):
    """
    Guarantees every customer appears at least once:
      1. Create one slot per customer (shuffled).
      2. Fill remaining slots with weighted sampling.
      3. Shuffle the combined array so order is random.
    """
    cust_ids = list(pool.keys())
    weights  = build_customer_weights(rng, pool)
    initial  = rng.permutation(cust_ids)           # all 10,000 once
    n_extra  = n_rows - NUM_CUSTOMERS              # 40,000 extra
    extra    = rng.choice(cust_ids, size=n_extra, p=weights)
    combined = np.concatenate([initial, extra])
    rng.shuffle(combined)
    return combined


def generate_prices(categories, rng):
    prices = np.empty(len(categories), dtype=float)
    for cat, p in CATEGORY_PRICE_PARAMS.items():
        mask = categories == cat
        if not mask.any():
            continue
        raw = rng.lognormal(mean=p["mean_log"], sigma=p["std_log"], size=mask.sum())
        prices[mask] = np.round(np.clip(raw, p["min"], p["max"]))
    return prices


def generate_discounts(categories, is_promotion, order_segments, rng):
    n         = len(categories)
    discounts = np.zeros(n, dtype=int)

    for cat, p in CATEGORY_DISCOUNT_PARAMS.items():
        mask = np.where(categories == cat)[0]
        if len(mask) == 0:
            continue

        promo_flags = is_promotion[mask]
        seg_list    = [order_segments[i] for i in mask]

        seg_mult   = np.array([SEGMENT_DISCOUNT_MULT[s] for s in seg_list])
        promo_mult = np.where(promo_flags, PROMOTION_DISCOUNT_BOOST, 1.0)
        row_prob   = np.clip(p["base_prob"] * seg_mult * promo_mult, 0.0, 0.92)

        gets_discount = rng.random(len(mask)) < row_prob
        n_disc        = gets_discount.sum()

        if n_disc > 0:
            promo_shift = np.where(promo_flags[gets_discount], PROMOTION_MEAN_BOOST, 0)
            means       = p["mean"] + promo_shift
            raw         = rng.normal(loc=means, scale=p["std"], size=n_disc)
            raw         = np.clip(np.round(raw).astype(int), 1, 50)
            row_discounts = np.zeros(len(mask), dtype=int)
            row_discounts[gets_discount] = raw
            discounts[mask] = row_discounts

    return discounts


def generate_quantities(categories, discounts, is_promotion, rng):
    base_mean  = np.interp(discounts.astype(float), _DISC_BREAK, _QTY_BREAK)
    cat_base   = np.array([CATEGORY_QTY_BASE[c] for c in categories], dtype=float)
    cat_sens   = np.array([CATEGORY_DISCOUNT_SENSITIVITY[c] for c in categories], dtype=float)
    zero_qty   = cat_base * 1.50
    delta      = (base_mean - 1.50) * cat_sens
    eff_mean   = (zero_qty + delta) * (cat_base / 1.50)
    eff_mean  *= np.where(is_promotion, 1.0 + PROMOTION_QTY_BOOST, 1.0)
    lambda_p   = np.clip(eff_mean - 1.0, 0.1, 13.0)
    qty        = rng.poisson(lam=lambda_p) + 1
    return np.clip(qty, 1, 15).astype(int)


def generate_logistics(categories, original_prices, quantities, rng):
    n    = len(categories)
    cost = np.empty(n, dtype=float)
    for cat, p in CATEGORY_LOGISTICS_PARAMS.items():
        mask = categories == cat
        if not mask.any():
            continue
        det   = p["base"] + p["value_factor"] * original_prices[mask] + p["qty_factor"] * quantities[mask]
        noise = rng.lognormal(mean=0.0, sigma=0.15, size=mask.sum())
        cost[mask] = np.round(det * noise, 2)
    return np.maximum(cost, 10.0)


# ─────────────────────────────────────────────────────────────────────────────
# VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def validate(df):
    errors  = []
    results = []

    def check(name, condition, detail=""):
        tag = "PASS" if condition else "FAIL"
        sym = "OK" if condition else "!!"
        results.append(f"  [{sym}] {name}" + (f"  ({detail})" if detail else ""))
        if not condition:
            errors.append(name)

    check("Row count = 50,000",          len(df) == 50_000,          f"got {len(df):,}")
    check("Column count = 9",            len(df.columns) == 9,       f"got {len(df.columns)}")
    check("No duplicate Order_ID",       df["Order_ID"].nunique() == len(df))
    check("No missing values",           df.isnull().sum().sum() == 0)
    check("Original_Price > 0",          (df["Original_Price"] > 0).all())
    check("Logistics_Cost > 0",          (df["Logistics_Cost"] > 0).all())
    qty_ok = df["Quantity"].between(1, 15).all()
    check("Quantity in [1, 15]",         qty_ok,
          f"min={df['Quantity'].min()}, max={df['Quantity'].max()}")
    check("Discount in [0%, 50%]",       df["Discount"].between(0, 50).all(),
          f"min={df['Discount'].min()}, max={df['Discount'].max()}")
    check("Selling_Price <= Orig_Price", (df["Selling_Price"] <= df["Original_Price"] + 0.01).all())
    check("Selling_Price > 0",           (df["Selling_Price"] > 0).all())
    n_cats = df["Product_Category"].nunique()
    check("Exactly 10 categories",       n_cats == 10, f"got {n_cats}")
    n_cust = df["Customer_ID"].nunique()
    check("~10,000 unique customers",    9_000 <= n_cust <= 10_500, f"got {n_cust:,}")
    promo_vals = set(df["Promotion_Period"].unique())
    check("Both Promotion & Non-Promo",  {"Promotion", "Non-Promotion"}.issubset(promo_vals))
    check("Discounted & non-disc orders", df["Discount"].eq(0).any() and df["Discount"].gt(0).any())
    freq = df["Customer_ID"].value_counts()
    check("Long-tail customer frequency",
          (freq <= 3).any() and freq.between(4,10).any() and (freq > 10).any(),
          f"1-3:{(freq<=3).sum()} | 4-10:{freq.between(4,10).sum()} | 10+:{(freq>10).sum()}")
    expected  = (df["Original_Price"] * (1 - df["Discount"] / 100)).round(2)
    max_err   = (df["Selling_Price"] - expected).abs().max()
    check("Selling_Price formula OK",    max_err < 1.0, f"max err = {max_err:.4f}")

    print("\n" + "=" * 60)
    print("  DATASET VALIDATION REPORT")
    print("=" * 60)
    for r in results:
        print(r)
    print("=" * 60)

    if errors:
        raise ValueError(f"Validation failed: {errors}")
    print("  All 16 checks PASSED — dataset is clean and ready.\n")


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY & DATA DICTIONARY
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(df):
    disc = df["Discount"].gt(0)
    print("\n" + "=" * 60)
    print("  DATASET SUMMARY REPORT")
    print("=" * 60)
    print(f"  Rows                  : {len(df):,}")
    print(f"  Columns               : {len(df.columns)}")
    print(f"  Unique Customers      : {df['Customer_ID'].nunique():,}")
    print(f"  Unique Categories     : {df['Product_Category'].nunique()}")
    print(f"  Discounted Orders     : {disc.sum():,}  ({disc.mean()*100:.1f}%)")
    print(f"  Non-Discounted Orders : {(~disc).sum():,}  ({(~disc).mean()*100:.1f}%)")
    promo_pct = (df["Promotion_Period"] == "Promotion").mean() * 100
    print(f"  Promotion Orders      : {(df['Promotion_Period']=='Promotion').sum():,}  ({promo_pct:.1f}%)")
    print(f"  Avg Original Price    : Rs {df['Original_Price'].mean():,.2f}")
    print(f"  Avg Discount          : {df['Discount'].mean():.2f}%")
    print(f"  Avg Selling Price     : Rs {df['Selling_Price'].mean():,.2f}")
    print(f"  Avg Quantity          : {df['Quantity'].mean():.2f}")
    print(f"  Avg Logistics Cost    : Rs {df['Logistics_Cost'].mean():,.2f}")
    print("=" * 60)

    print("\n  Category Distribution:")
    for cat, cnt in df["Product_Category"].value_counts().items():
        print(f"    {cat:<28} {cnt:>6,}  ({cnt/len(df)*100:.1f}%)")

    print("\n  Avg Discount by Category:")
    for cat, avg in df.groupby("Product_Category")["Discount"].mean().sort_values(ascending=False).items():
        print(f"    {cat:<28} {avg:.2f}%")

    print("\n  Avg Quantity by Discount Band:")
    bins   = [-1, 0, 10, 20, 30, 50]
    labels = ["0%", "1-10%", "11-20%", "21-30%", "31-50%"]
    df2 = df.copy()
    df2["_db"] = pd.cut(df2["Discount"], bins=bins, labels=labels)
    for band, qty in df2.groupby("_db", observed=True)["Quantity"].mean().items():
        print(f"    {str(band):<10}  avg qty = {qty:.2f}")
    print()


def print_data_dictionary():
    print("=" * 60)
    print("  DATA DICTIONARY")
    print("=" * 60)
    rows = [
        ("Order_ID",         "string",  "Unique order identifier. Format: ORD000001–ORD050000."),
        ("Product_Category", "string",  "One of 10 e-commerce product categories."),
        ("Original_Price",   "integer", "Listed price before any discount (Rs). Category-specific log-normal."),
        ("Discount",         "integer", "Discount percentage applied [0-50]. 0 = no discount."),
        ("Selling_Price",    "float",   "Actual price paid = Original_Price * (1 - Discount/100). 2 d.p."),
        ("Quantity",         "integer", "Units ordered [1-15]. Positively correlated with discount."),
        ("Logistics_Cost",   "float",   "Shipping & handling cost (Rs). Depends on category, price, qty."),
        ("Customer_ID",      "string",  "Customer identifier. Format: C00001. ~10,000 unique customers."),
        ("Promotion_Period", "string",  "Whether order was placed during a promotion event."),
    ]
    for col, dtype, desc in rows:
        print(f"  {col:<28} [{dtype:<8}]")
        print(f"    {desc}")
    print("=" * 60 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  E-Commerce Synthetic Dataset Generator")
    print("  Project: Does Discount Really Increase Profit?")
    print("=" * 60)

    print(f"\n  Seed = {RANDOM_SEED} | Rows = {NUM_ROWS:,} | Customers = {NUM_CUSTOMERS:,}")
    rng = np.random.default_rng(RANDOM_SEED)

    # 1. Customer pool
    print("\n[1/9] Building customer pool...")
    pool = build_customer_pool(rng)

    # 2. Sample customers per order (all 10,000 guaranteed at least once)
    print("[2/9] Assigning customers to orders (long-tail)...")
    order_cust     = sample_customers(rng, pool, NUM_ROWS)
    order_segments = [pool[c] for c in order_cust]

    # 3. Product categories
    print("[3/9] Assigning product categories...")
    categories = rng.choice(CATEGORIES, size=NUM_ROWS, p=CATEGORY_WEIGHTS)

    # 4. Promotion period
    print("[4/9] Assigning promotion periods...")
    is_promo    = rng.random(NUM_ROWS) < PROMOTION_WEIGHT
    promo_labels = np.where(is_promo, "Promotion", "Non-Promotion")

    # 5. Original price
    print("[5/9] Generating original prices (log-normal per category)...")
    orig_prices = generate_prices(categories, rng)

    # 6. Discount
    print("[6/9] Generating discounts (category + promo + segment driven)...")
    discounts = generate_discounts(categories, is_promo, order_segments, rng)

    # 7. Selling price (derived)
    print("[7/9] Computing selling prices...")
    sell_prices = np.round(orig_prices * (1 - discounts / 100), 2)

    # 8. Quantity
    print("[8/9] Generating quantities (Poisson + diminishing returns)...")
    quantities = generate_quantities(categories, discounts, is_promo, rng)

    # 9. Logistics cost
    print("[9/9] Generating logistics costs...")
    logistics = generate_logistics(categories, orig_prices, quantities, rng)

    # Assemble
    print("\n  Assembling final DataFrame...")
    order_ids = [f"ORD{str(i+1).zfill(6)}" for i in range(NUM_ROWS)]
    df = pd.DataFrame({
        "Order_ID":         order_ids,
        "Product_Category": categories,
        "Original_Price":   orig_prices.astype(int),
        "Discount":         discounts,
        "Selling_Price":    sell_prices,
        "Quantity":         quantities,
        "Logistics_Cost":   logistics,
        "Customer_ID":      order_cust,
        "Promotion_Period": promo_labels,
    })

    # Validate
    print("  Running validation checks...")
    validate(df)

    # Report
    print_summary(df)
    print_data_dictionary()

    # Save
    script_dir  = os.path.dirname(os.path.abspath(__file__))
    output_dir  = os.path.join(script_dir, "..", "data")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, OUTPUT_FILE)

    df.to_csv(output_path, index=False)
    abs_path = os.path.abspath(output_path)
    print(f"  Saved: {abs_path}")
    print(f"  Size : {len(df):,} rows x {len(df.columns)} columns\n")
    size_mb = os.path.getsize(abs_path) / (1024 * 1024)
    print(f"  File size: {size_mb:.2f} MB")
    print("\n  Done!\n")


if __name__ == "__main__":
    main()
