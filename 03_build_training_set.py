import pandas as pd
import numpy as np

from src.load_data import load_all_data
from src.features import build_all_features

from src.config import (
    PROCESSED_DIR,
    RANDOM_STATE,
    NEGATIVE_RATIO,
    TOP_GLOBAL_PRODUCTS,
    TOP_PRODUCTS_PER_DEPARTMENT,
    MAX_HISTORICAL_PRODUCTS_PER_USER
)


# ============================================================
# 0. LOAD DATA
# ============================================================

data = load_all_data()

orders = data["orders"]
prior = data["prior"]
train = data["train"]
products = data["products"]


# ============================================================
# 1. BUILD FEATURES
# ============================================================

print("\nBuilding features...")

(
    history,
    customer_features,
    product_features,
    cp_features,
    department_features
) = build_all_features(
    orders,
    prior,
    products
)

print("Features built successfully.")


# ============================================================
# 2. TARGET ORDERS
# ============================================================

target_orders = (
    orders[
        orders["eval_set"] == "train"
    ][
        [
            "order_id",
            "user_id",
            "order_number",
            "order_dow",
            "order_hour_of_day"
        ]
    ]
)

print(
    "\nTarget orders:",
    target_orders.shape
)


# ============================================================
# 3. POSITIVE EXAMPLES
# ============================================================

positive = (
    train[
        [
            "order_id",
            "product_id"
        ]
    ]
    .merge(
        target_orders,
        on="order_id",
        how="inner"
    )
)

positive["target"] = 1

positive = (
    positive[
        [
            "user_id",
            "product_id",
            "target"
        ]
    ]
    .drop_duplicates(
        ["user_id", "product_id"]
    )
)

print(
    "Positive examples:",
    len(positive)
)


# ============================================================
# 4. HISTORICAL CANDIDATES
# ============================================================

print("\nGenerating historical candidates...")

# Count how many times each customer bought each product
product_frequency = (
    history
    .groupby(
        ["user_id", "product_id"],
        as_index=False
    )
    .size()
    .rename(
        columns={
            "size": "frequency"
        }
    )
)

# Keep the most frequently purchased products for each customer
historical_candidates = (
    product_frequency
    .sort_values(
        [
            "user_id",
            "frequency"
        ],
        ascending=[
            True,
            False
        ]
    )
    .groupby(
        "user_id",
        group_keys=False
    )
    .head(
        MAX_HISTORICAL_PRODUCTS_PER_USER
    )
    [
        [
            "user_id",
            "product_id"
        ]
    ]
)

print(
    "Historical candidates:",
    len(historical_candidates)
)


# ============================================================
# 5. GLOBAL TOP PRODUCTS
# ============================================================

print("\nGenerating global candidates...")

global_top = (
    history
    .groupby("product_id")
    .size()
    .sort_values(
        ascending=False
    )
    .head(
        TOP_GLOBAL_PRODUCTS
    )
    .index
)

users = target_orders["user_id"].unique()

global_candidates = (
    pd.MultiIndex.from_product(
        [
            users,
            global_top
        ],
        names=[
            "user_id",
            "product_id"
        ]
    )
    .to_frame(
        index=False
    )
)

print(
    "Global candidates:",
    len(global_candidates)
)


# ============================================================
# 6. DEPARTMENT CANDIDATES
# ============================================================

print("\nGenerating department candidates...")

department_popularity = (
    history
    .groupby(
        [
            "department_id",
            "product_id"
        ],
        as_index=False
    )
    .size()
    .rename(
        columns={
            "size": "frequency"
        }
    )
)

# Top products within each department
top_department_products = (
    department_popularity
    .sort_values(
        [
            "department_id",
            "frequency"
        ],
        ascending=[
            True,
            False
        ]
    )
    .groupby(
        "department_id",
        group_keys=False
    )
    .head(
        TOP_PRODUCTS_PER_DEPARTMENT
    )
    [
        [
            "department_id",
            "product_id"
        ]
    ]
)

# Departments purchased by each customer
user_departments = (
    history[
        [
            "user_id",
            "department_id"
        ]
    ]
    .drop_duplicates()
)

department_candidates = (
    user_departments
    .merge(
        top_department_products,
        on="department_id",
        how="inner"
    )
    [
        [
            "user_id",
            "product_id"
        ]
    ]
)

print(
    "Department candidates:",
    len(department_candidates)
)


# ============================================================
# 7. UNION OF CANDIDATES
# ============================================================

print("\nCombining candidates...")

candidates = pd.concat(
    [
        historical_candidates,
        global_candidates,
        department_candidates,

        # Always keep all positive examples
        positive[
            [
                "user_id",
                "product_id"
            ]
        ]
    ],
    ignore_index=True
)

# Remove duplicate customer-product pairs
candidates = (
    candidates
    .drop_duplicates(
        [
            "user_id",
            "product_id"
        ]
    )
)

print(
    "Candidate pairs before target:",
    len(candidates)
)


# ============================================================
# 8. TARGET
# ============================================================

print("\nCreating target...")

# Merge target directly instead of constructing string keys
candidates = candidates.merge(
    positive[
        [
            "user_id",
            "product_id",
            "target"
        ]
    ],
    on=[
        "user_id",
        "product_id"
    ],
    how="left"
)

candidates["target"] = (
    candidates["target"]
    .fillna(0)
    .astype("int8")
)

print(
    "\nCandidate target distribution:"
)

print(
    candidates["target"].value_counts()
)


# ============================================================
# 9. NEGATIVE SAMPLING
# ============================================================

print("\nApplying negative sampling...")

positives = (
    candidates[
        candidates["target"] == 1
    ]
    .copy()
)

negatives = (
    candidates[
        candidates["target"] == 0
    ]
    .copy()
)

print(
    "Available positives:",
    len(positives)
)

print(
    "Available negatives:",
    len(negatives)
)


# Maximum number of negatives
max_negatives = (
    len(positives)
    * NEGATIVE_RATIO
)

print(
    "Maximum negatives to keep:",
    max_negatives
)


if len(negatives) > max_negatives:

    negatives = negatives.sample(
        n=max_negatives,
        random_state=RANDOM_STATE
    )

    print(
        "Negative examples sampled:",
        len(negatives)
    )

else:

    print(
        "All negative examples kept:",
        len(negatives)
    )


# Combine positives and sampled negatives
training = pd.concat(
    [
        positives,
        negatives
    ],
    ignore_index=True
)

# Shuffle
training = (
    training
    .sample(
        frac=1,
        random_state=RANDOM_STATE
    )
    .reset_index(drop=True)
)

print(
    "\nTraining size after negative sampling:",
    len(training)
)

print(
    "\nTarget distribution after negative sampling:"
)

print(
    training["target"].value_counts()
)

print(
    "\nTarget distribution (%):"
)

print(
    training["target"]
    .value_counts(
        normalize=True
    )
)


# ============================================================
# 10. JOIN CUSTOMER-PRODUCT FEATURES
# ============================================================

print("\nJoining customer-product features...")

training = training.merge(
    cp_features,
    on=[
        "user_id",
        "product_id"
    ],
    how="left"
)

print(
    "Shape after customer-product features:",
    training.shape
)


# ============================================================
# 11. JOIN CUSTOMER FEATURES
# ============================================================

print("\nJoining customer features...")

training = training.merge(
    customer_features,
    on="user_id",
    how="left"
)

print(
    "Shape after customer features:",
    training.shape
)


# ============================================================
# 12. JOIN PRODUCT FEATURES
# ============================================================

print("\nJoining product features...")

training = training.merge(
    product_features,
    on="product_id",
    how="left"
)

print(
    "Shape after product features:",
    training.shape
)


# ============================================================
# 13. PRODUCT METADATA
# ============================================================

print("\nJoining product metadata...")

training = training.merge(
    products[
        [
            "product_id",
            "aisle_id",
            "department_id"
        ]
    ],
    on="product_id",
    how="left"
)

print(
    "Shape after product metadata:",
    training.shape
)


# ============================================================
# 14. CUSTOMER-DEPARTMENT FEATURES
# ============================================================

print("\nJoining customer-department features...")

training = training.merge(
    department_features,
    on=[
        "user_id",
        "department_id"
    ],
    how="left"
)

print(
    "Shape after department features:",
    training.shape
)


# ============================================================
# 15. MISSING VALUES
# ============================================================

print("\nHandling missing values...")

count_features = [
    "customer_product_purchase_count",
    "product_purchase_count",
    "customer_department_purchase_count",
    "customer_total_orders"
]

for col in count_features:

    if col in training.columns:

        training[col] = (
            training[col]
            .fillna(0)
        )


rate_features = [
    "customer_product_reorder_rate",
    "product_reorder_rate",
    "customer_department_share"
]

for col in rate_features:

    if col in training.columns:

        training[col] = (
            training[col]
            .fillna(0)
        )


# ============================================================
# 16. OPTIMIZE NUMERIC TYPES
# ============================================================

print("\nOptimizing data types...")

for col in training.select_dtypes(
    include=["float64"]
).columns:

    training[col] = (
        training[col]
        .astype("float32")
    )


# ============================================================
# 17. SAVE
# ============================================================

output_path = (
    PROCESSED_DIR
    / "training_dataset.parquet"
)

training.to_parquet(
    output_path,
    index=False
)

print(
    f"\nTraining dataset saved to:\n{output_path}"
)

print(
    "\nFinal shape:",
    training.shape
)

print(
    "\nFinal target distribution:"
)

print(
    training["target"].value_counts()
)

print(
    "\nFinal target distribution (%):"
)

print(
    training["target"]
    .value_counts(
        normalize=True
    )
)

