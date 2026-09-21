import pandas as pd
import numpy as np
import joblib

from src.config import (
    PROCESSED_DIR,
    MODEL_DIR,
    TOP_N_RECOMMENDATIONS
)


# ============================================================
# LOAD
# ============================================================

df = pd.read_parquet(
    PROCESSED_DIR
    / "training_dataset.parquet"
)

model = joblib.load(
    MODEL_DIR
    / "random_forest_model.joblib"
)

features = joblib.load(
    MODEL_DIR
    / "features.joblib"
)


# ============================================================
# PREPARE
# ============================================================

X = (
    df[features]
    .replace(
        [np.inf, -np.inf],
        np.nan
    )
    .fillna(0)
)


# ============================================================
# PROPENSITY SCORE
# ============================================================

df["propensity_score"] = (
    model.predict_proba(X)[:, 1]
)


# ============================================================
# PERCENTILE SCORE
# ============================================================

df["score_percentile"] = (
    df["propensity_score"]
    .rank(
        pct=True
    )
)


# ============================================================
# DECILES
# ============================================================

df["score_decile"] = (
    pd.qcut(
        df["propensity_score"],
        10,
        labels=False,
        duplicates="drop"
    )
    + 1
)


# ============================================================
# MARKETING SEGMENT
# ============================================================

df["marketing_segment"] = pd.cut(
    df["propensity_score"],

    bins=[
        -np.inf,
        0.20,
        0.40,
        0.60,
        0.80,
        np.inf
    ],

    labels=[
        "D - Très faible",
        "C - Faible",
        "B - Moyenne",
        "A - Forte",
        "A+ - Très forte"
    ]
)


# ============================================================
# TOP RECOMMENDATIONS PER CUSTOMER
# ============================================================

recommendations = (
    df
    .sort_values(
        [
            "user_id",
            "propensity_score"
        ],
        ascending=[
            True,
            False
        ]
    )
    .groupby("user_id")
    .head(
        TOP_N_RECOMMENDATIONS
    )
)


# ============================================================
# SAVE
# ============================================================

recommendations[
    [
        "user_id",
        "product_id",
        "propensity_score",
        "score_decile",
        "marketing_segment"
    ]
].to_parquet(
    PROCESSED_DIR
    / "customer_product_scores.parquet",
    index=False
)


print(
    recommendations.head(20)
)







### SCORE DE PRIORITE MARKETING


df["marketing_priority"] = (
    df["propensity_score"]
    *
    np.log1p(
        df["customer_total_orders"]
    )
)

df = df.sort_values(
    "marketing_priority",
    ascending=False
)



############ SEGMENTATION MARKETING

def assign_marketing_action(row):

    score = row["propensity_score"]

    if score >= 0.80:

        return "Campagne personnalisée prioritaire"

    elif score >= 0.60:

        return "Recommandation personnalisée"

    elif score >= 0.40:

        return "Test A/B"

    else:

        return "Ne pas cibler"




recommendations[
    "marketing_action"
] = recommendations.apply(
    assign_marketing_action,
    axis=1
)

