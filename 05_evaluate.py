import pandas as pd
import numpy as np
import joblib

from src.config import (
    PROCESSED_DIR,
    MODEL_DIR
)

from src.evaluation import evaluate_model


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_parquet(
    PROCESSED_DIR
    / "training_dataset.parquet"
)

features = joblib.load(
    MODEL_DIR
    / "features.joblib"
)


# ============================================================
# SAME TEMPORAL SPLIT
# ============================================================

split_value = (
    df["last_customer_order"]
    .quantile(0.75)
)

valid_df = df[
    df["last_customer_order"]
    > split_value
].copy()


X_valid = (
    valid_df[features]
    .replace(
        [np.inf, -np.inf],
        np.nan
    )
    .fillna(0)
)

y_valid = (
    valid_df["target"]
)


# ============================================================
# MODELS
# ============================================================

models = {

    "Logistic Regression":
        joblib.load(
            MODEL_DIR
            / "logistic_model.joblib"
        ),

    "Random Forest":
        joblib.load(
            MODEL_DIR
            / "random_forest_model.joblib"
        )
}


results = []


# ============================================================
# EVALUATION
# ============================================================

for name, model in models.items():

    print(
        f"\nEvaluating {name}..."
    )

    probabilities = (
        model.predict_proba(
            X_valid
        )[:, 1]
    )

    metrics = evaluate_model(
        y_valid,
        probabilities,
        k=0.10
    )

    metrics["Model"] = name

    results.append(
        metrics
    )


results_df = pd.DataFrame(
    results
)

results_df = results_df[
    [
        "Model",
        "ROC_AUC",
        "PR_AUC",
        "Brier",
        "Precision@10%",
        "Recall@10%",
        "Lift@10%"
    ]
]

print(
    "\n========== MODEL COMPARISON ==========\n"
)

print(
    results_df.to_string(
        index=False
    )
)


results_df.to_csv(
    PROCESSED_DIR
    / "model_comparison.csv",
    index=False
)




###########   COURBE LIFT  #########
import numpy as np
import matplotlib.pyplot as plt


def plot_lift_curve(
    y_true,
    probabilities
):

    data = pd.DataFrame({
        "target": y_true,
        "score": probabilities
    })

    data = data.sort_values(
        "score",
        ascending=False
    )

    data["cumulative_positive"] = (
        data["target"]
        .cumsum()
    )

    total_positive = (
        data["target"]
        .sum()
    )

    data["capture_rate"] = (
        data["cumulative_positive"]
        / total_positive
    )

    data["population_rate"] = (
        np.arange(
            1,
            len(data) + 1
        )
        / len(data)
    )

    data["lift"] = (
        data["capture_rate"]
        / data["population_rate"]
    )

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        data["population_rate"],
        data["lift"]
    )

    plt.axhline(
        1,
        linestyle="--"
    )

    plt.xlabel(
        "Population ciblée"
    )

    plt.ylabel(
        "Lift"
    )

    plt.title(
        "Lift Curve - Propensity Model"
    )

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.show()


