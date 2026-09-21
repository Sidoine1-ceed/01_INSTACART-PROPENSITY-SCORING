import numpy as np

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    brier_score_loss
)


def evaluate_model(
    y_true,
    probabilities,
    k=0.10
):

    results = {}

    # --------------------------------------------------------
    # ROC AUC
    # --------------------------------------------------------

    results["ROC_AUC"] = (
        roc_auc_score(
            y_true,
            probabilities
        )
    )

    # --------------------------------------------------------
    # PR AUC
    # --------------------------------------------------------

    results["PR_AUC"] = (
        average_precision_score(
            y_true,
            probabilities
        )
    )

    # --------------------------------------------------------
    # BRIER SCORE
    # --------------------------------------------------------

    results["Brier"] = (
        brier_score_loss(
            y_true,
            probabilities
        )
    )

    # --------------------------------------------------------
    # PRECISION / RECALL AT K
    # --------------------------------------------------------

    n = len(probabilities)

    top_n = max(
        1,
        int(n * k)
    )

    order = np.argsort(
        probabilities
    )[::-1]

    top_indices = order[
        :top_n
    ]

    y_top = np.asarray(
        y_true
    )[top_indices]

    precision_at_k = (
        y_top.mean()
    )

    total_positive = (
        np.asarray(y_true)
        .sum()
    )

    captured_positive = (
        y_top.sum()
    )

    recall_at_k = (
        captured_positive
        / total_positive
        if total_positive > 0
        else 0
    )

    # --------------------------------------------------------
    # LIFT
    # --------------------------------------------------------

    baseline_rate = (
        np.asarray(y_true)
        .mean()
    )

    lift_at_k = (
        precision_at_k
        / baseline_rate
        if baseline_rate > 0
        else 0
    )

    results[
        f"Precision@{int(k*100)}%"
    ] = precision_at_k

    results[
        f"Recall@{int(k*100)}%"
    ] = recall_at_k

    results[
        f"Lift@{int(k*100)}%"
    ] = lift_at_k

    return results