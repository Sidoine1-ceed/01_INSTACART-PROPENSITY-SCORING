import json
import pandas as pd
import numpy as np

from src.config import PROCESSED_DIR, OUTPUT_DIR


# ============================================================
# CONFIGURATION
# ============================================================

TRAINING_PATH = PROCESSED_DIR / "training_dataset.parquet"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("PRÉPARATION DES DONNÉES DE DÉPLOIEMENT")
print("=" * 70)

print("\nLecture du training dataset...")
df = pd.read_parquet(TRAINING_PATH)

print(f"Dataset chargé : {df.shape[0]:,} lignes × {df.shape[1]} colonnes")


# ============================================================
# 1. KPI GLOBAUX
# ============================================================

print("\n[1/6] Calcul des KPI...")

kpis = {
    "couples_client_produit": int(len(df)),
    "variables": int(df.shape[1]),
    "clients": int(df["user_id"].nunique()) if "user_id" in df.columns else 0,
    "produits": int(df["product_id"].nunique()) if "product_id" in df.columns else 0,
    "departements": int(df["department_id"].nunique())
    if "department_id" in df.columns else 0,
}

with open(OUTPUT_DIR / "overview_kpis.json", "w", encoding="utf-8") as f:
    json.dump(kpis, f, ensure_ascii=False, indent=2)

print(kpis)


# ============================================================
# 2. PROFIL CLIENT
# ============================================================

print("\n[2/6] Préparation du profil client...")

customer_columns = [
    "user_id",
    "customer_total_orders",
    "unique_products",
    "avg_products_per_order",
    "avg_days_between_orders",
]

customer_columns = [
    c for c in customer_columns
    if c in df.columns
]

customer_profile = (
    df[customer_columns]
    .drop_duplicates(subset=["user_id"])
    .copy()
)

customer_profile.to_parquet(
    OUTPUT_DIR / "overview_customer_profile.parquet",
    index=False
)

print(
    f"Profil client : "
    f"{len(customer_profile):,} clients"
)


# ============================================================
# 3. PROFIL PRODUIT
# ============================================================

print("\n[3/6] Préparation du profil produit...")

product_columns = [
    "product_id",
    "product_reorder_rate",
    "product_purchase_count",
    "department_id",
]

product_columns = [
    c for c in product_columns
    if c in df.columns
]

product_profile = (
    df[product_columns]
    .drop_duplicates(subset=["product_id"])
    .copy()
)

product_profile.to_parquet(
    OUTPUT_DIR / "overview_product_profile.parquet",
    index=False
)

print(
    f"Profil produit : "
    f"{len(product_profile):,} produits"
)


# ============================================================
# 4. STATISTIQUES DÉPARTEMENTS
# ============================================================

print("\n[4/6] Préparation des statistiques départements...")

if "department_id" in df.columns:

    department_stats = (
        df.groupby("department_id")
        .agg(
            nb_produits=("product_id", "nunique")
            if "product_id" in df.columns else ("department_id", "size"),

            nb_clients=("user_id", "nunique")
            if "user_id" in df.columns else ("department_id", "size"),

            taux_reachat=("target", "mean")
            if "target" in df.columns else ("department_id", "mean"),
        )
        .reset_index()
    )

    department_stats.to_csv(
        OUTPUT_DIR / "overview_department_stats.csv",
        index=False
    )

    print(
        f"Statistiques départements : "
        f"{len(department_stats):,} départements"
    )


# ============================================================
# 5. RÉPARTITION TARGET
# ============================================================

print("\n[5/6] Préparation de la répartition target...")

if "target" in df.columns:

    target_balance = (
        df["target"]
        .value_counts()
        .rename_axis("target")
        .reset_index(name="count")
    )

    target_balance["label"] = target_balance["target"].map(
        {
            0: "Non-réachat",
            1: "Réachat",
        }
    )

    target_balance.to_csv(
        OUTPUT_DIR / "overview_target_balance.csv",
        index=False
    )

    print(target_balance)


# ============================================================
# 6. SCORE DE PROPENSION
# ============================================================

print("\n[6/6] Préparation de la distribution des scores...")

# Si le score existe déjà dans le dataset, on le conserve.
score_candidates = [
    "propensity_score",
    "score",
    "prediction",
    "prediction_score",
]

score_column = next(
    (c for c in score_candidates if c in df.columns),
    None
)

if score_column:

    scores = df[[score_column]].dropna().copy()
    scores.columns = ["propensity_score"]

    scores.to_parquet(
        OUTPUT_DIR / "overview_propensity_scores.parquet",
        index=False
    )

    print(
        f"Distribution des scores sauvegardée : "
        f"{len(scores):,} scores"
    )

else:
    print(
        "Aucune colonne de score trouvée dans le training dataset."
    )


print("\n" + "=" * 70)
print("PRÉPARATION TERMINÉE")
print("=" * 70)

print("\nFichiers créés :")

for file in sorted(OUTPUT_DIR.iterdir()):
    print(f"  ✓ {file.name}")