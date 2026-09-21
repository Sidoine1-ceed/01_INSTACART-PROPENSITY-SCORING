import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from pathlib import Path

from src.config import PROCESSED_DIR, MODEL_DIR


# ============================================================
# 1. Chargement
# ============================================================

print("Chargement des données...")

df = pd.read_parquet(
    PROCESSED_DIR / "training_dataset.parquet"
)

model = joblib.load(
    MODEL_DIR / "random_forest_model.joblib"
)

features = joblib.load(
    MODEL_DIR / "features.joblib"
)

print(f"Dataset : {len(df):,} lignes")
print(f"Features : {len(features)}")


# ============================================================
# 2. Petit échantillon pour SHAP
# ============================================================

# 500 lignes suffisent largement pour un graphique SHAP global.
SAMPLE_SIZE = 500

if len(df) > SAMPLE_SIZE:
    df_sample = df.sample(
        n=SAMPLE_SIZE,
        random_state=42
    )
else:
    df_sample = df.copy()

X = df_sample[features].fillna(0)

print(f"Échantillon SHAP : {len(X):,} lignes")


# ============================================================
# 3. Calcul SHAP
# ============================================================

print()
print("Calcul des valeurs SHAP...")

explainer = shap.TreeExplainer(model)

# API SHAP récente
shap_result = explainer(X)


# ============================================================
# 4. Sélection de la classe positive
# ============================================================

# Pour une classification binaire, les valeurs SHAP peuvent
# avoir la forme :
# (observations, features, classes)

if len(shap_result.values.shape) == 3:

    # Classe 1 = achat / réachat
    shap_values_plot = shap_result.values[:, :, 1]

else:

    shap_values_plot = shap_result.values


# ============================================================
# 5. Création du dossier output
# ============================================================

output_dir = Path(__file__).resolve().parent / "data" / "output"

output_dir.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 6. Graphique SHAP global
# ============================================================

print("Génération du graphique SHAP...")

plt.figure(figsize=(10, 8))

shap.summary_plot(
    shap_values_plot,
    X,
    show=False,
    max_display=15
)

plt.tight_layout()


# ============================================================
# 7. Sauvegarde
# ============================================================

output_path = output_dir / "shap_summary.png"

plt.savefig(
    output_path,
    dpi=150,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 8. Vérification
# ============================================================

print()
print("==============================================")
print("SHAP terminé avec succès !")
print("==============================================")
print(f"Graphique sauvegardé ici :")
print(output_path)
print()
print(f"Fichier créé : {output_path.exists()}")