
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

try:
    import plotly.express as px
except ImportError:
    px = None

from src.config import PROCESSED_DIR, MODEL_DIR


# ============================================================
# CONFIGURATION
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent
RAW_DIR = ROOT_DIR / "data" / "raw"
OUTPUT_DIR = ROOT_DIR / "data" / "output"
LOGO_PATH = ROOT_DIR / "logo.png"
LOGO_PATH1 = ROOT_DIR / "logo instacart.png"

st.set_page_config(
    page_title="Customer Propensity Scoring",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Palette inspirée du logo
NAVY = "#0B1F3A"
BLUE = "#0066CC"
CYAN = "#18BFEA"
TEAL = "#18A999"
VIOLET = "#635BFF"
ORANGE = "#FF8A3D"
PINK = "#E94B8A"
LIGHT_BLUE = "#EAF5FF"
LIGHT_CYAN = "#E8FAFE"
LIGHT_VIOLET = "#F0EEFF"
LIGHT_ORANGE = "#FFF2E8"
LIGHT_TEAL = "#EAF9F6"
TEXT = "#172033"
MUTED = "#667085"
BORDER = "#E6EAF0"


# ============================================================
# STYLE
# ============================================================

st.markdown(
    f"""
    <style>
    .stApp {{
        background: linear-gradient(180deg, #F8FBFF 0%, #FFFFFF 32%);
        color: {TEXT};
    }}

    .block-container {{
        padding-top: 1.2rem;
        padding-bottom: 2.5rem;
        max-width: 1450px;
    }}

    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #F7FBFF 0%, #FFFFFF 70%);
        border-right: 1px solid {BORDER};
    }}

    [data-testid="stSidebar"] .block-container {{
        padding-top: 1rem;
    }}

    .brand-header {{
        display: flex;
        align-items: center;
        gap: 18px;
        padding: 10px 0 4px 0;
        margin-bottom: 8px;
    }}

    .brand-title {{
        font-size: 2.25rem;
        line-height: 1.05;
        font-weight: 800;
        color: {NAVY};
        margin: 0;
    }}

    .brand-subtitle {{
        color: {MUTED};
        font-size: 1rem;
        margin-top: 7px;
    }}

    .page-kicker {{
        display: inline-block;
        padding: 5px 11px;
        border-radius: 999px;
        background: {LIGHT_CYAN};
        color: {BLUE};
        font-weight: 700;
        font-size: 0.78rem;
        letter-spacing: 0.02em;
        margin-bottom: 8px;
    }}

    .section-title {{
        font-size: 1.25rem;
        font-weight: 800;
        color: {NAVY};
        margin: 1.3rem 0 0.75rem 0;
    }}

    .section-subtitle {{
        color: {MUTED};
        margin-top: -0.4rem;
        margin-bottom: 0.9rem;
        font-size: 0.92rem;
    }}

    .insight-card {{
        border-radius: 14px;
        padding: 15px 17px;
        margin: 6px 0 14px 0;
        background: #FFFFFF;
        border: 1px solid {BORDER};
        box-shadow: 0 4px 14px rgba(11,31,58,0.05);
    }}

    .insight-blue {{ border-left: 5px solid {BLUE}; background: {LIGHT_BLUE}; }}
    .insight-cyan {{ border-left: 5px solid {CYAN}; background: {LIGHT_CYAN}; }}
    .insight-violet {{ border-left: 5px solid {VIOLET}; background: {LIGHT_VIOLET}; }}
    .insight-orange {{ border-left: 5px solid {ORANGE}; background: {LIGHT_ORANGE}; }}
    .insight-teal {{ border-left: 5px solid {TEAL}; background: {LIGHT_TEAL}; }}

    div[data-testid="stMetric"] {{
        background: #FFFFFF;
        border: 1px solid {BORDER};
        border-radius: 15px;
        padding: 13px 15px;
        box-shadow: 0 4px 14px rgba(11,31,58,0.04);
    }}

    div[data-testid="stMetricLabel"] {{
        color: {MUTED};
    }}

    div[data-testid="stMetricValue"] {{
        color: {NAVY};
    }}

    /* Navigation horizontale : rendu en barres cliquables */
    div[role="radiogroup"] {{
        gap: 7px;
        margin: 4px 0 18px 0;
        padding: 5px;
        background: #EEF4FA;
        border-radius: 14px;
        border: 1px solid {BORDER};
    }}

    div[role="radiogroup"] > label {{
        flex: 1;
        justify-content: center;
        border-radius: 10px;
        padding: 10px 18px !important;
        background: transparent;
        color: {NAVY};
        font-weight: 700;
        transition: all 0.15s ease;
    }}

    div[role="radiogroup"] > label:hover {{
        background: #FFFFFF;
        color: {BLUE};
    }}

    div[role="radiogroup"] > label:has(input:checked) {{
        background: #FFFFFF;
        color: {BLUE};
        box-shadow: 0 3px 10px rgba(0,102,204,0.14);
    }}

    div[role="radiogroup"] > label > div:first-child {{
        display: none;
    }}

    .filter-title {{
        color: {NAVY};
        font-weight: 800;
        font-size: 0.95rem;
        margin-bottom: 0.2rem;
    }}

    .filter-help {{
        color: {MUTED};
        font-size: 0.78rem;
        margin-bottom: 0.8rem;
    }}

    .sidebar-logo {{
        text-align: center;
        padding: 2px 0 10px 0;
    }}

    .small-muted {{
        color: {MUTED};
        font-size: 0.82rem;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data(show_spinner=False)
def load_scores():
    path = PROCESSED_DIR / "customer_product_scores.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {path}")
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def load_products():
    path = RAW_DIR / "products.csv"
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {path}")
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_departments():
    path = RAW_DIR / "departments.csv"
    if not path.exists():
        return pd.DataFrame(columns=["department_id", "department"])
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_training_dataset():
    path = PROCESSED_DIR / "training_dataset.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


@st.cache_resource(show_spinner=False)
def load_model():
    path = MODEL_DIR / "random_forest_model.joblib"
    if not path.exists():
        return None
    return joblib.load(path)


@st.cache_resource(show_spinner=False)
def load_features():
    path = MODEL_DIR / "features.joblib"
    if not path.exists():
        return []
    return joblib.load(path)


# ============================================================
# PREPARATION
# ============================================================

@st.cache_data(show_spinner=False)
def enrich_scores(scores, products, departments):
    product_cols = ["product_id", "product_name", "department_id"]
    available = [c for c in product_cols if c in products.columns]

    result = scores.merge(
        products[available],
        on="product_id",
        how="left",
    )

    if "department_id" in result.columns and "department" in departments.columns:
        result = result.merge(
            departments[["department_id", "department"]],
            on="department_id",
            how="left",
        )

    if "department" not in result.columns:
        result["department"] = "Non renseigné"

    result["department"] = result["department"].fillna("Non renseigné")
    return result


@st.cache_data(show_spinner=False)
def enrich_training(training_df, departments):
    if training_df is None:
        return None
    if "department_id" not in training_df.columns or "department" not in departments.columns:
        return training_df
    return training_df.merge(
        departments[["department_id", "department"]],
        on="department_id",
        how="left",
    )


def apply_filters_to_scores(
    df,
    departments_selected,
    segments_selected,
    deciles_selected,
    min_score,
):
    out = df

    if departments_selected:
        out = out[out["department"].isin(departments_selected)]

    if segments_selected and "marketing_segment" in out.columns:
        out = out[out["marketing_segment"].isin(segments_selected)]

    if deciles_selected and "score_decile" in out.columns:
        out = out[out["score_decile"].isin(deciles_selected)]

    if "propensity_score" in out.columns:
        out = out[out["propensity_score"] >= min_score]

    return out


def apply_filters_to_training(df, departments_selected):
    if df is None:
        return None
    if departments_selected and "department" in df.columns:
        return df[df["department"].isin(departments_selected)]
    return df


@st.cache_data(show_spinner=False)
def get_dataset_kpis(df):
    if df is None:
        return {}
    return {
        "rows": len(df),
        "variables": len(df.columns),
        "customers": df["user_id"].nunique() if "user_id" in df.columns else 0,
        "products": df["product_id"].nunique() if "product_id" in df.columns else 0,
        "departments": df["department_id"].nunique() if "department_id" in df.columns else 0,
    }


@st.cache_data(show_spinner=False)
def get_customer_profile(df):
    cols = [
        "user_id",
        "customer_total_orders",
        "unique_products",
        "unique_aisles",
        "unique_departments",
        "avg_products_per_order",
        "avg_days_between_orders",
    ]
    cols = [c for c in cols if c in df.columns]
    if "user_id" not in cols:
        return pd.DataFrame()
    return df[cols].drop_duplicates("user_id")


@st.cache_data(show_spinner=False)
def get_product_profile(df):
    cols = [
        "product_id",
        "product_purchase_count",
        "product_unique_users",
        "product_reorder_rate",
        "department",
    ]
    cols = [c for c in cols if c in df.columns]
    if "product_id" not in cols:
        return pd.DataFrame()
    return df[cols].drop_duplicates("product_id")


@st.cache_data(show_spinner=False)
def get_department_stats(df):
    if df is None or "department" not in df.columns:
        return pd.DataFrame()

    agg = {"product_id": "nunique"}
    if "user_id" in df.columns:
        agg["user_id"] = "nunique"
    if "product_reorder_rate" in df.columns:
        agg["product_reorder_rate"] = "mean"
    if "customer_product_reorder_rate" in df.columns:
        agg["customer_product_reorder_rate"] = "mean"

    result = df.groupby("department").agg(agg).reset_index()

    rename = {
        "product_id": "Produits",
        "user_id": "Clients",
        "product_reorder_rate": "Taux de réachat produit",
        "customer_product_reorder_rate": "Taux de réachat client-produit",
    }
    return result.rename(columns=rename)


@st.cache_data(show_spinner=False)
def get_target_balance(df):
    if df is None or "target" not in df.columns:
        return pd.DataFrame()
    result = (
        df["target"]
        .value_counts()
        .rename_axis("target")
        .reset_index(name="count")
    )
    result["Statut"] = result["target"].map({0: "Non acheté", 1: "Acheté"})
    return result


def format_pct(value):
    if pd.isna(value):
        return "—"
    return f"{value:.1%}"


def base_fig(fig, height=350):
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=55, b=20),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color=TEXT),
        title_font=dict(color=NAVY, size=16),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#EEF2F6")
    return fig


# ============================================================
# MODEL INSIGHTS
# ============================================================

@st.cache_data(show_spinner=False)
def load_shap_image():
    path = OUTPUT_DIR / "shap_summary.png"
    return path if path.exists() else None


# Le "_" devant model empêche Streamlit de tenter de hasher
# l'objet RandomForestClassifier pour le cache.
@st.cache_data(show_spinner=False)
def compute_lift_curve(
    _model,
    training_df,
    model_features,
    sample_size=100_000,
):
    if (
        _model is None
        or training_df is None
        or "target" not in training_df.columns
        or "customer_total_orders" not in training_df.columns
    ):
        return None

    split_value = training_df["customer_total_orders"].quantile(0.80)
    valid = training_df[
        training_df["customer_total_orders"] > split_value
    ].copy()

    if valid.empty:
        return None

    if len(valid) > sample_size:
        valid = valid.sample(sample_size, random_state=42)

    drop_cols = [
        "user_id",
        "product_id",
        "target",
        "first_order_number",
        "last_order_number",
        "department",
    ]

    X = valid.drop(
        columns=[c for c in drop_cols if c in valid.columns],
        errors="ignore",
    )
    y = valid["target"].astype(int)

    if model_features:
        missing = [c for c in model_features if c not in X.columns]
        if missing:
            return None
        X = X[model_features]

    X = X.fillna(0)

    try:
        scores_pred = _model.predict_proba(X)[:, 1]
    except Exception:
        return None

    evaluation = pd.DataFrame(
        {
            "target": y.to_numpy(),
            "score": scores_pred,
        }
    ).sort_values("score", ascending=False)

    baseline_rate = evaluation["target"].mean()
    positives = evaluation["target"].sum()

    if baseline_rate <= 0:
        return None

    total = len(evaluation)
    rows = []

    for pct in np.arange(0.05, 1.01, 0.05):
        n = max(1, int(total * pct))
        top = evaluation.head(n)
        precision = top["target"].mean()

        rows.append(
            {
                "Population ciblée": pct,
                "Precision": precision,
                "Lift": precision / baseline_rate,
                "Recall": (
                    top["target"].sum() / positives
                    if positives > 0
                    else 0
                ),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# LOAD
# ============================================================

try:
    scores = load_scores()
    products = load_products()
    departments = load_departments()
except Exception as exc:
    st.error(f"Impossible de charger les données : {exc}")
    st.stop()

scores = enrich_scores(scores, products, departments)
training_df = load_training_dataset()
training_df_enriched = enrich_training(training_df, departments)
model_features = load_features()


# ============================================================
# SIDEBAR — LOGO + FILTRES GLOBAUX
# ============================================================

with st.sidebar:
    if LOGO_PATH.exists():
        st.markdown('<div class="sidebar-logo">', unsafe_allow_html=True)
        st.image(str(LOGO_PATH), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        f"""
        <div style="text-align:center;">
            <div style="font-size:1.05rem;font-weight:800;color:{NAVY};">
                CUSTOMER PROPENSITY SCORING
            </div>
            <div class="small-muted">
                Next Best Product · Data Marketing
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    st.markdown(
        '<div class="filter-title">🎛️ Sélectionner</div>',
        unsafe_allow_html=True,
    )

    all_departments = sorted(
        [x for x in scores["department"].dropna().unique().tolist()]
    )

    departments_selected = st.multiselect(
        "Département",
        options=all_departments,
        default=[],
        key="global_departments",
    )

    all_segments = []
    if "marketing_segment" in scores.columns:
        all_segments = sorted(
            [x for x in scores["marketing_segment"].dropna().unique().tolist()]
        )

    segments_selected = st.multiselect(
        "Segment marketing",
        options=all_segments,
        default=[],
        key="global_segments",
    )

    all_deciles = []
    if "score_decile" in scores.columns:
        all_deciles = sorted(
            [int(x) for x in scores["score_decile"].dropna().unique()]
        )

    deciles_selected = st.multiselect(
        "Décile",
        options=all_deciles,
        default=[],
        format_func=lambda x: f"D{x}",
        key="global_deciles",
    )

    st.markdown(
        '<div class="filter-title">🎯 Score minimum</div>',
        unsafe_allow_html=True,
    )

    min_score_pct = st.slider(
        "Score minimum",
        min_value=0,
        max_value=100,
        value=0,
        step=5,
        format="%d%%",
        key="global_min_score_pct",
        label_visibility="collapsed",
    )

    # Conversion en score entre 0 et 1 pour le modèle
    min_score = min_score_pct / 100

    filtered_scores = apply_filters_to_scores(
        scores,
        departments_selected,
        segments_selected,
        deciles_selected,
        min_score,
    )

    filtered_training = apply_filters_to_training(
        training_df_enriched,
        departments_selected,
    )

    st.divider()

    st.markdown(
        f"""
        <div class="insight-card insight-cyan">
            <b>Filtres actifs</b><br>
            <span class="small-muted">
                {len(filtered_scores):,} lignes de scoring après filtrage.
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption("Modèle : Random Forest")
    st.caption("Unité de scoring : client × produit")


# ============================================================
# HEADER + NAVIGATION
# ============================================================

logo_col, title_col = st.columns([1.15, 7])

with logo_col:
    if LOGO_PATH1.exists():
        st.image(str(LOGO_PATH1), width=150)

with title_col:
    st.markdown(
        '<div class="page-kicker">DATA MARKETING · NEXT BEST PRODUCT</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="brand-title">Analyse du panier d achat chez Instacart</div>',
        unsafe_allow_html=True,
    )


# Visuellement, le contrôle est transformé en barre de navigation.
# Cela conserve un seul onglet actif et évite de calculer les 3 pages
# en même temps comme le ferait st.tabs().
page = st.radio(
    "Navigation",
    ["Overview", "Scoring", "Model Insights"],
    horizontal=True,
    label_visibility="collapsed",
    key="main_navigation",
)


# ============================================================
# OVERVIEW
# ============================================================

if page == "Overview":
    if filtered_training is None:
        st.warning(
            "training_dataset.parquet est introuvable. Les statistiques détaillées ne peuvent pas être affichées."
        )
        st.stop()

    kpis = get_dataset_kpis(filtered_training)

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        st.metric("Couples client-produit", f"{kpis['rows']:,}")
    with c2:
        st.metric("Variables", f"{kpis['variables']:,}")
    with c3:
        st.metric("Clients", f"{kpis['customers']:,}")
    with c4:
        st.metric("Produits", f"{kpis['products']:,}")
    with c5:
        st.metric("Départements", f"{kpis['departments']:,}")

    st.markdown(
        '<div class="insight-card insight-blue"><b>Lecture marketing :</b> le scoring travaille au niveau <b>client × produit</b>. L’objectif est de détecter les couples présentant les signaux les plus favorables à un prochain achat.</div>',
        unsafe_allow_html=True,
    )

    # -------- Client behavior --------

    customer_profile = get_customer_profile(filtered_training)

    a, b = st.columns(2)

    if not customer_profile.empty and px is not None:
        with a:
            col = "customer_total_orders"
            if col in customer_profile:
                fig = px.histogram(
                    customer_profile,
                    x=col,
                    nbins=25,
                    title="Distribution du nombre de commandes par client",
                    labels={col: "Nombre de commandes"},
                    color_discrete_sequence=[BLUE],
                )
                base_fig(fig, 340)
                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    config={"displayModeBar": False},
                )

        with b:
            x = "unique_products"
            if x in customer_profile:
                fig = px.histogram(
                    customer_profile,
                    x=x,
                    nbins=25,
                    title="Diversité du portefeuille produit",
                    labels={x: "Produits uniques"},
                    color_discrete_sequence=[CYAN],
                )
                base_fig(fig, 340)
                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    config={"displayModeBar": False},
                )

    c, d = st.columns(2)

    if not customer_profile.empty and px is not None:
        with c:
            x = "avg_products_per_order"
            if x in customer_profile:
                fig = px.histogram(
                    customer_profile,
                    x=x,
                    nbins=25,
                    title="Taille moyenne du panier",
                    labels={x: "Produits / commande"},
                    color_discrete_sequence=[TEAL],
                )
                base_fig(fig, 340)
                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    config={"displayModeBar": False},
                )

        with d:
            x = "avg_days_between_orders"
            if x in customer_profile:
                fig = px.histogram(
                    customer_profile,
                    x=x,
                    nbins=25,
                    title="Rythme de réachat",
                    labels={x: "Jours entre commandes"},
                    color_discrete_sequence=[VIOLET],
                )
                base_fig(fig, 340)
                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    config={"displayModeBar": False},
                )

    # -------- Product behavior --------
    product_profile = get_product_profile(filtered_training)
    p1, p2 = st.columns(2)

    if not product_profile.empty and px is not None:
        with p1:
            if "product_reorder_rate" in product_profile.columns:
                fig = px.histogram(
                    product_profile,
                    x="product_reorder_rate",
                    nbins=25,
                    title="Distribution du taux de réachat produit",
                    labels={"product_reorder_rate": "Taux de réachat"},
                    color_discrete_sequence=[ORANGE],
                )
                fig.update_xaxes(tickformat=".0%")
                base_fig(fig, 340)
                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    config={"displayModeBar": False},
                )

        with p2:
            if "product_purchase_count" in product_profile.columns:
                top_products = (
                    product_profile
                    .sort_values("product_purchase_count", ascending=False)
                    .head(10)
                    .sort_values("product_purchase_count")
                )

                fig = px.bar(
                    top_products,
                    x="product_purchase_count",
                    y="product_id",
                    orientation="h",
                    title="Top 10 produits par fréquence d’achat",
                    labels={
                        "product_purchase_count": "Achats",
                        "product_id": "Produit",
                    },
                    color="product_purchase_count",
                    color_continuous_scale=[LIGHT_BLUE, BLUE, NAVY],
                )
                base_fig(fig, 390)
                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    config={"displayModeBar": False},
                )

    # -------- Target / candidate mix --------
    target_balance = get_target_balance(filtered_training)
    t1, t2 = st.columns(2)

    with t1:
        if not target_balance.empty and px is not None:
            fig = px.pie(
                target_balance,
                names="Statut",
                values="count",
                title="Répartition des couples client-produit",
                color_discrete_sequence=[BLUE, CYAN],
                hole=0.52,
            )
            base_fig(fig, 350)
            st.plotly_chart(
                fig,
                use_container_width=True,
                config={"displayModeBar": False},
            )

    with t2:
        dept_stats = get_department_stats(filtered_training)

        if not dept_stats.empty and px is not None:
            top_depts = (
                dept_stats
                .sort_values("Produits", ascending=False)
                .head(10)
                .sort_values("Produits")
            )

            fig = px.bar(
                top_depts,
                x="Produits",
                y="department",
                orientation="h",
                title="Départements couvrant le plus de produits",
                labels={
                    "department": "Département",
                    "Produits": "Produits",
                },
                color="Produits",
                color_continuous_scale=[LIGHT_CYAN, CYAN, BLUE],
            )
            base_fig(fig, 390)
            st.plotly_chart(
                fig,
                use_container_width=True,
                config={"displayModeBar": False},
            )

    # -------- Reorder rate by department --------
    st.markdown(
        '<div class="section-title">🏷️ Les signaux de réachat ?</div>',
        unsafe_allow_html=True,
    )

    dept_stats = get_department_stats(filtered_training)

    if not dept_stats.empty and px is not None:
        metric_col = (
            "Taux de réachat client-produit"
            if "Taux de réachat client-produit" in dept_stats.columns
            else "Taux de réachat produit"
        )

        if metric_col in dept_stats.columns:
            chart_df = (
                dept_stats
                .dropna(subset=[metric_col])
                .sort_values(metric_col, ascending=False)
                .head(10)
                .sort_values(metric_col)
            )

            fig = px.bar(
                chart_df,
                x=metric_col,
                y="department",
                orientation="h",
                title="Top départements par taux moyen de réachat",
                labels={
                    "department": "Département",
                    metric_col: "Taux de réachat",
                },
                color=metric_col,
                color_continuous_scale=[LIGHT_VIOLET, VIOLET, NAVY],
            )
            fig.update_xaxes(tickformat=".0%")
            base_fig(fig, 420)
            st.plotly_chart(
                fig,
                use_container_width=True,
                config={"displayModeBar": False},
            )

    # -------- Score distribution --------
    if (
        not filtered_scores.empty
        and "propensity_score" in filtered_scores.columns
        and px is not None
    ):
        st.markdown(
            '<div class="section-title">📈 Distribution des scores de propension</div>',
            unsafe_allow_html=True,
        )

        score_sample = filtered_scores.sample(
            min(100_000, len(filtered_scores)),
            random_state=42,
        )

        fig = px.histogram(
            score_sample,
            x="propensity_score",
            nbins=30,
            title="Concentration des scores sur les couples client-produit",
            labels={"propensity_score": "Score de propension"},
            color_discrete_sequence=[BLUE],
        )
        fig.update_xaxes(tickformat=".0%")
        base_fig(fig, 350)
        st.plotly_chart(
            fig,
            use_container_width=True,
            config={"displayModeBar": False},
        )


# ============================================================
# SCORING
# ============================================================

elif page == "Scoring":
    st.markdown(
        '<div class="section-title">🎯 Scoring client</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-subtitle">Sélectionnez un client pour visualiser ses produits à plus forte appétence.</div>',
        unsafe_allow_html=True,
    )

    if filtered_scores.empty:
        st.warning("Aucune ligne de scoring ne correspond aux filtres globaux.")
        st.stop()

    customers = np.sort(
        filtered_scores["user_id"].dropna().unique()
    )

    customer_id = st.selectbox(
        "Sélectionner un client",
        customers,
        key="selected_customer",
    )

    customer_scores = (
        filtered_scores[
            filtered_scores["user_id"] == customer_id
        ]
        .sort_values("propensity_score", ascending=False)
        .head(10)
        .copy()
    )

    if customer_scores.empty:
        st.warning("Aucune recommandation disponible pour ce client.")
        st.stop()

    max_score = customer_scores["propensity_score"].max()
    mean_score = customer_scores["propensity_score"].mean()

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("Produits recommandés", len(customer_scores))
    with c2:
        st.metric("Score maximum", format_pct(max_score))
    with c3:
        st.metric("Score moyen", format_pct(mean_score))
    with c4:
        if "score_decile" in customer_scores.columns:
            best_decile = customer_scores["score_decile"].min()
            st.metric("Meilleur décile", f"D{int(best_decile)}")
        else:
            st.metric("Client", str(customer_id))

    st.markdown(
        '<div class="insight-card insight-blue"><b>Lecture CRM :</b> les produits du haut du classement correspondent aux signaux d’appétence les plus élevés pour ce client. Le score sert à prioriser les actions.</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-title">🎯 Top 10 recommandations</div>',
        unsafe_allow_html=True,
    )

    display_columns = [
        "product_name",
        "department",
        "propensity_score",
        "score_decile",
        "marketing_segment",
    ]

    display_columns = [
        c for c in display_columns
        if c in customer_scores.columns
    ]

    display_df = customer_scores[display_columns].copy()

    display_df = display_df.rename(
        columns={
            "product_name": "Produit",
            "department": "Département",
            "propensity_score": "Score de propension",
            "score_decile": "Décile",
            "marketing_segment": "Segment marketing",
        }
    )

    if "Score de propension" in display_df.columns:
        display_df["Score de propension"] = (
            display_df["Score de propension"].map(format_pct)
        )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
    )

    left, right = st.columns([1.35, 1])

    with left:
        if px is not None:
            chart_df = customer_scores.copy()
            chart_df["Produit"] = (
                chart_df["product_name"].fillna("Produit inconnu")
            )
            chart_df = chart_df.sort_values("propensity_score")

            fig = px.bar(
                chart_df,
                x="propensity_score",
                y="Produit",
                orientation="h",
                title="Intensité de l’appétence",
                labels={
                    "propensity_score": "Score",
                    "Produit": "",
                },
                color="propensity_score",
                color_continuous_scale=[
                    LIGHT_CYAN,
                    CYAN,
                    BLUE,
                    NAVY,
                ],
            )

            fig.update_xaxes(
                range=[0, 1],
                tickformat=".0%",
            )

            base_fig(fig, 440)
            st.plotly_chart(
                fig,
                use_container_width=True,
                config={"displayModeBar": False},
            )

    with right:
        if "department" in customer_scores.columns and px is not None:
            dept_mix = (
                customer_scores["department"]
                .value_counts()
                .rename_axis("Département")
                .reset_index(name="Produits recommandés")
            )

            fig = px.pie(
                dept_mix,
                names="Département",
                values="Produits recommandés",
                title="Mix des départements recommandés",
                color_discrete_sequence=[
                    BLUE,
                    CYAN,
                    TEAL,
                    VIOLET,
                    ORANGE,
                    PINK,
                ],
                hole=0.5,
            )

            base_fig(fig, 440)
            st.plotly_chart(
                fig,
                use_container_width=True,
                config={"displayModeBar": False},
            )

    top_product = customer_scores.iloc[0]

    score = top_product["propensity_score"]
    product_name = top_product.get("product_name", "ce produit")
    department = top_product.get(
        "department",
        "département inconnu",
    )

    st.markdown(
        '<div class="section-title">💡 Traduction marketing</div>',
        unsafe_allow_html=True,
    )

    if score >= 0.80:
        title = "Forte appétence détectée"
        action = (
            "Prioriser une recommandation personnalisée "
            "ou un scénario de réachat."
        )
        css = "insight-blue"
    elif score >= 0.60:
        title = "Appétence intéressante"
        action = (
            "Tester une personnalisation ou un A/B test "
            "avant généralisation."
        )
        css = "insight-cyan"
    else:
        title = "Appétence modérée à faible"
        action = (
            "Ne pas prioriser ce produit dans une pression "
            "commerciale immédiate."
        )
        css = "insight-orange"

    st.markdown(
        f"""
        <div class="insight-card {css}">
            <b>{title}</b><br><br>
            <b>{product_name}</b> · {department}<br>
            Score de propension : <b>{score:.1%}</b><br><br>
            → <b>Action :</b> {action}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# MODEL INSIGHTS
# ============================================================

else:
    st.markdown(
        '<div class="section-title">🔎 Model Insights</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-subtitle">Performance du modèle, explicabilité et capacité à concentrer les achats potentiels.</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-title">🏆 Performance de ranking</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("ROC-AUC", "0.752")
    with c2:
        st.metric("PR-AUC", "0.630")
    with c3:
        st.metric("Precision@10%", "74.8%")
    with c4:
        st.metric("Lift@10%", "2.10×")

    if px is not None:
        metric_df = pd.DataFrame(
            {
                "Métrique": [
                    "ROC-AUC",
                    "PR-AUC",
                    "Precision@10%",
                ],
                "Valeur": [
                    0.752,
                    0.630,
                    0.748,
                ],
            }
        )

        fig = px.bar(
            metric_df,
            x="Métrique",
            y="Valeur",
            title="Indicateurs clés de performance",
            labels={"Valeur": "Score"},
            color="Métrique",
            color_discrete_sequence=[
                BLUE,
                CYAN,
                VIOLET,
            ],
        )

        fig.update_yaxes(
            range=[0, 1],
            tickformat=".0%",
        )

        base_fig(fig, 350)
        st.plotly_chart(
            fig,
            use_container_width=True,
            config={"displayModeBar": False},
        )

    st.markdown(
        '<div class="insight-card insight-violet"><b>Lecture business :</b> à 10% de la population ciblée, le modèle concentre environ <b>2,10×</b> plus de positifs qu’un ciblage aléatoire. La Precision@10% est de <b>74,8%</b> sur l’échantillon de validation utilisé.</div>',
        unsafe_allow_html=True,
    )

    # -------- SHAP --------
    st.markdown(
        '<div class="section-title">🧠 SHAP — facteurs qui expliquent le scoring</div>',
        unsafe_allow_html=True,
    )

    shap_path = load_shap_image()

    if shap_path is not None:
        st.image(
            str(shap_path),
            use_container_width=True,
        )
        st.caption(
            "Importance globale des variables calculée avec SHAP sur le Random Forest."
        )
    else:
        st.warning(
            "Le fichier data/output/shap_summary.png n'a pas été trouvé. "
            "Lance le script SHAP avant de revenir sur cette page."
        )

    st.markdown(
        '<div class="insight-card insight-cyan"><b>Lecture métier :</b> les variables client-produit sont particulièrement pertinentes pour une logique de réachat : fréquence d’achat, taux de réachat, récence et part des commandes du client contenant le produit.</div>',
        unsafe_allow_html=True,
    )

    # -------- Lift --------
    st.markdown(
        '<div class="section-title">📈 Efficacité du ciblage</div>',
        unsafe_allow_html=True,
    )

    model = load_model()

    with st.spinner("Calcul de la courbe de lift..."):
        lift_df = compute_lift_curve(
            model,
            training_df,
            model_features,
        )

    if lift_df is not None and not lift_df.empty:
        if px is not None:
            plot_df = lift_df.copy()
            plot_df["Population ciblée"] = (
                plot_df["Population ciblée"] * 100
            )

            fig = px.line(
                plot_df,
                x="Population ciblée",
                y="Lift",
                markers=True,
                title="Lift selon la part de la population ciblée",
                labels={
                    "Population ciblée": "Population ciblée (%)",
                    "Lift": "Lift",
                },
                color_discrete_sequence=[BLUE],
            )

            fig.add_hline(
                y=1,
                line_dash="dash",
                annotation_text="Référence aléatoire",
            )

            fig.update_xaxes(ticksuffix="%")

            base_fig(fig, 390)
            st.plotly_chart(
                fig,
                use_container_width=True,
                config={"displayModeBar": False},
            )

        top10 = lift_df.iloc[
            np.argmin(
                np.abs(
                    lift_df["Population ciblée"].to_numpy()
                    - 0.10
                )
            )
        ]

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric(
                "Lift à 10%",
                f"{top10['Lift']:.2f}×",
            )

        with c2:
            st.metric(
                "Precision à 10%",
                format_pct(top10["Precision"]),
            )

        with c3:
            st.metric(
                "Recall à 10%",
                format_pct(top10["Recall"]),
            )

        st.markdown(
            '<div class="insight-card insight-teal"><b>Lecture CRM :</b> la courbe permet de visualiser jusqu’où réduire la population ciblée tout en conservant une concentration élevée d’acheteurs potentiels.</div>',
            unsafe_allow_html=True,
        )

    else:
        st.warning(
            "Impossible de reconstruire la courbe de lift. Vérifie que le modèle, "
            "features.joblib et training_dataset.parquet sont disponibles."
        )

    st.caption(
        "Important : la validation actuelle repose sur un split selon customer_total_orders. "
        "Il s'agit d'un split par profondeur d'historique et non d'un véritable backtest temporel."
    )

    # -------- Variable families --------
    st.markdown(
        '<div class="section-title">🧩 Lecture des familles de variables</div>',
        unsafe_allow_html=True,
    )

    family_df = pd.DataFrame(
        {
            "Famille": [
                "Client × produit",
                "Produit",
                "Client",
                "Client × département",
            ],
            "Rôle": [
                "Signal principal de réachat",
                "Popularité et répétition globale",
                "Intensité et habitudes d’achat",
                "Affinité avec la catégorie",
            ],
            "Exemples": [
                "fréquence, récence, reorder rate, share",
                "purchase count, unique users, reorder rate",
                "total orders, panier moyen, diversité",
                "department share, achats catégorie",
            ],
        }
    )

    st.dataframe(
        family_df,
        use_container_width=True,
        hide_index=True,
    )
