import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

try:
    import plotly.express as px
except ImportError:
    px = None

from src.config import PROCESSED_DIR


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


# ============================================================
# PALETTE
# ============================================================

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
        background: linear-gradient(
            180deg,
            #F8FBFF 0%,
            #FFFFFF 32%
        );
        color: {TEXT};
    }}

    .block-container {{
        padding-top: 1.2rem;
        padding-bottom: 2.5rem;
        max-width: 1450px;
    }}

    [data-testid="stSidebar"] {{
        background: linear-gradient(
            180deg,
            #F7FBFF 0%,
            #FFFFFF 70%
        );
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

    .insight-blue {{
        border-left: 5px solid {BLUE};
        background: {LIGHT_BLUE};
    }}

    .insight-cyan {{
        border-left: 5px solid {CYAN};
        background: {LIGHT_CYAN};
    }}

    .insight-violet {{
        border-left: 5px solid {VIOLET};
        background: {LIGHT_VIOLET};
    }}

    .insight-orange {{
        border-left: 5px solid {ORANGE};
        background: {LIGHT_ORANGE};
    }}

    .insight-teal {{
        border-left: 5px solid {TEAL};
        background: {LIGHT_TEAL};
    }}

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
        raise FileNotFoundError(
            f"Fichier introuvable : {path}"
        )

    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def load_products():

    path = RAW_DIR / "products.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {path}"
        )

    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_departments():

    path = RAW_DIR / "departments.csv"

    if not path.exists():
        return pd.DataFrame(
            columns=[
                "department_id",
                "department"
            ]
        )

    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_lift_curve():

    path = OUTPUT_DIR / "lift_curve.csv"

    if not path.exists():
        return None

    return pd.read_csv(path)


@st.cache_data
def load_overview_kpis():
    path = OUTPUT_DIR / "overview_kpis.json"
    if not path.exists():
        return None

    import json

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_overview_customer_profile():
    path = OUTPUT_DIR / "overview_customer_profile.parquet"

    if not path.exists():
        return None

    return pd.read_parquet(path)


@st.cache_data
def load_overview_product_profile():
    path = OUTPUT_DIR / "overview_product_profile.parquet"

    if not path.exists():
        return None

    return pd.read_parquet(path)


@st.cache_data
def load_overview_department_stats():
    path = OUTPUT_DIR / "overview_department_stats.csv"

    if not path.exists():
        return None

    return pd.read_csv(path)


@st.cache_data
def load_overview_target_balance():
    path = OUTPUT_DIR / "overview_target_balance.csv"

    if not path.exists():
        return None

    return pd.read_csv(path)

# ============================================================
# DATA PREPARATION
# ============================================================

@st.cache_data(show_spinner=False)
def enrich_scores(
    scores,
    products,
    departments
):

    product_cols = [
        "product_id",
        "product_name",
        "department_id",
    ]

    available = [
        c for c in product_cols
        if c in products.columns
    ]

    result = scores.merge(
        products[available],
        on="product_id",
        how="left",
    )

    if (
        "department_id" in result.columns
        and "department" in departments.columns
    ):

        result = result.merge(
            departments[
                [
                    "department_id",
                    "department"
                ]
            ],
            on="department_id",
            how="left",
        )

    if "department" not in result.columns:
        result["department"] = "Non renseigné"

    result["department"] = (
        result["department"]
        .fillna("Non renseigné")
    )

    return result


def apply_filters_to_scores(
    df,
    departments_selected,
    segments_selected,
    deciles_selected,
    min_score,
):

    out = df

    if departments_selected:
        out = out[
            out["department"].isin(
                departments_selected
            )
        ]

    if (
        segments_selected
        and "marketing_segment" in out.columns
    ):

        out = out[
            out["marketing_segment"].isin(
                segments_selected
            )
        ]

    if (
        deciles_selected
        and "score_decile" in out.columns
    ):

        out = out[
            out["score_decile"].isin(
                deciles_selected
            )
        ]

    if "propensity_score" in out.columns:

        out = out[
            out["propensity_score"] >= min_score
        ]

    return out


# ============================================================
# DESCRIPTIVE STATISTICS
# ============================================================

@st.cache_data(show_spinner=False)
def get_dataset_kpis(df):

    if df is None:
        return {}

    return {

        "rows": len(df),

        "variables": len(df.columns),

        "customers": (
            df["user_id"].nunique()
            if "user_id" in df.columns
            else 0
        ),

        "products": (
            df["product_id"].nunique()
            if "product_id" in df.columns
            else 0
        ),

        "departments": (
            df["department_id"].nunique()
            if "department_id" in df.columns
            else (
                df["department"].nunique()
                if "department" in df.columns
                else 0
            )
        ),
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

    cols = [
        c for c in cols
        if c in df.columns
    ]

    if "user_id" not in cols:
        return pd.DataFrame()

    return (
        df[cols]
        .drop_duplicates("user_id")
    )


@st.cache_data(show_spinner=False)
def get_product_profile(df):

    cols = [

        "product_id",

        "product_purchase_count",

        "product_unique_users",

        "product_reorder_rate",

        "department",

    ]

    cols = [
        c for c in cols
        if c in df.columns
    ]

    if "product_id" not in cols:
        return pd.DataFrame()

    return (
        df[cols]
        .drop_duplicates("product_id")
    )


@st.cache_data(show_spinner=False)
def get_department_stats(df):

    if (
        df is None
        or "department" not in df.columns
    ):
        return pd.DataFrame()

    agg = {
        "product_id": "nunique"
    }

    if "user_id" in df.columns:
        agg["user_id"] = "nunique"

    if "product_reorder_rate" in df.columns:
        agg[
            "product_reorder_rate"
        ] = "mean"

    if "customer_product_reorder_rate" in df.columns:
        agg[
            "customer_product_reorder_rate"
        ] = "mean"

    result = (
        df.groupby("department")
        .agg(agg)
        .reset_index()
    )

    rename = {

        "product_id":
            "Produits",

        "user_id":
            "Clients",

        "product_reorder_rate":
            "Taux de réachat produit",

        "customer_product_reorder_rate":
            "Taux de réachat client-produit",
    }

    return result.rename(
        columns=rename
    )


def format_pct(value):

    if pd.isna(value):
        return "—"

    return f"{value:.1%}"


def base_fig(
    fig,
    height=350
):

    fig.update_layout(

        height=height,

        margin=dict(
            l=10,
            r=10,
            t=55,
            b=20
        ),

        plot_bgcolor="white",

        paper_bgcolor="white",

        font=dict(
            color=TEXT
        ),

        title_font=dict(
            color=NAVY,
            size=16
        ),

        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )

    fig.update_xaxes(
        showgrid=False
    )

    fig.update_yaxes(
        gridcolor="#EEF2F6"
    )

    return fig


# ============================================================
# SHAP
# ============================================================

@st.cache_data(show_spinner=False)
def load_shap_image():

    path = OUTPUT_DIR / "shap_summary.png"

    return (
        path
        if path.exists()
        else None
    )


# ============================================================
# LOAD DATA
# ============================================================

try:

    scores = load_scores()

    products = load_products()

    departments = load_departments()

    overview_kpis = load_overview_kpis()

    overview_customer = load_overview_customer_profile()

    overview_product = load_overview_product_profile()

    overview_department = load_overview_department_stats()

    overview_target = load_overview_target_balance()


except Exception as exc:

    st.error(
        f"Impossible de charger les données : {exc}"
    )

    st.stop()


scores = enrich_scores(
    scores,
    products,
    departments
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    if LOGO_PATH.exists():

        st.markdown(
            '<div class="sidebar-logo">',
            unsafe_allow_html=True
        )

        st.image(
            str(LOGO_PATH),
            use_container_width=True
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )

    st.markdown(
        f"""<div style="font-size:1.05rem;font-weight:800;color:{NAVY};">
    CUSTOMER PROPENSITY SCORING
    </div>
    <div class="small-muted">
    Next Best Product · Data Marketing
    </div>""",
        unsafe_allow_html=True,
    )

    st.divider()

    st.markdown(
        '<div class="filter-title">🎛️ Sélectionner</div>',
        unsafe_allow_html=True,
    )

    all_departments = sorted(
        [
            x
            for x in
            scores["department"]
            .dropna()
            .unique()
            .tolist()
        ]
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
            [
                x
                for x in
                scores[
                    "marketing_segment"
                ]
                .dropna()
                .unique()
                .tolist()
            ]
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
            [
                int(x)
                for x in
                scores[
                    "score_decile"
                ]
                .dropna()
                .unique()
            ]
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

    min_score = (
        min_score_pct / 100
    )

    filtered_scores = (
        apply_filters_to_scores(
            scores,
            departments_selected,
            segments_selected,
            deciles_selected,
            min_score,
        )
    )

    st.divider()

    st.markdown(
        f"""<div class="insight-card insight-cyan">
    <b>Filtres actifs</b>
    <span class="small-muted">
    {len(filtered_scores):,} lignes de scoring
    </span>
    </div>""",
        unsafe_allow_html=True,
    )

    st.caption(
        "Modèle : Random Forest"
    )

    st.caption(
        "Unité de scoring : client × produit"
    )


# ============================================================
# HEADER
# ============================================================

logo_col, title_col = st.columns(
    [1.15, 7]
)

with logo_col:

    if LOGO_PATH1.exists():

        st.image(
            str(LOGO_PATH1),
            width=150
        )


with title_col:

    st.markdown(
        '<div class="page-kicker">DATA MARKETING · NEXT BEST PRODUCT</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="brand-title">Analyse du panier d achat chez Instacart</div>',
        unsafe_allow_html=True,
    )


# ============================================================
# NAVIGATION
# ============================================================

page = st.radio(

    "Navigation",

    [
        "Overview",
        "Scoring",
        "Model Insights"
    ],

    horizontal=True,

    label_visibility="collapsed",

    key="main_navigation",
)


# ============================================================
# OVERVIEW
# ============================================================
if page == "Overview":

    # ============================================================
    # KPI GLOBAUX
    # ============================================================

    if overview_kpis is not None:

        k1, k2, k3, k4, k5 = st.columns(5)

        k1.metric(
            "Couples client-produit",
            f"{overview_kpis['couples_client_produit']:,}".replace(",", " "),
        )

        k2.metric(
            "Variables",
            f"{overview_kpis['variables']:,}".replace(",", " "),
        )

        k3.metric(
            "Clients",
            f"{overview_kpis['clients']:,}".replace(",", " "),
        )

        k4.metric(
            "Produits",
            f"{overview_kpis['produits']:,}".replace(",", " "),
        )

        k5.metric(
            "Départements",
            f"{overview_kpis['departements']:,}".replace(",", " "),
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ============================================================
    # INSIGHT PRINCIPAL
    # ============================================================

    st.markdown(
        """
        <div class="insight-card insight-blue">
            <div class="small-muted">
                Le dataset combine les comportements historiques des clients,
                la diversité des produits achetés, la fréquence des commandes
                et les signaux de réachat. Ces variables permettent ensuite
                d'estimer la probabilité qu'un client soit intéressé par
                un produit donné.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ============================================================
    # COMPORTEMENT CLIENT
    # ============================================================

    st.markdown(
        '<div class="section-title">Comportement client</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-subtitle">
            Distribution, diversité du panier, taille moyenne des commandes
            et rythme de réachat.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if overview_customer is not None:

        c1, c2 = st.columns(2)

        # --------------------------------------------------------
        # Nombre de commandes
        # --------------------------------------------------------

        if "customer_total_orders" in overview_customer.columns:

            fig = px.histogram(
                overview_customer,
                x="customer_total_orders",
                nbins=30,
                title="Distribution du nombre de commandes par client",
                labels={
                    "customer_total_orders": "Nombre total de commandes",
                    "count": "Nombre de clients",
                },
            )

            fig = base_fig(fig)
            c1.plotly_chart(fig, use_container_width=True)

        # --------------------------------------------------------
        # Diversité produit
        # --------------------------------------------------------

        if "unique_products" in overview_customer.columns:

            fig = px.histogram(
                overview_customer,
                x="unique_products",
                nbins=30,
                title="Diversité produit par client",
                labels={
                    "unique_products": "Nombre de produits uniques",
                    "count": "Nombre de clients",
                },
            )

            fig = base_fig(fig)
            c2.plotly_chart(fig, use_container_width=True)

        c3, c4 = st.columns(2)

        # --------------------------------------------------------
        # Taille moyenne du panier
        # --------------------------------------------------------

        if "avg_products_per_order" in overview_customer.columns:

            fig = px.histogram(
                overview_customer,
                x="avg_products_per_order",
                nbins=30,
                title="Taille moyenne du panier",
                labels={
                    "avg_products_per_order": "Produits moyens par commande",
                    "count": "Nombre de clients",
                },
            )

            fig = base_fig(fig)
            c3.plotly_chart(fig, use_container_width=True)

        # --------------------------------------------------------
        # Rythme de réachat
        # --------------------------------------------------------

        if "avg_days_between_orders" in overview_customer.columns:

            fig = px.histogram(
                overview_customer,
                x="avg_days_between_orders",
                nbins=30,
                title="Rythme moyen de réachat",
                labels={
                    "avg_days_between_orders": "Jours moyens entre commandes",
                    "count": "Nombre de clients",
                },
            )

            fig = base_fig(fig)
            c4.plotly_chart(fig, use_container_width=True)

    # ============================================================
    # COMPORTEMENT PRODUIT
    # ============================================================

    st.markdown(
        '<div class="section-title">Comportement produit</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-subtitle">
            Quels produits sont les plus achetés et lesquels présentent
            les taux de réachat les plus élevés ?
        </div>
        """,
        unsafe_allow_html=True,
    )

    if overview_product is not None:

        p1, p2 = st.columns(2)

        # --------------------------------------------------------
        # Taux de réachat produit
        # --------------------------------------------------------

        if "product_reorder_rate" in overview_product.columns:

            fig = px.histogram(
                overview_product,
                x="product_reorder_rate",
                nbins=30,
                title="Distribution du taux de réachat produit",
                labels={
                    "product_reorder_rate": "Taux de réachat",
                    "count": "Nombre de produits",
                },
            )

            fig.update_xaxes(tickformat=".0%")
            fig = base_fig(fig)

            p1.plotly_chart(
                fig,
                use_container_width=True,
            )

        # --------------------------------------------------------
        # Top 10 produits
        # --------------------------------------------------------

        if "product_purchase_count" in overview_product.columns:

            top_products = (
                overview_product
                .sort_values(
                    "product_purchase_count",
                    ascending=False
                )
                .head(10)
                .copy()
            )

            if products is not None and "product_id" in products.columns:

                product_name_col = None

                for candidate in [
                    "product_name",
                    "name",
                    "product",
                ]:
                    if candidate in products.columns:
                        product_name_col = candidate
                        break

                if product_name_col:

                    top_products = top_products.merge(
                        products[
                            ["product_id", product_name_col]
                        ].drop_duplicates("product_id"),
                        on="product_id",
                        how="left",
                    )

                    top_products["label"] = (
                        top_products[product_name_col]
                        .fillna(
                            "Produit "
                            + top_products["product_id"].astype(str)
                        )
                    )

                else:
                    top_products["label"] = (
                        "Produit "
                        + top_products["product_id"].astype(str)
                    )

            else:

                top_products["label"] = (
                    "Produit "
                    + top_products["product_id"].astype(str)
                )

            top_products = top_products.sort_values(
                "product_purchase_count",
                ascending=True,
            )

            fig = px.bar(
                top_products,
                x="product_purchase_count",
                y="label",
                orientation="h",
                title="Top 10 des produits les plus achetés",
                labels={
                    "product_purchase_count": "Nombre d'achats",
                    "label": "",
                },
            )

            fig = base_fig(fig)

            p2.plotly_chart(
                fig,
                use_container_width=True,
            )

    # ============================================================
    # TARGET / MIX DES CANDIDATS
    # ============================================================

    st.markdown(
        '<div class="section-title">Réachat et candidats</div>',
        unsafe_allow_html=True,
    )

    t1, t2 = st.columns(2)

    # ------------------------------------------------------------
    # Répartition target
    # ------------------------------------------------------------

    if overview_target is not None:

        fig = px.pie(
            overview_target,
            names="label",
            values="count",
            title="Répartition réachat / non-réachat",
            hole=0.45,
        )

        fig = base_fig(fig)

        t1.plotly_chart(
            fig,
            use_container_width=True,
        )

    # ------------------------------------------------------------
    # Produits par département
    # ------------------------------------------------------------

    if overview_department is not None:

        department_plot = overview_department.copy()

        department_name_col = None

        if (
            departments is not None
            and "department_id" in departments.columns
        ):

            for candidate in [
                "department",
                "department_name",
                "name",
            ]:

                if candidate in departments.columns:
                    department_name_col = candidate
                    break

            if department_name_col:

                department_plot = department_plot.merge(
                    departments[
                        [
                            "department_id",
                            department_name_col,
                        ]
                    ].drop_duplicates("department_id"),
                    on="department_id",
                    how="left",
                )

        if department_name_col:

            department_plot["label"] = (
                department_plot[department_name_col]
                .fillna(
                    "Département "
                    + department_plot["department_id"].astype(str)
                )
            )

        else:

            department_plot["label"] = (
                "Département "
                + department_plot["department_id"].astype(str)
            )

        department_plot = (
            department_plot
            .sort_values("nb_produits", ascending=False)
            .head(10)
            .sort_values("nb_produits", ascending=True)
        )

        fig = px.bar(
            department_plot,
            x="nb_produits",
            y="label",
            orientation="h",
            title="Top départements par nombre de produits",
            labels={
                "nb_produits": "Nombre de produits",
                "label": "",
            },
        )

        fig = base_fig(fig)

        t2.plotly_chart(
            fig,
            use_container_width=True,
        )

    # ============================================================
    # STATISTIQUES DÉPARTEMENTS
    # ============================================================

    st.markdown(
        '<div class="section-title">Taux de réachat par département</div>',
        unsafe_allow_html=True,
    )

    if overview_department is not None:

        department_reorder = overview_department.copy()

        department_name_col = None

        if (
            departments is not None
            and "department_id" in departments.columns
        ):

            for candidate in [
                "department",
                "department_name",
                "name",
            ]:

                if candidate in departments.columns:
                    department_name_col = candidate
                    break

            if department_name_col:

                department_reorder = department_reorder.merge(
                    departments[
                        [
                            "department_id",
                            department_name_col,
                        ]
                    ].drop_duplicates("department_id"),
                    on="department_id",
                    how="left",
                )

        if department_name_col:

            department_reorder["label"] = (
                department_reorder[department_name_col]
                .fillna(
                    "Département "
                    + department_reorder["department_id"].astype(str)
                )
            )

        else:

            department_reorder["label"] = (
                "Département "
                + department_reorder["department_id"].astype(str)
            )

        department_reorder = department_reorder.sort_values(
            "taux_reachat",
            ascending=True,
        )

        fig = px.bar(
            department_reorder,
            x="taux_reachat",
            y="label",
            orientation="h",
            title="Taux de réachat moyen par département",
            labels={
                "taux_reachat": "Taux de réachat",
                "label": "",
            },
        )

        fig.update_xaxes(tickformat=".0%")
        fig = base_fig(fig)

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    # ============================================================
    # PROPENSITY SCORE
    # ============================================================

    st.markdown(
        '<div class="section-title">Distribution des scores de propension</div>',
        unsafe_allow_html=True,
    )

    score_data = scores.copy()

    if score_data is not None and "propensity_score" in score_data.columns:

        fig = px.histogram(
            score_data,
            x="propensity_score",
            nbins=40,
            title="Distribution des propensity scores",
            labels={
                "propensity_score": "Score de propension",
                "count": "Nombre de couples client-produit",
            },
        )

        fig.update_xaxes(tickformat=".0%")
        fig = base_fig(fig)

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

        st.markdown(
            """
            <div class="insight-card insight-cyan">
                <b>Lecture marketing</b><br><br>
                Le score de propension permet d'identifier les couples
                client-produit présentant la plus forte probabilité
                d'intérêt. Les scores élevés peuvent être utilisés pour
                prioriser les recommandations, les campagnes de
                cross-sell ou les scénarios de réachat.
            </div>
            """,
            unsafe_allow_html=True,
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

        st.warning(
            "Aucune ligne de scoring ne correspond aux filtres globaux."
        )

        st.stop()

    customers = np.sort(
        filtered_scores[
            "user_id"
        ]
        .dropna()
        .unique()
    )

    customer_id = st.selectbox(

        "Sélectionner un client",

        customers,

        key="selected_customer",
    )

    customer_scores = (

        filtered_scores[

            filtered_scores[
                "user_id"
            ] == customer_id

        ]

        .sort_values(
            "propensity_score",
            ascending=False
        )

        .head(10)

        .copy()
    )

    if customer_scores.empty:

        st.warning(
            "Aucune recommandation disponible pour ce client."
        )

        st.stop()

    max_score = (
        customer_scores[
            "propensity_score"
        ].max()
    )

    mean_score = (
        customer_scores[
            "propensity_score"
        ].mean()
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.metric(
            "Produits recommandés",
            len(customer_scores)
        )

    with c2:

        st.metric(
            "Score maximum",
            format_pct(max_score)
        )

    with c3:

        st.metric(
            "Score moyen",
            format_pct(mean_score)
        )

    with c4:

        if (
            "score_decile"
            in customer_scores.columns
        ):

            best_decile = (
                customer_scores[
                    "score_decile"
                ].min()
            )

            st.metric(
                "Meilleur décile",
                f"D{int(best_decile)}"
            )

        else:

            st.metric(
                "Client",
                str(customer_id)
            )

    st.markdown(
        """
        <div class="insight-card insight-blue">

        <b>Lecture CRM :</b>
        les produits du haut du classement correspondent
        aux signaux d’appétence les plus élevés pour ce client.

        Le score sert à prioriser les actions.

        </div>
        """,
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

    display_df = (
        customer_scores[
            display_columns
        ].copy()
    )

    display_df = display_df.rename(

        columns={

            "product_name":
                "Produit",

            "department":
                "Département",

            "propensity_score":
                "Score de propension",

            "score_decile":
                "Décile",

            "marketing_segment":
                "Segment marketing",
        }
    )

    if (
        "Score de propension"
        in display_df.columns
    ):

        display_df[
            "Score de propension"
        ] = (

            display_df[
                "Score de propension"
            ]

            .map(format_pct)
        )

    st.dataframe(

        display_df,

        use_container_width=True,

        hide_index=True,
    )

    left, right = st.columns(
        [1.35, 1]
    )

    with left:

        if px is not None:

            chart_df = (
                customer_scores.copy()
            )

            chart_df["Produit"] = (

                chart_df[
                    "product_name"
                ]

                .fillna(
                    "Produit inconnu"
                )
            )

            chart_df = (
                chart_df
                .sort_values(
                    "propensity_score"
                )
            )

            fig = px.bar(

                chart_df,

                x="propensity_score",

                y="Produit",

                orientation="h",

                title="Intensité de l’appétence",

                labels={

                    "propensity_score":
                    "Score",

                    "Produit":
                    "",
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

            base_fig(
                fig,
                440
            )

            st.plotly_chart(

                fig,

                use_container_width=True,

                config={
                    "displayModeBar": False
                },
            )

    with right:

        if (
            "department"
            in customer_scores.columns
            and px is not None
        ):

            dept_mix = (

                customer_scores[
                    "department"
                ]

                .value_counts()

                .rename_axis(
                    "Département"
                )

                .reset_index(
                    name="Produits recommandés"
                )
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

            base_fig(
                fig,
                440
            )

            st.plotly_chart(

                fig,

                use_container_width=True,

                config={
                    "displayModeBar": False
                },
            )

    top_product = (
        customer_scores.iloc[0]
    )

    score = (
        top_product[
            "propensity_score"
        ]
    )

    product_name = (
        top_product.get(
            "product_name",
            "ce produit"
        )
    )

    department = (
        top_product.get(
            "department",
            "département inconnu"
        )
    )

    st.markdown(
        '<div class="section-title">💡 Traduction marketing</div>',
        unsafe_allow_html=True,
    )

    if score >= 0.80:

        title = (
            "Forte appétence détectée"
        )

        action = (
            "Prioriser une recommandation "
            "personnalisée ou un scénario de réachat."
        )

        css = "insight-blue"

    elif score >= 0.60:

        title = (
            "Appétence intéressante"
        )

        action = (
            "Tester une personnalisation "
            "ou un A/B test avant généralisation."
        )

        css = "insight-cyan"

    else:

        title = (
            "Appétence modérée à faible"
        )

        action = (
            "Ne pas prioriser ce produit "
            "dans une pression commerciale immédiate."
        )

        css = "insight-orange"

    st.markdown(
    f"""<div class="insight-card {css}">
<b>{title}</b>
<br><br>
<b>{product_name}</b> · {department}
<br>
Score de propension : <b>{score:.1%}</b>
<br><br>
→ <b>Action :</b> {action}
</div>""",
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


    # ========================================================
    # PERFORMANCE
    # ========================================================

    st.markdown(
        '<div class="section-title">🏆 Performance de ranking</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.metric(
            "ROC-AUC",
            "0.8076"
        )

    with c2:

        st.metric(
            "PR-AUC",
            "0.7034"
        )

    with c3:

        st.metric(
            "Precision@10%",
            "82.0%"
        )

    with c4:

        st.metric(
            "Lift@10%",
            "2.30×"
        )


    if px is not None:

        metric_df = pd.DataFrame(

            {
                "Métrique": [

                    "ROC-AUC",

                    "PR-AUC",

                    "Precision@10%",

                ],

                "Valeur": [

                    0.8076,

                    0.7034,

                    0.8200,

                ],
            }
        )

        fig = px.bar(

            metric_df,

            x="Métrique",

            y="Valeur",

            title="Indicateurs clés de performance",

            labels={
                "Valeur":
                "Score"
            },

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

        base_fig(
            fig,
            350
        )

        st.plotly_chart(

            fig,

            use_container_width=True,

            config={
                "displayModeBar": False
            },
        )


    st.markdown(
        """
        <div class="insight-card insight-violet">

        <b>Lecture business :</b>
        à 10% de la population ciblée, le modèle
        concentre environ <b>2,30×</b> plus de positifs
        qu’un ciblage aléatoire.

        La Precision@10% est de
        <b>82,0%</b> sur l’échantillon de validation utilisé.

        </div>
        """,
        unsafe_allow_html=True,
    )


    # ========================================================
    # SHAP
    # ========================================================

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
            "Le fichier data/output/shap_summary.png "
            "n'a pas été trouvé."
        )


    st.markdown(
        """
        <div class="insight-card insight-cyan">

        <b>Lecture métier :</b>
        les variables client-produit sont particulièrement
        pertinentes pour une logique de réachat :

        fréquence d’achat, taux de réachat, récence
        et part des commandes du client contenant le produit.

        </div>
        """,
        unsafe_allow_html=True,
    )


    # ========================================================
    # LIFT
    # ========================================================

    st.markdown(
        '<div class="section-title">📈 Efficacité du ciblage</div>',
        unsafe_allow_html=True,
    )

    lift_df = load_lift_curve()

    if (
        lift_df is not None
        and not lift_df.empty
    ):

        if px is not None:

            plot_df = (
                lift_df.copy()
            )

            plot_df[
                "Population ciblée"
            ] = (

                plot_df[
                    "Population ciblée"
                ] * 100
            )

            fig = px.line(

                plot_df,

                x="Population ciblée",

                y="Lift",

                markers=True,

                title="Lift selon la part de la population ciblée",

                labels={

                    "Population ciblée":
                    "Population ciblée (%)",

                    "Lift":
                    "Lift",
                },

                color_discrete_sequence=[
                    BLUE
                ],
            )

            fig.add_hline(

                y=1,

                line_dash="dash",

                annotation_text=
                "Référence aléatoire",
            )

            fig.update_xaxes(
                ticksuffix="%"
            )

            base_fig(
                fig,
                390
            )

            st.plotly_chart(

                fig,

                use_container_width=True,

                config={
                    "displayModeBar": False
                },
            )


        top10 = lift_df.iloc[

            np.argmin(

                np.abs(

                    lift_df[
                        "Population ciblée"
                    ].to_numpy()

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

                format_pct(
                    top10["Precision"]
                ),
            )

        with c3:

            st.metric(

                "Recall à 10%",

                format_pct(
                    top10["Recall"]
                ),
            )


        st.markdown(
            """
            <div class="insight-card insight-teal">

            <b>Lecture CRM :</b>
            la courbe permet de visualiser jusqu’où réduire
            la population ciblée tout en conservant une
            concentration élevée d’acheteurs potentiels.

            </div>
            """,
            unsafe_allow_html=True,
        )

    else:

        st.warning(
            "Le fichier lift_curve.csv est introuvable. "
            "Lance 08_prepare_deployment.py."
        )


    st.caption(
        """
        Important : la validation actuelle repose sur un split
        selon customer_total_orders. Il s'agit d'un split par
        profondeur d'historique et non d'un véritable
        backtest temporel.
        """
    )


    # ========================================================
    # VARIABLE FAMILIES
    # ========================================================

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