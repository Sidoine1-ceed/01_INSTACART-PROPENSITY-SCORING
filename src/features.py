####### Feature engineering (construction des variables) ##############

##############################################
# Variables Client
    # total_orders
    # avg_days_between_orders
    # unique_products
    #unique_departments
    # unique_aisles
    # avg_products_per_order
    # preferred_day_of_week
    # preferred_hour
    # preferred_department

# Variables Produit
    # product_purchase_count
    # product_reorder_rate
    # product_unique_users

# Variables Client × Produit
    # customer_product_purchase_count
    # customer_product_reorder_rate
    # days_since_customer_last_bought_product
    # share_of_customer_orders_containing_product
    # product_streak

#########################################



import numpy as np
import pandas as pd


def build_base_history(orders, prior):

    history = prior.merge(
        orders[
            [
                "order_id",
                "user_id",
                "order_number",
                "order_dow",
                "order_hour_of_day",
                "days_since_prior_order"
            ]
        ],
        on="order_id",
        how="left"
    )

    return history


# ============================================================
# CUSTOMER FEATURES
# ============================================================

def build_customer_features(history):

    # --------------------------------------------------------
    # Une ligne par commande
    # --------------------------------------------------------
    customer_orders = (
        history
        .groupby("order_id", as_index=False)
        .agg(
            user_id=("user_id", "first"),
            order_number=("order_number", "first"),
            order_dow=("order_dow", "first"),
            order_hour_of_day=("order_hour_of_day", "first"),
            days_since_prior_order=("days_since_prior_order", "first")
        )
    )

    # --------------------------------------------------------
    # Variables client
    # --------------------------------------------------------
    customer_features = (
        customer_orders
        .groupby("user_id")
        .agg(
            total_orders=("order_id", "nunique"),

            avg_days_between_orders=(
                "days_since_prior_order",
                "mean"
            ),

            median_days_between_orders=(
                "days_since_prior_order",
                "median"
            ),

            avg_order_number=(
                "order_number",
                "mean"
            )
        )
        .reset_index()
    )

    # --------------------------------------------------------
    # Nombre de produits différents
    # --------------------------------------------------------
    products_per_customer = (
        history
        .groupby("user_id")["product_id"]
        .nunique()
        .reset_index(
            name="unique_products"
        )
    )

    customer_features = customer_features.merge(
        products_per_customer,
        on="user_id",
        how="left"
    )

    # --------------------------------------------------------
    # Nombre d'aisles différentes
    # --------------------------------------------------------
    if "aisle_id" in history.columns:

        unique_aisles = (
            history
            .groupby("user_id")["aisle_id"]
            .nunique()
            .reset_index(
                name="unique_aisles"
            )
        )

        customer_features = customer_features.merge(
            unique_aisles,
            on="user_id",
            how="left"
        )

    # --------------------------------------------------------
    # Nombre de départements différents
    # --------------------------------------------------------
    if "department_id" in history.columns:

        unique_departments = (
            history
            .groupby("user_id")["department_id"]
            .nunique()
            .reset_index(
                name="unique_departments"
            )
        )

        customer_features = customer_features.merge(
            unique_departments,
            on="user_id",
            how="left"
        )

    # --------------------------------------------------------
    # Taille moyenne du panier
    # --------------------------------------------------------
    basket_size = (
        history
        .groupby(
            ["user_id", "order_id"]
        )
        .size()
        .groupby("user_id")
        .mean()
        .reset_index(
            name="avg_products_per_order"
        )
    )

    customer_features = customer_features.merge(
        basket_size,
        on="user_id",
        how="left"
    )

    # --------------------------------------------------------
    # Jour préféré
    # --------------------------------------------------------
    preferred_day = (
        customer_orders
        .groupby(
            ["user_id", "order_dow"]
        )
        .size()
        .reset_index(name="count")
        .sort_values(
            ["user_id", "count"],
            ascending=[True, False]
        )
        .drop_duplicates("user_id")
        [["user_id", "order_dow"]]
        .rename(
            columns={
                "order_dow":
                "preferred_day_of_week"
            }
        )
    )

    customer_features = customer_features.merge(
        preferred_day,
        on="user_id",
        how="left"
    )

    # --------------------------------------------------------
    # Heure préférée
    # --------------------------------------------------------
    preferred_hour = (
        customer_orders
        .groupby(
            ["user_id", "order_hour_of_day"]
        )
        .size()
        .reset_index(name="count")
        .sort_values(
            ["user_id", "count"],
            ascending=[True, False]
        )
        .drop_duplicates("user_id")
        [["user_id", "order_hour_of_day"]]
        .rename(
            columns={
                "order_hour_of_day":
                "preferred_hour"
            }
        )
    )

    customer_features = customer_features.merge(
        preferred_hour,
        on="user_id",
        how="left"
    )

    return customer_features

# ============================================================
# PRODUCT FEATURES
# ============================================================

def build_product_features(history):

    product_features = (
        history
        .groupby("product_id")
        .agg(
            product_purchase_count=(
                "order_id",
                "count"
            ),

            product_unique_users=(
                "user_id",
                "nunique"
            ),

            product_reorder_rate=(
                "reordered",
                "mean"
            )
        )
        .reset_index()
    )

    return product_features


# ============================================================
# CUSTOMER × PRODUCT FEATURES
# ============================================================

def build_customer_product_features(history):

    # --------------------------------------------------------
    # Basic frequency
    # --------------------------------------------------------

    cp = (
        history
        .groupby(
            ["user_id", "product_id"]
        )
        .agg(
            customer_product_purchase_count=(
                "order_id",
                "count"
            ),

            customer_product_reorder_rate=(
                "reordered",
                "mean"
            ),

            first_order_number=(
                "order_number",
                "min"
            ),

            last_order_number=(
                "order_number",
                "max"
            ),

            last_purchase_dow=(
                "order_dow",
                "last"
            )
        )
        .reset_index()
    )

    # --------------------------------------------------------
    # Last purchase
    # --------------------------------------------------------

    last_order_by_customer = (
        history
        .groupby("user_id")["order_number"]
        .max()
        .reset_index(
            name="last_customer_order"
        )
    )

    cp = cp.merge(
        last_order_by_customer,
        on="user_id",
        how="left"
    )

    cp["orders_since_last_purchase"] = (
        cp["last_customer_order"]
        - cp["last_order_number"]
    )

    # --------------------------------------------------------
    # Share of customer orders containing product
    # --------------------------------------------------------

    total_customer_orders = (
        history
        .groupby("user_id")["order_id"]
        .nunique()
        .reset_index(
            name="customer_total_orders"
        )
    )

    cp = cp.merge(
        total_customer_orders,
        on="user_id",
        how="left"
    )

    cp["share_of_customer_orders_containing_product"] = (
        cp["customer_product_purchase_count"]
        / cp["customer_total_orders"]
    )

    # --------------------------------------------------------
    # Purchase interval
    # --------------------------------------------------------

    cp["avg_orders_between_purchases"] = np.where(
        cp["customer_product_purchase_count"] > 1,
        (
            cp["last_order_number"]
            - cp["first_order_number"]
        )
        / (
            cp["customer_product_purchase_count"] - 1
        ),
        np.nan
    )

    return cp


# ============================================================
# DEPARTMENT FEATURES
# ============================================================

def build_customer_department_features(history):

    customer_department = (
        history
        .groupby(
            ["user_id", "department_id"]
        )
        .agg(
            customer_department_purchase_count=(
                "order_id",
                "count"
            )
        )
        .reset_index()
    )

    customer_total = (
        history
        .groupby("user_id")
        .size()
        .reset_index(
            name="customer_total_product_purchases"
        )
    )

    customer_department = customer_department.merge(
        customer_total,
        on="user_id",
        how="left"
    )

    customer_department[
        "customer_department_share"
    ] = (
        customer_department[
            "customer_department_purchase_count"
        ]
        / customer_department[
            "customer_total_product_purchases"
        ]
    )

    return customer_department




def compute_product_streak(history):

    orders_product = (
        history[
            [
                "user_id",
                "order_number",
                "product_id"
            ]
        ]
        .drop_duplicates()
        .sort_values(
            [
                "user_id",
                "product_id",
                "order_number"
            ]
        )
    )

    def calculate_streak(group):

        order_numbers = (
            group["order_number"]
            .values
        )

        if len(order_numbers) == 0:
            return 0

        streak = 1

        for i in range(
            len(order_numbers) - 1,
            0,
            -1
        ):

            if (
                order_numbers[i]
                - order_numbers[i - 1]
                == 1
            ):
                streak += 1

            else:
                break

        return streak

    streak = (
        orders_product
        .groupby(
            [
                "user_id",
                "product_id"
            ]
        )
        .apply(
            calculate_streak,
            include_groups=False
        )
        .reset_index(
            name="product_streak"
        )
    )

    return streak




# ============================================================
# COMPLETE FEATURE TABLE
# ============================================================

def build_all_features(
    orders,
    prior,
    products
):

    history = build_base_history(
        orders,
        prior
    )

    # Add product metadata
    history = history.merge(
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

    customer_features = build_customer_features(
        history
    )

    product_features = build_product_features(
        history
    )

    cp_features = build_customer_product_features(
        history
    )

    department_features = (
        build_customer_department_features(
            history
        )
    )


    return (
        history,
        customer_features,
        product_features,
        cp_features,
        department_features
    )


