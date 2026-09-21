import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.load_data import load_all_data


data = load_all_data()

orders = data["orders"]
prior = data["prior"]
train = data["train"]
products = data["products"]
aisles = data["aisles"]
departments = data["departments"]


# ============================================================
# 1. DIMENSIONS
# ============================================================

print("\n========== DATASET SIZE ==========\n")

for name, df in data.items():
    print(f"{name:<15}: {df.shape}")


# ============================================================
# 2. USERS
# ============================================================

print("\n========== USERS ==========\n")

print(
    "Number of users:",
    orders["user_id"].nunique()
)

print(
    "Average orders/user:",
    orders.groupby("user_id")["order_number"]
    .max()
    .mean()
)


# ============================================================
# 3. ORDERS
# ============================================================

order_counts = (
    orders
    .groupby("user_id")["order_number"]
    .max()
)

plt.figure(figsize=(10, 5))

sns.histplot(
    order_counts,
    bins=30
)

plt.title("Number of orders per customer")
plt.xlabel("Number of orders")
plt.ylabel("Customers")

plt.tight_layout()

plt.savefig(
    "data/output/orders_per_customer.png",
    dpi=150
)

plt.show()


# ============================================================
# 4. PURCHASED PRODUCTS
# ============================================================

product_frequency = (
    prior
    .groupby("product_id")
    .size()
    .sort_values(ascending=False)
)

top_products = product_frequency.head(20)

plt.figure(figsize=(12, 6))

top_products.sort_values().plot(kind="barh")

plt.title("Top 20 most purchased products")

plt.xlabel("Number of purchases")

plt.tight_layout()

plt.savefig(
    "data/output/top_products.png",
    dpi=150
)

plt.show()


# ============================================================
# 5. DEPARTMENTS
# ============================================================

department_mapping = (
    products
    .merge(
        departments,
        on="department_id",
        how="left"
    )
)

department_frequency = (
    prior
    .merge(
        department_mapping[
            ["product_id", "department_id", "department"]
        ],
        on="product_id",
        how="left"
    )
    .groupby("department")["product_id"]
    .count()
    .sort_values(ascending=False)
)

plt.figure(figsize=(12, 7))

department_frequency.plot(kind="barh")

plt.title("Purchases by department")

plt.xlabel("Number of purchases")

plt.tight_layout()

plt.savefig(
    "data/output/purchases_by_department.png",
    dpi=150
)

plt.show()