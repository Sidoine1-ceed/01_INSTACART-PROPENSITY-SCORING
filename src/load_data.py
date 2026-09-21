import pandas as pd

from src.config import (
    ORDERS_PATH,
    PRIOR_PATH,
    TRAIN_PATH,
    PRODUCTS_PATH,
    AISLES_PATH,
    DEPARTMENTS_PATH
)


def load_orders():
    return pd.read_csv(ORDERS_PATH)


def load_prior():
    return pd.read_csv(PRIOR_PATH)


def load_train():
    return pd.read_csv(TRAIN_PATH)


def load_products():
    return pd.read_csv(PRODUCTS_PATH)


def load_aisles():
    return pd.read_csv(AISLES_PATH)


def load_departments():
    return pd.read_csv(DEPARTMENTS_PATH)


def load_all_data():

    orders = load_orders()
    prior = load_prior()
    train = load_train()
    products = load_products()
    aisles = load_aisles()
    departments = load_departments()

    return {
        "orders": orders,
        "prior": prior,
        "train": train,
        "products": products,
        "aisles": aisles,
        "departments": departments
    }


if __name__ == "__main__":

    data = load_all_data()

    for name, df in data.items():

        print(
            f"{name:<15} "
            f"{df.shape[0]:>10,} rows "
            f"{df.shape[1]:>3} columns"
        )