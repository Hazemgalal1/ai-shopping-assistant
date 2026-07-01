"""
Data Collection Script
Role: Hazem Galal & Ahmed Osama (Data Engineers)
"""

import os
import json
import logging
import pandas as pd
import requests
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

REQUIRED_FIELDS = ["name", "description", "price", "category", "rating"]


def load_from_csv(filepath: str) -> pd.DataFrame:
    """Load dataset from local CSV file (e.g. downloaded from Kaggle)."""
    logger.info(f"Loading data from {filepath}")
    df = pd.read_csv(filepath)
    logger.info(f"Loaded {len(df)} records with columns: {list(df.columns)}")
    return df


def validate_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate that required fields exist.
    Attempts to auto-map common column name variants.
    """
    column_map = {
        "product_name": "name",
        "title": "name",
        "product_title": "name",
        "about_product": "description",
        "product_description": "description",
        "discounted_price": "price",
        "actual_price": "price",
        "selling_price": "price",
        "product_price": "price",
        "category": "category",
        "main_category": "category",
        "rating": "rating",
        "average_rating": "rating",
        "review_rating": "rating",
    }

    df = df.rename(columns={k: v for k, v in column_map.items() if k in df.columns})

    missing = [f for f in REQUIRED_FIELDS if f not in df.columns]
    if missing:
        logger.warning(f"Missing fields after mapping: {missing}")
        for field in missing:
            df[field] = None
    else:
        logger.info("Schema validation passed — all required fields present.")

    return df[REQUIRED_FIELDS + [c for c in df.columns if c not in REQUIRED_FIELDS]]


def generate_sample_dataset(n: int = 500) -> pd.DataFrame:
    """
    Generate a sample dataset for development/testing purposes.
    Replace this with real Kaggle data in production.
    """
    import random
    import numpy as np

    random.seed(42)
    np.random.seed(42)

    categories = ["Electronics", "Clothing", "Books", "Home & Kitchen", "Sports", "Beauty", "Toys"]

    products = []
    for i in range(n):
        cat = random.choice(categories)
        products.append({
            "name": f"Sample Product {i+1} ({cat})",
            "description": (
                f"High quality {cat.lower()} product with excellent features. "
                f"Perfect for everyday use. Durable and affordable."
            ),
            "price": round(random.uniform(5.0, 500.0), 2),
            "category": cat,
            "rating": round(random.uniform(1.0, 5.0), 1),
        })

    df = pd.DataFrame(products)
    logger.info(f"Generated {len(df)} sample products.")
    return df


def save_raw(df: pd.DataFrame, filename: str = "products_raw.csv"):
    path = RAW_DIR / filename
    df.to_csv(path, index=False)
    logger.info(f"Raw data saved to {path}")
    return path


def collect(source_csv: str = None) -> pd.DataFrame:
    """
    Main collection function.
    - If source_csv provided: load from file
    - Otherwise: use sample generator (dev mode)
    """
    if source_csv and os.path.exists(source_csv):
        df = load_from_csv(source_csv)
    else:
        logger.warning("No CSV provided — using generated sample dataset.")
        df = generate_sample_dataset()

    df = validate_schema(df)
    save_raw(df)
    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=None, help="Path to Kaggle CSV file")
    args = parser.parse_args()
    df = collect(source_csv=args.csv)
    print(df.head())
    print(f"\nShape: {df.shape}")
