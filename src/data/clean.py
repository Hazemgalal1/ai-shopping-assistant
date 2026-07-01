"""
Data Cleaning & Preprocessing
Role: Hazem Galal & Ahmed Osama (Data Engineers)
"""

import re
import logging
import pandas as pd
import numpy as np
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def clean_price(price_series: pd.Series) -> pd.Series:
    """Clean price column — remove currency symbols, convert to float."""
    def _parse(val):
        if pd.isna(val):
            return np.nan
        val = str(val).replace(",", "").replace("$", "").replace("£", "").replace("₹", "").strip()
        match = re.search(r"[\d.]+", val)
        return float(match.group()) if match else np.nan

    return price_series.apply(_parse)


def clean_rating(rating_series: pd.Series) -> pd.Series:
    """Normalize rating to 0–5 scale."""
    def _parse(val):
        if pd.isna(val):
            return np.nan
        val = str(val).strip().split(" ")[0]
        try:
            r = float(val)
            if r > 5:
                r = r / 2  # handle 10-point scales
            return round(min(max(r, 0.0), 5.0), 1)
        except ValueError:
            return np.nan

    return rating_series.apply(_parse)


def clean_text(text_series: pd.Series) -> pd.Series:
    """Lowercase, remove special chars, normalize whitespace."""
    def _clean(val):
        if pd.isna(val):
            return ""
        val = str(val).lower()
        val = re.sub(r"[^a-z0-9\s,.\'-]", " ", val)
        val = re.sub(r"\s+", " ", val).strip()
        return val

    return text_series.apply(_clean)


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates(subset=["name"], keep="first")
    logger.info(f"Removed {before - len(df)} duplicate rows")
    return df


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Fill or drop missing values."""
    before = len(df)

    # Drop rows missing name or description (critical fields)
    df = df.dropna(subset=["name", "description"])
    df = df[df["name"].str.strip() != ""]
    df = df[df["description"].str.strip() != ""]

    # Fill missing price with category median
    df["price"] = df.groupby("category")["price"].transform(
        lambda x: x.fillna(x.median())
    )
    df["price"] = df["price"].fillna(df["price"].median())

    # Fill missing rating with global mean
    df["rating"] = df["rating"].fillna(df["rating"].mean().round(1))

    # Fill missing category
    df["category"] = df["category"].fillna("Uncategorized")

    logger.info(f"Dropped {before - len(df)} rows with missing critical fields")
    return df


def normalize_price(df: pd.DataFrame) -> pd.DataFrame:
    """Add normalized price column (0–1 min-max)."""
    min_p, max_p = df["price"].min(), df["price"].max()
    df["price_normalized"] = ((df["price"] - min_p) / (max_p - min_p)).round(4)
    return df


def create_text_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Combine name + description + category into a single search field."""
    df["search_text"] = (
        df["name"] + " " + df["category"] + " " + df["description"]
    )
    return df


def clean(input_path: str = "data/raw/products_raw.csv") -> pd.DataFrame:
    """Full cleaning pipeline."""
    logger.info("Starting data cleaning pipeline...")

    df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(df)} rows")

    # Step 1: Clean individual columns
    df["price"] = clean_price(df["price"])
    df["rating"] = clean_rating(df["rating"])
    df["name"] = clean_text(df["name"])
    df["description"] = clean_text(df["description"])
    df["category"] = df["category"].str.strip().str.title().fillna("Uncategorized")

    # Step 2: Remove duplicates
    df = remove_duplicates(df)

    # Step 3: Handle missing values
    df = handle_missing(df)

    # Step 4: Filter outliers
    df = df[df["price"] > 0]
    df = df[df["rating"].between(0, 5)]

    # Step 5: Feature engineering
    df = normalize_price(df)
    df = create_text_feature(df)

    # Step 6: Reset index
    df = df.reset_index(drop=True)
    df.insert(0, "product_id", df.index)

    # Save
    output_path = PROCESSED_DIR / "products_clean.csv"
    df.to_csv(output_path, index=False)

    logger.info(f"Cleaned data saved: {output_path} | Shape: {df.shape}")
    return df


if __name__ == "__main__":
    df = clean()
    print(df.head())
    print(f"\nShape: {df.shape}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    print(f"\nCategories: {df['category'].unique()}")
