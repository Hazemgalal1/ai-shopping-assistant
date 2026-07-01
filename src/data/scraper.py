"""
Real Product Scraper
Scrapes product data from open sources (no auth required)
Sources: Open Food Facts API, Fake Store API, Books to Scrape
"""

import time
import logging
import requests
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)


# ─── Source 1: Fake Store API (Electronics, Clothing, Jewelry) ───────────────

def scrape_fakestore(max_products: int = 20) -> list[dict]:
    """Free open API with real-looking product data."""
    logger.info("Scraping Fake Store API...")
    try:
        resp = requests.get("https://fakestoreapi.com/products", timeout=10)
        resp.raise_for_status()
        items = resp.json()

        products = []
        for item in items[:max_products]:
            products.append({
                "name": item.get("title", "")[:100],
                "description": item.get("description", "")[:300],
                "price": float(item.get("price", 0)),
                "category": item.get("category", "General").title(),
                "rating": float(item.get("rating", {}).get("rate", 3.0)),
            })

        logger.info(f"Fakestore: {len(products)} products")
        return products
    except Exception as e:
        logger.warning(f"Fakestore scrape failed: {e}")
        return []


# ─── Source 2: DummyJSON Products API ────────────────────────────────────────

def scrape_dummyjson(max_products: int = 100) -> list[dict]:
    """DummyJSON — free API with 100 products across many categories."""
    logger.info("Scraping DummyJSON API...")
    try:
        products = []
        limit = 30
        skip = 0

        while len(products) < max_products:
            resp = requests.get(
                f"https://dummyjson.com/products?limit={limit}&skip={skip}",
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            items = data.get("products", [])
            if not items:
                break

            for item in items:
                products.append({
                    "name": item.get("title", "")[:100],
                    "description": item.get("description", "")[:300],
                    "price": float(item.get("price", 0)),
                    "category": item.get("category", "General").replace("-", " ").title(),
                    "rating": float(item.get("rating", 3.0)),
                })

            skip += limit
            if skip >= data.get("total", 0):
                break
            time.sleep(0.3)

        logger.info(f"DummyJSON: {len(products)} products")
        return products[:max_products]
    except Exception as e:
        logger.warning(f"DummyJSON scrape failed: {e}")
        return []


# ─── Source 3: Open Library (Books) ──────────────────────────────────────────

def scrape_books(subjects: list[str] = None, max_per_subject: int = 20) -> list[dict]:
    """Open Library API — real book data."""
    if subjects is None:
        subjects = ["python", "machine learning", "business", "self help"]

    logger.info("Scraping Open Library...")
    products = []

    for subject in subjects:
        try:
            resp = requests.get(
                f"https://openlibrary.org/subjects/{subject.replace(' ', '_')}.json?limit={max_per_subject}",
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            works = data.get("works", [])

            for work in works:
                title = work.get("title", "")
                authors = work.get("authors", [{}])
                author = authors[0].get("name", "Unknown") if authors else "Unknown"
                desc = f"A book about {subject} by {author}."

                products.append({
                    "name": title[:100],
                    "description": desc[:300],
                    "price": round(10 + (hash(title) % 40), 2),  # Simulated price $10-$50
                    "category": "Books",
                    "rating": round(3.5 + (hash(title) % 15) / 10, 1),
                })

            time.sleep(0.5)
        except Exception as e:
            logger.warning(f"OpenLibrary subject '{subject}' failed: {e}")

    logger.info(f"Books: {len(products)} products")
    return products


# ─── Main scraper ─────────────────────────────────────────────────────────────

def scrape_all(target: int = 300) -> pd.DataFrame:
    """Scrape from all sources and combine."""
    logger.info(f"Starting scrape — target: {target} products")
    all_products = []

    # Source 1
    all_products.extend(scrape_fakestore(20))

    # Source 2
    all_products.extend(scrape_dummyjson(200))

    # Source 3
    all_products.extend(scrape_books(max_per_subject=20))

    if not all_products:
        logger.warning("All scraping failed — falling back to sample data")
        from src.data.collect import generate_sample_dataset
        df = generate_sample_dataset(target)
        df.to_csv(RAW_DIR / "products_raw.csv", index=False)
        return df

    df = pd.DataFrame(all_products)

    # Basic dedup
    df = df.drop_duplicates(subset=["name"])
    df = df[df["name"].str.strip() != ""]
    df = df.reset_index(drop=True)

    out = RAW_DIR / "products_raw.csv"
    df.to_csv(out, index=False)
    logger.info(f"Saved {len(df)} real products → {out}")

    # Summary
    print(f"\n{'='*40}")
    print(f"Total products scraped: {len(df)}")
    print(f"Categories: {df['category'].nunique()}")
    print(f"\nBy category:")
    print(df['category'].value_counts().to_string())
    print(f"{'='*40}\n")

    return df


if __name__ == "__main__":
    df = scrape_all()
    print(df.head(10))
