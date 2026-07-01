"""
Product Recommendation Engine — Content-Based Filtering
Role: Mohamed Ahmed Morsy (ML Engineer)
"""

import logging
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class RecommendationEngine:
    def __init__(self):
        self.df: pd.DataFrame = None
        self.embeddings: np.ndarray = None

    def load(self, df: pd.DataFrame, embeddings: np.ndarray):
        """Load dataset and precomputed embeddings."""
        self.df = df.reset_index(drop=True)
        self.embeddings = embeddings
        logger.info(f"Recommendation engine loaded: {len(df)} products")

    # ─── Recommend by product_id ───────────────────────────────────────────────

    def recommend(self, product_id: int, top_k: int = 5) -> list[dict]:
        """
        Return top_k similar products for a given product_id.
        Uses cosine similarity on sentence embeddings.
        """
        if product_id not in self.df["product_id"].values:
            logger.warning(f"product_id {product_id} not found")
            return []

        idx = self.df[self.df["product_id"] == product_id].index[0]
        query_emb = self.embeddings[idx].reshape(1, -1)

        scores = cosine_similarity(query_emb, self.embeddings).flatten()
        scores[idx] = -1  # exclude self

        top_indices = scores.argsort()[::-1][:top_k]
        return self._format_results(top_indices, scores)

    # ─── Recommend by category ─────────────────────────────────────────────────

    def recommend_by_category(self, category: str, top_k: int = 5) -> list[dict]:
        """Return top-rated products in a category."""
        cat_df = self.df[self.df["category"].str.lower() == category.lower()]
        if cat_df.empty:
            return []
        top = cat_df.nlargest(top_k, "rating")
        return top[["product_id", "name", "description", "price", "category", "rating"]].to_dict("records")

    # ─── Complementary products ────────────────────────────────────────────────

    def recommend_complementary(self, product_id: int, top_k: int = 5) -> list[dict]:
        """
        Recommend products from DIFFERENT categories (cross-sell).
        """
        if product_id not in self.df["product_id"].values:
            return []

        idx = self.df[self.df["product_id"] == product_id].index[0]
        source_category = self.df.iloc[idx]["category"]

        # Filter to other categories
        other_df = self.df[self.df["category"] != source_category]
        if other_df.empty:
            return []

        other_indices = other_df.index.tolist()
        query_emb = self.embeddings[idx].reshape(1, -1)
        other_embs = self.embeddings[other_indices]

        scores = cosine_similarity(query_emb, other_embs).flatten()
        top_local = scores.argsort()[::-1][:top_k]
        top_global = [other_indices[i] for i in top_local]

        return self._format_results(top_global, dict(zip(top_global, scores[top_local])))

    # ─── Helpers ──────────────────────────────────────────────────────────────

    def _format_results(self, indices, scores) -> list[dict]:
        results = []
        for idx in indices:
            if idx < 0 or idx >= len(self.df):
                continue
            row = self.df.iloc[idx]
            score = scores[idx] if isinstance(scores, np.ndarray) else scores.get(idx, 0.0)
            results.append({
                "product_id": int(row["product_id"]),
                "name": row["name"],
                "description": row["description"],
                "price": float(row["price"]),
                "category": row["category"],
                "rating": float(row["rating"]),
                "similarity_score": round(float(score), 4),
            })
        return results
