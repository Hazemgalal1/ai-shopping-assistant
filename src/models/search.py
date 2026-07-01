"""
NLP Search Model — TF-IDF + Sentence Embeddings + FAISS
Role: Mohamed Ahmed Morsy (ML Engineer)
"""

import logging
import numpy as np
import pandas as pd
import joblib
import faiss
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDINGS_DIR = Path("models/embeddings")
TFIDF_DIR = Path("models/tfidf")
EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
TFIDF_DIR.mkdir(parents=True, exist_ok=True)


class SearchEngine:
    def __init__(self):
        self.df: pd.DataFrame = None
        self.tfidf_vectorizer: TfidfVectorizer = None
        self.tfidf_matrix = None
        self.embedder: SentenceTransformer = None
        self.embeddings: np.ndarray = None
        self.faiss_index = None

    # ─── Build ────────────────────────────────────────────────────────────────

    def build(self, df: pd.DataFrame):
        """Build TF-IDF and embedding indexes from clean dataframe."""
        self.df = df.reset_index(drop=True)
        logger.info(f"Building search index for {len(df)} products...")

        self._build_tfidf()
        self._build_embeddings()
        self._build_faiss()

        logger.info("Search engine built successfully.")

    def _build_tfidf(self):
        logger.info("Building TF-IDF index...")
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=10_000,
            ngram_range=(1, 2),
            stop_words="english",
        )
        corpus = self.df["search_text"].fillna("").tolist()
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(corpus)
        joblib.dump(self.tfidf_vectorizer, TFIDF_DIR / "tfidf_vectorizer.pkl")
        logger.info("TF-IDF index built.")

    def _build_embeddings(self):
        logger.info(f"Building sentence embeddings with {MODEL_NAME}...")
        self.embedder = SentenceTransformer(MODEL_NAME)
        corpus = self.df["search_text"].fillna("").tolist()
        self.embeddings = self.embedder.encode(
            corpus, batch_size=64, show_progress_bar=True, normalize_embeddings=True
        )
        np.save(EMBEDDINGS_DIR / "product_embeddings.npy", self.embeddings)
        logger.info(f"Embeddings shape: {self.embeddings.shape}")

    def _build_faiss(self):
        logger.info("Building FAISS index...")
        dim = self.embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dim)  # Inner Product (cosine on normalized vecs)
        self.faiss_index.add(self.embeddings.astype("float32"))
        faiss.write_index(self.faiss_index, str(EMBEDDINGS_DIR / "faiss_index.bin"))
        logger.info(f"FAISS index built with {self.faiss_index.ntotal} vectors.")

    # ─── Load ─────────────────────────────────────────────────────────────────

    def load(self, df: pd.DataFrame):
        """Load pre-built indexes."""
        self.df = df.reset_index(drop=True)
        self.tfidf_vectorizer = joblib.load(TFIDF_DIR / "tfidf_vectorizer.pkl")
        self.embeddings = np.load(EMBEDDINGS_DIR / "product_embeddings.npy")
        self.faiss_index = faiss.read_index(str(EMBEDDINGS_DIR / "faiss_index.bin"))
        self.embedder = SentenceTransformer(MODEL_NAME)
        logger.info("Search engine loaded from disk.")

    # ─── Search ───────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = 10,
        method: str = "hybrid",
        min_price: float = None,
        max_price: float = None,
        category: str = None,
        min_rating: float = None,
    ) -> list[dict]:
        """
        Search products.
        method: 'tfidf' | 'semantic' | 'hybrid'
        """
        if method == "tfidf":
            results = self._tfidf_search(query, top_k * 3)
        elif method == "semantic":
            results = self._semantic_search(query, top_k * 3)
        else:
            results = self._hybrid_search(query, top_k * 3)

        # Apply filters
        results = self._apply_filters(results, min_price, max_price, category, min_rating)

        return results[:top_k]

    def _tfidf_search(self, query: str, top_k: int) -> list[dict]:
        query_vec = self.tfidf_vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        top_indices = scores.argsort()[::-1][:top_k]
        return self._format_results(top_indices, scores, "tfidf")

    def _semantic_search(self, query: str, top_k: int) -> list[dict]:
        query_emb = self.embedder.encode([query], normalize_embeddings=True).astype("float32")
        scores, indices = self.faiss_index.search(query_emb, top_k)
        return self._format_results(indices[0], scores[0], "semantic")

    def _hybrid_search(self, query: str, top_k: int) -> list[dict]:
        """Combine TF-IDF and semantic scores (weighted average)."""
        # TF-IDF scores
        query_vec = self.tfidf_vectorizer.transform([query])
        tfidf_scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        # Semantic scores
        query_emb = self.embedder.encode([query], normalize_embeddings=True).astype("float32")
        sem_scores, sem_indices = self.faiss_index.search(query_emb, len(self.df))
        semantic_scores = np.zeros(len(self.df))
        for idx, score in zip(sem_indices[0], sem_scores[0]):
            if idx >= 0:
                semantic_scores[idx] = score

        # Normalize TF-IDF to [0, 1]
        if tfidf_scores.max() > 0:
            tfidf_scores = tfidf_scores / tfidf_scores.max()

        # Hybrid: 40% TF-IDF + 60% Semantic
        hybrid_scores = 0.4 * tfidf_scores + 0.6 * semantic_scores
        top_indices = hybrid_scores.argsort()[::-1][:top_k]

        return self._format_results(top_indices, hybrid_scores, "hybrid")

    def _format_results(self, indices, scores, method: str) -> list[dict]:
        results = []
        for idx, score in zip(indices, scores):
            if idx < 0 or idx >= len(self.df):
                continue
            row = self.df.iloc[idx]
            results.append({
                "product_id": int(row.get("product_id", idx)),
                "name": row["name"],
                "description": row["description"],
                "price": float(row["price"]),
                "category": row["category"],
                "rating": float(row["rating"]),
                "score": round(float(score), 4),
                "method": method,
            })
        return results

    def _apply_filters(self, results, min_price, max_price, category, min_rating):
        filtered = []
        for r in results:
            if min_price is not None and r["price"] < min_price:
                continue
            if max_price is not None and r["price"] > max_price:
                continue
            if category and r["category"].lower() != category.lower():
                continue
            if min_rating is not None and r["rating"] < min_rating:
                continue
            filtered.append(r)
        return filtered
