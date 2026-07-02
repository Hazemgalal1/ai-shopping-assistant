"""
Agent Tools — split into CATALOG_TOOLS (internal catalog) and WEB_TOOLS (live web search)
so each specialized agent only sees the tools relevant to its own role.
"""

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

# ─── Catalog Tools ──────────────────────────────────────────────────────────

CATALOG_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_products",
            "description": (
                "Search the internal product catalog using natural language query. "
                "ALWAYS pass min_price/max_price/category/min_rating if the user mentioned any."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query"},
                    "top_k": {"type": "integer", "default": 5},
                    "method": {
                        "type": "string",
                        "enum": ["hybrid", "semantic", "tfidf"],
                        "default": "hybrid"
                    },
                    "min_price": {"type": "number"},
                    "max_price": {"type": "number"},
                    "category": {"type": "string"},
                    "min_rating": {"type": "number"},
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "filter_products",
            "description": (
                "Filter/browse the internal catalog by category, price, or rating, "
                "without a specific search query."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {"type": "string"},
                    "min_price": {"type": "number"},
                    "max_price": {"type": "number"},
                    "min_rating": {"type": "number"},
                    "sort_by": {"type": "string", "enum": ["rating", "price_asc", "price_desc"], "default": "rating"},
                    "top_k": {"type": "integer", "default": 5}
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_product_details",
            "description": "Get full details of a specific catalog product by ID.",
            "parameters": {
                "type": "object",
                "properties": {"product_id": {"type": "integer"}},
                "required": ["product_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "compare_products",
            "description": "Compare 2-3 catalog products side by side by their IDs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_ids": {
                        "type": "array", "items": {"type": "integer"},
                        "minItems": 2, "maxItems": 3
                    }
                },
                "required": ["product_ids"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "recommend_similar",
            "description": "Get catalog recommendations similar to a given product ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {"type": "integer"},
                    "top_k": {"type": "integer", "default": 5}
                },
                "required": ["product_id"]
            }
        }
    },
]

# ─── Web Tools ──────────────────────────────────────────────────────────────

WEB_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search_products",
            "description": (
                "Search the REAL, LIVE web for real products, current prices, brands, and reviews. "
                "Returns page titles, links, and snippets."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Specific web search query with constraints baked in, e.g. "
                            "'best laptop under $500 2026 review' or 'أفضل لابتوب اقل من 20000 جنيه'"
                        )
                    },
                    "top_k": {"type": "integer", "default": 5}
                },
                "required": ["query"]
            }
        }
    },
]

TOOLS = CATALOG_TOOLS + WEB_TOOLS  # kept for backward compatibility


# ─── Tool Executor ─────────────────────────────────────────────────────────────

class ToolExecutor:
    """Executes tool calls made by any agent. Connected to search/recommend engines + live web search."""

    def __init__(self, search_engine, rec_engine, product_df):
        self.search_engine = search_engine
        self.rec_engine = rec_engine
        self.df = product_df

    def execute(self, tool_name: str, tool_args: dict) -> Any:
        logger.info(f"🔧 Calling: {tool_name}({json.dumps(tool_args, ensure_ascii=False)})")
        try:
            if tool_name == "search_products":
                return self._search_products(**tool_args)
            elif tool_name == "web_search_products":
                return self._web_search_products(**tool_args)
            elif tool_name == "filter_products":
                return self._filter_products(**tool_args)
            elif tool_name == "get_product_details":
                return self._get_product_details(**tool_args)
            elif tool_name == "compare_products":
                return self._compare_products(**tool_args)
            elif tool_name == "recommend_similar":
                return self._recommend_similar(**tool_args)
            else:
                return {"error": f"Unknown tool: {tool_name}"}
        except Exception as e:
            logger.error(f"Tool error: {e}")
            return {"error": str(e)}

    def _search_products(
        self, query: str, top_k: int = 5, method: str = "hybrid",
        min_price=None, max_price=None, category=None, min_rating=None
    ):
        results = self.search_engine.search(
            query=query, top_k=top_k, method=method,
            min_price=min_price, max_price=max_price,
            category=category, min_rating=min_rating,
        )
        return {"found": len(results), "products": results}

    def _web_search_products(self, query: str, top_k: int = 5):
        top_k = min(top_k or 5, 8)
        try:
            from ddgs import DDGS
        except ImportError:
            return {"error": "Web search library not installed (ddgs).", "found": 0, "web_results": []}

        results = []
        try:
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=top_k, safesearch="moderate"):
                    results.append({
                        "title": r.get("title", ""),
                        "url": r.get("href", ""),
                        "snippet": (r.get("body", "") or "")[:300],
                    })
        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return {"error": f"Web search failed: {e}", "found": 0, "web_results": []}

        return {"found": len(results), "web_results": results}

    def _filter_products(
        self, category=None, min_price=None, max_price=None,
        min_rating=None, sort_by="rating", top_k=5
    ):
        df = self.df.copy()
        if category:
            df = df[df["category"].str.lower().str.contains(category.lower(), na=False)]
        if min_price is not None:
            df = df[df["price"] >= min_price]
        if max_price is not None:
            df = df[df["price"] <= max_price]
        if min_rating is not None:
            df = df[df["rating"] >= min_rating]

        if sort_by == "rating":
            df = df.sort_values("rating", ascending=False)
        elif sort_by == "price_asc":
            df = df.sort_values("price", ascending=True)
        elif sort_by == "price_desc":
            df = df.sort_values("price", ascending=False)

        results = df.head(top_k)[
            ["product_id", "name", "description", "price", "category", "rating"]
        ].to_dict("records")
        return {"found": len(results), "products": results}

    def _get_product_details(self, product_id: int):
        row = self.df[self.df["product_id"] == product_id]
        if row.empty:
            return {"error": f"Product {product_id} not found"}
        r = row.iloc[0]
        return {
            "product_id": int(r["product_id"]), "name": r["name"],
            "description": r["description"], "price": float(r["price"]),
            "category": r["category"], "rating": float(r["rating"]),
        }

    def _compare_products(self, product_ids: list):
        products = []
        for pid in product_ids[:3]:
            row = self.df[self.df["product_id"] == pid]
            if not row.empty:
                r = row.iloc[0]
                products.append({
                    "product_id": int(r["product_id"]), "name": r["name"],
                    "price": float(r["price"]), "rating": float(r["rating"]),
                    "category": r["category"], "description": r["description"][:100],
                })
        if not products:
            return {"error": "No products found"}
        cheapest = min(products, key=lambda x: x["price"])
        best_rated = max(products, key=lambda x: x["rating"])
        return {"products": products, "cheapest": cheapest["product_id"], "highest_rated": best_rated["product_id"]}

    def _recommend_similar(self, product_id: int, top_k: int = 5):
        results = self.rec_engine.recommend(product_id=product_id, top_k=top_k)
        return {"found": len(results), "products": results}