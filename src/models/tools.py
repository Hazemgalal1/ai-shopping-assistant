"""
Agent Tools — Functions the AI Agent can call autonomously
Each tool has a clear description so the LLM knows when to use it
"""

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

# ─── Tool Definitions (sent to Groq as tools) ─────────────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_products",
            "description": (
                "Search for products using natural language query. "
                "Use this when the user asks to find, look for, or search products. "
                "Returns a list of relevant products."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query, e.g. 'wireless headphones' or 'سماعات لاسلكية'"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to return (default: 5)",
                        "default": 5
                    },
                    "method": {
                        "type": "string",
                        "enum": ["hybrid", "semantic", "tfidf"],
                        "description": "Search method. Use 'hybrid' by default.",
                        "default": "hybrid"
                    }
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
                "Filter and browse products by category, price range, or rating. "
                "Use this when the user specifies constraints like 'under $100', "
                "'in Electronics category', or 'rating above 4'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "Product category, e.g. 'Electronics', 'Clothing', 'Books'"
                    },
                    "min_price": {
                        "type": "number",
                        "description": "Minimum price in USD"
                    },
                    "max_price": {
                        "type": "number",
                        "description": "Maximum price in USD"
                    },
                    "min_rating": {
                        "type": "number",
                        "description": "Minimum rating (0-5)"
                    },
                    "sort_by": {
                        "type": "string",
                        "enum": ["rating", "price_asc", "price_desc"],
                        "default": "rating"
                    },
                    "top_k": {
                        "type": "integer",
                        "default": 5
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_product_details",
            "description": (
                "Get full details of a specific product by its ID. "
                "Use this when the user asks about a specific product or wants more info."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {
                        "type": "integer",
                        "description": "The product ID to look up"
                    }
                },
                "required": ["product_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "compare_products",
            "description": (
                "Compare multiple products side by side. "
                "Use this when the user wants to compare 2-3 products, "
                "or after searching to compare the top results."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "product_ids": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "List of 2-3 product IDs to compare",
                        "minItems": 2,
                        "maxItems": 3
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
            "description": (
                "Get product recommendations similar to a given product. "
                "Use this when the user likes a product and wants similar options, "
                "or wants alternatives."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {
                        "type": "integer",
                        "description": "The reference product ID"
                    },
                    "top_k": {
                        "type": "integer",
                        "default": 5
                    }
                },
                "required": ["product_id"]
            }
        }
    },
]


# ─── Tool Executor ─────────────────────────────────────────────────────────────

class ToolExecutor:
    """
    Executes tool calls made by the agent.
    Connected to the actual search/recommend engines.
    """

    def __init__(self, search_engine, rec_engine, product_df):
        self.search_engine = search_engine
        self.rec_engine = rec_engine
        self.df = product_df

    def execute(self, tool_name: str, tool_args: dict) -> Any:
        """Execute a tool call and return the result."""
        logger.info(f"🔧 Agent calling: {tool_name}({json.dumps(tool_args, ensure_ascii=False)})")

        try:
            if tool_name == "search_products":
                return self._search_products(**tool_args)

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

    def _search_products(self, query: str, top_k: int = 5, method: str = "hybrid"):
        results = self.search_engine.search(query=query, top_k=top_k, method=method)
        return {
            "found": len(results),
            "products": results,
        }

    def _filter_products(
        self, category=None, min_price=None, max_price=None,
        min_rating=None, sort_by="rating", top_k=5
    ):
        df = self.df.copy()

        if category:
            df = df[df["category"].str.lower() == category.lower()]
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
            "product_id": int(r["product_id"]),
            "name": r["name"],
            "description": r["description"],
            "price": float(r["price"]),
            "category": r["category"],
            "rating": float(r["rating"]),
        }

    def _compare_products(self, product_ids: list):
        products = []
        for pid in product_ids[:3]:
            row = self.df[self.df["product_id"] == pid]
            if not row.empty:
                r = row.iloc[0]
                products.append({
                    "product_id": int(r["product_id"]),
                    "name": r["name"],
                    "price": float(r["price"]),
                    "rating": float(r["rating"]),
                    "category": r["category"],
                    "description": r["description"][:100],
                })

        if not products:
            return {"error": "No products found"}

        cheapest = min(products, key=lambda x: x["price"])
        best_rated = max(products, key=lambda x: x["rating"])

        return {
            "products": products,
            "cheapest": cheapest["product_id"],
            "highest_rated": best_rated["product_id"],
        }

    def _recommend_similar(self, product_id: int, top_k: int = 5):
        results = self.rec_engine.recommend(product_id=product_id, top_k=top_k)
        return {"found": len(results), "products": results}
