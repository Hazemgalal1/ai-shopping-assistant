"""
CatalogAgent — specialized sub-agent that ONLY searches the internal product catalog.
It has no knowledge of, or access to, the web.
"""

import os
import json
import logging
from groq import Groq
from src.models.tools import CATALOG_TOOLS, ToolExecutor

logger = logging.getLogger(__name__)
GROQ_MODEL = "llama-3.3-70b-versatile"

CATALOG_SYSTEM_PROMPT = """You are a specialized catalog-search sub-agent for a shopping assistant.
You ONLY have access to our INTERNAL product catalog.

Available categories (use these EXACT values, nothing else):
{available_categories}

Rules:
- Decide which catalog tools to call to satisfy the request (search_products, filter_products,
  compare_products, recommend_similar, get_product_details).
- Pass any price/category/rating constraints as parameters.
- ONLY set the category filter if it clearly matches one of the exact categories above.
- Call tools until you have enough results, then stop.
- If you find 0 relevant results, don't invent products — just stop.
"""


class CatalogAgent:
    MAX_ROUNDS = 3

    def __init__(self, tool_executor: ToolExecutor):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not set.")
        self.client = Groq(api_key=api_key)
        self.executor = tool_executor
        try:
            self.available_categories = ", ".join(
                sorted(self.executor.df["category"].dropna().unique().tolist())
            )
        except Exception:
            self.available_categories = "Electronics, Clothing, Books, Sports, Home"

    def run(self, user_message: str) -> dict:
        system = CATALOG_SYSTEM_PROMPT.format(available_categories=self.available_categories)
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_message},
        ]
        tool_calls_log = []
        products = []

        for _ in range(self.MAX_ROUNDS):
            response = self.client.chat.completions.create(
                model=GROQ_MODEL, messages=messages, tools=CATALOG_TOOLS,
                tool_choice="auto", max_tokens=512, temperature=0.2,
            )
            message = response.choices[0].message
            if response.choices[0].finish_reason == "stop" or not message.tool_calls:
                break

            messages.append({
                "role": "assistant", "content": message.content or "",
                "tool_calls": [
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                    for tc in message.tool_calls
                ],
            })
            for tc in message.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}
                result = self.executor.execute(tc.function.name, args)
                if "products" in result:
                    products.extend(result["products"])
                tool_calls_log.append({"tool": tc.function.name, "args": args, "result_count": result.get("found", 1)})
                messages.append({
                    "role": "tool", "tool_call_id": tc.id,
                    "content": json.dumps(result, ensure_ascii=False, default=str),
                })
                logger.info(f"  ✓ [Catalog] {tc.function.name} → {result.get('found', '?')} results")

        return {"products": products, "tool_calls": tool_calls_log}