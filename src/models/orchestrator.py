"""
Orchestrator — routes each user message to the CatalogAgent and/or WebSearchAgent,
then composes a single final natural-language reply from their combined results.
Top-level entry point used by app.py.
"""

import os
import json
import logging
from groq import Groq
from src.models.memory import memory_store
from src.models.catalog_agent import CatalogAgent
from src.models.web_agent import WebSearchAgent
from src.models.tools import ToolExecutor

logger = logging.getLogger(__name__)
GROQ_MODEL = "llama-3.3-70b-versatile"

ROUTE_TOOL = [{
    "type": "function",
    "function": {
        "name": "route_request",
        "description": "Decide which data source(s) are needed to answer the user's shopping request.",
        "parameters": {
            "type": "object",
            "properties": {
                "use_catalog": {
                    "type": "boolean",
                    "description": "True if our internal product catalog could contain relevant products."
                },
                "use_web": {
                    "type": "boolean",
                    "description": (
                        "True if a live web search is needed — specific real brand/model, "
                        "current market prices, 'best X' comparisons, or a category unlikely "
                        "to be in a generic sample catalog."
                    )
                },
            },
            "required": ["use_catalog", "use_web"],
        },
    },
}]

COMPOSER_SYSTEM_PROMPT = """You are the final response-writer for an AI shopping assistant.
You are given the user's message plus raw results from two specialist sub-agents:
- catalog_products: items from our internal product catalog (if any)
- web_results: live web search results with title/url/snippet (if any)

Write ONE helpful, natural, conversational reply based ONLY on this data.
Rules:
- If the user wrote in Arabic → reply in Arabic. Otherwise reply in English.
- Mention concrete names/prices/ratings for catalog_products.
- Mention titles and brief content for web_results, noting they're from the web.
- Never invent products, prices, or specs not present in the given data.
- If both lists are empty, say so honestly and ask a clarifying question.
- Keep it focused — a short paragraph, not a wall of text.

{memory_summary}
"""


class ShoppingOrchestrator:
    """Multi-agent orchestrator: Router → CatalogAgent / WebSearchAgent → Composer."""

    def __init__(self, tool_executor: ToolExecutor):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not set.")
        self.client = Groq(api_key=api_key)
        self.catalog_agent = CatalogAgent(tool_executor)
        self.web_agent = WebSearchAgent(tool_executor)

    def _route(self, user_message: str) -> dict:
        try:
            response = self.client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": "You are a routing classifier for a shopping assistant. Always call route_request."},
                    {"role": "user", "content": user_message},
                ],
                tools=ROUTE_TOOL,
                tool_choice={"type": "function", "function": {"name": "route_request"}},
                max_tokens=200,
                temperature=0,
            )
            tc = response.choices[0].message.tool_calls[0]
            args = json.loads(tc.function.arguments)
            use_catalog = bool(args.get("use_catalog", True))
            use_web = bool(args.get("use_web", False))
            if not use_catalog and not use_web:
                use_catalog = True
            return {"use_catalog": use_catalog, "use_web": use_web}
        except Exception as e:
            logger.warning(f"Routing failed, defaulting to catalog+web: {e}")
            return {"use_catalog": True, "use_web": True}

    def run(self, user_message: str, session_id: str = "default") -> dict:
        memory = memory_store.get_or_create(session_id)
        memory.add_user_message(user_message)

        route = self._route(user_message)
        logger.info(f"Route decision: {route}")

        tool_calls_log = []
        products = []
        web_results = []

        if route["use_catalog"]:
            catalog_out = self.catalog_agent.run(user_message)
            products.extend(catalog_out["products"])
            tool_calls_log.extend([{**tc, "agent": "catalog"} for tc in catalog_out["tool_calls"]])

        # Fall back to web if catalog gave nothing, even if router didn't ask for web
        if route["use_web"] or (route["use_catalog"] and not products):
            web_out = self.web_agent.run(user_message)
            web_results.extend(web_out["web_results"])
            tool_calls_log.extend([{**tc, "agent": "web"} for tc in web_out["tool_calls"]])

        composer_input = {
            "user_message": user_message,
            "catalog_products": products[:8],
            "web_results": web_results[:8],
        }
        system = COMPOSER_SYSTEM_PROMPT.format(memory_summary=memory.get_memory_summary())
        try:
            comp_response = self.client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": json.dumps(composer_input, ensure_ascii=False, default=str)},
                ],
                max_tokens=700,
                temperature=0.4,
            )
            final_reply = comp_response.choices[0].message.content or "Sorry, I couldn't generate a response."
        except Exception as e:
            logger.error(f"Composer failed: {e}")
            final_reply = "Sorry, something went wrong while composing the response."

        product_ids = [p.get("product_id") for p in products if "product_id" in p]
        memory.add_assistant_message(final_reply, products_shown=product_ids)

        return {
            "reply": final_reply,
            "tool_calls_made": tool_calls_log,
            "products_found": products[:10],
            "web_results_found": web_results[:8],
            "route": route,
            "rounds": 1,
            "session_id": session_id,
        }