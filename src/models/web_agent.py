"""
WebSearchAgent — specialized sub-agent that ONLY does live web search.
It has no knowledge of, or access to, the internal catalog.
"""

import os
import json
import logging
from groq import Groq
from src.models.tools import WEB_TOOLS, ToolExecutor

logger = logging.getLogger(__name__)
GROQ_MODEL = "llama-3.3-70b-versatile"

WEB_SYSTEM_PROMPT = """You are a specialized live-web-search sub-agent for a shopping assistant.
You ONLY have access to real-time web search.

Rules:
- Turn the user's request into one or more effective web search queries (bake in constraints
  like price, brand, "2026", "best", "review" directly into the query text).
- Call web_search_products (you may call it more than once with different queries, e.g. once
  in English and once in Arabic if the user wrote Arabic).
- Do not invent prices or specs that are not present in the search snippets.
"""


class WebSearchAgent:
    MAX_ROUNDS = 3

    def __init__(self, tool_executor: ToolExecutor):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not set.")
        self.client = Groq(api_key=api_key)
        self.executor = tool_executor

    def run(self, user_message: str) -> dict:
        messages = [
            {"role": "system", "content": WEB_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]
        tool_calls_log = []
        web_results = []

        for _ in range(self.MAX_ROUNDS):
            response = self.client.chat.completions.create(
                model=GROQ_MODEL, messages=messages, tools=WEB_TOOLS,
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
                if "web_results" in result:
                    web_results.extend(result["web_results"])
                tool_calls_log.append({"tool": tc.function.name, "args": args, "result_count": result.get("found", 1)})
                messages.append({
                    "role": "tool", "tool_call_id": tc.id,
                    "content": json.dumps(result, ensure_ascii=False, default=str),
                })
                logger.info(f"  ✓ [Web] {tc.function.name} → {result.get('found', '?')} results")

        return {"web_results": web_results, "tool_calls": tool_calls_log}