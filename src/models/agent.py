"""
Shopping Agent — Agentic AI with Autonomous Tool Use
The agent thinks, decides which tools to call, executes them, and generates a final response.
Uses Groq function calling (tool use) API.
"""

import os
import json
import logging
from groq import Groq
from src.models.memory import ConversationMemory, memory_store
from src.models.tools import TOOLS, ToolExecutor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

GROQ_MODEL = "llama-3.3-70b-versatile"

AGENT_SYSTEM_PROMPT = """You are an intelligent AI shopping agent. You have access to a product catalog and several tools.

Your job:
1. Understand what the user wants
2. Decide which tools to call (you can call multiple tools)
3. Use the tool results to give a helpful, natural response

Tool usage rules:
- Always search before recommending
- If user mentions a price limit → use filter_products with max_price
- If user wants comparison → search first, then compare top results
- If user asks about a specific product ID → use get_product_details
- You can call tools sequentially (search → compare → recommend)
- Never make up products — only use results from tools

Response rules:
- Be conversational and helpful
- Mention specific product names, prices, and ratings from tool results
- If user writes in Arabic → respond in Arabic
- Keep responses focused and clear
- If tools return no results → say so honestly

{memory_summary}
"""


class ShoppingAgent:
    """
    Agentic AI that autonomously decides what tools to call.
    Supports multi-step reasoning: search → filter → compare → recommend
    """

    MAX_TOOL_ROUNDS = 5  # prevent infinite loops

    def __init__(self, tool_executor: ToolExecutor):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not set.")
        self.client = Groq(api_key=api_key)
        self.executor = tool_executor

    def run(self, user_message: str, session_id: str = "default") -> dict:
        """
        Main agent loop:
        1. Send message + tools to LLM
        2. If LLM calls a tool → execute it, feed result back
        3. Repeat until LLM gives final text response
        Returns: {reply, tool_calls_made, products_found}
        """
        memory = memory_store.get_or_create(session_id)
        memory.add_user_message(user_message)

        # Build messages
        system = AGENT_SYSTEM_PROMPT.format(
            memory_summary=memory.get_memory_summary()
        )
        messages = [
            {"role": "system", "content": system},
            *memory.get_context_messages(),
        ]

        tool_calls_log = []
        all_products = []

        # ── Agentic loop ──────────────────────────────────────────────────────
        for round_num in range(self.MAX_TOOL_ROUNDS):
            logger.info(f"Agent round {round_num + 1}")

            response = self.client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                max_tokens=1024,
                temperature=0.3,
            )

            message = response.choices[0].message
            finish_reason = response.choices[0].finish_reason

            # ── No more tool calls → final answer ─────────────────────────────
            if finish_reason == "stop" or not message.tool_calls:
                final_reply = message.content or "I couldn't find what you're looking for."
                logger.info(f"Agent finished after {round_num + 1} rounds | tools used: {len(tool_calls_log)}")

                # Save to memory
                product_ids = [p.get("product_id") for p in all_products if "product_id" in p]
                memory.add_assistant_message(final_reply, products_shown=product_ids)

                return {
                    "reply": final_reply,
                    "tool_calls_made": tool_calls_log,
                    "products_found": all_products[:10],
                    "rounds": round_num + 1,
                    "session_id": session_id,
                }

            # ── Execute tool calls ─────────────────────────────────────────────
            # Add assistant message with tool calls
            messages.append({
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        }
                    }
                    for tc in message.tool_calls
                ]
            })

            # Execute each tool and add result
            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                try:
                    tool_args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    tool_args = {}

                # Execute
                result = self.executor.execute(tool_name, tool_args)

                # Collect products
                if "products" in result:
                    all_products.extend(result["products"])

                # Log
                tool_calls_log.append({
                    "tool": tool_name,
                    "args": tool_args,
                    "result_count": result.get("found", 1),
                })

                # Add tool result to messages
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result, ensure_ascii=False, default=str),
                })

                logger.info(f"  ✓ {tool_name} → {result.get('found', '?')} results")

        # If we hit max rounds
        logger.warning("Agent hit MAX_TOOL_ROUNDS")
        return {
            "reply": "I ran into an issue processing your request. Please try again.",
            "tool_calls_made": tool_calls_log,
            "products_found": all_products[:10],
            "rounds": self.MAX_TOOL_ROUNDS,
            "session_id": session_id,
        }
