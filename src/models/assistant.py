"""
Groq Conversational AI Engine — v2
With Multi-turn Memory + Arabic Support + Smart Context
"""

import os
import logging
from groq import Groq
from src.models.memory import ConversationMemory, memory_store

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

GROQ_MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT_EN = """You are a helpful AI shopping assistant. Help users find products,
compare options, and make smart purchasing decisions.

Guidelines:
- Be conversational and friendly
- Reference specific products from the context provided
- Mention price and rating when relevant
- Keep responses concise (3-5 sentences) unless a detailed comparison is needed
- Never make up products not in the context
- If no products found, say so honestly
- Remember what the user said earlier in the conversation
{memory_summary}
"""

SYSTEM_PROMPT_AR = """أنت مساعد تسوق ذكي. مهمتك مساعدة المستخدم في إيجاد المنتجات المناسبة ومقارنتها.

تعليمات مهمة:
- تكلم دايماً بالعربي
- كن ودود وطبيعي في كلامك
- اذكر المنتجات الموجودة في الـ context بالتحديد
- اذكر السعر والتقييم دايماً
- ردودك تكون مختصرة ومفيدة (3-5 جمل)
- لو مفيش منتجات مناسبة، قول ده بصراحة
- افتكر اللي المستخدم قاله قبل كده في المحادثة
{memory_summary}
"""


class ConversationalAssistant:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not set in environment variables.")
        self.client = Groq(api_key=api_key)

    def _get_system_prompt(self, memory: ConversationMemory) -> str:
        summary = memory.get_memory_summary()
        if memory.preferences.language == "ar":
            return SYSTEM_PROMPT_AR.format(memory_summary=summary)
        return SYSTEM_PROMPT_EN.format(memory_summary=summary)

    def _format_products_context(self, products: list[dict]) -> str:
        if not products:
            return "No relevant products found in the catalog."
        lines = ["Relevant products from our catalog:\n"]
        for i, p in enumerate(products[:5], 1):
            lines.append(
                f"{i}. {p['name'].title()}\n"
                f"   Price: ${p['price']:.2f} | Rating: {p['rating']:.1f}/5 | Category: {p['category']}\n"
                f"   {p['description'][:100]}...\n"
            )
        return "\n".join(lines)

    def chat(
        self,
        user_message: str,
        session_id: str = "default",
        retrieved_products: list[dict] = None,
    ) -> str:
        # Get or create memory for this session
        memory = memory_store.get_or_create(session_id)

        # Build augmented message with product context
        if retrieved_products:
            context = self._format_products_context(retrieved_products)
            augmented = f"{user_message}\n\n[PRODUCT CONTEXT]\n{context}"
        else:
            augmented = user_message

        # Add user message to memory
        product_ids = [p["product_id"] for p in (retrieved_products or [])]
        memory.add_user_message(augmented, products_shown=product_ids)

        # Build messages: system + conversation history
        messages = [
            {"role": "system", "content": self._get_system_prompt(memory)},
            *memory.get_context_messages(),
        ]

        # Call Groq
        response = self.client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            max_tokens=512,
            temperature=0.7,
        )
        reply = response.choices[0].message.content

        # Save assistant reply to memory
        memory.add_assistant_message(reply, products_shown=product_ids)

        logger.info(f"[{session_id}] lang={memory.preferences.language} | reply={len(reply)} chars")
        return reply

    def generate_comparison(self, products: list[dict], language: str = "en") -> str:
        context = self._format_products_context(products)
        if language == "ar":
            prompt = f"قارن المنتجات دي وقول أيهم الأنسب ولماذا:\n\n{context}"
        else:
            prompt = f"Compare these products and recommend the best one:\n\n{context}"

        response = self.client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_EN.format(memory_summary="")},
                {"role": "user", "content": prompt},
            ],
            max_tokens=600,
            temperature=0.5,
        )
        return response.choices[0].message.content

    def generate_product_description(self, product: dict, language: str = "en") -> str:
        if language == "ar":
            prompt = (
                f"اكتب وصف تسويقي جذاب بالعربي للمنتج ده:\n"
                f"الاسم: {product['name']}\n"
                f"الفئة: {product['category']}\n"
                f"السعر: ${product['price']:.2f}\n"
                f"التقييم: {product['rating']}/5\n"
                f"الوصف الأصلي: {product['description']}\n"
                f"اكتب وصف في 3-4 جمل بيوضح المميزات."
            )
        else:
            prompt = (
                f"Write an engaging product description for:\n"
                f"Name: {product['name']} | Category: {product['category']}\n"
                f"Price: ${product['price']:.2f} | Rating: {product['rating']}/5\n"
                f"Original: {product['description']}\n"
                f"Keep it under 80 words, highlight key benefits."
            )

        response = self.client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_EN.format(memory_summary="")},
                {"role": "user", "content": prompt},
            ],
            max_tokens=200,
            temperature=0.8,
        )
        return response.choices[0].message.content
