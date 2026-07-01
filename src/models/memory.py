"""
Multi-turn Memory Manager
Handles conversation history, user preferences, and context across turns
"""

import json
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class UserPreferences:
    """Learned preferences from conversation history."""
    preferred_categories: list[str] = field(default_factory=list)
    max_price: Optional[float] = None
    min_rating: Optional[float] = None
    language: str = "en"  # 'en' or 'ar'
    viewed_products: list[int] = field(default_factory=list)
    liked_products: list[int] = field(default_factory=list)


@dataclass
class MemoryEntry:
    role: str        # 'user' or 'assistant'
    content: str
    timestamp: float
    products_shown: list[int] = field(default_factory=list)  # product_ids shown in this turn


class ConversationMemory:
    """
    Manages multi-turn memory for a single user session.
    - Stores full conversation history
    - Learns user preferences automatically
    - Provides smart context window (last N turns)
    - Detects language (Arabic / English)
    """

    MAX_HISTORY = 20      # max turns to keep
    CONTEXT_WINDOW = 6    # turns sent to LLM each time

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.history: list[MemoryEntry] = []
        self.preferences = UserPreferences()
        self.created_at = time.time()

    # ─── Add messages ────────────────────────────────────────────────────────

    def add_user_message(self, content: str, products_shown: list[int] = None):
        self._detect_language(content)
        self._extract_preferences_from_query(content)
        self.history.append(MemoryEntry(
            role="user",
            content=content,
            timestamp=time.time(),
            products_shown=products_shown or [],
        ))
        self._trim()

    def add_assistant_message(self, content: str, products_shown: list[int] = None):
        self.history.append(MemoryEntry(
            role="assistant",
            content=content,
            timestamp=time.time(),
            products_shown=products_shown or [],
        ))
        if products_shown:
            self.preferences.viewed_products.extend(products_shown)
            # Keep only last 20 viewed
            self.preferences.viewed_products = self.preferences.viewed_products[-20:]
        self._trim()

    # ─── Get context for LLM ─────────────────────────────────────────────────

    def get_context_messages(self) -> list[dict]:
        """Return last N turns formatted for Groq API."""
        recent = self.history[-self.CONTEXT_WINDOW:]
        return [{"role": e.role, "content": e.content} for e in recent]

    def get_memory_summary(self) -> str:
        """Build a summary of what we know about the user — injected into system prompt."""
        parts = []

        if self.preferences.language == "ar":
            parts.append("User prefers Arabic language responses.")

        if self.preferences.preferred_categories:
            parts.append(f"User is interested in: {', '.join(self.preferences.preferred_categories)}.")

        if self.preferences.max_price:
            parts.append(f"User's budget is around ${self.preferences.max_price:.0f}.")

        if self.preferences.min_rating:
            parts.append(f"User prefers products rated above {self.preferences.min_rating}.")

        if self.preferences.viewed_products:
            parts.append(f"User has seen products: {self.preferences.viewed_products[-5:]}.")

        if not parts:
            return ""

        return "USER MEMORY:\n" + "\n".join(f"- {p}" for p in parts)

    # ─── Language detection ──────────────────────────────────────────────────

    def _detect_language(self, text: str):
        arabic_chars = sum(1 for c in text if "\u0600" <= c <= "\u06ff")
        if arabic_chars > len(text) * 0.2:
            self.preferences.language = "ar"
        else:
            self.preferences.language = "en"

    # ─── Auto preference extraction ──────────────────────────────────────────

    def _extract_preferences_from_query(self, text: str):
        text_lower = text.lower()

        # Price extraction
        import re
        price_match = re.search(r"under\s*\$?(\d+)|أقل من\s*(\d+)|budget.*?(\d+)", text_lower)
        if price_match:
            price = next(g for g in price_match.groups() if g)
            self.preferences.max_price = float(price)

        # Rating preference
        if "good rating" in text_lower or "highly rated" in text_lower or "تقييم عالي" in text:
            self.preferences.min_rating = 4.0

        # Category hints
        category_keywords = {
            "Electronics": ["laptop", "phone", "tablet", "headphone", "camera", "لاب توب", "موبايل"],
            "Clothing": ["shirt", "shoes", "dress", "جاكيت", "هدوم"],
            "Books": ["book", "novel", "كتاب"],
            "Sports": ["gym", "running", "sport", "رياضة"],
            "Home & Kitchen": ["kitchen", "home", "مطبخ", "بيت"],
        }
        for category, keywords in category_keywords.items():
            if any(kw in text_lower for kw in keywords):
                if category not in self.preferences.preferred_categories:
                    self.preferences.preferred_categories.append(category)

    def _trim(self):
        if len(self.history) > self.MAX_HISTORY:
            self.history = self.history[-self.MAX_HISTORY:]

    # ─── Reset ───────────────────────────────────────────────────────────────

    def reset(self):
        self.history = []
        self.preferences = UserPreferences()

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "turns": len(self.history),
            "preferences": asdict(self.preferences),
        }


class MemoryStore:
    """Global store for all active sessions."""

    def __init__(self, session_ttl_seconds: int = 3600):
        self._store: dict[str, ConversationMemory] = {}
        self._ttl = session_ttl_seconds

    def get_or_create(self, session_id: str) -> ConversationMemory:
        self._cleanup_expired()
        if session_id not in self._store:
            self._store[session_id] = ConversationMemory(session_id)
        return self._store[session_id]

    def reset(self, session_id: str):
        if session_id in self._store:
            self._store[session_id].reset()

    def delete(self, session_id: str):
        self._store.pop(session_id, None)

    def _cleanup_expired(self):
        now = time.time()
        expired = [
            sid for sid, mem in self._store.items()
            if now - mem.created_at > self._ttl
        ]
        for sid in expired:
            del self._store[sid]

    def stats(self) -> dict:
        return {
            "active_sessions": len(self._store),
            "sessions": [m.to_dict() for m in self._store.values()],
        }


# Global singleton
memory_store = MemoryStore()
