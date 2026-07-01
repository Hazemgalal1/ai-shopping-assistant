"""
AI Shopping Assistant — Streamlit Cloud Deployment
كل الـ logic جوا Streamlit مباشرة — بدون FastAPI
"""

import os
import uuid
import time
import logging
import numpy as np
import pandas as pd
import streamlit as st
import requests

# ─── Page Config (لازم تكون أول حاجة) ───────────────────────────────────────
st.set_page_config(
    page_title="AI Shopping Assistant",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Load GROQ_API_KEY من Streamlit Secrets ───────────────────────────────────
if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header { font-size:2rem; font-weight:700; color:#1a1a2e; }
    .sub-header { font-size:0.95rem; color:#888; margin-bottom:1.5rem; }
    .product-card {
        background:#f8f9fa; border-radius:12px; padding:1rem;
        margin-bottom:0.8rem; border-left:4px solid #4F8EF7;
    }
    .product-name { font-weight:600; font-size:1rem; color:#1a1a2e; }
    .badge {
        display:inline-block; padding:2px 8px; border-radius:12px;
        font-size:0.75rem; font-weight:600; margin-right:4px;
    }
    .badge-category { background:#e8f0fe; color:#4F8EF7; }
    .badge-price    { background:#d4edda; color:#155724; }
    .badge-rating   { background:#fff3cd; color:#856404; }
    .section-title {
        font-size:1.1rem; font-weight:600; color:#1a1a2e;
        border-bottom:2px solid #e9ecef; padding-bottom:0.3rem;
        margin-bottom:0.8rem;
    }
    .memory-pill {
        background:#f3e8ff; color:#6f42c1; border-radius:8px;
        padding:4px 10px; font-size:0.78rem; margin:2px;
        display:inline-block;
    }
    .tool-step {
        background:#e8f5e9; border-left:3px solid #28a745;
        padding:6px 12px; border-radius:4px; margin:4px 0;
        font-size:0.85rem; font-family:monospace;
    }
    .arabic-text { direction:rtl; text-align:right; }
    .stAlert { border-radius:10px; }
</style>
""", unsafe_allow_html=True)

# ─── Load Models (cached — مرة واحدة بس) ─────────────────────────────────────

@st.cache_resource(show_spinner="⏳ جاري تحميل البيانات والنماذج...")
def load_all():
    """تحميل كل حاجة مرة واحدة وتخزينها في الـ cache."""
    from src.data.collect import generate_sample_dataset
    from src.data.clean import (
        clean_price, clean_rating, clean_text,
        normalize_price, create_text_feature, remove_duplicates, handle_missing
    )
    from src.models.search import SearchEngine
    from src.models.recommend import RecommendationEngine
    from src.models.tools import ToolExecutor
    from src.models.agent import ShoppingAgent
    from src.models.assistant import ConversationalAssistant

    # ── Data ──────────────────────────────────────────────────────────────────
    # محاولة تحميل Kaggle data لو موجودة
    data_path = "data/processed/products_clean.csv"
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
    else:
        # توليد sample data وتنظيفها
        raw = generate_sample_dataset(300)

        raw["price"]       = clean_price(raw["price"].astype(str))
        raw["rating"]      = clean_rating(raw["rating"].astype(str))
        raw["name"]        = clean_text(raw["name"])
        raw["description"] = clean_text(raw["description"])

        raw = remove_duplicates(raw)
        raw = handle_missing(raw)
        raw = normalize_price(raw)
        raw = create_text_feature(raw)
        raw = raw.reset_index(drop=True)
        raw.insert(0, "product_id", raw.index)

        df = raw

    # ── Search Engine ─────────────────────────────────────────────────────────
    emb_path   = "models/embeddings/product_embeddings.npy"
    idx_path   = "models/embeddings/faiss_index.bin"
    tfidf_path = "models/tfidf/tfidf_vectorizer.pkl"

    search_engine = SearchEngine()

    if os.path.exists(emb_path) and os.path.exists(tfidf_path):
        search_engine.load(df)
    else:
        search_engine.build(df)

    embeddings = search_engine.embeddings

    # ── Recommendation Engine ─────────────────────────────────────────────────
    rec_engine = RecommendationEngine()
    rec_engine.load(df, embeddings)

    # ── Agent & Assistant ─────────────────────────────────────────────────────
    tool_executor = ToolExecutor(
        search_engine=search_engine,
        rec_engine=rec_engine,
        product_df=df,
    )

    agent     = None
    assistant = None

    if os.environ.get("GROQ_API_KEY"):
        try:
            agent     = ShoppingAgent(tool_executor=tool_executor)
            assistant = ConversationalAssistant()
        except Exception as e:
            st.warning(f"Groq not available: {e}")

    return df, search_engine, rec_engine, agent, assistant


# تحميل كل حاجة
df, search_engine, rec_engine, agent, assistant = load_all()

# ─── Memory store ─────────────────────────────────────────────────────────────
from src.models.memory import memory_store

# ─── Session State ────────────────────────────────────────────────────────────
if "agent_messages"   not in st.session_state: st.session_state.agent_messages   = []
if "chat_messages"    not in st.session_state: st.session_state.chat_messages     = []
if "agent_session_id" not in st.session_state: st.session_state.agent_session_id  = "agent_" + str(uuid.uuid4())[:8]
if "chat_session_id"  not in st.session_state: st.session_state.chat_session_id   = "chat_"  + str(uuid.uuid4())[:8]
if "language"         not in st.session_state: st.session_state.language           = "en"

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🛍️ AI Shopping Assistant")
    st.markdown(f"**📦 {len(df):,} products loaded**")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["🤖 AI Agent", "💬 AI Chat", "🔍 Search", "💡 Recommendations", "⚖️ Compare", "🔧 Browse"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("**🌐 Language**")
    lang = st.radio("", ["English 🇬🇧", "العربية 🇪🇬"], horizontal=True, label_visibility="collapsed")
    st.session_state.language = "ar" if "العربية" in lang else "en"

    st.markdown("---")
    st.markdown("**🔧 Filters**")
    cats = ["All"] + sorted(df["category"].unique().tolist())
    sidebar_category  = st.selectbox("Category", cats)
    sidebar_min_price = st.number_input("Min Price ($)", min_value=0.0,   value=0.0,   step=5.0)
    sidebar_max_price = st.number_input("Max Price ($)", min_value=0.0,   value=500.0, step=5.0)
    sidebar_min_rating= st.slider("Min Rating", 0.0, 5.0, 0.0, 0.5)

    # Memory summary
    mem = memory_store.get_or_create(st.session_state.agent_session_id)
    prefs = mem.preferences
    if prefs.preferred_categories or prefs.max_price:
        st.markdown("---")
        st.markdown("**🧠 Memory:**")
        for cat in prefs.preferred_categories:
            st.markdown(f'<span class="memory-pill">📦 {cat}</span>', unsafe_allow_html=True)
        if prefs.max_price:
            st.markdown(f'<span class="memory-pill">💰 ${prefs.max_price:.0f}</span>', unsafe_allow_html=True)

    st.markdown("---")
    st.caption("DEPI Generative AI Track 2026")

# ─── Helper ───────────────────────────────────────────────────────────────────
def render_product_card(p: dict, show_score: bool = True):
    score = p.get("score") or p.get("similarity_score")
    score_html = f'<span class="badge" style="background:#f3e8ff;color:#6f42c1;">Score: {score:.3f}</span>' if score and show_score else ""
    card_html = (
        f'<div class="product-card">'
        f'<div class="product-name">{str(p.get("name","")).title()}</div>'
        f'<div style="margin-top:6px;">'
        f'<span class="badge badge-category">{p.get("category","")}</span>'
        f'<span class="badge badge-price">${p.get("price",0):.2f}</span>'
        f'<span class="badge badge-rating">⭐ {p.get("rating",0):.1f}</span>'
        f'{score_html}'
        f'</div>'
        f'<div style="margin-top:8px;color:#777;font-size:0.82rem;">'
        f'{str(p.get("description",""))[:150]}...'
        f'</div>'
        f'</div>'
    )
    st.markdown(card_html, unsafe_allow_html=True)

is_arabic = st.session_state.language == "ar"

# ═══════════════════════════════════════════════════════════════════════════════
# PAGES
# ═══════════════════════════════════════════════════════════════════════════════

# ── 🤖 AI Agent ───────────────────────────────────────────────────────────────
if "🤖 AI Agent" in page:
    st.markdown('<div class="main-header">🤖 AI Shopping Agent</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header arabic-text">الـ Agent بيفكر لوحده — بيقرر أيه الـ tools يشغّلها تلقائياً</div>'
        if is_arabic else
        '<div class="sub-header">The Agent thinks autonomously — decides which tools to call and in what order</div>',
        unsafe_allow_html=True,
    )

    if agent is None:
        st.error("⚠️ GROQ_API_KEY مش موجود — حط الـ key في Streamlit Secrets أو .streamlit/secrets.toml" if is_arabic
                 else "⚠️ GROQ_API_KEY not found. Add it to Streamlit Secrets.")
        st.code('GROQ_API_KEY = "gsk_xxxxxxxxxxxxxxxxxxxx"', language="toml")
        st.stop()

    with st.expander("⚙️ إزاي بيشتغل الـ Agent؟" if is_arabic else "⚙️ How does the Agent work?"):
        st.markdown("""
        ```
        1. بتكتب طلبك بشكل طبيعي
        2. الـ Agent يحلل ويقرر أيه الـ tools
        3. بيشغّلهم تلقائياً بالترتيب الصح
        4. بيولّد رد شامل من النتايج
        ```
        **Tools:** 🔍 search | 🔧 filter | ⚖️ compare | 💡 recommend | 📋 details
        """)

    col1, col2 = st.columns([5, 1])
    with col2:
        if st.button("🗑️ Clear"):
            st.session_state.agent_messages = []
            memory_store.reset(st.session_state.agent_session_id)
            st.rerun()

    for msg in st.session_state.agent_messages:
        with st.chat_message(msg["role"]):
            if is_arabic and msg["role"] == "assistant":
                st.markdown(f'<div class="arabic-text">{msg["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(msg["content"])

            if msg.get("tool_calls"):
                with st.expander(f"🧠 {len(msg['tool_calls'])} tool calls | {msg.get('rounds',1)} rounds"):
                    for i, tc in enumerate(msg["tool_calls"], 1):
                        st.markdown(f'<div class="tool-step">Step {i}: {tc["tool"]}({tc["args"]}) → {tc["result_count"]} results</div>',
                                    unsafe_allow_html=True)

            if msg.get("products"):
                with st.expander(f"📦 {len(msg['products'])} products", expanded=True):
                    for p in msg["products"][:5]:
                        render_product_card(p)

    placeholder = "اطلب أي حاجة... مثلاً: عايز لاب توب للشغل تحت 500 دولار وتقييم عالي" if is_arabic \
                  else "Ask anything... e.g. Find me the best laptop under $500 with high ratings"

    user_input = st.chat_input(placeholder)

    if user_input:
        st.session_state.agent_messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            status = st.status("🧠 Agent is thinking..." if not is_arabic else "🧠 الـ Agent بيفكر...", expanded=True)
            try:
                result = agent.run(user_message=user_input, session_id=st.session_state.agent_session_id)

                for tc in result.get("tool_calls_made", []):
                    status.write(f"🔧 `{tc['tool']}` → {tc['result_count']} results")

                status.update(
                    label=f"✅ Done — {result.get('rounds',1)} rounds | {len(result.get('tool_calls_made',[]))} tools",
                    state="complete",
                )

                reply    = result["reply"]
                products = result.get("products_found", [])

                if is_arabic:
                    st.markdown(f'<div class="arabic-text">{reply}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(reply)

                if products:
                    with st.expander(f"📦 {len(products)} products", expanded=True):
                        for p in products[:5]:
                            render_product_card(p)

                st.session_state.agent_messages.append({
                    "role": "assistant",
                    "content": reply,
                    "tool_calls": result.get("tool_calls_made", []),
                    "products": products,
                    "rounds": result.get("rounds", 1),
                })

            except Exception as e:
                status.update(label="❌ Error", state="error")
                st.error(str(e))

    if not st.session_state.agent_messages:
        st.markdown("---")
        st.markdown("**جرب:**" if is_arabic else "**Try:**")
        examples = [
            "عايز لاب توب تحت 500 دولار وتقييمه فوق 4",
            "قارنلي أحسن 3 منتجات في Electronics",
            "Find me the cheapest highly-rated product in Sports",
            "I need a gift under $30, suggest something and explain why",
        ]
        cols = st.columns(2)
        for i, ex in enumerate(examples):
            with cols[i % 2]:
                if st.button(ex, use_container_width=True, key=f"ag_{i}"):
                    st.session_state.agent_messages.append({"role": "user", "content": ex})
                    st.rerun()


# ── 💬 AI Chat ────────────────────────────────────────────────────────────────
elif "💬 AI Chat" in page:
    if is_arabic:
        st.markdown('<div class="main-header arabic-text">💬 مساعد التسوق الذكي</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header arabic-text">اسألني بالعربي — هساعدك تلاقي أحسن منتج</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="main-header">💬 AI Shopping Chat</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">Chat naturally — RAG-powered responses</div>', unsafe_allow_html=True)

    if assistant is None:
        st.error("⚠️ GROQ_API_KEY مش موجود" if is_arabic else "⚠️ GROQ_API_KEY not found.")
        st.stop()

    col1, col2 = st.columns([5, 1])
    with col2:
        if st.button("🗑️ Clear"):
            st.session_state.chat_messages = []
            memory_store.reset(st.session_state.chat_session_id)
            st.rerun()

    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            if is_arabic and msg["role"] == "assistant":
                st.markdown(f'<div class="arabic-text">{msg["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(msg["content"])
            if msg.get("products"):
                with st.expander(f"📦 {len(msg['products'])} products"):
                    for p in msg["products"]:
                        render_product_card(p)

    placeholder = "اكتب سؤالك..." if is_arabic else "Ask me anything..."
    user_input  = st.chat_input(placeholder)

    if user_input:
        st.session_state.chat_messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("جاري التفكير..." if is_arabic else "Thinking..."):
                retrieved = search_engine.search(query=user_input, top_k=5, method="hybrid")
                reply = assistant.chat(
                    user_message=user_input,
                    session_id=st.session_state.chat_session_id,
                    retrieved_products=retrieved,
                )

            if is_arabic:
                st.markdown(f'<div class="arabic-text">{reply}</div>', unsafe_allow_html=True)
            else:
                st.markdown(reply)

            if retrieved:
                with st.expander(f"📦 {len(retrieved)} related products"):
                    for p in retrieved[:5]:
                        render_product_card(p)

            st.session_state.chat_messages.append({
                "role": "assistant",
                "content": reply,
                "products": retrieved[:5],
            })

    if not st.session_state.chat_messages:
        st.markdown("---")
        examples = [
            "عايز موبايل كويس برخص سعر",
            "إيه أحسن منتج في Electronics؟",
            "What's a good laptop for students?",
            "Recommend something under $50",
        ]
        cols = st.columns(2)
        for i, ex in enumerate(examples):
            with cols[i % 2]:
                if st.button(ex, use_container_width=True, key=f"ch_{i}"):
                    st.session_state.chat_messages.append({"role": "user", "content": ex})
                    st.rerun()


# ── 🔍 Search ─────────────────────────────────────────────────────────────────
elif "🔍 Search" in page:
    st.markdown('<div class="main-header">🔍 Search Products</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([4, 1])
    with col1:
        query = st.text_input("", placeholder="ابحث عن أي منتج / Search for anything...", label_visibility="collapsed")
    with col2:
        method = st.selectbox("", ["hybrid", "semantic", "tfidf"], label_visibility="collapsed")

    top_k = st.slider("Results", 5, 30, 10)

    if st.button("🔍 Search", type="primary", use_container_width=True) and query:
        with st.spinner("Searching..."):
            results = search_engine.search(
                query=query, top_k=top_k, method=method,
                min_price=sidebar_min_price if sidebar_min_price > 0 else None,
                max_price=sidebar_max_price if sidebar_max_price < 500 else None,
                category=sidebar_category if sidebar_category != "All" else None,
                min_rating=sidebar_min_rating if sidebar_min_rating > 0 else None,
            )
        st.markdown(f'<div class="section-title">Found {len(results)} results for "{query}"</div>',
                    unsafe_allow_html=True)
        for p in results:
            render_product_card(p)


# ── 💡 Recommendations ────────────────────────────────────────────────────────
elif "💡 Recommendations" in page:
    st.markdown('<div class="main-header">💡 Recommendations</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])
    with col1:
        product_id = st.number_input("Product ID", min_value=0, value=0, step=1)
    with col2:
        mode = st.selectbox("Mode", ["similar", "complementary", "category"])
    top_k = st.slider("Number", 3, 15, 5)

    if st.button("Get Recommendations", type="primary"):
        if mode == "similar":
            results = rec_engine.recommend(int(product_id), top_k)
        elif mode == "complementary":
            results = rec_engine.recommend_complementary(int(product_id), top_k)
        else:
            row = df[df["product_id"] == int(product_id)]
            cat = row.iloc[0]["category"] if not row.empty else "Electronics"
            results = rec_engine.recommend_by_category(cat, top_k)

        st.markdown(f'<div class="section-title">{len(results)} recommendations</div>', unsafe_allow_html=True)
        for p in results:
            render_product_card(p)


# ── ⚖️ Compare ────────────────────────────────────────────────────────────────
elif "⚖️ Compare" in page:
    st.markdown('<div class="main-header">⚖️ Compare Products</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1: id1 = st.number_input("Product ID 1", min_value=0, value=0, step=1)
    with col2: id2 = st.number_input("Product ID 2", min_value=0, value=1, step=1)
    with col3: id3 = st.number_input("Product ID 3 (optional)", min_value=-1, value=-1, step=1)

    col_a, col_b = st.columns(2)

    with col_a:
        if st.button("⚖️ Compare", type="primary", use_container_width=True):
            ids = [int(id1), int(id2)]
            if id3 >= 0: ids.append(int(id3))

            products = []
            for pid in ids:
                row = df[df["product_id"] == pid]
                if not row.empty:
                    r = row.iloc[0]
                    products.append({
                        "product_id": int(r["product_id"]),
                        "name": r["name"],
                        "price": float(r["price"]),
                        "rating": float(r["rating"]),
                        "category": r["category"],
                        "description": r["description"],
                    })

            if products:
                cheapest    = min(products, key=lambda x: x["price"])["product_id"]
                best_rated  = max(products, key=lambda x: x["rating"])["product_id"]
                cols        = st.columns(len(products))

                for col, p in zip(cols, products):
                    with col:
                        badges = ""
                        if p["product_id"] == cheapest:   badges += "💰 Cheapest  "
                        if p["product_id"] == best_rated: badges += "⭐ Best Rated"
                        st.markdown(f"""
                        <div class="product-card">
                            <div class="product-name">{p['name'].title()}</div>
                            <div style="color:#28a745;font-size:0.8rem;margin:4px 0;">{badges}</div>
                            <b>Price:</b> ${p['price']:.2f}<br>
                            <b>Rating:</b> ⭐ {p['rating']:.1f}/5<br>
                            <b>Category:</b> {p['category']}
                        </div>
                        """, unsafe_allow_html=True)

                # Summary table
                st.markdown("---")
                st.dataframe(pd.DataFrame([{
                    "Name": p["name"].title()[:25],
                    "Price": f"${p['price']:.2f}",
                    "Rating": f"⭐ {p['rating']:.1f}",
                    "Category": p["category"],
                } for p in products]), use_container_width=True)

    with col_b:
        if st.button("🤖 AI Analysis", type="secondary", use_container_width=True):
            if assistant is None:
                st.error("GROQ_API_KEY needed for AI Analysis")
            else:
                ids = [int(id1), int(id2)]
                if id3 >= 0: ids.append(int(id3))
                products = []
                for pid in ids:
                    row = df[df["product_id"] == pid]
                    if not row.empty:
                        r = row.iloc[0]
                        products.append({
                            "name": r["name"], "price": float(r["price"]),
                            "rating": float(r["rating"]), "category": r["category"],
                            "description": r["description"],
                        })
                with st.spinner("AI is analyzing..."):
                    analysis = assistant.generate_comparison(products, language=st.session_state.language)
                st.markdown("### 🤖 AI Says:")
                st.markdown(analysis)


# ── 🔧 Browse ─────────────────────────────────────────────────────────────────
elif "🔧 Browse" in page:
    st.markdown('<div class="main-header">🔧 Browse Products</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])
    with col1: sort_by = st.selectbox("Sort by", ["rating", "price_asc", "price_desc"])
    with col2: top_k   = st.slider("Results", 10, 100, 20)

    if st.button("Browse", type="primary", use_container_width=True):
        result = df.copy()

        if sidebar_category != "All":
            result = result[result["category"].str.lower() == sidebar_category.lower()]
        if sidebar_min_price > 0:
            result = result[result["price"] >= sidebar_min_price]
        if sidebar_max_price < 500:
            result = result[result["price"] <= sidebar_max_price]
        if sidebar_min_rating > 0:
            result = result[result["rating"] >= sidebar_min_rating]

        if sort_by == "rating":       result = result.sort_values("rating", ascending=False)
        elif sort_by == "price_asc":  result = result.sort_values("price",  ascending=True)
        elif sort_by == "price_desc": result = result.sort_values("price",  ascending=False)

        result = result.head(top_k)
        st.markdown(f'<div class="section-title">{len(result)} products</div>', unsafe_allow_html=True)

        for _, p in result.iterrows():
            render_product_card(p.to_dict(), show_score=False)
