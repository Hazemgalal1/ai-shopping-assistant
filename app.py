"""
AI Shopping Assistant — Streamlit Cloud Deployment
Unified single-page chat: Multi-Agent architecture
(Orchestrator -> CatalogAgent / WebSearchAgent -> Composer), all in one Agent.
"""

import os
import uuid
import pandas as pd
import streamlit as st

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
    .web-card {
        background:#fff8f0; border-radius:12px; padding:1rem;
        margin-bottom:0.8rem; border-left:4px solid #f78e4f;
    }
    .product-name { font-weight:600; font-size:1rem; color:#1a1a2e; }
    .badge {
        display:inline-block; padding:2px 8px; border-radius:12px;
        font-size:0.75rem; font-weight:600; margin-right:4px;
    }
    .badge-category { background:#e8f0fe; color:#4F8EF7; }
    .badge-price    { background:#d4edda; color:#155724; }
    .badge-rating   { background:#fff3cd; color:#856404; }
    .badge-web      { background:#ffe8d6; color:#a5540d; }
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
    from src.models.orchestrator import ShoppingOrchestrator

    data_path = "data/processed/products_clean.csv"
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
    else:
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

    emb_path   = "models/embeddings/product_embeddings.npy"
    tfidf_path = "models/tfidf/tfidf_vectorizer.pkl"

    search_engine = SearchEngine()
    if os.path.exists(emb_path) and os.path.exists(tfidf_path):
        search_engine.load(df)
    else:
        search_engine.build(df)

    rec_engine = RecommendationEngine()
    rec_engine.load(df, search_engine.embeddings)

    tool_executor = ToolExecutor(
        search_engine=search_engine,
        rec_engine=rec_engine,
        product_df=df,
    )

    agent = None
    if os.environ.get("GROQ_API_KEY"):
        try:
            agent = ShoppingOrchestrator(tool_executor=tool_executor)
        except Exception as e:
            st.warning(f"Groq not available: {e}")

    return df, agent


df, agent = load_all()

from src.models.memory import memory_store

# ─── Session State ────────────────────────────────────────────────────────────
if "messages"   not in st.session_state: st.session_state.messages   = []
if "session_id" not in st.session_state: st.session_state.session_id = "sess_" + str(uuid.uuid4())[:8]
if "language"   not in st.session_state: st.session_state.language   = "en"

# ─── Sidebar (مبسّط) ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🛍️ AI Shopping Assistant")
    st.markdown(f"**📦 {len(df):,} internal products + 🌐 live web search**")
    st.markdown("---")

    lang = st.radio("🌐 Language", ["English 🇬🇧", "العربية 🇪🇬"], horizontal=True, label_visibility="collapsed")
    st.session_state.language = "ar" if "العربية" in lang else "en"

    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        memory_store.reset(st.session_state.session_id)
        st.rerun()

    mem = memory_store.get_or_create(st.session_state.session_id)
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

is_arabic = st.session_state.language == "ar"

# ─── Helpers ──────────────────────────────────────────────────────────────────

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


def render_web_result_card(w: dict):
    title = str(w.get("title", "")).strip() or "Untitled"
    url = w.get("url", "#")
    snippet = str(w.get("snippet", ""))[:220]
    card_html = (
        f'<div class="web-card">'
        f'<div class="product-name">{title}</div>'
        f'<div style="margin-top:4px;"><span class="badge badge-web">🌐 Web</span></div>'
        f'<div style="margin-top:8px;color:#777;font-size:0.82rem;">{snippet}...</div>'
        f'<div style="margin-top:8px;"><a href="{url}" target="_blank">{url}</a></div>'
        f'</div>'
    )
    st.markdown(card_html, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN CHAT — كل حاجة في مكان واحد
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="main-header">🛍️ AI Shopping Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header arabic-text">اسأل عن أي منتج — الكتالوج الداخلي أو أي حاجة على الإنترنت</div>'
    if is_arabic else
    '<div class="sub-header">Ask about anything — our catalog or live results from the web</div>',
    unsafe_allow_html=True,
)

if agent is None:
    st.error("⚠️ GROQ_API_KEY مش موجود — حط الـ key في Streamlit Secrets" if is_arabic
             else "⚠️ GROQ_API_KEY not found. Add it to Streamlit Secrets.")
    st.code('GROQ_API_KEY = "gsk_xxxxxxxxxxxxxxxxxxxx"', language="toml")
    st.stop()

with st.expander("⚙️ إزاي بيشتغل؟" if is_arabic else "⚙️ How does this work?"):
    st.markdown(
        "بيدور الأول في الكتالوج الداخلي، ولو مش لاقي حاجة مناسبة (أو طلبت منتج حقيقي/سعر حالي) بيدور على الإنترنت تلقائيًا.\n\n"
        "**البنية:** Router بيقرر مين يشتغل → CatalogAgent (كتالوج داخلي) و/أو WebSearchAgent (بحث حي على الإنترنت) → "
        "Composer بيجمع النتائج ويطلع رد واحد."
        if is_arabic else
        "It searches the internal catalog first, and automatically falls back to a live web "
        "search when the catalog doesn't have what you need (or you ask for real-world prices).\n\n"
        "**Architecture:** A Router decides which specialist to use → CatalogAgent (internal data) "
        "and/or WebSearchAgent (live web) run → a Composer merges everything into one final reply."
    )

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if is_arabic and msg["role"] == "assistant":
            st.markdown(f'<div class="arabic-text">{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(msg["content"])

        if msg.get("tool_calls"):
            with st.expander(f"🧠 {len(msg['tool_calls'])} tool calls | {msg.get('rounds',1)} rounds"):
                for i, tc in enumerate(msg["tool_calls"], 1):
                    agent_tag = f"[{tc['agent']}] " if tc.get("agent") else ""
                    st.markdown(
                        f'<div class="tool-step">Step {i}: {agent_tag}{tc["tool"]}({tc["args"]}) → {tc["result_count"]} results</div>',
                        unsafe_allow_html=True,
                    )

        if msg.get("products"):
            with st.expander(f"📦 {len(msg['products'])} catalog products", expanded=True):
                for p in msg["products"][:5]:
                    render_product_card(p)

        if msg.get("web_results"):
            with st.expander(f"🌐 {len(msg['web_results'])} web results", expanded=True):
                for w in msg["web_results"][:5]:
                    render_web_result_card(w)

placeholder = (
    "اطلب أي حاجة... مثلاً: عايز أفضل لاب توب بأفضل سعر، أو أي حاجة تانية غير الإلكترونيكس"
    if is_arabic else
    "Ask anything... e.g. Find me the best laptop at the best price, or anything outside Electronics"
)
user_input = st.chat_input(placeholder)

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        status = st.status("🧠 Thinking..." if not is_arabic else "🧠 بفكر...", expanded=True)
        try:
            result = agent.run(user_message=user_input, session_id=st.session_state.session_id)

            route = result.get("route")
            if route:
                status.write(f"🧭 Route → catalog: {route.get('use_catalog')} | web: {route.get('use_web')}")

            for tc in result.get("tool_calls_made", []):
                agent_tag = f"[{tc['agent']}] " if tc.get("agent") else ""
                status.write(f"🔧 {agent_tag}`{tc['tool']}` → {tc['result_count']} results")

            status.update(
                label=f"✅ Done — {len(result.get('tool_calls_made',[]))} tools used",
                state="complete",
            )

            reply       = result["reply"]
            products    = result.get("products_found", [])
            web_results = result.get("web_results_found", [])

            if is_arabic:
                st.markdown(f'<div class="arabic-text">{reply}</div>', unsafe_allow_html=True)
            else:
                st.markdown(reply)

            if products:
                with st.expander(f"📦 {len(products)} catalog products", expanded=True):
                    for p in products[:5]:
                        render_product_card(p)

            if web_results:
                with st.expander(f"🌐 {len(web_results)} web results", expanded=True):
                    for w in web_results[:5]:
                        render_web_result_card(w)

            st.session_state.messages.append({
                "role": "assistant",
                "content": reply,
                "tool_calls": result.get("tool_calls_made", []),
                "products": products,
                "web_results": web_results,
                "rounds": result.get("rounds", 1),
            })

        except Exception as e:
            status.update(label="❌ Error", state="error")
            st.error(str(e))

if not st.session_state.messages:
    st.markdown("---")
    st.markdown("**جرب:**" if is_arabic else "**Try:**")
    examples = [
        "عايز أفضل لاب توب بأفضل سعر",
        "قارنلي أحسن 3 منتجات في Electronics عندنا",
        "Find me a good pair of running shoes online",
        "I need a gift under $30 from our catalog, suggest something",
    ]
    cols = st.columns(2)
    for i, ex in enumerate(examples):
        with cols[i % 2]:
            if st.button(ex, use_container_width=True, key=f"ex_{i}"):
                st.session_state.messages.append({"role": "user", "content": ex})
                st.rerun()