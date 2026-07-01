# 🛍️ AI Shopping Assistant — Streamlit Cloud

## 🚀 Deploy on Streamlit Cloud

### Step 1: رفع على GitHub
```bash
git init
git add .
git commit -m "initial commit"
git remote add origin https://github.com/USERNAME/ai-shopping-assistant
git push -u origin main
```

### Step 2: Streamlit Cloud
1. روح على **share.streamlit.io**
2. اضغط **New app**
3. اختار الـ repo
4. الـ Main file: `app.py`
5. اضغط **Deploy**

### Step 3: GROQ_API_KEY
في Streamlit Cloud:
1. **Settings** → **Secrets**
2. الصق:
```toml
GROQ_API_KEY = "gsk_xxxxxxxxxxxxxxxxxxxx"
```

احصل على API key مجاناً: **console.groq.com**

---

## 💻 Local Run

```bash
pip install -r requirements.txt
# حط الـ API key في .streamlit/secrets.toml
streamlit run app.py
```

---

## 📁 Structure

```
app.py                    ← الـ entry point الوحيد
requirements.txt
.streamlit/
    config.toml
    secrets.toml          ← GROQ_API_KEY (لا ترفعه على GitHub)
src/
    models/
        agent.py          ← 🤖 Agentic AI
        tools.py          ← Tool definitions
        assistant.py      ← Conversational AI
        memory.py         ← Multi-turn memory
        search.py         ← NLP search
        recommend.py      ← Recommendations
    data/
        collect.py
        clean.py
        scraper.py
```

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🤖 AI Agent | بيفكر ويقرر أيه الـ tools تلقائياً |
| 💬 AI Chat | محادثة طبيعية مع RAG |
| 🔍 Search | Hybrid TF-IDF + Semantic |
| 💡 Recommend | Similar + Complementary |
| ⚖️ Compare | مقارنة + AI Analysis |
| 🔧 Browse | Filter by price/category/rating |
| 🇦🇪 Arabic | يفهم ويرد بالعربي |
| 🧠 Memory | بيفتكر المحادثة |
