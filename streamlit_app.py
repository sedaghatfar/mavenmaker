# streamlit_app.py — Streamlit Cloud (no persistence)
# ──────────────────────────────────────────────────────────────────────────────
# - No filesystem persistence (everything is in st.session_state only)
# - CSV from repo (meatmaven_specials.csv) or user upload
# - Groq via st.secrets["GROQ_API_KEY"] (falls back to env if needed)
# - Kosher/high-protein guardrails, analytics, exports
# - Meal planner shows appetizing MEAL TITLES; shopping aggregates base items
# - Default meal plan length = 3 days
# ──────────────────────────────────────────────────────────────────────────────

import os
import re
import json
import uuid
import random
from datetime import datetime

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

try:
    from groq import Groq  # optional
except Exception:
    Groq = None

# ── Config for Streamlit Cloud (GitHub) ───────────────────────────────────────
DATA_FILE = "meatmaven_specials.csv"  # lives in your repo root
RANDOM_SEED = 42

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MeatMaven Meal Planner 🥩",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styles ────────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
:root { --brand1:#FF6B6B; --brand2:#4ECDC4; --bg:#f8f9fa; }

.main-header {
    background: linear-gradient(90deg, var(--brand1), var(--brand2));
    padding: 1.5rem; border-radius: 14px; text-align: center;
    color: white; margin-bottom: 1.25rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.10);
}

.feature-card, .recipe-card, .sidebar-section {
    background: white; border-radius: 12px; padding: 1rem 1.25rem; margin: .75rem 0;
    box-shadow: 0 2px 10px rgba(0,0,0,.06);
    border: 1px solid #eef1f5;
}
.sidebar-section { background: var(--bg); border-left: 4px solid var(--brand2); }
.recipe-card { background: #fff; }
.small-muted { color:#6c757d; font-size:.92rem; }

.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] {
  height: 50px; padding: 0 18px; background-color: #f0f2f6; border-radius: 10px;
}
.stTabs [aria-selected="true"] { background-color: var(--brand1); color: white; }

.code-wrap pre { white-space: pre-wrap; }
</style>
""",
    unsafe_allow_html=True,
)

# ── Utils ─────────────────────────────────────────────────────────────────────
def koshernize_name(name: str) -> str:
    # Soft guardrails: avoid pork terms in labels
    return "Turkey Tenderloin" if "pork" in str(name).lower() else str(name)

def clean_markdown_output(text: str) -> str:
    text = re.sub(r"\*{3,}", "**", text or "")
    text = re.sub(r"\n\s*\n\s*\n", "\n\n", text).strip()
    return text

def human_time() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M")

def value_score_row(row) -> float:
    try:
        return float(row["protein"]) / float(row["price"])
    except Exception:
        return 0.0

# ── Caching ───────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Normalize
    if "title" not in df.columns:
        df["title"] = [f"Item {i+1}" for i in range(len(df))]
    df["title"] = df["title"].astype(str)

    if "price" in df.columns:
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
    else:
        df["price"] = float("nan")

    if df["price"].isna().any():
        st.warning("⚠️ Some prices were non-numeric or missing and set to NaN.")

    # Synthetic nutrition if missing
    if "protein" not in df.columns:
        random.seed(RANDOM_SEED)
        df["protein"] = [round(random.uniform(20, 35), 1) for _ in range(len(df))]
    if "calories" not in df.columns:
        random.seed(RANDOM_SEED + 1)
        df["calories"] = [random.randint(250, 450) for _ in range(len(df))]

    # Derived
    df["title"] = df["title"].apply(koshernize_name)
    df["display"] = df["title"] + " - $" + df["price"].round(2).astype(str)
    df["value_score"] = df.apply(value_score_row, axis=1).round(2)
    return df

@st.cache_resource(show_spinner=False)
def init_groq_client():
    # Prefer Streamlit secrets, fallback to env var
    api_key = None
    if st.secrets is not None and "GROQ_API_KEY" in st.secrets:
        api_key = st.secrets["GROQ_API_KEY"]
    if not api_key:
        api_key = os.getenv("GROQ_API_KEY")
    if not Groq or not api_key:
        return None
    return Groq(api_key=api_key)

# ── Session State Bootstrap (no disk persistence) ─────────────────────────────
if "meal_history" not in st.session_state:
    st.session_state.meal_history = []
if "favorites" not in st.session_state:
    st.session_state.favorites = []
if "current_plan" not in st.session_state:
    st.session_state.current_plan = None

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="main-header"><h1>🥩 MeatMaven Recipe Intelligence</h1>'
    '<p>Your AI-powered culinary companion for exceptional meal planning</p></div>',
    unsafe_allow_html=True,
)

# ── Data Ingestion (GitHub file or upload) ────────────────────────────────────
data_col1, data_col2 = st.columns([0.65, 0.35])
with data_col1:
    up_file = st.file_uploader(
        "📥 Upload a specials CSV (optional)",
        type=["csv"],
        help="Columns: title, price, [protein], [calories]",
    )
with data_col2:
    st.caption("If you don't upload, the app will load `meatmaven_specials.csv` from the repo.")

if up_file is not None:
    try:
        df = ensure_columns(pd.read_csv(up_file))
    except Exception as e:
        st.error(f"❌ Error reading uploaded CSV: {e}")
        st.stop()
else:
    try:
        df = ensure_columns(pd.read_csv(DATA_FILE))
    except FileNotFoundError:
        # Provide a friendly sample if the repo file is missing
        sample = {
            "title": [
                "London Broil", "Lamb Shoulder", "Pepper Steak",
                "Organic Chicken Breast", "Chicken Drumsticks", "Beef Brisket"
            ],
            "price": [17.99, 24.99, 15.49, 12.99, 10.99, 21.99],
            "protein": [28.0, 26.0, 27.0, 31.0, 29.0, 27.5],
            "calories": [250, 294, 265, 165, 210, 300],
        }
        df = ensure_columns(pd.DataFrame(sample))
        st.info("📁 Using sample data. Commit `meatmaven_specials.csv` to your repo to use live data.")
    except Exception as e:
        st.error(f"❌ Error reading `{DATA_FILE}`: {e}")
        st.stop()

# ── Sidebar Controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("## 🎯 Meal Planning Hub")
    st.markdown("### 🔒 Core Dietary Standards")
    st.markdown("- ✅ **Kosher Certified**\n- ✅ **High Protein Focus**\n- ✅ **Premium Quality**")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("### 🌐 Model Settings")
    client = init_groq_client()
    has_api = client is not None
    st.toggle(
        "Use Groq (if key set)",
        value=has_api,
        key="use_groq",
        help="If off or no key, app uses fast demo mode.",
    )
    model_choice = st.selectbox(
        "Groq Model",
        options=[
            "llama-3.1-8b-instant",
            "llama-3.1-70b-versatile",
            "llama-guard-3-8b",
        ],
        index=0,
        disabled=not (has_api and st.session_state.use_groq),
        help="8B is snappy; 70B gives richer detail.",
    )
    temperature = st.slider(
        "Creativity (temperature)", 0.0, 1.0, 0.7, 0.05,
        disabled=not (has_api and st.session_state.use_groq),
    )
    max_tokens = st.slider(
        "Max tokens", 256, 4096, 1600, 64,
        disabled=not (has_api and st.session_state.use_groq),
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("### 🎛️ Dietary & Prefs")
    low_carb = st.checkbox("🥬 Low-Carb Focused", value=False)
    dietary_restrictions = st.multiselect(
        "Additional Dietary Needs",
        ["Gluten-Free", "Keto-Friendly", "Paleo", "Whole30", "Anti-Inflammatory"],
    )
    cuisine_types = st.multiselect(
        "🌍 Cuisine Preferences",
        ["American", "Asian Fusion", "Israeli", "Mediterranean", "Mexican", "Persian",
         "Spanish", "Italian", "Indian", "French", "Thai", "Lebanese"]
    )
    cooking_appliances = st.multiselect(
        "🔥 Preferred Cooking Tools",
        ["Instant Pot", "Air Fryer", "Oven", "Stovetop", "Grill", "Slow Cooker",
         "Sous Vide", "Smoker", "Cast Iron"]
    )
    family_size = st.slider("👥 Family Size", 1.0, 12.0, 4.0, 0.5)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("### ⚙️ Generation Settings")
    num_recipes = st.slider("📝 Recipe Ideas Count", 1, 8, 3)
    meal_type = st.selectbox("🍽️ Meal Focus", ["Any", "Breakfast", "Lunch", "Dinner", "Snack", "Appetizer"])
    difficulty = st.selectbox("👨‍🍳 Cooking Skill Level", ["Any", "Beginner", "Intermediate", "Advanced"])
    include_nutrition = st.checkbox("📊 Nutritional Analysis", value=True)
    include_steps = st.checkbox("📋 Detailed Instructions", value=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ── Catalog Filter Bar ────────────────────────────────────────────────────────
st.markdown("### 🧭 Browse & Filter Items")
fil1, fil2, fil3, fil4 = st.columns(4)
with fil1:
    q = st.text_input("Search title", placeholder="e.g., salmon, ribeye")
with fil2:
    price_min = float(df["price"].min()) if df["price"].notna().any() else 0.0
    price_max = float(df["price"].max()) if df["price"].notna().any() else 100.0
    if price_min == price_max:
        price_min = max(0.0, price_min - 0.01)
        price_max = price_max + 0.01
    price_range = st.slider("Price range ($)", price_min, price_max, (price_min, price_max))
with fil3:
    protein_min = st.number_input("Min protein (g)", value=float(df["protein"].min()), step=1.0)
with fil4:
    sort_by = st.selectbox("Sort by", ["title", "price", "protein", "calories", "value_score"], index=4)

mask_title = df["title"].str.contains(q, case=False, na=False) if q else True
mask_price = df["price"].between(price_range[0], price_range[1], inclusive="both") if df["price"].notna().any() else True
mask_protein = df["protein"] >= protein_min
mask = mask_title & mask_price & mask_protein

catalog_df = df[mask].sort_values(sort_by, ascending=True if sort_by in ["title", "calories"] else False)
st.dataframe(
    catalog_df[["title", "price", "protein", "calories", "value_score"]]
    .rename(columns={"value_score": "protein_per_$"}),
    use_container_width=True, hide_index=True
)

# ── Helpers: Prompt Builder & Meal Title Generator ────────────────────────────
def build_recipe_prompt(selected_title: str, prompt_style: str) -> str:
    diet_constraints = "Kosher, High Protein"
    if low_carb:
        diet_constraints += ", Low-Carb"
    if dietary_restrictions:
        diet_constraints += f", {', '.join(dietary_restrictions)}"

    extra = []
    if cuisine_types:
        extra.append(f"Draw inspiration from {', '.join(cuisine_types)} cuisines.")
    if cooking_appliances:
        extra.append(f"Optimize for cooking with: {', '.join(cooking_appliances)}.")
    if meal_type != "Any":
        extra.append(f"Focus on {meal_type.lower()} preparations.")
    if difficulty != "Any":
        extra.append(f"Keep recipes at a {difficulty.lower()} skill level.")

    templates = {
        "Family-Friendly Classics":
            f"Create {{N}} beloved, {diet_constraints} family recipes using {selected_title}. "
            f"Familiar flavors, simple techniques. Serves {{S}}.",
        "Gourmet Culinary Adventure":
            f"Design {{N}} sophisticated, {diet_constraints} dishes featuring {selected_title}. "
            f"Restaurant-quality with impressive presentation. Serves {{S}}.",
        "Quick & Healthy Weeknight Meals":
            f"Generate {{N}} nutritious, {diet_constraints} weeknight recipes using {selected_title}. "
            f"Max 30 minutes, minimal cleanup. Serves {{S}}.",
        "Budget-Conscious Cooking":
            f"Develop {{N}} economical, {diet_constraints} recipes with {selected_title}. "
            f"Stretch the budget without losing flavor. Serves {{S}}.",
        "International Fusion":
            f"Create {{N}} globally-inspired, {diet_constraints} fusion dishes with {selected_title}. "
            f"Blend traditional techniques creatively. Serves {{S}}.",
        "Meal Prep Master":
            f"Design {{N}} {diet_constraints} meal-prep recipes using {selected_title}. "
            f"Batchable, 3–5 day storage. Serves {{S}}.",
    }
    base = templates.get(prompt_style, templates["Family-Friendly Classics"])
    base = base.replace("{N}", str(num_recipes)).replace("{S}", str(int(family_size)))
    full = base + " " + " ".join(extra)

    if include_nutrition:
        full += " Include detailed nutrition per serving (calories, protein, carbs, fat, fiber, sodium)."
    if include_steps:
        full += " Provide clear step-by-step instructions with timing and technique tips."
    else:
        full += " Provide a concise overview and ingredient list."

    full += (
        " Strictly avoid non-kosher ingredients (no pork/shellfish, separate meat/dairy, "
        "no mixing meat with dairy). Use pareve or meat-friendly substitutions when needed. "
        "Format each recipe with: Title, Ingredients, Instructions, Tips, and Nutrition."
    )
    return full

def classify_item(base_item: str) -> tuple[str, str]:
    """Return (protein, cut_or_item) for nicer titles."""
    s = base_item.lower()
    if "lamb" in s:
        return "Lamb", base_item
    if "london broil" in s:
        return "Beef", "London Broil"
    if "brisket" in s:
        return "Beef", "Brisket"
    if "flanken" in s or "short rib" in s:
        return "Beef", "Short Ribs"
    if "pepper steak" in s:
        return "Beef", "Pepper Steak"
    if "beef" in s:
        return "Beef", base_item
    if "drumstick" in s or "thigh" in s or "breast" in s or "chicken" in s or "legs" in s:
        return "Chicken", base_item
    if "turkey" in s:
        return "Turkey", base_item
    if "salmon" in s:
        return "Salmon", base_item
    if "fish" in s:
        return "Fish", base_item
    return base_item.split()[0].title(), base_item

def generate_meal_title(base_item: str, plan_style: str, cuisines: list[str], meal_focus: str) -> str:
    """
    Return a short, appetizing meal title built around the base ingredient.
    Uses Groq if available/toggled; otherwise curated rule-based fallback.
    Keeps kosher by avoiding obvious dairy-meat mixes and non-kosher items.
    """
    protein, cut = classify_item(base_item)

    # Prefer AI if available
    if 'client' in globals() and client and st.session_state.get("use_groq"):
        try:
            cuisines_txt = ", ".join(cuisines) if cuisines else "any cuisine"
            meal_focus_txt = meal_focus if meal_focus != "Any" else "any meal"
            prompt = (
                "Create ONE short, appetizing kosher meal title (6–10 words) using the base item.\n"
                f"- Base item: {base_item}\n"
                f"- Protein: {protein}\n"
                f"- Cut/Style: {cut}\n"
                f"- Theme: {plan_style}\n"
                f"- Cuisine hints: {cuisines_txt}\n"
                f"- Meal focus: {meal_focus_txt}\n"
                "- Kosher: no pork/shellfish; no mixing meat+dairy; prefer pareve sauces (e.g., tahini).\n"
                "- Output only the title. No extra text."
            )
            resp = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.1-8b-instant",
                temperature=0.6,
                max_tokens=32,
            )
            title = (resp.choices[0].message.content or "").strip()
            title = re.sub(r"^[\"'“”]+|[\"'“”]+$", "", title)
            if 4 <= len(title.split()) <= 14:
                return title
        except Exception:
            pass  # fall through to rule-based

    # Curated fallback patterns for tastier names
    rand = random.Random(RANDOM_SEED + hash(base_item) % 100000)

    patterns_by_protein = {
        "Lamb": [
            "Moroccan Lamb Stew Salad Bowls",
            "Lamb and Chickpea Tagine",
            "Za’atar Grilled Lamb with Herb Couscous",
            "Sumac-Rubbed Lamb over Roasted Vegetables",
        ],
        "Beef": [
            f"{cut} Skewers with Herbed Tahini Sauce" if "London Broil" in cut else "London Broil Skewers with Herbed Tahini Sauce",
            "Pepper Steak Lettuce Wraps",
            f"Korean Bulgogi {cut}" if cut not in ("Pepper Steak",) else "Korean Bulgogi London Broil",
            "One-Pan Pepper Steak with Roasted Vegetables",
            "Garlic-Herb Beef with Cauliflower Pilaf",
        ],
        "Chicken": [
            "Za’atar Roast Chicken with Herbed Couscous",
            "Garlic-Lemon Chicken over Roasted Veggies",
            "Sheet-Pan Chicken with Sumac Onions",
            "Honey-Chili Chicken with Turmeric Rice",
        ],
        "Turkey": [
            "Citrus-Herb Turkey over Quinoa Tabbouleh",
            "Smoky Turkey Skewers with Tahini Drizzle",
        ],
        "Salmon": [
            "Herb-Crusted Salmon with Lemon Dill Slaw",
            "Sumac-Dusted Salmon over Charred Broccolini",
        ],
        "Fish": [
            "Chili-Lime Fish with Tomato-Cucumber Salad",
            "Mediterranean Spiced Fish over Olive Couscous",
        ],
    }

    pool = patterns_by_protein.get(protein, [
        f"Herb-Crusted {cut} with Olive & Tomato Salad",
        f"Spice-Rubbed {cut} with Garlic Green Beans",
    ])
    return rand.choice(pool)

def demo_recipes(selected_title: str, n: int = 3) -> str:
    out = []
    for i in range(n):
        out.append(
            f"""### {selected_title} — Chef's Idea {i+1}

**Ingredients**
- 1 lb {selected_title.lower()}
- 2 cloves garlic, minced
- 1 tbsp olive oil
- Salt & pepper
- Fresh herbs (rosemary/thyme)

**Instructions**
1) Season {selected_title.lower()} with salt & pepper.
2) Sear in hot pan 2–3 min/side (or air-fry 8–12 min @ 400°F), rest 5 min.
3) Finish with herbs & garlic in last minute.

**Tips**
- Keep it kosher: pair with veggie sides or non-dairy sauces.
- High-protein swap: add egg-white scramble on the side.

**Estimated Nutrition (per serving)**
~320 kcal • 28g protein • 2g carbs • 22g fat
---
"""
        )
    return "\n".join(out)

def add_favorite(name: str, content: str):
    fav = {
        "id": str(uuid.uuid4()),
        "name": name,
        "content": content,
        "date_saved": human_time(),
    }
    st.session_state.favorites.insert(0, fav)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["💡 Recipe Ideas", "🗓️ Meal Plans", "📊 Analytics", "⭐ Favorites", "🛒 Shopping"]
)

# ── Tab 1: Recipe Ideas ───────────────────────────────────────────────────────
with tab1:
    st.header("🍳 Intelligent Recipe Generation")

    c1, c2 = st.columns([2, 1])
    with c1:
        selected_item = st.selectbox(
            "🥩 Choose Your Premium Cut",
            options=catalog_df["display"],
            index=0 if len(catalog_df) else None,
        )
        prompt_style = st.selectbox(
            "🎨 Recipe Style & Approach",
            [
                "Family-Friendly Classics",
                "Gourmet Culinary Adventure",
                "Quick & Healthy Weeknight Meals",
                "Budget-Conscious Cooking",
                "International Fusion",
                "Meal Prep Master",
            ],
        )
    with c2:
        if selected_item:
            row = df[df["display"] == selected_item].iloc[0]
            st.markdown(
                f"""
                <div class="feature-card">
                    <h4>📋 Item Details</h4>
                    <p><strong>Price:</strong> ${row['price']:.2f}</p>
                    <p><strong>Protein:</strong> {row['protein']} g</p>
                    <p><strong>Calories:</strong> {int(row['calories'])}</p>
                    <p class="small-muted"><strong>Protein/$:</strong> {row['value_score']}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    gen = st.button("🚀 Generate Amazing Recipes", use_container_width=True, type="primary")
    if gen and selected_item:
        selected_title = selected_item.split(" - $")[0]
        with st.spinner("🧠 AI Chef is crafting your personalized recipes..."):
            recipe_md = None
            if client and st.session_state.use_groq:
                try:
                    prompt = build_recipe_prompt(selected_title, prompt_style)
                    chat_completion = client.chat.completions.create(
                        messages=[{"role": "user", "content": prompt}],
                        model=model_choice,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    recipe_md = clean_markdown_output(chat_completion.choices[0].message.content)
                except Exception as e:
                    st.error(f"API Error (using demo recipes): {e}")
                    recipe_md = demo_recipes(selected_title, n=num_recipes)
            else:
                recipe_md = demo_recipes(selected_title, n=num_recipes)

        st.markdown("---")
        st.subheader(f"✨ Gourmet Recipes for {selected_title}")
        st.markdown(f'<div class="recipe-card">{recipe_md}</div>', unsafe_allow_html=True)

        # History (session-only)
        history_item = {
            "id": str(uuid.uuid4()),
            "item": selected_title,
            "style": prompt_style,
            "content": recipe_md,
            "timestamp": human_time(),
        }
        st.session_state.meal_history.insert(0, history_item)

        a1, a2, a3, a4 = st.columns(4)
        with a1:
            if st.button("⭐ Save to Favorites"):
                add_favorite(selected_title, recipe_md)
                st.toast("Added to favorites!", icon="⭐")
        with a2:
            st.download_button(
                "⬇️ Download (Markdown)",
                data=recipe_md.encode("utf-8"),
                file_name=f"{selected_title.replace(' ', '_').lower()}_recipes.md",
                mime="text/markdown",
            )
        with a3:
            st.download_button(
                "⬇️ Download (TXT)",
                data=re.sub(r"[#*_`]", "", recipe_md).encode("utf-8"),
                file_name=f"{selected_title.replace(' ', '_').lower()}_recipes.txt",
                mime="text/plain",
            )
        with a4:
            if st.button("🔄 New Variations"):
                st.experimental_rerun()

# ── Tab 2: Meal Plans (MEAL TITLES; default 3 days) ──────────────────────────
with tab2:
    st.header("🗓️ Intelligent Meal Planning")

    c1, c2 = st.columns([2, 1])
    with c1:
        num_days_plan = st.slider("📅 Plan Duration (Days)", 1, 14, 3)  # default = 3
        meal_types_for_plan = st.multiselect(
            "🍽️ Meals to Include",
            ["Breakfast", "Lunch", "Dinner", "Snacks"],
            default=["Lunch", "Dinner"],  # typically two per day, matching your examples
        )
        plan_style = st.selectbox(
            "🎯 Meal Plan Theme",
            ["Balanced Variety", "High Protein Focus", "Quick & Easy", "Gourmet Week", "Budget Friendly", "Meal Prep Focused"],
        )
        allow_repeats = st.checkbox("Allow item repeats across the plan", value=True, help="If off, planner will sample unique items.")
    with c2:
        total_meals = max(1, len(meal_types_for_plan)) * num_days_plan
        st.markdown(
            f"""
            <div class="feature-card">
              <h4>📊 Plan Summary</h4>
              <p><strong>Duration:</strong> {num_days_plan} days</p>
              <p><strong>Meals/Day:</strong> {len(meal_types_for_plan)}</p>
              <p><strong>Total Meals:</strong> {total_meals}</p>
              <p><strong>Family Size:</strong> {int(family_size)} people</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    make_plan = st.button("🎲 Create My Meal Plan", use_container_width=True, type="primary")
    if make_plan:
        if not meal_types_for_plan:
            st.error("Please choose at least one meal type.")
        else:
            pool = list(df["title"])
            if not allow_repeats and len(pool) < total_meals:
                st.error(f"Need {total_meals} unique items, but only {len(pool)} available. Enable repeats or reduce scope.")
            else:
                random.seed(RANDOM_SEED)
                plan_rows = []
                # Keep separate mapping for shopping list aggregation
                planned_items_for_shopping = []

                for day in range(1, num_days_plan + 1):
                    row = {"Day": f"{day}"}  # numeric day index like your example
                    for mt in meal_types_for_plan:
                        # choose base item
                        if allow_repeats:
                            base_item = random.choice(pool)
                        else:
                            pick = random.choice(pool)
                            base_item = pick
                            pool.remove(pick)

                        # build meal title (AI or curated rule-based)
                        meal_title = generate_meal_title(base_item, plan_style, cuisine_types, mt)

                        # show pretty name in the grid; remember base item for shopping
                        # column header = meal type (e.g., Lunch, Dinner)
                        row[mt] = meal_title
                        planned_items_for_shopping.append(base_item)
                    plan_rows.append(row)

                # Reorder columns as Day, then selected meal types
                plan_df = pd.DataFrame(plan_rows, columns=["Day"] + meal_types_for_plan)
                st.dataframe(plan_df, use_container_width=True, hide_index=True)

                # Store plan (display) and shopping items (session only)
                st.session_state.current_plan = plan_df.to_dict(orient="list")
                st.session_state.plan_items_for_shopping = planned_items_for_shopping
                st.success("🎉 Meal plan generated! Check the Shopping tab for your grocery list.")

                # Export plan
                plan_csv = plan_df.to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ Download Plan (CSV)", data=plan_csv, file_name="meal_plan.csv", mime="text/csv")

# ── Tab 3: Analytics ─────────────────────────────────────────────────────────
with tab3:
    st.header("📊 Nutrition & Price Analytics")
    tab3a, tab3b = st.tabs(["📈 Nutrition Analysis", "💰 Price Insights"])

    with tab3a:
        fig = px.scatter(
            df, x="protein", y="calories",
            size="price", hover_name="title",
            title="Protein vs Calories (bubble=size=price)",
            color="value_score", color_continuous_scale="viridis",
        )
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

        mc1, mc2, mc3 = st.columns(3)
        with mc1:
            st.metric("Highest Protein", df.loc[df["protein"].idxmax(), "title"], f"{df['protein'].max():.1f} g")
        with mc2:
            st.metric("Lowest Calories", df.loc[df["calories"].idxmin(), "title"], f"{int(df['calories'].min())} kcal")
        with mc3:
            st.metric("Best Value", df.loc[df["value_score"].idxmax(), "title"], f"{df['value_score'].max():.2f} g/$")

    with tab3b:
        max_n = max(1, min(20, len(df)))
        default_n = min(10, max_n)
        fig2 = px.bar(
            df.nlargest(default_n, "price"),
            x="title", y="price", title=f"Top {default_n} Prices",
            color="price", color_continuous_scale="reds",
        )
        fig2.update_xaxes(tickangle=45)
        fig2.update_layout(height=420, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig2, use_container_width=True)

        pc1, pc2, pc3 = st.columns(3)
        with pc1:
            st.metric("Average Price", f"${df['price'].mean():.2f}")
        with pc2:
            st.metric("Most Expensive", df.loc[df["price"].idxmax(), "title"], f"${df['price'].max():.2f}")
        with pc3:
            st.metric("Best Deal (cheap)", df.loc[df["price"].idxmin(), "title"], f"${df['price'].min():.2f}")

# ── Tab 4: Favorites ─────────────────────────────────────────────────────────
with tab4:
    st.header("⭐ My Recipe Favorites")

    if st.session_state.favorites:
        for fav in list(st.session_state.favorites):
            with st.expander(f"⭐ {fav['name']} — Saved {fav['date_saved']}"):
                st.markdown(fav["content"])
                cols = st.columns(3)
                with cols[0]:
                    if st.button("🗑️ Remove", key=f"rm_{fav['id']}"):
                        st.session_state.favorites = [f for f in st.session_state.favorites if f["id"] != fav["id"]]
                        st.experimental_rerun()
                with cols[1]:
                    st.download_button(
                        "⬇️ Download (MD)",
                        data=fav["content"].encode("utf-8"),
                        file_name=f"{fav['name'].replace(' ','_').lower()}_{fav['id'][:8]}.md",
                        mime="text/markdown",
                        key=f"dl_{fav['id']}",
                    )
                with cols[2]:
                    st.code(f"Saved: {fav['date_saved']}", language="text")
    else:
        st.info("🔖 No favorites yet. Generate some recipes and save your winners!")

    st.markdown("### 📚 Recent Recipe History")
    if st.session_state.meal_history:
        hist_df = pd.DataFrame(
            [{"Item": h["item"], "Style": h["style"], "Generated": h["timestamp"]} for h in st.session_state.meal_history[:10]]
        )
        st.dataframe(hist_df, use_container_width=True, hide_index=True)
        st.download_button(
            "⬇️ Download History (JSON)",
            data=json.dumps(st.session_state.meal_history, ensure_ascii=False, indent=2),
            file_name="recipe_history.json",
            mime="application/json",
        )
    else:
        st.info("📝 No history yet — go create something tasty!")

# ── Tab 5: Shopping (uses base items even if plan shows fancy titles) ────────
with tab5:
    st.header("🛒 Smart Shopping Assistant")

    if st.session_state.current_plan is not None:
        st.subheader("📋 Generated Shopping List")
        plan_df = pd.DataFrame(st.session_state.current_plan)
        st.dataframe(plan_df, use_container_width=True, hide_index=True)

        # Flatten meal items (use mapped base items if available)
        if "plan_items_for_shopping" in st.session_state and st.session_state.plan_items_for_shopping:
            plan_items = list(st.session_state.plan_items_for_shopping)
        else:
            # Fallback: parse titles from the grid (older behavior)
            plan_items = []
            for col in plan_df.columns:
                if col == "Day":
                    continue
                plan_items.extend(plan_df[col].dropna().tolist())

        # Aggregate unique items w/ info
        shopping = []
        for item in sorted(set(plan_items)):
            if item in df["title"].values:
                info = df[df["title"] == item].iloc[0]
                shopping.append({
                    "Item": item,
                    "Est Price": round(float(info["price"]), 2),
                    "Protein (g)": float(info["protein"]),
                    "Qty": "1 lb",
                })
        shop_df = pd.DataFrame(shopping)
        if not shop_df.empty:
            st.dataframe(shop_df, use_container_width=True, hide_index=True)
            total_cost = shop_df["Est Price"].sum()
            st.metric("💰 Estimated Total Cost", f"${total_cost:.2f}")

            # Exports
            st.download_button(
                "⬇️ Download Shopping (CSV)",
                data=shop_df.to_csv(index=False).encode("utf-8"),
                file_name="shopping_list.csv",
                mime="text/csv",
            )
            md_lines = [f"- [ ] {r['Item']} — {r['Qty']} — ${r['Est Price']:.2f}" for _, r in shop_df.iterrows()]
            st.download_button(
                "⬇️ Download Shopping (Markdown)",
                data="\n".join(md_lines).encode("utf-8"),
                file_name="shopping_list.md",
                mime="text/markdown",
            )
        else:
            st.info("No recognized items found in plan (did you upload a CSV with matching titles?).")
    else:
        st.info("🛒 Generate a meal plan first to create your shopping list!")

    # Manual add
    st.markdown("### ➕ Custom Shopping List")
    with st.expander("Add items manually"):
        custom = st.multiselect("Select additional items", df["title"].tolist())
        if custom:
            cdf = df[df["title"].isin(custom)][["title", "price", "protein"]].rename(
                columns={"title": "Item", "price": "Est Price", "protein": "Protein (g)"}
            )
            st.dataframe(cdf, use_container_width=True, hide_index=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    """
<div style="text-align:center; padding: 1.25rem; background: #f8f9fa; border-radius: 10px; margin-top: .5rem;">
  <h4>🥩 MeatMaven Recipe Intelligence</h4>
  <p><strong>Disclaimer:</strong> AI-generated recipes are suggestions only. Always verify cooking temps and food safety.</p>
  <p><em>Demo application. Not affiliated with The MeatMaven.</em></p>
  <p>📧 Feedback: <a href="mailto:Matt@tevunah.com">Matt@tevunah.com</a></p>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div style="text-align:center; margin-top: 10px;">
  <a href="https://groq.com" target="_blank" rel="noopener noreferrer">
    <img src="https://console.groq.com/powered-by-groq.svg" alt="Powered by Groq" style="width: 120px; height: auto;" />
  </a>
</div>
""",
    unsafe_allow_html=True,
)
