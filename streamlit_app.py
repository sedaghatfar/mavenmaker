import streamlit as st
from groq import Groq
import pandas as pd
import random
import re

# Load your CSV
data_file = "meatmaven.csv"
df = pd.read_csv(data_file)

# Combine title and price for display
df["display"] = df["title"] + " - " + df["price"].astype(str)

# Initialize Groq client
client = Groq(
    api_key=st.secrets["GROQ_API_KEY"],
)

# Enhanced Sidebar
st.sidebar.image("/mount/src/mavenmaker/mavenlogo.svg", width=200)  # Add your logo
st.sidebar.markdown("## 🍽️ Meal Planning Toolkit")

# Fixed Dietary Restrictions
st.sidebar.markdown("### Fixed Dietary Foundations")
st.sidebar.write("✅ Kosher")
st.sidebar.write("✅ Dairy-Free")
st.sidebar.write("✅ High Protein")

# Low-Carb Option
low_carb = st.sidebar.checkbox("Low-Carb Option", value=False)

# Cuisine Type Filter
cuisine_types = st.sidebar.multiselect(
    "Preferred Cuisines",
    ["American", "Asian", "Israeli", "Persian", "Spanish"]
)

# Cooking Appliance Selection
st.sidebar.markdown("### Cooking Appliances")
cooking_appliances = st.sidebar.multiselect(
    "Preferred Cooking Methods",
    ["Instant Pot", "Air Fryer", "Oven", "Stovetop"]
)

# Family Size
family_size = st.sidebar.slider("Family Size", 1.0, 10.0, 4.0, 0.5)

# Main App
st.title("MeatMaven 🥩 Recipe Intelligence")
st.write("Personalized meal planning powered by AI and your favorite proteins!")

# Prompt Enhancement Dropdown
prompt_style = st.selectbox(
    "Recipe Idea Generation Style",
    [
        "Family-Friendly Classics", 
        "Gourmet Culinary Adventure", 
        "Quick & Healthy Weeknight Meals", 
        "Budget-Conscious Cooking"
    ]
)

# Display the table for user selection
selected_item = st.selectbox("Choose an item:", options=df["display"], index=0)

# Enhanced Prompt Generation Function
def generate_enhanced_prompt(selected_title, prompt_style, low_carb, cuisine_types, cooking_appliances, family_size):
    # Base dietary constraints
    diet_constraints = "Kosher, Dairy-Free, High Protein"
    if low_carb:
        diet_constraints += ", Low-Carb"
    
    # Incorporate cuisine and cooking method if specified
    additional_context = ""
    if cuisine_types:
        additional_context += f"Consider {', '.join(cuisine_types)} cuisine styles. "
    if cooking_appliances:
        additional_context += f"Use {', '.join(cooking_appliances)} for cooking. "
    
    prompt_templates = {
        "Family-Friendly Classics": 
            f"Create 3 kid-approved, {diet_constraints} recipe ideas using {selected_title}. "
            f"Prepare for a family of {family_size}. {additional_context}",
        
        "Gourmet Culinary Adventure": 
            f"Design 3 sophisticated, {diet_constraints} culinary creations featuring {selected_title} "
            f"with restaurant-quality presentation. Family size: {family_size}. {additional_context}",
        
        "Quick & Healthy Weeknight Meals": 
            f"Generate 3 nutritious, {diet_constraints} recipes using {selected_title} "
            f"that can be prepared quickly for {family_size} people. {additional_context}",
        
        "Budget-Conscious Cooking": 
            f"Develop 3 economical, {diet_constraints} meal ideas with {selected_title} "
            f"that maximize flavor while minimizing cost. Serves {family_size}. {additional_context}"
    }
    
    return prompt_templates.get(prompt_style, prompt_templates["Family-Friendly Classics"])

# Function to clean up LLM output
def clean_markdown_output(text):
    # Remove any stray asterisks or markdown formatting artifacts
    text = re.sub(r'\*+', '', text)
    return text

# Recipe Ideas Button
if st.button("🍳 Get Recipe Ideas"):
    selected_title = selected_item.split(" - ")[0]
    enhanced_prompt = generate_enhanced_prompt(
        selected_title, prompt_style, low_carb, 
        cuisine_types, cooking_appliances, family_size
    )
    
    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": enhanced_prompt}],
        model="gemma2-9b-it",
    )
    
    st.markdown(clean_markdown_output(chat_completion.choices[0].message.content))

# Surprise Meal Plan Button
if st.button("🎲 Surprise Meal Plan"):
    random_items = random.sample(list(df["title"]), 3)
    random_items_str = ", ".join(random_items)
    
    surprise_prompt = (
        f"Act as an expert nutritionist. Create a unique, diverse Kosher, dairy-free, "
        f"high protein meal plan for a family of {family_size}. "
        f"Use {random_items_str} as the main proteins in creative, varied ways. "
        f"{'Add low-carb considerations.' if low_carb else ''} "
        f"{'Incorporate ' + ', '.join(cuisine_types) + ' cuisine styles.' if cuisine_types else 'Use diverse international cuisine inspirations.'} "
        f"{'Suggest recipes using ' + ', '.join(cooking_appliances) + '.' if cooking_appliances else 'Vary cooking techniques.'} "
        "Ensure NO repeated meal ideas across days. Create an easy-to-read lunch and dinner plan "
        "with completely different recipes for each meal. Format:\n\n"
        "Day | Lunch | Dinner\n"
        "1 | Mediterranean Spiced Chicken Salad | Korean-Style Beef Bulgogi Lettuce Wraps\n"
        "2 | Middle Eastern Lamb Kebab Plate | Moroccan Spiced Fish Tagine\n"
        "3 | Mexican Carnitas-Style Pulled Beef Tacos | Thai-Inspired Chicken Stir Fry\n\n"
        "Provide a detailed shopping list with exact quantities for the family size, "
        "highlighting unique ingredients for each recipe. Avoid generic or repeated ingredients."
    )
    
    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": surprise_prompt}],
        model="gemma2-9b-it",
    )
    
    st.markdown("### Surprise 3-Day Meal Plan 🍽️")
    st.markdown(clean_markdown_output(chat_completion.choices[0].message.content))

st.write("Note: LLMs may hallucinate and do not fully understand all dietary nuances.")

# Groq Branding
st.markdown(
    """
    <a href="https://groq.com" target="_blank" rel="noopener noreferrer">
      <img
        src="https://groq.com/wp-content/uploads/2024/03/PBG-mark1-color.svg"
        alt="Powered by Groq for fast inference."
        style="width: 200px;"
      />
    </a>
    """,
    unsafe_allow_html=True,
)
