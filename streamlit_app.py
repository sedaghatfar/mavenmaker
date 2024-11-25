import streamlit as st
from groq import Groq
import pandas as pd
import random

# Load your CSV
data_file = "meatmaven.csv"
df = pd.read_csv(data_file)

# Combine title and price for display
df["display"] = df["title"] + " - " + df["price"].astype(str)

# Initialize Groq client
client = Groq(
    api_key=st.secrets["GROQ_API_KEY"],
)

# -- Sidebar for family size
st.sidebar.markdown("## How big is your family?")
family_size = st.sidebar.slider("Family Size", 1.0, 10.0, 4.0, 0.5)

# Streamlit app
st.title("Recipe Idea Generator 🍴")
st.write(
    "Select an item from the table to get recipe ideas or click the 'Surprise Meal Plan' button for inspiration!"
)

# Display the table for user selection
selected_item = st.selectbox("Choose an item:", options=df["display"], index=0)

# Button to trigger Groq API call for selected item
if st.button("Get Recipe Ideas"):
    # Extract the title from the selected item
    selected_title = selected_item.split(" - ")[0]
    
    # Call Groq API
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": f"You are an expert nutritionist - can you give me 3 recipe ideas using {selected_title} that are Kosher, Dairy free, high protein, and low in carbs and sugar, No cheeses. In a table format also please give what the grocery list should be for a family of 4",
            }
        ],
        model="llama3-8b-8192",
    )
    
    # Display the response
    st.markdown(chat_completion.choices[0].message.content)

# Surprise Meal Plan button
if st.button("Surprise Meal Plan"):
    # Randomly select 3 items from the CSV
    random_items = random.sample(list(df["title"]), 3)
    random_items_str = ", ".join(random_items)
    
    # Call Groq API for the meal plan
    chat_completion = client.chat.completions.create(
        messages = [
            {
                "role": "user",
                "content": (
                    "Act as an expert nutritionist. Please create a Kosher, dairy-free, "
                    "high protein low carb lunch and dinner plan for a family of {family_size} "
                    "using {selected_title} as the mains. Then create an easy-to-read grocery list. "
                    "Make batches that are easy to reheat. and use the following format as a style guide\n\n"
                    "Day | Lunch | Dinner\n"
                    "1 | Salsa Chicken with black beans and rice | Lean shoulder Burgers\n"
                    "2 | Salsa Chicken with black beans and rice | London Broil Fajita-Style with Sauteed Onions and Bell Peppers\n"
                    "3 | Pasta Chicken (cubes) and mixed vegetables | London Broil Fajita-Style with Sauteed Onions and Bell Peppers\n\n"
                    "Shopping list:\n"
                    "- Meat Maven Marinated London Broil 2 lbs\n"
                    "- Chicken Cubes - Nuggets  - 1.5 lbs\n"
                    "- Grain-Finished Ground Shoulder - Super Lean 2 lbs\n"
                    "- Chicken Cutlets - Family Pack 2 lbs\n\n"
                    "- Salsa 16oz\n"
                    "- dry black beans\n"
                    "- 2 cups basmati rice\n"
                    "- frozen mixed vegetables\n"
                    "- 2 Onions\n"
                    "- 2 Bell peppers\n"
                    "- 1 pack Burger Buns"
        ).format(family_size=family_size, selected_title=selected_title)
    }
],
        model="gemma2-9b-it",
    )
    
    # Display the meal plan response
    st.markdown("### Surprise 3-Day Meal Plan 🍽️")
    st.markdown(chat_completion.choices[0].message.content)

# Add Groq branding at the bottom
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
