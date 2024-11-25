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

# Streamlit app
st.title("Recipe Idea Generator 🍴")
st.write("Select an item from the table to get recipe ideas or click the 'Surprise Meal Plan' button for inspiration!")

# Display the table for user selection
selected_item = st.selectbox(
    "Choose an item:", options=df["display"], index=0
)

# Button to trigger Groq API call for selected item
if st.button("Get Recipe Ideas"):
    # Extract the title from the selected item
    selected_title = selected_item.split(" - ")[0]
    
    # Call Groq API
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": f"You are an expert nutritionist - can you give me 3 recipe ideas using {selected_title} that are Kosher, high protein, and low in carbs and sugar. Which means no dairy and meat in the same meal. Also please give an easy to digest shopping list at the bottom",
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
        messages=[
            {
                "role": "user",
                "content": f"Can you give me a 3-day lunch and Dinner meal plan using {random_items_str} that is Kosher, high protein, and low in carbs and sugar In a table format?",
            }
        ],
        model="llama3-8b-8192",
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
