import streamlit as st
from groq import Groq
import pandas as pd

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
st.write("Select an item from the table to get recipe ideas!")

# Display the table for user selection
selected_item = st.selectbox(
    "Choose an item:", options=df["display"], index=0
)

# Button to trigger Groq API call
if st.button("Get Recipe Ideas"):
    # Extract the title from the selected item
    selected_title = selected_item.split(" - ")[0]
    
    # Call Groq API
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": f"Can you give me 3 recipe ideas using {selected_title} that are Kosher, high protein, and low in carbs and sugar?",
            }
        ],
        model="llama3-8b-8192",
    )
    
    # Display the response
    st.markdown(chat_completion.choices[0].message.content)
