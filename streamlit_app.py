import streamlit as st
from groq import Groq

# Show title and description.
st.title("💬 Chatbot")
st.write(
    "This is a simple chatbot that uses OpenAI's GPT-3.5 model to generate responses. "
    "To use this app, you need to provide an OpenAI API key, which you can get [here](https://platform.openai.com/account/api-keys). "
    "You can also learn how to build this app step by step by [following our tutorial](https://docs.streamlit.io/develop/tutorials/llms/build-conversational-apps)."
)


client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "Can you give me a 3 recipe ideas using Grain-Finished Beef Stew that is Kosher, high protein and low in carbs and sugar",
        }
    ],
    model="llama3-8b-8192",
)

print(chat_completion.choices[0].message.content)

st.markdown(chat_completion.choices[0].message.content)