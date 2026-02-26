import os
from dotenv import load_dotenv
load_dotenv()


import streamlit as st
from langchain_rag.graph import app_graph

st.set_page_config(page_title="RAG SQL Chatbot", layout="wide")
st.title("🤖 RAG SQL Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Show chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

def show_products(df):

    for _, row in df.iterrows():

        col1, col2 = st.columns([1, 3])

        # Image (if exists)
        with col1:
            if "image_url" in df.columns and row["image_url"]:
                st.image(row["image_url"], width=150)
            else:
                st.image("https://via.placeholder.com/150", width=150)

        # Info
        with col2:

            # Name
            if "name" in df.columns:
                st.markdown(f"### {row['name']}")

            # Dynamically show fields
            for col in df.columns:

                if col in ["image_url", "name"]:
                    continue

                value = row[col]

                st.write(f"**{col.capitalize()}**: {value}")

        st.divider()

# User input
user_input = st.chat_input("Ask your database...")

if user_input:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Prepare state
    state = {"question": user_input}

    # Initialize response safely
    response = None
    try:
        response = app_graph.invoke(state)
    except RuntimeError as e:
        if "StopIteration" in str(e):
            # Generator ended correctly; ignore
            pass
        else:
            st.error(f"Runtime error: {e}")
    except Exception as e:
        st.error(f"Unexpected error: {e}")

    # Safely access response
    if isinstance(response, dict):
        sql_output = response.get("sql")
        df = response.get("result")
        error = response.get("error")
    else:
        st.error("No response from model")

    # Format assistant message

    # st.markdown("### 🧠 Generated SQL")
    # st.code(sql_output, language="sql")

    if error:
        st.error(error)

    elif df is not None:
        show_products(df)

    else:
        st.warning("No data found.")
