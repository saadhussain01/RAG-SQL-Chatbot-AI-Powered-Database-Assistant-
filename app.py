import streamlit as st
from langchain_rag.graph import app_graph
import pandas as pd

# Page config
st.set_page_config(page_title="RAG SQL Bot", layout="wide")

st.title("🤖 Pet Store AI Assistant")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


# ===============================
# Display Chat History
# ===============================
for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):

        if msg["type"] == "text":
            st.markdown(msg["content"])

        elif msg["type"] == "product":

            st.image(msg["image"], width=250)

            st.markdown(f"### {msg['name']}")
            st.write(f"💰 Rs {msg['price']}")
            st.markdown(f"[🔗 View Product]({msg['url']})")

            st.divider()


# ===============================
# Show Product Grid (Recommendations)
# ===============================
def show_recommendations(data, title):

    if not data:
        return

    st.subheader(title)

    df = pd.DataFrame(data)

    cols = st.columns(3)

    for i, (_, row) in enumerate(df.iterrows()):

        with cols[i % 3]:

            img = row.get("image_url") or "https://via.placeholder.com/300"

            st.image(img, use_container_width=True)

            st.markdown(f"### {row.get('name','')}")

            if "price" in row:
                st.write(f"💰 Rs {row['price']}")

            if "product_url" in row:
                st.markdown(f"[🔗 View Product]({row['product_url']})")

            st.markdown("---")


# ===============================
# Chat Input
# ===============================
user_input = st.chat_input("Ask me about pets or products...")


if user_input:

    # Show user message
    st.session_state.messages.append({
        "role": "user",
        "type": "text",
        "content": user_input
    })

    with st.chat_message("user"):
        st.markdown(user_input)

    # Run LangGraph
    state = {"question": user_input}

    try:
        result = app_graph.invoke(state)
        output = result.get("result", [])

    except Exception as e:
        st.error(f"System Error: {e}")
        st.stop()


    # ===============================
    # CASE 1: Normal Chat Reply
    # ===============================
    if isinstance(output, list) and len(output) > 0 and output[0].get("type") == "text":

        reply = output[0]["value"]

        st.session_state.messages.append({
            "role": "assistant",
            "type": "text",
            "content": reply
        })

        with st.chat_message("assistant"):
            st.markdown(reply)


    # ===============================
    # CASE 2: Database Results
    # ===============================
    elif isinstance(output, list) and len(output) > 0:

        main_results = []
        recommendations = []

        # Split main + recommended
        for i, row in enumerate(output):

            if i < 3:
                main_results.append(row)
            else:
                recommendations.append(row)


        # Show Main Results
        for row in main_results:

            with st.chat_message("assistant"):

                # Image
                img = row.get("image_url") or "https://via.placeholder.com/300"
                st.image(img, width=250)

                # Name
                st.markdown(f"## 🐾 {row.get('name','')}")

                # Details
                if "species" in row:
                    st.write(f"**Species:** {row['species']}")

                if "breed" in row:
                    st.write(f"**Breed:** {row['breed']}")

                if "age" in row:
                    st.write(f"**Age:** {row['age']} years")

                if "price" in row:
                    st.write(f"💰 **Price:** Rs {row['price']}")

                if "status" in row:
                    st.write(f"📌 **Status:** {row['status']}")

                # Product link
                if "product_url" in row:
                    st.markdown(f"🔗 [View Product]({row['product_url']})")

                st.divider()


        # Save main results to history
        for row in main_results:

            st.session_state.messages.append({
                "role": "assistant",
                "type": "product",
                "name": row.get("name", ""),
                "price": row.get("price", ""),
                "image": row.get("image_url"),
                "url": row.get("product_url")
            })


        # ===============================
        # Recommendations Section
        # ===============================
        if recommendations:

            st.markdown("## ⭐ You May Also Like")

            show_recommendations(
                recommendations,
                "Recommended For You"
            )


    # ===============================
    # CASE 3: No Result
    # ===============================
    else:

        st.warning("❌ No data found. Try different keywords.")