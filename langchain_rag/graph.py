from langgraph.graph import StateGraph, END
from langchain_rag.llm import llm
from langchain_rag.embeddings import build_vectorstore
from utils.sql_utils import execute_sql

# Build vectorstore
vectorstore = build_vectorstore()


class ChatState(dict):
    question: str
    context: str
    sql: str
    result: list
    mode: str   # "chat" or "db"


# -------------------------
# Detect Intent
# -------------------------
def detect_intent(state):

    q = state["question"].lower()

    greetings = ["hi", "hello", "hey", "how are you"]

    if any(word in q for word in greetings):
        return {"mode": "chat"}

    return {"mode": "db"}


def extract_keyword(question):

    stopwords = ["show", "find", "me", "all", "the", "a", "an", "please"]

    words = question.lower().split()

    keywords = [w for w in words if w not in stopwords]

    return keywords[-1] if keywords else ""


# -------------------------
# Normal Chat
# -------------------------
def normal_chat(state):

    prompt = f"""
You are a friendly chatbot.

User: {state['question']}

Reply naturally. Also encourage user to ask about pets/products.
"""

    reply = llm.invoke(prompt)

    return {"result": [{"type": "text", "value": reply}]}


# -------------------------
# Retrieve Schema
# -------------------------
def retrieve_context(state):

    docs = vectorstore.similarity_search(
        state["question"],
        k=2
    )

    context = "\n".join(d.page_content for d in docs)

    return {"context": context}


# -------------------------
# Generate SQL
# -------------------------
def generate_sql(state):

    prompt = f"""
You are a professional MySQL expert.

Database schema:
{state.get("context","")}

Rules:
- Write only SELECT queries
- No explanation
- No markdown
- Always use LIKE with %
- Search in: name, category, species, breed

User question:
{state["question"]}

Examples:

show accessories
SELECT * FROM products
WHERE category LIKE '%accessories%';

show dog products
SELECT * FROM products
WHERE name LIKE '%dog%'
OR category LIKE '%dog%';

show persian cat
SELECT * FROM pets
WHERE breed LIKE '%persian%';

Now write SQL:

SQL:
"""

    sql = llm.invoke(prompt)

    print("=== GENERATED SQL ===")
    print(sql)

    return {"sql": sql.strip()}


# -------------------------
# Run SQL
# -------------------------
def run_sql(state):

    question = state.get("question", "").lower()
    sql = state.get("sql", "").strip()

    words = question.split()
    main_word = words[-1] if words else ""

    if not sql or len(sql) < 10:

        sql = f"""
        SELECT id, name, price, image_url, product_url
        FROM pets
        WHERE
            name LIKE '%{main_word}%'
            OR species LIKE '%{main_word}%'
            OR breed LIKE '%{main_word}%'

        UNION

        SELECT id, name, price, image_url, product_url
        FROM products
        WHERE
            name LIKE '%{main_word}%'
            OR category LIKE '%{main_word}%'
        """

    df, error = execute_sql(sql)

    if error:
        return {"error": error}

    # Recommendations
    rec_sql = f"""
    SELECT id, name, price, image_url, product_url
    FROM products
    ORDER BY RAND()
    LIMIT 4
    """

    rec_df, _ = execute_sql(rec_sql)

    return {
        "data": df,
        "recommendations": rec_df
    }

# -------------------------
# Router
# -------------------------
def router(state):

    if state["mode"] == "chat":
        return "chat"

    return "db"


# -------------------------
# Build Graph
# -------------------------
graph = StateGraph(ChatState)

graph.add_node("intent", detect_intent)
graph.add_node("chat", normal_chat)

graph.add_node("retrieve", retrieve_context)
graph.add_node("generate", generate_sql)
graph.add_node("execute", run_sql)

graph.set_entry_point("intent")

graph.add_conditional_edges(
    "intent",
    router,
    {
        "chat": "chat",
        "db": "retrieve"
    }
)

graph.add_edge("chat", END)

graph.add_edge("retrieve", "generate")
graph.add_edge("generate", "execute")
graph.add_edge("execute", END)

app_graph = graph.compile()

__all__ = ["app_graph"]