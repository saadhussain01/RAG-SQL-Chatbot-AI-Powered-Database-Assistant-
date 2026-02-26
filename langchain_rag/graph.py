from langgraph.graph import StateGraph, END

from langchain_rag.llm import llm
from langchain_rag.embeddings import build_vectorstore
from utils.sql_utils import execute_sql


# Build vectorstore once
vectorstore = build_vectorstore()


class ChatState(dict):
    question: str
    context: str
    sql: str
    result: str


# -------------------------
# Retrieve Schema
# -------------------------
def retrieve_context(state):

    docs = vectorstore.similarity_search(
        state["question"],
        k=1
    )

    context = "\n".join(d.page_content for d in docs)

    print("\n=== RETRIEVED CONTEXT ===")
    print(context)

    return {"context": context}


# -------------------------
# Generate SQL
# -------------------------
def generate_sql(state):

    prompt = f"""
    You are an expert MySQL database engineer and data analyst.

    Your job:
    Convert the user's natural language question into a valid MySQL SELECT query.

    You must think carefully about:
    - User intent
    - Relationships between tables
    - Required joins
    - Filters
    - Aggregations
    - Sorting
    - Limits

    ==================================================

    DATABASE SCHEMA:

    {state["context"]}

    ==================================================

    RULES (VERY IMPORTANT):

    1. ONLY generate SELECT queries.
    2. NEVER use INSERT, UPDATE, DELETE, DROP, ALTER, CREATE.
    3. NEVER explain anything.
    4. NEVER use markdown.
    5. Return ONLY pure SQL.
    6. Always use correct column names.
    7. Use JOINs when data is in multiple tables.
    8. If images or links are relevant, include image_url and product_url.
    9. If user asks for "best", "top", "highest", use ORDER BY + LIMIT.
    10. If user asks for "cheap", "lowest", use ORDER BY ASC.
    11. If user asks for "recent", use ORDER BY date DESC.
    12. If user asks for "total", "sum", use aggregation.
    13. If user asks something unclear, make the best logical assumption.

    ==================================================

    USER QUESTION:
    {state["question"]}

    ==================================================

    THINK INTERNALLY ABOUT THE QUERY.

    Then output ONLY the SQL query.

    SQL:
    """

    sql = llm.invoke(prompt)

    print("=== RAW LLM OUTPUT ===")
    print(sql)


    if not sql or len(sql.strip()) < 5:
        print("❌ Empty SQL from LLM")

        return {
            "sql": "",
            "result": "LLM failed to generate SQL"
        }

    # Safety cleanup
    sql = sql.strip()

    if not sql.endswith(";"):
        sql += ";"

    return {"sql": sql}


# -------------------------
# Run SQL
# -------------------------
def run_sql(state):

    sql = state.get("sql", "").strip()

    if not sql:
        return {"result": None, "error": "No SQL generated."}

    df, error = execute_sql(sql)

    return {
        "result": df,
        "error": error
    }

# -------------------------
# Graph
# -------------------------
graph = StateGraph(ChatState)

graph.add_node("retrieve", retrieve_context)
graph.add_node("generate", generate_sql)
graph.add_node("execute", run_sql)

graph.set_entry_point("retrieve")

graph.add_edge("retrieve", "generate")
graph.add_edge("generate", "execute")
graph.add_edge("execute", END)

app_graph = graph.compile()

__all__ = ["app_graph"]