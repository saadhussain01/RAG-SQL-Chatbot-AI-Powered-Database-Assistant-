import pandas as pd
from db.connection import engine
from sqlalchemy import text

# Allowed keywords
SAFE_KEYWORDS = ["select"]

def is_safe_sql(query: str) -> bool:
    q = query.lower()
    forbidden = ["insert", "update", "delete", "drop", "alter", "truncate", "create", "replace"]
    return all(word not in q for word in forbidden)

def execute_sql(query: str):

    if not is_safe_sql(query):
        return None, "❌ Unsafe SQL blocked"

    try:
        with engine.connect() as conn:
            res = conn.execute(text(query))

            rows = res.fetchall()
            cols = res.keys()

            df = pd.DataFrame(rows, columns=cols)

            if df.empty:
                return None, "No results."

            return df, None

    except Exception as e:
        return None, f"SQL Error: {str(e)}"