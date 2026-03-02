import pandas as pd
from db.connection import engine
from sqlalchemy import text


def is_safe_sql(query):
    forbidden = ["insert", "update", "delete", "drop", "alter"]
    return not any(word in query.lower() for word in forbidden)


def execute_sql(query: str):

    if not is_safe_sql(query):
        return None, "❌ Unsafe SQL blocked"

    try:
        with engine.connect() as conn:
            res = conn.execute(text(query))
            rows = res.fetchall()
            cols = res.keys()

            df = pd.DataFrame(rows, columns=cols)

            return df, None

    except Exception as e:
        return None, str(e)