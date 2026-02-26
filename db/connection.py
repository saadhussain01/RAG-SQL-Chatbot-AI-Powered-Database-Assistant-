from sqlalchemy import create_engine, text
from config import settings

DB_URL = (
    f"mysql+pymysql://{settings.MYSQL_USER}:"
    f"{settings.MYSQL_PASSWORD}@{settings.MYSQL_HOST}:"
    f"{settings.MYSQL_PORT}/{settings.MYSQL_DB}"
)

engine = create_engine(DB_URL)
print("✅ Database connection established")
