from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import os


SCHEMA_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "schema.txt"
)


def get_schema_text():
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        return f.read()


def build_vectorstore():

    schema_text = get_schema_text()

    if not schema_text.strip():
        raise ValueError("schema.txt is empty!")

    docs = [Document(page_content=schema_text)]

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(docs, embeddings)

    return vectorstore