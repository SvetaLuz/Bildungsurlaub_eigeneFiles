
import re
import sqlite3
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field


_dotenv_path = find_dotenv(usecwd=True) or find_dotenv()
if _dotenv_path:
    load_dotenv(_dotenv_path)


def _resolve_db_path(db_path: str | None = None) -> str:
    if db_path:
        return str(Path(db_path))

    env_path = (Path(str(Path.cwd())) / "coffee_sales.db")
    if env_path.exists():
        return str(env_path)

    local_path = Path(__file__).resolve().parent / "coffee_sales.db"
    return str(local_path)

#%% Pydantic SQL-Query Model
class SQLQuery(BaseModel):
    sql_query: str = Field(description="The SQL query to answer the user question.")

#%%
def fetch_information_from_db(query: str, db_path: str | None = None):
    conn = sqlite3.connect(_resolve_db_path(db_path))
    cursor = conn.cursor()
    try:
        cursor.execute(query)
        res = cursor.fetchall()
        conn.close()
        return res
    except Exception as e:
        conn.close()
        print(f"Error executing query: {e}")
        return []
#%%
def create_sql_query(user_query: str, sql_table_info: str):
    model = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a SQL expert and can create a SQL query to answer the user's question. You will be given a SQL table information and a user question. You will need to create a SQL query to answer the user question. You will return a JSON with an object sql_query"),
            ("user", "SQL table information: {sql_table_info} \n User question: {user_query}"),
        ]
    )
    chain = prompt | model | JsonOutputParser(pydantic_object=SQLQuery)
    return chain.invoke({"sql_table_info": sql_table_info, "user_query": user_query})


def validate_readonly_sql(sql: str) -> None:
    """Validate that the SQL is read-only and safe to execute.

    Allowed: SELECT or WITH ... SELECT.
    Blocked: multi-statement and destructive keywords.
    """
    if not sql or not sql.strip():
        raise ValueError("Empty SQL query")

    stripped = sql.strip()
    upper = stripped.upper()

    if not (upper.startswith("SELECT") or upper.startswith("WITH")):
        raise ValueError("Only SELECT queries are allowed")

    # Disallow multiple statements. Allow a single trailing semicolon.
    semi = stripped.find(";")
    if semi != -1 and stripped[semi:].strip() != ";":
        raise ValueError("Multiple SQL statements are not allowed")

    blocked = [
        "DROP",
        "DELETE",
        "UPDATE",
        "INSERT",
        "ALTER",
        "CREATE",
        "ATTACH",
        "DETACH",
        "PRAGMA",
        "VACUUM",
        "REINDEX",
        "TRUNCATE",
        "REPLACE",
    ]
    pattern = r"\\b(" + "|".join(blocked) + r")\\b"
    if re.search(pattern, upper):
        raise ValueError("Unsafe SQL keyword detected")

#%%
def rag(user_query: str, sql_table_info: str):
    # 1. Retrieval
    sql_query = create_sql_query(user_query, sql_table_info)
    print(sql_query['sql_query'])
    retrieved_information = fetch_information_from_db(query=sql_query['sql_query'])
    print(f"Retrieved Info: {retrieved_information}")
    # 2. Augmentation
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    # prepare chain
    messages = [
        ("system", """
        You are an expert on coffee and you can answer questions based on data from the table. please answer purely based on the retrieved information.
        Here is the information from the database call {retrieved_information}.
        The underlying SQL-query is {sql_query}.
        """),
        ("user", "{user_query}")
    ]
    prompt_template = ChatPromptTemplate.from_messages(messages)
    chain = prompt_template | llm | StrOutputParser()
    # 3. Generation
    model_response = chain.invoke({"retrieved_information": retrieved_information, "user_query": user_query, "sql_query": sql_query['sql_query']})
    return model_response


def generate_answer_from_retrieval(
    user_query: str,
    retrieved_information: list[tuple],
    sql_query: str,
):
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    messages = [
        (
            "system",
            """
        You are an expert on coffee and you can answer questions based on data from the table. please answer purely based on the retrieved information.
        Here is the information from the database call {retrieved_information}.
        The underlying SQL-query is {sql_query}.
        """,
        ),
        ("user", "{user_query}"),
    ]
    prompt_template = ChatPromptTemplate.from_messages(messages)
    chain = prompt_template | llm | StrOutputParser()
    return chain.invoke(
        {
            "retrieved_information": retrieved_information,
            "user_query": user_query,
            "sql_query": sql_query,
        }
    )


def rag_with_outputs(
    user_query: str,
    sql_table_info: str,
    db_path: str | None = None,
):
    """Return structured outputs for UIs: (sql_query, rows, answer)."""
    sql_result = create_sql_query(user_query, sql_table_info)
    sql_query = sql_result["sql_query"]
    validate_readonly_sql(sql_query)
    rows = fetch_information_from_db(query=sql_query, db_path=db_path)
    answer = generate_answer_from_retrieval(user_query, rows, sql_query)
    return sql_query, rows, answer


if __name__ == "__main__":
    # Minimal smoke demo when run directly; not executed on import.
    sql_table_info = """
    coffee_sales (
        sale_id INT PRIMARY KEY,
        product_name VARCHAR(100),
        origin_country VARCHAR(50),
        bean_type VARCHAR(50),
        roast_level VARCHAR(20),
        price_per_kg DECIMAL(6,2),
        quantity_kg DECIMAL(6,2),
        sale_date DATE,
        customer_id INT,
        region VARCHAR(50),
        organic BOOLEAN,
        certification VARCHAR(50)
    );
    """
    q = "SELECT name FROM sqlite_master WHERE type='table';"
    print(fetch_information_from_db(q))

