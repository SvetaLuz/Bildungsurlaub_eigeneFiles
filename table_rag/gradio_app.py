from __future__ import annotations

import sqlite3
import sys
from pathlib import Path
from typing import Any

import gradio as gr
import pandas as pd

# Ensure local imports work when executed from repo root
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from table_rag import rag_with_outputs  # noqa: E402


SQL_TABLE_INFO = """
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
""".strip()


def _resolve_db_path(db_path: str | None = None) -> str:
    if db_path:
        return str(Path(db_path))

    cwd_db = Path.cwd() / "coffee_sales.db"
    if cwd_db.exists():
        return str(cwd_db)

    local_db = _THIS_DIR / "coffee_sales.db"
    return str(local_db)


def _get_column_names(sql: str, db_path: str) -> list[str]:
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute(sql)
        if cursor.description is None:
            return []
        return [d[0] for d in cursor.description]
    finally:
        conn.close()


def _rows_to_dataframe(rows: list[tuple], columns: list[str]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    if columns and len(columns) == len(rows[0]):
        return pd.DataFrame(rows, columns=columns)
    return pd.DataFrame(rows)


def run_question(question: str, db_path: str, progress: gr.Progress = gr.Progress()):
    """Gradio callback: returns (sql, df, answer, error)."""

    question = (question or "").strip()
    if not question:
        return "", pd.DataFrame(), "", "Bitte eine Frage eingeben."

    resolved_db = _resolve_db_path(db_path.strip() if db_path else None)
    if not Path(resolved_db).exists():
        return "", pd.DataFrame(), "", f"❌ Datenbank nicht gefunden: {resolved_db}"

    try:
        progress(0.1, desc="Erzeuge SQL...")
        sql, rows, answer = rag_with_outputs(question, SQL_TABLE_INFO, db_path=resolved_db)

        progress(0.6, desc="Formatiere Resultset...")
        columns = _get_column_names(sql, resolved_db)
        df = _rows_to_dataframe(rows, columns)

        progress(0.9, desc="Fertig")
        return sql, df, answer, ""
    except Exception as e:
        return "", pd.DataFrame(), "", f"❌ Fehler: {type(e).__name__}: {e}"


def build_app() -> gr.Blocks:
    with gr.Blocks() as demo:
        gr.Markdown(
            """
# ☕ Coffee Sales Table-RAG (Gradio)

Stelle eine Frage in natürlicher Sprache. Die App zeigt:
- den generierten SQL-Query,
- die Rohdaten aus SQLite,
- die finale Antwort.

**Hinweis:** Es werden nur read-only SQL-Queries erlaubt (SELECT / WITH ... SELECT).
""".strip()
        )

        with gr.Row():
            question = gr.Textbox(
                label="Frage",
                placeholder="z.B. What's the average price of organic coffee compared to non-organic?",
                lines=3,
            )

        with gr.Row():
            db_path = gr.Textbox(
                label="DB-Pfad (optional)",
                placeholder="Leer lassen für Standard: coffee_sales.db",
                value="",
            )
            submit = gr.Button("Ask Question", variant="primary")

        sql_out = gr.Code(label="Generated SQL", language="sql")
        df_out = gr.Dataframe(label="Database Results", interactive=False, wrap=True)
        answer_out = gr.Markdown(label="AI Answer")
        error_out = gr.Markdown(label="Errors")

        gr.Examples(
            examples=[
                "What is the total sales over all time?",
                "What's the average price of organic coffee compared to non-organic?",
                "How does the price depend on the roast level?",
                "Which country has the highest sales?",
            ],
            inputs=question,
            label="Examples",
        )

        submit.click(
            fn=run_question,
            inputs=[question, db_path],
            outputs=[sql_out, df_out, answer_out, error_out],
        )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch()
