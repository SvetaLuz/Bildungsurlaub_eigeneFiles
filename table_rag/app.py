import streamlit as st
import sqlite3
import pandas as pd
import os
from table_rag import fetch_information_from_db, create_sql_query, rag, SQLQuery

# Page configuration
st.set_page_config(
    page_title="Coffee Sales RAG Query System",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Table schema information
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
"""

def validate_sql_query(query: str) -> bool:
    """Ensure the query is a SELECT statement only."""
    query_upper = query.strip().upper()
    return query_upper.startswith('SELECT')

def get_column_names(query: str) -> list:
    """Get column names from the database query."""
    conn = sqlite3.connect('coffee_sales.db')
    cursor = conn.cursor()
    try:
        cursor.execute(query)
        columns = [description[0] for description in cursor.description]
        conn.close()
        return columns
    except Exception:
        conn.close()
        return []

def format_results(results: list, columns: list) -> pd.DataFrame:
    """Convert query results to a pandas DataFrame."""
    if not results or not columns:
        return pd.DataFrame()
    return pd.DataFrame(results, columns=columns)

# Check if database exists
DB_PATH = 'coffee_sales.db'
if not os.path.exists(DB_PATH):
    st.error(f"❌ Database file '{DB_PATH}' not found. Please ensure the database exists in the current directory.")
    st.stop()

# Initialize session state
if 'query_history' not in st.session_state:
    st.session_state.query_history = []

# Sidebar
with st.sidebar:
    st.title("☕ Coffee Sales RAG")
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This application uses **Retrieval-Augmented Generation (RAG)** 
    to answer questions about coffee sales data.
    
    Ask questions in natural language, and the system will:
    1. Generate a SQL query
    2. Retrieve data from the database
    3. Generate a natural language answer
    """)
    st.markdown("---")
    st.markdown("### Example Questions")
    st.markdown("""
    - What is the total sales over all time?
    - What's the average price of organic coffee compared to non-organic?
    - How does the price depend on the roast level?
    - Which country has the highest sales?
    """)
    st.markdown("---")
    st.markdown("### Database Schema")
    with st.expander("View Schema"):
        st.code(SQL_TABLE_INFO, language="sql")

# Main content
st.title("☕ Coffee Sales Query System")
st.markdown("Ask questions about coffee sales data using natural language.")

# Query input section
col1, col2 = st.columns([4, 1])
with col1:
    user_query = st.text_area(
        "Enter your question:",
        placeholder="e.g., What is the total sales over all time?",
        height=100,
        key="query_input"
    )
with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    submit_button = st.button("🔍 Ask Question", type="primary", use_container_width=True)

# Process query
if submit_button and user_query:
    if not user_query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Processing your question..."):
            try:
                # Step 1: Generate SQL query
                with st.expander("🔧 Step 1: Generating SQL Query", expanded=True):
                    try:
                        sql_result = create_sql_query(user_query, SQL_TABLE_INFO)
                        generated_sql = sql_result['sql_query']
                        
                        # Validate SQL query
                        if not validate_sql_query(generated_sql):
                            st.error("⚠️ Security: Only SELECT queries are allowed!")
                            st.stop()
                        
                        st.markdown("**Generated SQL Query:**")
                        st.code(generated_sql, language="sql")
                    except Exception as e:
                        st.error(f"Error generating SQL query: {str(e)}")
                        st.stop()
                
                # Step 2: Execute query and retrieve results
                with st.expander("📊 Step 2: Database Results", expanded=True):
                    try:
                        results = fetch_information_from_db(generated_sql)
                        
                        if results:
                            columns = get_column_names(generated_sql)
                            df = format_results(results, columns)
                            
                            st.markdown(f"**Retrieved {len(results)} row(s):**")
                            st.dataframe(df, use_container_width=True)
                            
                            # Show summary statistics for numeric columns
                            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
                            if len(numeric_cols) > 0:
                                with st.expander("📈 Summary Statistics"):
                                    st.dataframe(df[numeric_cols].describe(), use_container_width=True)
                        else:
                            st.info("No results found.")
                            retrieved_info = []
                    except Exception as e:
                        st.error(f"Error executing query: {str(e)}")
                        st.stop()
                        retrieved_info = []
                
                # Step 3: Generate natural language answer
                with st.expander("💬 Step 3: AI-Generated Answer", expanded=True):
                    try:
                        answer = rag(user_query, SQL_TABLE_INFO)
                        st.markdown("**Answer:**")
                        st.success(answer)
                    except Exception as e:
                        st.error(f"Error generating answer: {str(e)}")
                
                # Add to query history
                st.session_state.query_history.append({
                    'query': user_query,
                    'sql': generated_sql,
                    'timestamp': pd.Timestamp.now()
                })
                
            except Exception as e:
                st.error(f"An unexpected error occurred: {str(e)}")
                st.exception(e)

# Query history section
if st.session_state.query_history:
    st.markdown("---")
    st.subheader("📜 Query History")
    
    # Show last 5 queries
    recent_queries = st.session_state.query_history[-5:]
    for idx, item in enumerate(reversed(recent_queries), 1):
        with st.expander(f"Query #{len(st.session_state.query_history) - len(recent_queries) + idx}: {item['query'][:50]}..."):
            st.markdown(f"**Question:** {item['query']}")
            st.markdown("**SQL Query:**")
            st.code(item['sql'], language="sql")
    
    if st.button("Clear History"):
        st.session_state.query_history = []
        st.rerun()

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>Powered by LangChain, Groq (Llama 3.3), and Streamlit</div>",
    unsafe_allow_html=True
)

