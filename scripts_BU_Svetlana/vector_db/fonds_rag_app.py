"""
Streamlit GUI for Fonds RAG (FAISS) helper.

- Chat UI (like your movie example)
- Uses FAISS index (projekt_fonds_docs_chunks or env FONDS_FAISS_DIR)
- Uses OpenAIEmbeddings + ChatOpenAI
"""

# %% packages
import os
import streamlit as st
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# ---------------------------
# RAG backend (your logic)
# ---------------------------
embedding_model = OpenAIEmbeddings()
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _find_repo_root(start_dir: str) -> str:
    current = os.path.abspath(start_dir)
    while True:
        if os.path.exists(os.path.join(current, "pyproject.toml")) or os.path.exists(
            os.path.join(current, ".git")
        ):
            return current
        parent = os.path.dirname(current)
        if parent == current:
            return start_dir
        current = parent


def _looks_like_faiss_dir(path: str) -> bool:
    return (
        os.path.isdir(path)
        and os.path.exists(os.path.join(path, "index.faiss"))
        and os.path.exists(os.path.join(path, "index.pkl"))
    )


def _resolve_index_dir() -> str:
    env_path = os.getenv("FONDS_FAISS_DIR")
    if env_path:
        return env_path

    candidates = [
        os.path.join(_SCRIPT_DIR, "projekt_fonds_docs_chunks"),
        os.path.join(os.getcwd(), "projekt_fonds_docs_chunks"),
        os.path.join(_find_repo_root(_SCRIPT_DIR), "projekt_fonds_docs_chunks"),
    ]

    for c in candidates:
        if _looks_like_faiss_dir(c):
            return c

    return os.path.join(_SCRIPT_DIR, "projekt_fonds_docs_chunks")


@st.cache_resource(show_spinner=False)
def _get_vectordb_cached():
    index_dir = _resolve_index_dir()
    return FAISS.load_local(
        index_dir, embeddings=embedding_model, allow_dangerous_deserialization=True
    )


def _retrieve_context(user_query: str, k: int = 3):
    vectordb = _get_vectordb_cached()
    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": k})
    docs = retriever.invoke(input=user_query)  # -> list[Document]
    context_str = "; ".join([d.page_content for d in docs])
    return context_str, docs


def rag(user_query: str, k: int = 3) -> str:
    context_str, _docs = _retrieve_context(user_query=user_query, k=k)

    message = [
        (
            "system",
            """
Du bist ein Fonds-Spezialist.
Du lieferst fundsspezifische Produktinformationen.
Antworte ausschließlich auf Basis der Kontextinfos.
Wenn die Antwort nicht so konfident ist, sag: 'Konfidenzgrad gering, bitte überprüfen'.
Wenn der Kontext die Frage nicht beantworten lässt, sag: 'die Frage kann nicht anhand der Kontextinformationen beantwortet werden'.
""",
        ),
        ("user", "Userquery: {userquery}\n\nKontextinformationen: {context_info}"),
    ]
    prompt_template = ChatPromptTemplate.from_messages(message)

    model = ChatOpenAI(model="gpt-5.2")
    chain = prompt_template | model | StrOutputParser()

    return chain.invoke({"userquery": user_query, "context_info": context_str})


# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Fonds RAG Chat", page_icon="📄", layout="wide")
st.title("📄 Fonds Info Helper")

with st.sidebar:
    st.header("Einstellungen")
    k = st.slider("Anzahl Kontext-Chunks (k)", min_value=1, max_value=10, value=3, step=1)
    show_context = st.checkbox("Kontext-Chunks anzeigen", value=True)
    show_index_path = st.checkbox("Index-Pfad anzeigen", value=False)

    if show_index_path:
        st.caption("Aufgelöster Index-Pfad:")
        st.code(_resolve_index_dir())

# Chat state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

user_query = st.chat_input("Stelle eine Frage zu einem Fonds (z.B. 'Wie hoch ist das Fondsvolumen von ...?')")

if user_query:
    # user message
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # assistant response
    with st.chat_message("assistant"):
        with st.spinner("Suche Kontext & generiere Antwort..."):
            try:
                context_str, docs = _retrieve_context(user_query, k=k)
                answer = rag(user_query, k=k)

                st.markdown(answer)

                if show_context:
                    st.divider()
                    st.subheader("🔎 Verwendete Kontext-Chunks")
                    for i, d in enumerate(docs, start=1):
                        meta = d.metadata or {}
                        source = meta.get("source") or meta.get("file_path") or meta.get("path") or ""
                        page = meta.get("page", meta.get("page_number", ""))
                        header = f"Chunk {i}"
                        if source or page != "":
                            header += f" — {source} {(' / Seite ' + str(page)) if page != '' else ''}"
                        with st.expander(header, expanded=(i == 1)):
                            st.write(d.page_content)

            except Exception as e:
                st.error(f"Fehler: {e}")
                answer = f"Fehler: {e}"

    st.session_state.messages.append({"role": "assistant", "content": answer})
