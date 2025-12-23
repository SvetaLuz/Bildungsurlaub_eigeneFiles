#%% packages
import os
import re
import hashlib
from datetime import datetime, timezone
from langchain.document_loaders import TextLoader
 
script_path = os.path.abspath(__file__)
current_path = os.path.dirname(script_path)
current_path
 

 
#%% load a single file from web
# from langchain_community.document_loaders import WebBaseLoader
# url = "https://www.gutenberg.org/cache/epub/46/pg46.txt"
# loader = WebBaseLoader(web_path=url)
# docs = loader.load()
 
#%%
# docs[0].metadata
 
#%% load multiple file from a folder
from langchain.document_loaders import DirectoryLoader
# loader = DirectoryLoader(path= os.path.join(current_path, "data"), loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"})
# docs = loader.load()
# docs
 
#%% Wikipedia
# ladet 5 Artikel zum Thema "Generative KI" herunter
# nutzt den entsprechenden Loader von Langchain
# from langchain.document_loaders import WikipediaLoader
# loader = WikipediaLoader(query="Generative KI", load_max_docs=5, lang="de")
# docs = loader.load()
#%%
# docs[0].metadata
 
#%% load multiple files from a folder of different types
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_community.document_loaders import UnstructuredPowerPointLoader


PROJECT_DOCS_DIRNAME = "projekt_fonds_docs"


def _safe_str(value) -> str:
    if value is None:
        return ""
    return str(value)


def _normalize_path(path: str) -> str:
    # Deterministic normalization across Windows/Unix separators.
    return os.path.normpath(path).replace("\\", "/")


def infer_fonds_id(source_path: str, base_dir: str) -> str:
    """Infer fonds_id from folder structure under projekt_fonds_docs.

    Expected layouts:
      - projekt_fonds_docs/<fonds_id>/<file>
      - projekt_fonds_docs/<file>  (fallback: filename stem)
    """
    try:
        rel = os.path.relpath(source_path, base_dir)
    except Exception:
        rel = source_path
    rel_norm = _normalize_path(rel)
    parts = [p for p in rel_norm.split("/") if p and p not in (".", "..")]
    if len(parts) >= 2:
        return parts[0]
    stem = os.path.splitext(os.path.basename(source_path))[0]
    return stem


def compute_doc_id(source_path: str, base_dir: str) -> str:
    """Stable doc_id: sha1 of relative path under base_dir."""
    try:
        rel = os.path.relpath(source_path, base_dir)
    except Exception:
        rel = source_path
    rel_norm = _normalize_path(rel).lower()
    return hashlib.sha1(rel_norm.encode("utf-8")).hexdigest()[:16]


def infer_fonds_name(source_path: str, base_dir: str, fonds_id: str) -> str:
    """Infer a human-friendly fonds name.

    Heuristics:
      - If docs are stored under projekt_fonds_docs/<fonds_id>/..., use fonds_id as name.
      - Otherwise derive from filename, stripping common suffix patterns like codes/ISIN.
    """
    # Folder-based layout: folder name is typically already a good display name.
    try:
        rel = os.path.relpath(source_path, base_dir)
        rel_norm = _normalize_path(rel)
        parts = [p for p in rel_norm.split("/") if p and p not in (".", "..")]
        if len(parts) >= 2:
            return fonds_id
    except Exception:
        pass

    filename = os.path.splitext(os.path.basename(source_path))[0]
    # Common pattern in your files: "<Name> _ <Code> _ <ISIN>"
    if " _ " in filename:
        return filename.split(" _ ")[0].strip()
    # Fallback: drop trailing ISIN-like tokens or short codes separated by spaces.
    cleaned = re.sub(r"\s+([A-Z]{2}[A-Z0-9]{10}|[A-Z0-9]{4,8})\s*$", "", filename).strip()
    return cleaned or fonds_id


def file_modified_date(source_path: str) -> str | None:
    try:
        mtime = os.path.getmtime(source_path)
    except Exception:
        return None
    dt = datetime.fromtimestamp(mtime, tz=timezone.utc)
    return dt.date().isoformat()


_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def naive_short_summary(text: str, max_chars: int = 320) -> str:
    """Heuristic 1–2 sentence summary (LLM-free)."""
    cleaned = " ".join(_safe_str(text).split())
    if not cleaned:
        return ""
    sentences = _SENT_SPLIT_RE.split(cleaned)
    summary = " ".join(sentences[:2]).strip()
    if not summary:
        summary = cleaned
    if len(summary) > max_chars:
        summary = summary[: max_chars - 1].rstrip() + "…"
    return summary


def naive_content_type(text: str) -> str:
    """Simple keyword-based classification. Replace with LLM classification if needed."""
    t = _safe_str(text).lower()
    if any(k in t for k in ("definition", "begriff", "ist ", "bedeutet")):
        return "Definition"
    if any(k in t for k in ("vorgehensweise", "schritt", "anleitung", "workflow", "prozess")):
        return "Vorgehensweise"
    if any(k in t for k in ("markt", "prognose", "ausblick", "szenario", "entwicklung")):
        return "Marktanalyse"
    if any(k in t for k in ("risiko", "volatil", "verlust", "drawdown")):
        return "Risikohinweis"
    if any(k in t for k in ("anlagepolitik", "strategie", "ziel", "investment", "benchmark")):
        return "Anlagestrategie"
    return "Unklassifiziert"
 
 
def select_loader(file_path: str):
    # extract file extension from file_path
    _, ext = os.path.splitext(file_path)
 
    # use different loaders depending on extension
    if ext == ".txt":
        return TextLoader(file_path=file_path, encoding="utf-8")
   
    if ext == ".pdf":
        return PyMuPDF4LLMLoader(file_path=file_path, mode="single")
    if ext == ".pptx":
        return UnstructuredPowerPointLoader(file_path= file_path)
 
    raise ValueError(f"Unsupported Filetype: {ext} for {file_path}")
 
loader = DirectoryLoader(
    path=os.path.join(current_path, "projekt_fonds_docs"),
    #path=os.path.join(current_path),
    loader_cls=select_loader,
    show_progress=True,
    use_multithreading=True,
    exclude="*.py"
)
docs = loader.load()
docs
# Metadaten von docs ansehen
print(f"Anzahl der geladenen Dokumente: {len(docs)}")
for i, doc in enumerate(docs):
    print(f"\nDokument {i}:")
    print(f"Metadaten: {doc.metadata}")
    print(f"Inhaltslänge: {len(doc.page_content)} Zeichen")
 
# %%
 
 
#%% Data Chunking
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
base_docs_dir = os.path.join(current_path, PROJECT_DOCS_DIRNAME)
splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=3000 / 5)
# [Document, Document]  -> längere [Document, Document]
docs_splitted = splitter.split_documents(docs)
# Anzahl chunks
len(docs_splitted)
 
#%% size of split
doc_sizes = [len(doc.page_content) for doc in docs_splitted]
import seaborn as sns
sns.histplot(doc_sizes)
 
#%%
from pprint import pprint
pprint(docs[0].page_content)

# --- Chunk-Metadaten anreichern (vor Embedding) ---
# Zielstruktur (pro Chunk) in doc.metadata:
#   - chunk_id (stabil)
#   - doc_id (stabil)
#   - fonds_id
#   - source_path (voll)
#   - source_name (Dateiname)
#   - location ("Seite X" oder "Char a-b")
#   - date (YYYY-MM-DD, file modified)
#   - content_type (heuristisch)
#   - short_summary (heuristisch)

doc_chunk_counters: dict[str, int] = {}
chunk_ids: list[str] = []
chunk_metadatas: list[dict] = []

for chunk in docs_splitted:
    source_path = _safe_str(chunk.metadata.get("source") or chunk.metadata.get("file_path") or "")
    source_path_abs = source_path
    if source_path_abs and not os.path.isabs(source_path_abs):
        # DirectoryLoader usually provides abs; this keeps behavior stable.
        source_path_abs = os.path.abspath(source_path_abs)

    fonds_id = infer_fonds_id(source_path_abs or source_path, base_docs_dir)
    fonds_name = infer_fonds_name(source_path_abs or source_path, base_docs_dir, fonds_id)
    doc_id = compute_doc_id(source_path_abs or source_path or chunk.metadata.get("source", ""), base_docs_dir)

    next_index = doc_chunk_counters.get(doc_id, 0)
    doc_chunk_counters[doc_id] = next_index + 1
    chunk_id = f"{doc_id}:{next_index:05d}"

    # Location: prefer explicit page/section info if available.
    page = chunk.metadata.get("page")
    total_pages = chunk.metadata.get("total_pages")
    if page is not None:
        location = f"Seite {page}"
    elif total_pages is not None:
        location = f"Dokument (Seiten 1-{total_pages})"
    else:
        location = "Unbekannt"

    date_str = file_modified_date(source_path_abs) if source_path_abs else None

    enriched = dict(chunk.metadata)
    enriched.update(
        {
            "chunk_id": chunk_id,
            "chunk_index": next_index,
            "doc_id": doc_id,
            "fonds_id": fonds_id,
            "fonds_name": fonds_name,
            "source_path": source_path_abs or source_path,
            "source_name": os.path.basename(source_path_abs or source_path) if (source_path_abs or source_path) else "",
            "location": location,
            "date": date_str,
            "content_type": naive_content_type(chunk.page_content),
            "short_summary": naive_short_summary(chunk.page_content),
            "summary_generated_by": "heuristic",
        }
    )

    chunk.metadata = enriched
    chunk_ids.append(chunk_id)
    chunk_metadatas.append(enriched)



 
#%% Embeddings
from langchain_openai.embeddings import OpenAIEmbeddings
embedding_model = OpenAIEmbeddings()

# for every chunk: get text content 
docs_pagecontents = [d.page_content for d in docs_splitted]
# create semantic vector (embeddings) for every chunk
docs_embedded = []
import time
for i, d in enumerate(docs_splitted):
    time.sleep(0.3) # against time out
    print(i)
    emb = embedding_model.embed_query(d.page_content)
    docs_embedded.append(emb)




 
#%% storing the data in data base
 
from langchain_community.vectorstores import FAISS

# pairs of text chunk / embedding (+ metadaten + stabile IDs)
docs_emb_pairs = list(zip(docs_pagecontents, docs_embedded))
vectordb = FAISS.from_embeddings(
    text_embeddings=docs_emb_pairs,
    embedding=embedding_model,
    metadatas=chunk_metadatas,
    ids=chunk_ids,
)
 
vectordb.save_local(folder_path="projekt_fonds_docs_chunks")
# %% testen
# find similar chunks to the query
retriever = vectordb.as_retriever(search_type ="similarity", search_kwargs ={"k": 3})
# kwargs = zusätzliche Parameter für die Suche
res = retriever.invoke(input="Anlagepolitik von Dynamisch?")
res
len((res[0]).page_content)
 
#%%