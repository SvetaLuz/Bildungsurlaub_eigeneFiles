#%% packages
import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
 
script_path = os.path.abspath(__file__)
current_path = os.path.dirname(script_path)
current_path
 
# %% load a single, local file
# loader = TextLoader(file_path=os.path.join(current_path, "data", "christmascarol.txt"), encoding="utf-8")
# docs = loader.load()
 
# #%% full book content
# docs[0].page_content
# # %%
# docs[0].metadata
 
#%% load a single file from web
# from langchain_community.document_loaders import WebBaseLoader
# url = "https://www.gutenberg.org/cache/epub/46/pg46.txt"
# loader = WebBaseLoader(web_path=url)
# docs = loader.load()
 
#%%
# docs[0].metadata
 
#%% load multiple file from a folder
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
from langchain_community.document_loaders import CSVLoader, UnstructuredPowerPointLoader
 
 
def select_loader(file_path: str):
    # extract file extension from file_path
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
 
    # use different loaders depending on extension
    if ext == ".txt":
        return TextLoader(file_path=file_path, encoding="utf-8")
   
    if ext == ".pdf":
        return PyMuPDF4LLMLoader(file_path=file_path, mode="single")
    if ext == ".csv":
        return CSVLoader(file_path=file_path, encoding="utf-8")
    if ext == ".pptx":
        return UnstructuredPowerPointLoader(file_path= file_path)
 
    raise ValueError(f"Unsupported Filetype: {ext} for {file_path}")
 
loader = DirectoryLoader(
    path=os.path.join(current_path, "data"),
    loader_cls=select_loader,
    show_progress=True,
    use_multithreading=True,
    exclude="*.py"
)
if __name__ == "__main__":
    docs = loader.load()
 
 
# %%
 
 
if __name__ == "__main__":
    # Data Chunking
    from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=4000 / 5)
    # [Document, Document]  -> längere [Document, Document]
    docs_splitted = splitter.split_documents(docs)
    len(docs_splitted)

    #size of split
    doc_sizes = [len(doc.page_content) for doc in docs_splitted]
    import seaborn as sns

    sns.histplot(doc_sizes)

    #
    from pprint import pprint

    pprint(docs[0].page_content)

    # Embeddings
    from langchain_openai.embeddings import OpenAIEmbeddings

    embedding_model = OpenAIEmbeddings()

    docs_pagecontents = [d.page_content for d in docs_splitted]

    docs_embedded = []
    import time

    for i, d in enumerate(docs_splitted):
        time.sleep(0.5) # um nicht an die grenze zu kommen
        print(i)
        emb = embedding_model.embed_query(d.page_content)
        docs_embedded.append(emb)

    #%% pairs of text chunk / embedding
    docs_emb_pairs = zip(docs_pagecontents, docs_embedded)

    #%% storing the data
    from langchain_community.vectorstores import FAISS

    vectordb = FAISS.from_embeddings(text_embeddings=docs_emb_pairs, embedding=embedding_model)

    vectordb.save_local(folder_path="weltliteratur")

    # %% testen
    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    res = retriever.invoke(input="Wer ist Ahab?")
    res

    #%%