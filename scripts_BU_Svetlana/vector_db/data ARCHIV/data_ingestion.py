#%% packages
import os
from langchain.document_loaders import TextLoader
 
script_path = os.path.abspath(__file__)
current_path = os.path.dirname(script_path)
current_path
 
# %% load a single, local file
loader = TextLoader(file_path=os.path.join(current_path, "data", "christmas_dickens.txt"), encoding="utf-8")
docs = loader.load()
 
#%% full book content
docs[0].page_content
# %%
docs[0].metadata
 
#%% load a single file from web
from langchain_community.document_loaders import WebBaseLoader
url = "https://www.gutenberg.org/cache/epub/46/pg46.txt"
loader = WebBaseLoader(web_path=url)
docs = loader.load()
 
#%%
docs[0].metadata
 
#%% load multiple file from a folder
from langchain.document_loaders import DirectoryLoader
loader = DirectoryLoader(path= os.path.join(current_path, "data"), loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"})
docs = loader.load()
docs
   

#%% Wikipedia
# ladet 5 Artikel zum Thema "Generative KI" herunter
# nutzt den entsprechenden Loader von Langchain
from langchain_community.document_loaders import WikipediaLoader

loader = WikipediaLoader(
    query="Generative KI",
    lang="de"
)

docs = loader.load()


# %%

#%% load multiple files from a folder of different types
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_community.document_loaders import CSVLoader
 
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
    raise ValueError(f"Unsupported Filetype: {ext} for {file_path}")
 
loader = DirectoryLoader(path=os.path.join(current_path, "data"), loader_cls=select_loader, show_progress=True, use_multithreading=True)
docs = loader.load()
# %%
docs
# %%
docs[1].metadata

#%%










##############################################################
# %% ppt (noch nicht selbst ausprobiert, aber das geht - laut Bert Gollnick)
# todo: ppt abspeichern im Ordner


from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_community.document_loaders import UnstructuredPowerPointLoader
 
 
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
docs = loader.load()
 
 
# %%
docs[3].metadata
#%%
######################################################################
# %% Data Chunking
######################################################################
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter
)

# einfacher Character Splitter
splitter = CharacterTextSplitter(
    chunk_size=4000,
    chunk_overlap=4000 // 5
)

# Dokumente in Chunks aufteilen
docs_splitted = splitter.split_documents(docs)

# Anzahl der erzeugten Chunks
len(docs_splitted)

#%%


# %%
