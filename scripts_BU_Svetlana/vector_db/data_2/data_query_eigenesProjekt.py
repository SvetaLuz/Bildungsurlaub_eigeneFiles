#%% packages
from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings
 
#%% embedding model and vector db
embedding_model = OpenAIEmbeddings()
 
vectordb = FAISS.load_local("db_eigenesProjekt", embeddings=embedding_model, allow_dangerous_deserialization=True)
# %% testen
retriever = vectordb.as_retriever(
    search_type ="similarity",
    search_kwargs ={"k": 3})
res = retriever.invoke(input="Was ist die Investmentstrategie?")
res

# %%


vectordb.similarity_search_with_relevance_scores(query="Wer ist Ahab?")
# %%

