#%% packages
from langchain_community.vectorstores import FAISS
from langchain_core.messages import SystemMessage
from langchain_openai.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()

# FAISS → Vektor-Datenbank (Similarity Search)
# OpenAIEmbeddings → wandelt Text → numerische Vektoren
# dotenv → lädt API-Keys (z. B. OPENAI_API_KEY) aus .env
# SystemMessage → steuert das Verhalten des LLMs (Rolle, Regeln)
 
#%% embedding model and vector db
embedding_model = OpenAIEmbeddings()
 
vectordb = FAISS.load_local("weltliteratur_big_chunks", embeddings=embedding_model, allow_dangerous_deserialization=True)
# %% testen
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()
from langchain_core.output_parsers import StrOutputParser

# Folgende funktionen werden in RAG genutzt:
# Retrieval – relevante Textstellen suchen
# Augmentation – Prompt mit Kontext bauen: Erzwingt Grounding = Frage + Kontext
# Generation – Antwort durch LLM erzeugen

def rag(user_query: str) -> str:
    # 1. Retrieval: sucht 3 ähnlichste Textstücke 
    retriever = vectordb.as_retriever(
        search_type ="similarity",
        search_kwargs ={"k": 3})
    res = retriever.invoke(input=user_query) # -> [Document(page_content=)]
    # benötigt: str = "Kontextinfos: context1; context2; context3"
    # Extrahiert nur den Text, Baut daraus einen einzigen Kontext-String
    res_str = "; ".join([r.page_content for r in res])
 
    # 2. Augmentation
    message = [
        ("system", """
        Du bist ein freundlicher RAG-Chatbot Assistent.
        Du antwortest auf die Userquery auf Basis von bereitgestellten Kontextinformationen.
        Antworte ausschließlich auf Basis der Kontextinfos.
        Wenn der Kontext die Frage nicht beantworten lässt, sag 'weiß ich nicht'.
        """),
        ("user", "Userquery {userquery}, Kontextinformationen {context_info}")
    ]
    prompt_template = ChatPromptTemplate.from_messages(message)
 
    # 3. Modell und Chain
    model = ChatOpenAI(model="gpt-4o-mini")
 
    chain = prompt_template | model | StrOutputParser()
    rag_result = chain.invoke({"userquery": user_query, "context_info": res_str})
    return rag_result
 
rag(user_query="Wie heißt der alte Mann in A christmas carol?")
 
 
# %% Exkurs
test_liste = ["context 1", "context 2", "context 3"]
"; ".join(test_liste)
 
#%% Exkurs: Aufbau der Messages
from langchain_core.messages import HumanMessage, SystemMessage
message = [
    SystemMessage(content="Du bist ein assistent"),# Variante 1: Dictionary
        # {
        #     "role": "system",
        #     "content": "du bist ein assistent"
        # },
        # Variante 2: Tuple
        # ("system", """
        # Du bist ein freundlicher RAG-Chatbot Assistent.
        # Du antwortest auf die Userquery auf Basis von bereitgestellten Kontextinformationen.
        # Antworte ausschließlich auf Basis der Kontextinfos.
        # Wenn der Kontext die Frage nicht beantworten lässt, sag 'weiß ich nicht'.
        # """),
        # Variante 3: Langchain Message
        SystemMessage(content="Du bist ein assistent"),
        ("user", "Userquery {userquery}, Kontextinformationen {context_info}")
    ]
prompt_template = ChatPromptTemplate.from_messages(message)
prompt_template
 
# %%