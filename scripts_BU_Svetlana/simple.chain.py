#%% pakete
import os
from dotenv import load_dotenv
load_dotenv()
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
 
#%% Prompt Template
prompt_template = ChatPromptTemplate.from_messages([
    ("system", """
        Erzähle ein Hokku über die Figur an dem Ort. 
    """),
    ("user", "character: {character}, location: {location}")
])
 
 
# %% Modellinstanz
MODEL_NAME="llama-3.3-70b-versatile"
model = ChatGroq(model=MODEL_NAME)
 
#%% Chain erstellen
chain = prompt_template | model
# %% chain invocation
character = "Donald Trump"
location = "Taiwan"

res = chain.invoke({"character": character, "location": location})
 
#%% Ergebnis
from pprint import pprint
pprint(res.content)
# %%
