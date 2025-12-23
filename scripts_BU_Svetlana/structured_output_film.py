#%% pakete
import os
from dotenv import load_dotenv
load_dotenv()
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

#%% klare Struktur der Ausgabe für das Modell definieren

class MyMovieOutput(BaseModel):
    title: str = Field(..., description="The title of the movie")
    director: str = Field(..., description="The director of the movie")
    release_year: int = Field(..., description="The year the movie was released")
    genre: str = Field(..., description="The genre of the movie")
    actors: list[str] = Field(..., description="5 relevant actors in the movie")

output_parser = PydanticOutputParser[MyMovieOutput](pydantic_object=MyMovieOutput)

class MyMovieOutputs(BaseModel):
    movies: list[MyMovieOutput]
 
 
#%%
pprint(output_parser.get_format_instructions())
#%% Prompt Template
prompt_template = ChatPromptTemplate.from_messages([
    ("system", """
        Du lieferst Filminformation auf Basis der übergebenen Handlung. 
     {{format_instructions}}
    """),
    ("user", "Handlung: <<{handlung}>>")
]).partial(format_instructions=output_parser.get_format_instructions())
prompt_template
 
 
# %% Modellinstanz
MODEL_NAME="llama-3.3-70b-versatile"
model = ChatGroq(model=MODEL_NAME)
 
#%% Chain erstellen
chain = prompt_template | model
# %% chain invocation
handlung = "Tango"

res = chain.invoke({"handlung": handlung})
 
#%% Ergebnis
from pprint import pprint
pprint(res.content)
# %%


for r in res.model_dump()["movies"]:
    print(r['title'])
    print(r['director'])
    print("-"*20)
# %%
