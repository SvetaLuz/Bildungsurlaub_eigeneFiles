
#%% pakete
import os
from dotenv import load_dotenv
load_dotenv()
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from pprint import pprint
 
#%% output parser
class MyMovieOutput(BaseModel):
    title: str = Field(description="The movie title")
    director: str = Field(description="The director of the movie in the form 'surname, firstname'", examples=["Cameron, James"])
    actors: list[str] = Field(description="The 5 most relevant actors of the movie")
    release_year: int
 
class MyMovieOutputs(BaseModel):
    movies: list[MyMovieOutput]
 
output_parser = PydanticOutputParser(pydantic_object=MyMovieOutputs)
 
#%%
pprint(output_parser.get_format_instructions())
 
#%% Prompt Template
prompt_template = ChatPromptTemplate.from_messages([
    ("system", """
        Du lieferst Filminformationen auf Basis der übergebenen Handlung.
        Gib die 5 relevantesten Filme an.
        {format_instructions}
    """),
    ("user", "Handlung: <<{handlung}>>")
]).partial(format_instructions=output_parser.get_format_instructions())
prompt_template
 
 
# %% Modellinstanz
MODEL_NAME="llama-3.3-70b-versatile"
model = ChatGroq(model=MODEL_NAME)
 
#%% Chain erstellen
chain = prompt_template | model | output_parser
# %% chain invocation
handlung = "tango"
res = chain.invoke({"handlung": handlung})
 
# Ergebnis
from pprint import pprint
pprint(res)
 
 
# %%
for r in res.model_dump()["movies"]:
    print(r['title'])
    print(r['director'])
    print("-"*20)