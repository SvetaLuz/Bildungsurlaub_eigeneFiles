#%% packages
import streamlit as st
import os
from dotenv import load_dotenv
load_dotenv()
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
 
 
#%% output parser
class MyMovieOutput(BaseModel):
    title: str = Field(description="The movie title")
    director: str = Field(description="The director of the movie in the form 'surname, firstname'", examples=["Cameron, James"])
    actors: list[str] = Field(description="The 5 most relevant actors of the movie")
    release_year: int
 
class MyMovieOutputs(BaseModel):
    movies: list[MyMovieOutput]
 
output_parser = PydanticOutputParser(pydantic_object=MyMovieOutputs)
 
#%% Prompt Template
prompt_template = ChatPromptTemplate.from_messages([
    ("system", """
        Du lieferst Filminformationen auf Basis der übergebenen Handlung.
        Gib die 5 relevantesten Filme an.
        
        {format_instructions}
    """),
    ("user", "Handlung: <<{handlung}>>")
]).partial(format_instructions=output_parser.get_format_instructions())
 
 
# %% Modellinstanz
MODEL_NAME="llama-3.3-70b-versatile"
model = ChatGroq(model=MODEL_NAME)
 
#%% Chain erstellen
chain = prompt_template | model | output_parser
# %% chain invocation
st.title("Welcher Film ist es?")
 
filmbeschreibung = st.chat_input("Say something")
if filmbeschreibung:
    with st.chat_message("user"):
        st.write(f"Nutzerbeschreibung: {filmbeschreibung}")
    res = chain.invoke({"handlung": filmbeschreibung})
    with st.chat_message("ai"):
        for r in res.model_dump()["movies"]:
            st.markdown(f"**Titel:** {r['title']}")
            st.markdown(f"**Regisseur:** {r['director']}")
            st.divider()
            # st.write(r['actors'])
            # st.write(r['release_year'])
            # st.write("-"*20)
# # Ergebnis
# from pprint import pprint
# pprint(res)
# %%
