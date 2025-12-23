#%%
import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
from langchain_ollama import ChatOllama
#from langchain_groq import ChatGroq
from pprint import pprint

#%%

os.getenv("GROQ_API_KEY")
# %% Modellinstanz (konkrete Implementierung von Klasse)
MODEL_NAME = "deepseek-r1:1.5b"
model = ChatOllama(model=MODEL_NAME)

#%%
res = model.invoke("Was ist der Nutzen von KI?")
# %%
res.model_dump()
#%%
pprint(res.content)
#%% Wie teuer war die Anfrage?
input_cost = 0.59 /1E6 # $ /1M token
output_cost = 0.79/ 1E6 # $/1M token
input_token = res.model_dump()['response_metadata']['token_usage']['prompt_tokens']
output_token = res.model_dump()['response_metadata']['token_usage']['completion_tokens']
 
cost = input_cost * input_token + output_cost * output_token
print(f"Die Anfrage kostete {cost:.5f} $")
# %%

##############################


#%% pakete
import os
from dotenv import load_dotenv
load_dotenv()
from langchain_openai import ChatOpenAI
from pprint import pprint
 
 
# testen, ob der Key zur Verfügung steht
#os.getenv("GROQ_API_KEY")
 
 
# %% Modellinstanz
MODEL_NAME="gpt-4o-mini-2024-07-18"
model = ChatOpenAI(model=MODEL_NAME)


# %% model invocation

# %% model invocation
user_input = input("Was möchtest du fragen? ")
 
chat_history = []
while user_input != "exit":
    res = model.invoke(chat_history + [{"role": "user", "content": user_input}])
    chat_history.append({"role": "user", "content": user_input})
    chat_history.append({"role": "assistant", "content": res.content})
    print(res.content)
    user_input = input("Was möchtest du fragen? ")


#%%

res = model.invoke("Was ist der Nutzen von KI?")
 
#%% Ergebnis
res.model_dump()
 
#%%
pprint(res.content)
 
#%% Wie teuer war die Anfrage?
input_cost = 0.15 /1E6 # $ /1M token
output_cost = 0.6/ 1E6 # $ /1M token
input_token = res.model_dump()['response_metadata']['token_usage']['prompt_tokens']
output_token = res.model_dump()['response_metadata']['token_usage']['completion_tokens']
 
cost = input_cost * input_token + output_cost * output_token
print(f"Die Anfrage kostete {cost:.5f} $")
 
 
# %%