#%% packages
import pandas as pd
import numpy as np
from langchain_openai.embeddings import OpenAIEmbeddings
from sklearn.manifold import TSNE
from dotenv import load_dotenv
import seaborn as sns
import matplotlib.pyplot as plt
load_dotenv()
 
# %%
df = pd.read_csv("data/genres_database.csv", sep=";")
genres = df["Genre"].tolist()
genres
 
#%% embedding model instance
embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")
genres_embedded = embedding_model.embed_documents(genres)
#%%
len(genres_embedded[0])
 
#%% Reduktion von 1536 Dimensionen (pro Genre) auf 2
# Konvertiere die Embeddings in ein numpy Array
embeddings_array = np.array(genres_embedded)
 
# Erstelle t-SNE Modell mit 2 Dimensionen
tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(genres_embedded)-1))
 
# Reduziere die Dimensionen
genres_2d = tsne.fit_transform(embeddings_array)
 
# Überprüfe die Form (sollte (Anzahl Genres, 2) sein)
genres_2d.shape
 
#%% Visualisieren der Ergebnisse
# Erstelle DataFrame für die Visualisierung
df_plot = pd.DataFrame({
    'x': genres_2d[:, 0],
    'y': genres_2d[:, 1],
    'Genre': genres
})
 
# Erstelle den Plot mit seaborn
plt.figure(figsize=(14, 10))
sns.scatterplot(data=df_plot, x='x', y='y', s=100, alpha=0.6)
 
# Füge Textlabels für jedes Genre hinzu
for i, row in df_plot.iterrows():
    plt.text(row['x'], row['y'], row['Genre'],
             fontsize=9, ha='center', va='bottom')
 
plt.xlabel('t-SNE Dimension 1', fontsize=12)
plt.ylabel('t-SNE Dimension 2', fontsize=12)
plt.title('Genre Embeddings Visualisierung', fontsize=14)
plt.tight_layout()
plt.show()
 
#%% Similarity Search für einen Suchbegriff
# %% Similarity Search für einen Suchbegriff
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

user_query = "Lindy"

# Embed the user query
user_query_embedding = embedding_model.embed_query(user_query)

# Compute cosine similarity between query and each genre embedding
similarities = cosine_similarity(
    [user_query_embedding],      # shape (1, embedding_dim)
    embeddings_array             # shape (n_genres, embedding_dim)
).flatten()

# Sort genres by similarity
top_n = 5
top_indices = similarities.argsort()[::-1][:top_n]

print(f"Top ähnliche Genres zu '{user_query}':")
for idx in top_indices:
    print(f"{df['Genre'].iloc[idx]}: Ähnlichkeit = {similarities[idx]:.3f}")

# %%
print(df.columns)

# %%
