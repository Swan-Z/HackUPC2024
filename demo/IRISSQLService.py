import os, pandas as pd
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine, text

username = 'demo'
password = 'demo'
hostname = os.getenv('IRIS_HOSTNAME', 'localhost')
port = '1972' 
namespace = 'USER'
CONNECTION_STRING = f"iris://{username}:{password}@{hostname}:{port}/{namespace}"

engine = create_engine(CONNECTION_STRING)

df = pd.read_csv('../data/medicine_dataset.csv')
#show the first row
df.head()

# Clean data
# Remove the specified columns
#df.drop(['currency'], axis=1, inplace=True)
# Remove rows without a price
#df.dropna(subset=['price'], inplace=True)
# Ensure values in 'price' are numbers
#df = df[pd.to_numeric(df['price'], errors='coerce').notna()]

# Drop the first column
df.drop(columns=df.columns[0], inplace=True)

# Replace NaN values in other columns with an empty string
df.fillna('', inplace=True)

# Combina las filas 'sideEffect1' hasta 'sideEffect41' en una sola columna llamada 'side_effects'
df['substitutes'] = df.apply(lambda row: ' '.join(row['substitute0':'substitute4']), axis=1)
df['side_effects'] = df.apply(lambda row: ' '.join(row['sideEffect1':'sideEffect41']), axis=1)
df['uses'] = df.apply(lambda row: ' '.join(row['use0':'use4']), axis=1)

# Elimina las columnas originales
df.drop(columns=df.loc[:, 'sideEffect0':'sideEffect41'].columns, inplace=True)
df.drop(columns=df.loc[:, 'substitute0':'substitute4'].columns, inplace=True)
df.drop(columns=df.loc[:, 'use0':'use4'].columns, inplace=True)

# Muestra las primeras filas del DataFrame actualizado
df.head()

with engine.connect() as conn:
    with conn.begin():# Load 
        sql = f"""
                CREATE TABLE scotch_reviews (
        name VARCHAR(255),
        substitutes VARCHAR(255),
        sideEffects VARCHAR(255),
        uses VARCHAR(255),
        Chemical Class VARCHAR(255),
        Habit Forming VARCHAR(255),
        Theraputic Class VARCHAR(255),
        Action Class VARCHAR(255)
        )
                """
        result = conn.execute(text(sql))

model = SentenceTransformer('all-MiniLM-L6-v2') 
embeddings = model.encode(df['uses'].tolist(), normalize_embeddings=True)
df['uses_vector'] = embeddings.tolist()
df.head()
