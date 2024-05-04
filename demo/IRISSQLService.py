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

# Lee las primeras 10 líneas del archivo CSV
df = pd.read_csv('../data/final_data.csv').head(10)
#show the first row
df.head()

# Clean data
# Remove the specified columns
df.drop(['Total_applicants'], axis=1, inplace=True)
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
                CREATE TABLE job_description (
        title VARCHAR(255),
        role VARCHAR(255),
        work VARCHAR(255),
        need VARCHAR(255)
        )
                """
        result = conn.execute(text(sql))

model = SentenceTransformer('all-MiniLM-L6-v2') 
embeddings = model.encode(df['work'].tolist(), normalize_embeddings=True)
df['work_vector'] = embeddings.tolist()
df.head()

with engine.connect() as conn:
    with conn.begin():
        for index, row in df.iterrows():
            sql = text("""
                INSERT INTO job_description(title, role, work, need, work_vector) 
                VALUES (:name, :category, :review_point, :price, :description, TO_VECTOR(:description_vector))
            """)
            conn.execute(sql, {
                'name': row['name'], 
                'category': row['category'], 
                'review_point': row['review.point'], 
                'price': row['price'], 
                'description': row['description'], 
                'description_vector': str(row['description_vector'])
            })

