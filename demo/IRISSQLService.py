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
df = pd.read_csv('../data/dataset.csv', encoding='ISO-8859-1')

#show the first row
df.head()

# Clean data
# Remove the specified columns
# df.drop(['company'], axis=1, inplace=True)
# Remove rows without a price
#df.dropna(subset=['price'], inplace=True)
# Ensure values in 'price' are numbers
#df = df[pd.to_numeric(df['price'], errors='coerce').notna()]

# Replace NaN values in other columns with an empty string
df.fillna('', inplace=True)

# Combina las filas 'sideEffect1' hasta 'sideEffect41' en una sola columna llamada 'side_effects'
# df['substitutes'] = df.apply(lambda row: ' '.join(row['substitute0':'substitute4']), axis=1)
# df['side_effects'] = df.apply(lambda row: ' '.join(row['sideEffect1':'sideEffect41']), axis=1)
# df['uses'] = df.apply(lambda row: ' '.join(row['use0':'use4']), axis=1)

# Elimina las columnas originales
# df.drop(columns=df.loc[:, 'sideEffect0':'sideEffect41'].columns, inplace=True)
# df.drop(columns=df.loc[:, 'substitute0':'substitute4'].columns, inplace=True)
# df.drop(columns=df.loc[:, 'use0':'use4'].columns, inplace=True)



# with engine.connect() as conn:
#     with conn.begin():
#         sql = f"""
#                 CREATE TABLE jobAnnouncement (
#         title VARCHAR(255),
#         qualification VARCHAR(255),
#         job_description VARCHAR(255),
#         need VARCHAR(255)
#         )
#                 """
#         result = conn.execute(text(sql))

model = SentenceTransformer('all-MiniLM-L6-v2') 
embeddings = model.encode(df['job_description'].tolist(), normalize_embeddings=True)
df['job_vector'] = embeddings.tolist()
df.head()

with engine.connect() as conn:
    with conn.begin():
        for index, row in df.iterrows():
            sql = text("""
                INSERT INTO jobAnnouncement(title, role, job_description, need, job_vector) 
                VALUES (:title, :role, :job_description, :need, TO_VECTOR(:job_vector))
            """)
            conn.execute(sql, {
                'title': row['title'],
                'qualification': row['qualification'],
                'job_description': row['job_description'],
                'need': row['need'],
                'job_vector': str(row['job_vector'])
            })

description_search = "I need a software engineer with experience in Python and Java"
search_vector = model.encode(description_search, normalize_embeddings=True).tolist() 

with engine.connect() as conn:
    with conn.begin():
        sql = text("""
            SELECT TOP 3 * FROM job_description 
            ORDER BY VECTOR_DOT_PRODUCT(job_vector, TO_VECTOR(:search_vector)) DESC
        """)

        results = conn.execute(sql, {'search_vector': str(search_vector)}).fetchall()

print(results)