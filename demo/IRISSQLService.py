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

df.fillna('', inplace=True)

# with engine.connect() as conn:
#     with conn.begin():
#         sql = f"""
#                 DROP TABLE IF EXISTS jobAnnouncement
        
#                 """
#         result = conn.execute(text(sql))
# print(df.columns)

# with engine.connect() as conn:
#     with conn.begin():
#         sql = f"""
#                 CREATE TABLE jobAnnouncement (
#         title VARCHAR(2000),
#         qualification VARCHAR(2000),
#         job VARCHAR(2000),
#         need VARCHAR(2000),
#         job_vector VECTOR(DOUBLE, 384)
#         )
#                 """
#         result = conn.execute(text(sql))


model = SentenceTransformer('all-MiniLM-L6-v2') 
embeddings = model.encode(df['job'].tolist(), normalize_embeddings=True)
df['job_vector'] = embeddings.tolist()






with engine.connect() as conn:
    with conn.begin():
        for index, row in df.iterrows():
            sql = text("""
                INSERT INTO jobAnnouncement(title, qualification, job, need, job_vector) 
                VALUES (:title, :qualification, :job, :need, TO_VECTOR(:job_vector))
            """)
            conn.execute(sql, {
                'title': row['title'],
                'qualification': row['qualification'],
                'job': row['job'],
                'need': row['need'],
                'job_vector': str(row['job_vector'])
            })

description_search = "I need a software engineer with experience in Python and Java"
search_vector = model.encode(description_search, normalize_embeddings=True).tolist() 

with engine.connect() as conn:
    with conn.begin():
        sql = text("""
            SELECT TOP 3 job FROM jobAnnouncement
            ORDER BY VECTOR_DOT_PRODUCT(job_vector, TO_VECTOR(:search_vector)) DESC
        """)
        
        results = conn.execute(sql, {'search_vector': str(search_vector)}).fetchall()

print(results)