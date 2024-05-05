# api_module.py

from flask import Flask, jsonify, request
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
from service_module import handle_gpt_query
from sqlalchemy import create_engine, text
import os

app = Flask(__name__)
CORS(app, origins='*')

username = 'demo'
password = 'demo'
hostname = os.getenv('IRIS_HOSTNAME', 'localhost')
port = '1972'
namespace = 'USER'
CONNECTION_STRING = f"iris://{username}:{password}@{hostname}:{port}/{namespace}"

engine = create_engine(CONNECTION_STRING)
model = SentenceTransformer('all-MiniLM-L6-v2')

@app.route('/api/job_posting', methods=['POST'])
def job_posting():
    data = request.json
    position = data['position']
    company = data['company']
    #similarity search
    search_vector = model.encode(position, normalize_embeddings=True).tolist()
    with engine.connect() as conn:
        with conn.begin():
            sql = text("""
                SELECT TOP 3 job FROM jobAnnouncement
                ORDER BY VECTOR_DOT_PRODUCT(job_vector, TO_VECTOR(:search_vector)) DESC
            """)
            results = conn.execute(sql, {'search_vector': str(search_vector)}).fetchall()
    #TODO: handle the response from the similarity search
    result = handle_gpt_query(position, company, results)
    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(port=8000, debug=True)