from flask import Flask, request, jsonify
from llama_index import SimpleDirectoryReader, StorageContext
from llama_index.indices.vector_store import VectorStoreIndex
from llama_iris import IRISVectorStore
import os
from dotenv import load_dotenv

load_dotenv(override=True)

app = Flask(__name__)

# Load documents
documents = SimpleDirectoryReader("../data/paul_graham").load_data()

# Initialize IRIS Vector Store
username = 'demo'
password = 'demo' 
hostname = os.getenv('IRIS_HOSTNAME', 'localhost')
port = '1972' 
namespace = 'USER'
CONNECTION_STRING = f"iris://{username}:{password}@{hostname}:{port}/{namespace}"

vector_store = IRISVectorStore.from_params(
    connection_string=CONNECTION_STRING,
    table_name="jobAnnoucement",
    embed_dim=1536,  # openai embedding dimension
)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Build index
index = VectorStoreIndex.from_documents(
    documents, 
    storage_context=storage_context, 
    show_progress=True
)

# Define API endpoint
@app.route('/query', methods=['GET'])
def query():
    query_text = request.args.get('text')
    if query_text:
        response = index.as_query_engine().query(query_text)
        return jsonify(response)
    else:
        return jsonify({"error": "Query parameter 'text' is required."}), 400

if __name__ == '__main__':
    app.run(debug=True)