from llama_index import SimpleDirectoryReader, StorageContext, ServiceContext
from llama_index.indices.vector_store import VectorStoreIndex
from llama_iris import IRISVectorStore

import getpass
import os
from dotenv import load_dotenv

load_dotenv(override=True)

# if not os.environ.get("OPENAI_API_KEY"):
#     os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API Key:")

documents = SimpleDirectoryReader("../data/paul_graham").load_data()
print("Document ID:", documents[0].doc_id)

username = 'demo'
password = 'demo' 
hostname = os.getenv('IRIS_HOSTNAME', 'localhost')
port = '1972' 
namespace = 'USER'
CONNECTION_STRING = f"iris://{username}:{password}@{hostname}:{port}/{namespace}"

vector_store = IRISVectorStore.from_params(
    connection_string=CONNECTION_STRING,
    table_name="jobAnnouncement",
    embed_dim=1536,  # openai embedding dimension
)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
# service_context = ServiceContext.from_defaults(
#     embed_model=embed_model, llm=None
# )

index = VectorStoreIndex.from_documents(
    documents, 
    storage_context=storage_context, 
    show_progress=True, 
    # service_context=service_context,
)
query_engine = index.as_query_engine()

response = query_engine.query("What did the author do?")

import textwrap
print(textwrap.fill(str(response), 100))

response = query_engine.query("What happened in the mid 1980s?")
print(textwrap.fill(str(response), 100))

from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app, origins='http://localhost:8100')

# Define a route to handle queries
@app.route('/query', methods=['POST'])
def handle_query():
    # Obtain the query text from the request
    query_text = request.json.get('query')
    
    # Perform the query using the query_engine
    response = query_engine.query(query_text)
    
    # Return the response as JSON
    return jsonify({'response': str(response)})

if __name__ == '__main__':
    app.run(debug=True)



