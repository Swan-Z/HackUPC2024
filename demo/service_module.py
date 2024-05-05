# service_module.py

from llama_index import SimpleDirectoryReader, StorageContext, ServiceContext
from llama_index.indices.vector_store import VectorStoreIndex
from llama_iris import IRISVectorStore
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS
import textwrap
import getpass
import os
from openai import OpenAI

# openai_api_key = os.environ.get("OPENAI_API_KEY")
# if not os.environ.get("OPENAI_API_KEY"):
#     os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API Key:")
# client = OpenAI(api_key=openai_api_key)

def query_gpt(prompt, max_tokens=1000):
    try:
        response = client.chat.completions.create(model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens)
        return response.choices[0].message.content.strip()
    except Exception as e:
        return str(e)

def handle_gpt_query(position, company_description, job_description):
    prompt = f"""
        Write a job posting for a {position} position, including the company description, job description,
        and required skills. Please also identify related roles to this position. Ensure that the job posting 
        is detailed and provides a clear overview of the company, the responsibilities of the role, and the 
        essential skills needed. Additionally, specify other positions that are related to this job. Your response
        should be comprehensive and informative, offering a well-rounded view of the job and its requirements.
        Besides, provide a brief description of this company{company_description} and put it at the beginning of the job posting.
        What's more, I will provide you some relevant information about the job. 
        job description:{job_description}.
    """
    response = query_gpt(prompt)
    return response