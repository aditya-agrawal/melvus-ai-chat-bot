#!/usr/bin/env python
# coding: utf-8

# In[1]:


# first run installations and data download
# ! llama-index-llms-openai llama-index-readers-file llama-index 
# ! mkdir -p './data/10k/'
# ! curl 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/examples/data/10k/uber_2021.pdf' -O 'data/10k/uber_2021.pdf'
# ! curl 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/examples/data/10k/lyft_2021.pdf' -O 'data/10k/lyft_2021.pdf'
# ! mv 'lyft_2021.pdf' 'uber_2021.pdf' './data/10k/'


# In[1]:


import openai
openai.api_key = ""


# In[2]:


from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI


# In[12]:


import os
import flask
import flask_cors
from tqdm import tqdm
from urllib.parse import urljoin, urlparse
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.tools import QueryEngineTool, ToolMetadata

from pymilvus import connections
from llama_index.vector_stores.milvus import MilvusVectorStore
from web2pdf import BeautifulSoupFlexibleScraper


# In[4]:


from pymilvus import connections 
connections.connect("default", host='localhost', port='19530')


# In[6]:


# Scrape pages
#url_parent = "https://milvus.io/docs"
#
#def web2pdf_indepth(url_parent):
#    scraper = BeautifulSoupFlexibleScraper()
#    child_urls = scraper.get_child_urls(url_parent, depth=2)
#    
#    for url in tqdm(child_urls):
#        text, image_sources = scraper.scrape_page(url)
#        if text is not None:
#            # Define the path for the PDF file
#            parsed_url = urlparse(url)
#            directory_path = os.path.join('data', parsed_url.netloc, parsed_url.path.strip('/'))
#            os.makedirs(directory_path, exist_ok=True)
#            pdf_path = os.path.join(directory_path, os.path.basename(url) + '.pdf')
#            
#            # Save the page as a PDF
#            scraper.save_page_as_pdf(url, text, pdf_path)
#
#web2pdf_indepth(url_parent)


# In[5]:


try:
    storage_context = StorageContext.from_defaults(
        persist_dir="./storage/milvus_io"
    )
    index = load_index_from_storage(storage_context)
    index_loaded = True
except:
    index_loaded = False


# In[6]:


index_loaded


# In[7]:


if not index_loaded:
    # load data
    print("Reading docs...")
    docs = SimpleDirectoryReader(input_dir="./data/milvus.io", recursive=True,).load_data()
    print("Building index...")
    # build index
    vector_store = MilvusVectorStore(dim=1536, collection_name="milvus_io", overwrite=True)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(docs, storage_context=storage_context)

    # persist index
    index.storage_context.persist(persist_dir="./storage/milvus_io")


# In[8]:


engine = index.as_query_engine(similarity_top_k=3)


# In[9]:


query_engine_tools = [
    QueryEngineTool(
        query_engine=engine,
        metadata=ToolMetadata(
            name="milvus_ask",
            description=("""
            Provides information about connected Milvus documentation.
            Use a detailed plain text question as input to the tool.
            """
            ),
        ),
    ),
]


# In[10]:


llm = OpenAI(model="gpt-4")

agent = ReActAgent.from_tools(
    query_engine_tools,
    llm=llm,
    verbose=True,
    # context=context
)


# In[18]:


from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS

app = Flask(__name__)

CORS(app)  # Enable CORS for all routes

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Simple API Interaction</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f0f0f0;
        }
        form {
            margin-bottom: 20px;
        }
        input[type="text"] {
            font-size: 16px;
            padding: 10px;
            width: 300px;
            margin-right: 10px;
            border: 2px solid #007bff;
            border-radius: 4px;
        }
        button {
            padding: 10px 20px;
            font-size: 16px;
            background-color: #007bff;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }
        button:hover {
            background-color: #0056b3;
        }
        #response {
            padding: 20px;
            background-color: #ddd;
            border: 1px solid #ccc;
            border-radius: 4px;
            margin-top: 20px;
        }
    </style>
</head>
<body>
    <form action="/chat" method="POST">
        <input type="text" name="userinput" id="user_input">
        <button type="submit">Submit</button>
    </form>
    <div id="response">{{ response|safe }}</div>
</body>
</html>

"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/chat', methods=['POST'])
def get_data():
    user_input = request.form['userinput']
    print(user_input)
    response = agent.chat(user_input)
    return render_template_string(HTML_TEMPLATE, response=response)


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5001, debug=True)


# In[ ]:




