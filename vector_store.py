import logging
from typing import List

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from time import sleep

EMBED_DELAY = 0.02  # 20 milliseconds


# This is to get the Streamlit app to use less CPU while embedding documents into Chromadb.



# This happens all at once, not ideal for large datasets.
def create_vector_db(texts, openai_api_key, save_file, embeddings=None):
    if not texts:
        logging.warning("Empty texts passed in to create vector database")
    # Select embeddings
    if not embeddings:
        # To use HuggingFace embeddings instead:
        # from langchain_community.embeddings import HuggingFaceEmbeddings
        # embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        # openai_api_key = os.environ["OPENAI_API_KEY"]
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key, model="text-embedding-3-small",  chunk_size=300)

    # Create a vectorstore from documents
    # this will be a chroma collection with a default name.
    db = FAISS.from_documents(texts, embeddings)
    # db.save_local('faiss_emb')
    db.save_local(save_file)
    
    return db


def find_similar(vs, query):
    docs = vs.similarity_search(query)
    return docs

