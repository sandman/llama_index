import logging
import os
import sys

import chromadb
import openai

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from dotenv import find_dotenv, load_dotenv

from llama_index import (
    ServiceContext,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.vector_stores import ChromaVectorStore

load_dotenv(find_dotenv())

# Parse documents into configurable chunk size
service_context = ServiceContext.from_defaults(chunk_size=1000)

openai.api_key = os.getenv("OPENAI_API_KEY")
openai.log = None

chroma_client = chromadb.PersistentClient()
chroma_collection = chroma_client.create_collection("quickstart")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

documents = SimpleDirectoryReader("data").load_data()

if documents is not None:
    print("Loaded {} documents".format(len(documents)))
    # Check if storage context exists
    # rebuild storage context
    try:
        storage_context = StorageContext.from_defaults(persist_dir="./storage")
        # load index
        print("Loading index from storage")
        index = load_index_from_storage(storage_context)
    except FileNotFoundError:
        # Create a new index
        # NOTE: Do not pass the storage_context below if you want to use
        # the default vector store
        index = VectorStoreIndex.from_documents(
            documents,
            service_context=service_context,
            storage_context=storage_context,
        )
        print("No storage index found, created new index")
        # Persist the index to disk
        index.storage_context.persist()


query_engine = index.as_query_engine()
response = query_engine.query(
    "Who is the author and what are his credentials?"
)
print(response)
