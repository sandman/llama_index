import logging
import os
import sys

import openai

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from dotenv import find_dotenv, load_dotenv

from llama_index import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)

load_dotenv(find_dotenv())

openai.api_key = os.getenv("OPENAI_API_KEY")
openai.log = None

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
        index = VectorStoreIndex.from_documents(documents)
        print("No storage index found, created new index")
        # Persist the index to disk
        index.storage_context.persist()


query_engine = index.as_query_engine()
response = query_engine.query(
    "Who is the author and what are his credentials?"
)
print(response)
