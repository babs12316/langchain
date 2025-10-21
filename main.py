from dotenv import load_dotenv
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
import asyncio
from typing import List
from langchain_core.documents import Document
from langchain_core.runnables import chain

os.environ["TOKENIZERS_PARALLELISM"] = "false"

load_dotenv()

# https://docs.langchain.com/oss/python/langchain/knowledge-base
"""
 steps
1. load the document using document loader such as PyPdfloader, CSVloader, GithubFileLoader
2. split the document into chunks using Text splitters such as RecursiveCharacterTextSplitter (uses paragraph, comma), CharacterTextSplitter(length based), RecursiveJsonSplitter
3. select embedding model
4. select vector store
5. pass embeddings to vector store
6. add documents created in step 2 in vector store using  vector_store.add_documents

"""

# get and load the document

loader = PyPDFLoader('nike.pdf')
docs = loader.load()

print("length", len(docs))

print(docs[1].page_content[:1])

print(docs[1].metadata)
"""
document is too large or coarse split it into chunks The overlap helps mitigate the possibility of separating a
statement from important context related to it. We set add_start_index=True so that the character index where each
 split Document starts within the initial Document is preserved as metadata attribute “start_index”.
"""
text_spliter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)

all_splits = text_spliter.split_documents(docs)

print("length of chunk is", len(all_splits))
print("first chunk is", all_splits[0])

embeddings = HuggingFaceEmbeddings(model="sentence-transformers/all-mpnet-base-v2")

vector_store = InMemoryVectorStore(embeddings)
"""
 in this step 
  1. vector store will convert chunks into vectors using selected embedding model
  2. insert into vector and metadata into  store and
  3.  returns list of ids of each stored document( chunk)
"""
ids = vector_store.add_documents(documents=all_splits)

print("ids are", ids[0])


async def get_query_result():
    results1 = await vector_store.asimilarity_search("When was Nike incorporated?")
    print(results1[0])


results = vector_store.similarity_search("How many distribution centers does Nike have in the US?")
print(results[0])
asyncio.run(get_query_result())
results2 = vector_store.similarity_search_with_score("What was Nike's revenue in 2023?")
doc, score = results2[0]
print(f"Score:{score}")
print(doc)

embeddings1 = embeddings.embed_query("How were Nike's margins impacted in 2?")
results3 = vector_store.similarity_search_by_vector(embeddings1)
print("result3", results3[0])

