import os

from langchain_community.document_loaders import GithubFileLoader
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
token = os.getenv("GITHUB_ACCESS_TOKEN")
repo = os.getenv("REPO")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

loader = GithubFileLoader(
    access_token=token,
    repo=repo,
    branch="main",
    # f her is file path ex. src/utils/helper.py
    file_filter=lambda f: f.endswith(".md")

)
docs = loader.load()
print(f"load documents length {len(docs)} ")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=100,
    add_start_index=True
)

split_docs = text_splitter.split_documents(documents=docs)

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vector_store = InMemoryVectorStore(embedding)

vector_store.add_documents(split_docs)

retriever = vector_store.as_retriever(search_kwargs={"k": 1})

query = "can you tell me which tech stack used in this repo?"

result = retriever.invoke(query)

print("result is", result[0].page_content)
