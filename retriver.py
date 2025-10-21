from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

# Load environment variables
load_dotenv()

# Step 1: Load your PDF
loader = PyPDFLoader("nike.pdf")
docs = loader.load()
print(f" Loaded {len(docs)} pages from PDF.")

# Step 2: Split the PDF into smaller chunks (to fit model context)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True
)
all_splits = text_splitter.split_documents(docs)
print(f" Created {len(all_splits)} text chunks.")

# Step 3: Create embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Step 4: Create vector store and add PDF chunks
vector_store = InMemoryVectorStore(embedding=embeddings)
vector_store.add_documents(all_splits)
print(" All chunks added to vector store.")

# Step 5: Create retriever
"""
This converts your vector store (which contains embedded text chunks) into a retriever object.
A retriever is simply an abstraction — instead of you manually calling:

vector_store.similarity_search("your query")


you can just do:

retriever.invoke("your query")

The retriever handles everything internally — embedding the query, performing similarity search, and returning relevant documents.

k → the number of most similar documents (chunks) you want to retrieve.

So:

search_kwargs={"k": 3}

means →

For each query, return the top 3 most relevant chunks from the vector store based on vector similarity.
"""
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# Step 6: Ask a question (retrieval)
query = "When was Nike incorporated?"
results = retriever.invoke(query)

# Step 7: Display retrieved chunks
print("\n Retrieved Results:")
for index, doc in enumerate(results):
    print(f"{index}\n")
    print(doc.page_content[:300])
