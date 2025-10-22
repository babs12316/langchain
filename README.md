Langchain concepts
# 🧠 LangChain GitHub Retriever Demo

This project demonstrates how to **build a simple Retrieval-Augmented Generation (RAG)** system using a GitHub repository as the source of documents.  
The main goal is to fetch content from `README.md` files in the repo, embed it, store it in a vector store, and retrieve relevant context for user queries.

## 🚀 Tech Stack

- 🐍 **Python 3.9+**
- 🧰 **LangChain** – framework for building applications with LLMs
- 🪄 **Hugging Face Sentence Transformers** – for generating vector embeddings  
- 💾 **InMemoryVectorStore** – lightweight and fast vector store for local development
- 📂 **GitHub File Loader** – loads files directly from a public or private GitHub repository
- ✂️ **Recursive Character Text Splitter** – splits text into manageable chunks

## 🧠 How It Works

1. The loader fetches Markdown files from the GitHub repository (e.g., this `README.md`).
2. The text is split into overlapping chunks to preserve context.
3. Embeddings are generated using `sentence-transformers/all-mpnet-base-v2`.
4. The chunks and their embeddings are stored in the in-memory vector store.
5. A retriever is used to fetch the most relevant chunks based on user queries.
6. The LLM can then use this retrieved content to answer the question.

## 🧪 Example Query

You can ask:

```text
"What programming language is used in this repository?"
