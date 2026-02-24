from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document

from dotenv import load_dotenv
import os
load_dotenv()

os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY")

documents = [
    Document(page_content="LangChain helps developers build LLM applications easily."),
    Document(page_content="Chroma is a vector database that allows you to store and query embeddings efficiently."),
    Document(page_content="Google Generative AI Embeddings provide powerful vector representations for text data."),
    Document(page_content="LangChain provides a simple interface to work with various vector databases, including Chroma.")
]

embedding_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

vector_store = Chroma.from_documents(
    documents=documents,
    embedding=embedding_model,
    collection_name="my_collection",
    persist_directory="Retrievers/chroma_db"
)

retriever = vector_store.as_retriever(search_kwargs={"k": 2})

query = "What is Chroma used for?"
results = retriever.invoke(query)

for i,doc in enumerate(results):
    print(f"\n---Result {i+1}---")
    print(f"Content: \n{doc.page_content}...")

