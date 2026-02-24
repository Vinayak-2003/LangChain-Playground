from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from dotenv import load_dotenv
import os
load_dotenv()

os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY")

docs = [
    Document(page_content="LangChain helps developers build LLM applications easily."),
    Document(page_content="LangChain makes it easy to build applications with LLMs"),
    Document(page_content="Chroma is a vector database that allows you to store and query embeddings efficiently."),
    Document(page_content="Embeddings are vector representations of text data"),
    Document(page_content="MMR helps to diversify the retrieved results by considering both relevance and diversity."),
    Document(page_content="LangChain supports Chroma, FAISS, Pinecone, and more vector databases.")
]

embedding_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

vector_store = FAISS.from_documents(
    documents=docs,
    embedding=embedding_model
)

retriever = vector_store.as_retriever(
    search_type="mmr",                              # enables MMR search
    search_kwargs={"k": 3, "lambda_mult": 0.5}        # k = top results, lambda_mult = relevance-diversity balanace (1 = normal search, 0 = max diversity)
)

query = "What is langchain?"
results = retriever.invoke(query)

for i,doc in enumerate(results):
    print(f"\n---Result {i+1}---")
    print(f"Content: \n{doc.page_content}...")

