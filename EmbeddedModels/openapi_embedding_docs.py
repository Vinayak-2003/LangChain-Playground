from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(model="text-embedding-3-large", dimension=32)

documents = [
    "Delhi is the capital of India.",
    "Paris is the capital of France.",
    "Berlin is the capital of Germany."
]

result = embedding.embed_documents(documents)
print(str(result))