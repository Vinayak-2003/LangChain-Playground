from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

text = "Delhi is the capital of India."

documents = [
    "Delhi is the capital of India.",
    "Paris is the capital of France.",
    "Berlin is the capital of Germany."
]

result = embedding.embed_query(text)
print(str(result))