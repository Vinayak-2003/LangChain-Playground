from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(model="qwen/qwen3-32b")
response = model.invoke("Write 5 line poem on cricket")
print(response.content)