from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1-0528",
    task="text-generation",
)

model = ChatHuggingFace(llm=llm)

template = PromptTemplate(
    template="Greet the user in 5 different langauges, his name is {name}",
    input_variables=['name']
)

user_name = input("Enter your name: ")
prompt = template.invoke({"name": user_name})
response = model.invoke(prompt)

print(response.content)