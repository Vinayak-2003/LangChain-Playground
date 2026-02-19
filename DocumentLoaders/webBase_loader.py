from langchain_community.document_loaders import WebBaseLoader
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(model="qwen/qwen3-32b")
parser = StrOutputParser()
prompt = PromptTemplate(
    template="Write s brief summary pointers for the following article: \n {article}",
    input_variables=["article"]
)

url = "https://reference.langchain.com/python/langchain/"

loader = WebBaseLoader(url)

docs = loader.load()

print(len(docs))
print(type(docs), type(docs[0]))

chain = prompt | model | parser
output = chain.invoke({'article': docs[0].page_content})
print(output)

# print(docs)