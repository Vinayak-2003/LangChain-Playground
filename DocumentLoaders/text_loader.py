from langchain_community.document_loaders import TextLoader
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(model="qwen/qwen3-32b")
parser = StrOutputParser()
prompt = PromptTemplate(
    template="Write a summary for the following poem: \n {poem}",
    input_variables=["poem"]
)

loader = TextLoader(
    "DocumentLoaders/cricket.txt",
    encoding="utf-8",
)

docs = loader.load()

print(docs)
print(type(docs))
print(len(docs))
print(type(docs[0]))
print(docs[0].page_content)
print(docs[0].metadata)


chain = prompt | model | parser
output = chain.invoke({'poem': docs[0].page_content})
print(output)