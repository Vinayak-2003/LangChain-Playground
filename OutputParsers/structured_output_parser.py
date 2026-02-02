from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1-0528",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

class Facts(BaseModel):
    fact1: str = Field(description="The first interesting fact")
    fact2: str = Field(description="The second interesting fact")
    fact3: str = Field(description="The third interesting fact")

parser = JsonOutputParser(output_schema=Facts)

template = PromptTemplate(
    template="Give me three interesting facts about {subject} \n {format_instruction}",
    input_variables=["subject"],
    partial_variables={"format_instruction": parser.get_format_instructions()}
)

chain = template | model | parser
result = chain.invoke({'subject': 'space exploration'})
print(result)