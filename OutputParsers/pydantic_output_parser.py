from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1-0528",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

class Person(BaseModel):
    name: str = Field(description="The name of the fictional person")
    age: int = Field(gt=18, description="The age of the fictional person")
    city: str = Field(description="The city where the fictional person lives")

parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template="Generate the name, age and city of a fictional {place} person \n {format_instruction}",
    input_variables=["place"],
    partial_variables={"format_instruction": parser.get_format_instructions()}
)

prompt = template.invoke({'place': 'Indian'})
print(prompt)

result = model.invoke(prompt)
final_result = parser.parse(result.content)

print(final_result)


# chain = template | model | parser
# result = chain.invoke({'place': 'Indian'})