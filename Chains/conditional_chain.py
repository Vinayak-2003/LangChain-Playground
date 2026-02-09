from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal
from langchain_core.runnables import RunnableBranch, RunnableLambda
from dotenv import load_dotenv

load_dotenv()

class Sentiment(BaseModel):
    sentiment: Literal['positive', 'negative'] = Field(description="Give me the sentiment of the feedback as positive or negative")

model1 = ChatGroq(model="qwen/qwen3-32b")

str_parser = StrOutputParser()
pydantic_parser = PydanticOutputParser(pydantic_object=Sentiment)

prompt1 = PromptTemplate(
    template="Classify the sentiment of the following feedback text into positive or negative \n {feedback} \n {format_instruction}",
    input_variables=["feedback"],
    partial_variables={"format_instruction": pydantic_parser.get_format_instructions()}
)

classifier_chain = prompt1 | model1 | pydantic_parser

prompt2 = PromptTemplate(
    template="Generate a response for the positive feedback \n {feedback}",
    input_variables=["feedback"]
)

prompt3 = PromptTemplate(
    template="Generate a response for the negative feedback \n {feedback}",
    input_variables=["feedback"]
)

feedback_chain = RunnableBranch(
    (lambda x:x.sentiment == 'positive', prompt2 | model1 | str_parser),
    (lambda x:x.sentiment == 'negative', prompt3 | model1 | str_parser),
    RunnableLambda(lambda x: "Could not determin sentiment")
)

chain = classifier_chain | feedback_chain

result = chain.invoke({"feedback": "The product quality is terrible and the customer service was unhelpful."})

print(result)

chain.get_graph().print_ascii()