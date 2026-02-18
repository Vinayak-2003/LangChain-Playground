from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableSequence
from dotenv import load_dotenv
import json

load_dotenv()

model = ChatGroq(model="qwen/qwen3-32b")
parser = StrOutputParser()

prompt1 = PromptTemplate(
    template="Create a tweet for posting on X on the topic {topic}",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="Create a linkedIn post on the topic {topic}",
    input_variables=["topic"]
)

chain1 = RunnableSequence(prompt1, model, parser)
chain2 = RunnableSequence(prompt2, model, parser)

parallel_chain = RunnableParallel({'tweet': chain1, 'linkedin_post': chain2})
output = parallel_chain.invoke({'topic': 'Artificial Intelligence'})

with open("Runnables/tweet_linkedin_output.json", "w") as f:
    json.dump(output, f)

print("Generated content saved to tweet_linkedin_output.json")