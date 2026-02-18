from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda
from dotenv import load_dotenv

load_dotenv()

def word_count(text: str):
    return len(text.split())

prompt1 = PromptTemplate(
    template="Write a joke about {topic}",
    input_variables=["topic"]
)

model = ChatGroq(model="qwen/qwen3-32b")

parser = StrOutputParser()

joke_chain = RunnableSequence(prompt1, model, parser)

parallel_chain = RunnableParallel({
    "joke": RunnablePassthrough(),
    "words_count": RunnableLambda(word_count)
})

final_chain = RunnableSequence(joke_chain, parallel_chain)
output = final_chain.invoke({'topic': 'dogs'})
print(output)