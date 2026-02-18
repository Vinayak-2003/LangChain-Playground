from abc import ABC, abstractmethod
import random

class Runnable(ABC):

    @abstractmethod
    def invoke(input_data):
        pass

# demo implementation of LLM
class demoLLM(Runnable):
    def __init__(self):
        print("LLM created")

    def invoke(self, prompt):
        response_list = [
            'Delhi is the capital of India',
            'IPL is a cricket tournament',
            'AI stands for Artificial Intelligence',
        ]

        return {'response': random.choice(response_list)}

    def predict(self, prompt):
        response_list = [
            'Delhi is the capital of India',
            'IPL is a cricket tournament',
            'AI stands for Artificial Intelligence',
        ]

        return {'response': random.choice(response_list)}
    
# demo implementation of PromptTemplate
class demoPromptTemplate(Runnable):
    def __init__(self, template, input_variables):
        self.template = template
        self.input_variables = input_variables

    def invoke(self, input_dict):
        return self.template.format(**input_dict)

    def format(self, input_dict):
        return self.template.format(**input_dict)
    
class demoStrOutputParser(Runnable):
    def __init__(self):
        pass

    def invoke(self, input_data):
        return input_data["response"]
    
class RunnableConnector(Runnable):
    def __init__(self, runnable_list):
        self.runnable_list = runnable_list

    def invoke(self, input_data):

        for runnable in self.runnable_list:
            input_data = runnable.invoke(input_data)

        return input_data


# creating llm object
llm = demoLLM()

response = llm.predict("What is the capital of India?")

# creating prompt template object
template = demoPromptTemplate(
    template="write a {length} poem about {topic}",
    input_variables=["length", "topic"]
)

parser = demoStrOutputParser()

chain = RunnableConnector(runnable_list=[template, llm, parser])
output = chain.invoke({"length": "short", "topic": "nature"})
# print(output)


# connecting multiple runnables together
template1 = demoPromptTemplate(
    template="Write a joke about {topic}",
    input_variables=["topic"]
)

template2 = demoPromptTemplate(
    template="Explain the following joke {response}",
    input_variables=["response"]
)

chain1 = RunnableConnector([template1, llm])
chain2 = RunnableConnector([template2, llm, parser])

final_chain = RunnableConnector([chain1, chain2])
response = final_chain.invoke({"topic": "programming"})
print("Multiple Runnable: ", response)