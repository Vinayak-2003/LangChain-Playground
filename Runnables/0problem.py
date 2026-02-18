import random

# demo implementation of LLM
class demoLLM:
    def __init__(self):
        print("LLM created")

    def predict(self, prompt):
        response_list = [
            'Delhi is the capital of India',
            'IPL is a cricket tournament',
            'AI stands for Artificial Intelligence',
        ]

        return {'response': random.choice(response_list)}
    
# demo implementation of PromptTemplate
class demoPromptTemplate:
    def __init__(self, template, input_variables):
        self.template = template
        self.input_variables = input_variables

    def format(self, input_dict):
        return self.template.format(**input_dict)
    
# demo implementation of LLMChain
class demoLLMChain:
    def __init__(self, llm, prompt):
        self.llm = llm
        self.prompt = prompt

    def run(self, input_dict):
        final_prompt = self.prompt.format(input_dict)
        result = self.llm.predict(final_prompt)
        return result['response']
    

# creating llm object
llm = demoLLM()

response = llm.predict("What is the capital of India?")
print(response)

# creating prompt template object
template = demoPromptTemplate(
    template="write a {length} poem about {topic}",
    input_variables=["length", "topic"]
)

prompt = template.format({"length": "short", "topic": "nature"})
print(prompt)

prompt_response = llm.predict(prompt)
print(prompt_response)

# creating llm chain object
chain = demoLLMChain(llm=llm, prompt=template)
chain_response = chain.run({"length": "short", "topic": "nature"})
print(chain_response)

# this creates a problem to the deveopers as we are not able to call the LLM twice in the same code, we need to create a new object of the LLM to call it again, this is not efficient and also not practical in real world scenarios.