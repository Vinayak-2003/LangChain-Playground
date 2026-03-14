from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
import requests
from dotenv import load_dotenv
import os

load_dotenv()

os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY")

# tool creation
@tool
def multiply(a: int, b: int) -> int:
    """Multiplies two numbers to return their product."""
    return a * b

llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview")

# tool binding
llm_with_tools = llm.bind_tools([multiply])

# tool calling
normal_response = llm_with_tools.invoke("Hi, how are you?")
"""
[{'type': 'text', 'text': "I'm doing well, thank you for asking! How can I help you today?", 'extras': {'signature': 'Ev8BCvwBAb4+9vu/TZCy7kDqQT96Un3EutD2yT0c71okZllnrclOEuqlcYdUE7CM/+kfwxjQUxb9qvZ5uowzzNedZVt/njYS9nWJkBj21CQ49nR7s0KkHqzsf08Mrq5EQ/CtIoyWZKZqGXW2SDXWvSeAfbdON8uGnLxb0NIjW1w0UCPqPs4H8V4jM7mdaGL0Xhxkw8xJqYeXS1MCTJeI6rUy2z30pJFR6m9778FZOiJWlxNtliADFzvfyhd7yn5/O70UgTx3E6tqEEVVTGr1uVyVADL6nLOevssOwWImIR2sTSTvJ+my+6g+44SIwh4Ou8e8IWBvT5R01mxjHOA9/dSJ'}}] additional_kwargs={} response_metadata={'finish_reason': 'STOP', 'model_name': 'gemini-3-flash-preview', 'safety_ratings': [], 'model_provider': 'google_genai'} id='lc_run--019ceb8e-ff13-7c61-991a-1014c56e6b73-0' tool_calls=[] invalid_tool_calls=[] usage_metadata={'input_tokens': 64, 'output_tokens': 74, 'total_tokens': 138, 'input_token_details': {'cache_read': 0}, 'output_token_details': {'reasoning': 56}}
"""


query = HumanMessage("What is the product of 5 and 7?")
messages = [query]

multiply_response = llm_with_tools.invoke(messages)
messages.append(multiply_response)
"""
content=[] additional_kwargs={'function_call': {'name': 'multiply', 'arguments': '{"b": 7, "a": 5}'}, '__gemini_function_call_thought_signatures__': {'ca77c1f1-7628-4956-a0cb-5d1106a85341': 'EpcBCpQBAb4+9vsAOY+wr6mUXXiC+myvTXobDbzjejB63Rx6Nm8jueKNV+q7utOP6YOy6eWwnXBhN0WxoyIVWswN5XJ6A8HW2g1nGOhbZugJNcHhYc8KvOZVW6IGDU2o9Wvy/ZfD2kJDykz8fCjyT0ikXwn+a6ddtk6zYLKdX38grBcE1IwyLOOj8RCShxOaP89M7g+3I3tH2A=='}} response_metadata={'finish_reason': 'STOP', 'model_name': 'gemini-3-flash-preview', 'safety_ratings': [], 'model_provider': 'google_genai'} id='lc_run--019ceb8f-0862-7e93-a9e2-6c7e955d6511-0' tool_calls=[{'name': 'multiply', 'args': {'b': 7, 'a': 5}, 'id': 'ca77c1f1-7628-4956-a0cb-5d1106a85341', 'type': 'tool_call'}] invalid_tool_calls=[] usage_metadata={'input_tokens': 69, 'output_tokens': 42, 'total_tokens': 111, 'input_token_details': {'cache_read': 0}, 'output_token_details': {'reasoning': 26}}
"""

"""
print(multiply_response.tool_calls[0])

{
    "name": "multiply",
    "args": {
        "b": 7,
        "a": 5
    },
    "id": "22642258-29f7-49e6-a8af-936550405b1b",
    "type": "tool_call"
}
"""

"""
print(multiply_response.tool_calls[0]["args"])

{
    "b": 7,
    "a": 5
}
"""

# tool execution
tool_response = multiply.invoke(multiply_response.tool_calls[0])
messages.append(tool_response)
"""
content='35' name='multiply' tool_call_id='83072f16-a33e-48dc-9447-5716e3ce5b06'
"""


"""
print(messages)

[HumanMessage(content='What is the product of 5 and 7?', additional_kwargs={}, response_metadata={}), AIMessage(content=[], additional_kwargs={'function_call': {'name': 'multiply', 'arguments': '{"a": 5, "b": 7}'}, '__gemini_function_call_thought_signatures__': {'2650a57c-6c6a-4ebe-b449-f2a870b12612': 'Eo4BCosBAb4+9vtV0qsOyn4by2wVSWrb4dZLyhTYEfNTTj8I4Yxk2xPwubvxg+rdVvWUHaqWyr7soiaDRN/6tPwNftkEci7htS3Ixmu+V8XSdyE1cDXKREJIVZpcEHhvYubiXF8OzL0tUjckzPXmwaGtvMBcE2uWo/SkNSh1oCxAcMzKDTH6BTi6EnLFn0X22w=='}}, response_metadata={'finish_reason': 'STOP', 'model_name': 'gemini-3-flash-preview', 'safety_ratings': [], 'model_provider': 'google_genai'}, id='lc_run--019cebca-b223-7473-a14f-4923f322b0c2-0', tool_calls=[{'name': 'multiply', 'args': {'a': 5, 'b': 7}, 'id': '2650a57c-6c6a-4ebe-b449-f2a870b12612', 'type': 'tool_call'}], invalid_tool_calls=[], usage_metadata={'input_tokens': 69, 'output_tokens': 41, 'total_tokens': 110, 'input_token_details': {'cache_read': 0}, 'output_token_details': {'reasoning': 25}}), ToolMessage(content='35', name='multiply', tool_call_id='2650a57c-6c6a-4ebe-b449-f2a870b12612')]
"""

result = llm_with_tools.invoke(messages)
print(result.content[0]["text"])

"""
print(result)

content=[{'type': 'text', 'text': 'The product of 5 and 7 is 35.'}] additional_kwargs={} response_metadata={'finish_reason': 'STOP', 'model_name': 'gemini-3-flash-preview', 'safety_ratings': [], 'model_provider': 'google_genai'} id='lc_run--019cebcd-0175-7831-a25f-4fb366777b19-0' tool_calls=[] invalid_tool_calls=[] usage_metadata={'input_tokens': 154, 'output_tokens': 13, 'total_tokens': 167, 'input_token_details': {'cache_read': 0}}
"""