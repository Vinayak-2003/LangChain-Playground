from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_core.tools import InjectedToolArg
from typing import Annotated
from dotenv import load_dotenv
import os
import requests
import json

load_dotenv()
CURRENCY_API_KEY = os.getenv("CURRENCY_API_KEY")

# tool creation
@tool
def get_conversion_factor(base_currency: str, target_currency: str) -> float:
    """
    This function fetches the currency conversion factor between a given base currency and a target currency.
    """
    try:
        url = f"https://v6.exchangerate-api.com/v6/{CURRENCY_API_KEY}/pair/{base_currency}/{target_currency}"
        response = requests.get(url)
        return response.json()
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    

@tool
def convert_currency(base_amount: float, 
                     conversion_rate: Annotated[float, InjectedToolArg]
                    ) -> float:
    """
    This function returns the final amount after conversion providing the base amount and the conversion rate.
    """
    return base_amount * conversion_rate

# tool binding
llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview")
llm_with_tools = llm.bind_tools([get_conversion_factor, convert_currency])

groq_llm = ChatGroq(model="llama-3.3-70b-versatile")
groq_llm_with_tools = groq_llm.bind_tools([get_conversion_factor, convert_currency])

messages = [HumanMessage("What is the conversion factor between USD and INR, and based on that convert 10 USD to INR")]

# tool calling
ai_message = groq_llm_with_tools.invoke(messages)
messages.append(ai_message)

"""
print(ai_message.tool_calls)
----------------- before InjectedToolArg -----------------

[
    {
        "name": "get_conversion_factor",
        "args": {
            "base_currency": "USD",
            "target_currency": "INR"
        },
        "id": "cmhqpentc",
        "type": "tool_call"
    },
    {
        "name": "convert_currency",
        "args": {
            "base_amount": 10,
            "conversion_rate": 82.89        # here it automatically used the value of past record and not the real time value
        },
        "id": "fsr9j4tc3",
        "type": "tool_call"
    }
]


----------------- after InjectedToolArg -----------------

[
    {
        "name": "get_conversion_factor",
        "args": {
            "base_currency": "USD",
            "target_currency": "INR"
        },
        "id": "cq85h6qsa",
        "type": "tool_call"
    },
    {
        "name": "convert_currency",
        "args": {
            "base_amount": 10
        },
        "id": "z6h704cby",
        "type": "tool_call"
    }
]
"""

for tool_call in ai_message.tool_calls:
    if tool_call['name'] == "get_conversion_factor":
        tool_message1 = get_conversion_factor.invoke(tool_call)
        messages.append(tool_message1)
        """
        ToolMessage(content='{"result": "success", "documentation": "https://www.exchangerate-api.com/docs", 
        "terms_of_use": "https://www.exchangerate-api.com/terms", "time_last_update_unix": 1773446402, 
        "time_last_update_utc": "Sat, 14 Mar 2026 00:00:02 +0000", "time_next_update_unix": 1773532802, 
        "time_next_update_utc": "Sun, 15 Mar 2026 00:00:02 +0000", "base_code": "USD", "target_code": "INR", 
        "conversion_rate": 92.5544}', name='get_conversion_factor', tool_call_id='wwc6vs7mb')
        """
        conversion_rate = json.loads(tool_message1.content)['conversion_rate']
    
    if tool_call["name"] == "convert_currency":
        tool_call["args"]["conversion_rate"] = conversion_rate
        tool_message2 = convert_currency.invoke(tool_call)
        messages.append(tool_message2)
        """
        ToolMessage(content='925.544', name='convert_currency', tool_call_id='b75n6v0c8')
        """
 
result = groq_llm_with_tools.invoke(messages)
print(result.content)
"""
The conversion factor between USD and INR is 92.5544. Based on this conversion factor, 10 USD is equivalent to 925.544 INR.
"""

"""
This is not an AI agent as we have done a lot of decision making and values injection manually
"""