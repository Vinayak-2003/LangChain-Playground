from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_classic.agents.agent import AgentExecutor
from langchain_classic.agents.react.agent import create_react_agent
from langchain_classic import hub
from langchain_core.tools import tool
import requests
from dotenv import load_dotenv
import os

load_dotenv()

WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")

# Step 1: Tool creation
search_tool = DuckDuckGoSearchRun()

@tool
def get_weather_forecast(location: str) -> str:
    """
    This function fetches the current weather forecast for a given location or city.
    """
    url = f"https://api.weatherapi.com/v1/current.json?key={WEATHER_API_KEY}&q={location}&aqi=no"
    response = requests.get(url)
    return response.json()


groq_llm = ChatGroq(model="llama-3.3-70b-versatile")

# Step  2: Pull the ReAct prompt from langchain hub
# pulls the standard ReAct agent  (famous design pattern agent)
react_prompt = hub.pull("hwchase17/react")

# Step 3: Create the ReAct agent manually with the pulled prompt
# AIM -> Thinking
agent = create_react_agent(
    llm=groq_llm,
    tools=[search_tool, get_weather_forecast],
    prompt=react_prompt
)

# Step 4: wrap it with AgentExecutor
# AIM -> Doing
agent_executor = AgentExecutor(
    agent=agent,
    tools=[search_tool, get_weather_forecast],
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=5
)

# Step 5: Invoke
response = agent_executor.invoke({"input": "What is the capital of Uttar Pradesh and what is the weather there?"})
print(response)

 