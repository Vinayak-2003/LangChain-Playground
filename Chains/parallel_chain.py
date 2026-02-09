from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from dotenv import load_dotenv

load_dotenv()

model1 = ChatGroq(model="qwen/qwen3-32b")
model2 = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")
parser = StrOutputParser()

prompt1 = PromptTemplate(
    template="Generate short and simple notes from the following text \n {text}",
    input_variables=["text"]
)

prompt2 = PromptTemplate(
    template="Generate 5 short question answers from the following text \n {text}",
    input_variables=["text"]
)

prompt3 = PromptTemplate(
    template="Merge the provided notes and quiz into a single document \n notes -> {notes} \n quiz -> {quiz}",
    input_variables=["notes", "quiz"]
)

parallel_chain = RunnableParallel({
    "notes": prompt1 | model1 | parser,
    "quiz": prompt2 | model2 | parser
})

merge_chain = prompt3 | model1  | parser

chain = parallel_chain | merge_chain | parser

text = """
Artificial Intelligence (AI) has evolved from simple rule-based systems to models that can create, reason and act independently. They have developed from AI Agents to Generative AI and Agentic AI. While they sound similar, each serves a different purpose and represents a unique stage in AI capability.

What is Generative AI?
Generative AI refers to models that can create new content such as text, images, videos or code by learning from existing data. It doesn't perform real actions, it simply provides responses based on available data.

Trained on large datasets using deep learning.
Generates creative outputs (text, design, audio, etc.).
Works based on user prompts and learned patterns.
Common examples: ChatGPT, DALL·E, Midjourney, Gemini.
Example: When you ask, "What is the price of Emirates flight from New York to Delhi tomorrow?"

generative_al_with_only_llm2
Generative AI
The LLM predicts and generates a possible answer based on patterns learned from its training data. However, since these models are trained on past data and do not have real-time access to flight prices or current APIs, their answers may be outdated or inaccurate. The model might respond with something like: “I do not know the latest price.”

What is AI Agent?
AI Agents are systems designed to perform specific tasks automatically using defined instructions and external tools. Traditional AI Agents function within strict guidelines i.e they process inputs and respond based on preprogrammed rules, making them predictable but limited in adaptability.

Modern AI Agents, however combine tool access and automation to handle slightly more complex tasks, though they still rely on human-defined objectives and boundaries.

Example: When you ask, “Tell me the cheapest Emirates flight from New York to Delhi tomorrow”

generative_al_with_only_llm
AI Agent
How it works:

Connects with APIs like MakeMyTrip to fetch flight data.
Filters results according to predefined rules such as price or class.
Returns the cheapest available options.
What is Agentic AI?
Agentic AI goes beyond simple generation, it can reason, plan and take actions automatically. Agentic AI goes beyond executing commands. It determines objectives, refines strategies and adapts based on external factors making it more flexible and dynamic.

Uses reasoning and planning to decide next steps.
Can use tools, perform web searches or execute actions automatically.
Works through multi-step reasoning and feedback loops.
Often built on top of Gen AI models (like GPT-4 or Gemini 2.0).
Example: When you ask, “Book me a flight for my 7-day trip to New Delhi in May. The weather should be sunny all days, budget below $1600 and no layovers.”

multi_ai_booking_agents
Agentic AI
How it works:

Understands the goal: interprets the intent behind the user's request.
Plans a strategy: breaks the goal into smaller steps (check weather, find flights, compare prices).
Checks visa validity: verifies if the traveller's visa is active or expired before booking.
Interacts with tools/APIs: uses real-time data sources like AccuWeather and MakeMyTrip.
Adapts to context: if flights exceed the budget or weather doesn't match, it adjusts the plan.
Executes or recommends: completes the booking or suggests the best available alternative.
Comparison between Gen AI vs AI Agents vs Agentic AI
Feature	Generative AI	AI Agents	Agentic AI
Primary Function	Generates new content (text, images, videos, code).	Performs specific predefined tasks automatically.	Plans, reasons and acts independently toward a goal.
Dependency	Works based on trained data.	Depends on predefined rules and APIs.	Uses reasoning, planning and feedback loops.
Tool Access	No real-time access to tools or APIs.	Limited access to tools for specific actions.	Dynamic access to multiple tools and APIs.
Learning Capability	Learns from historical data (not real-time).	No continuous learning; follows instructions.	Continuously adapts to context and outcomes.
Example Use Case	Writing articles, creating designs.	Fetching flight data or sending alerts.	Booking flights based on preferences and constraints.
Autonomy Level	Low	Medium	High
Example Tools/Models	ChatGPT, DALL·E, Midjourney	Customer service bots, automation systems	AutoGPT, Devin AI, Gemini 2.0 with planning capabilities
When to Use What
1. Generative AI

Best for creating new content like articles, images, music or code.
Useful when real-time data or tool access is not required.
Example: Writing an essay or designing a logo.
2. AI Agents

Ideal for automating predefined tasks that follow fixed logic.
Works well when tasks involve APIs or databases but require no reasoning.
Example: Fetching flight data or scheduling a meeting.
3. Agentic AI

Suitable for complex, goal-oriented workflows that require reasoning and planning.
Can adapt to changes — like adjusting the plan if the visa is expired or the weather isn’t suitable.
Example: Planning and booking a trip under budget, checking real-time data and executing actions automatically.
"""

result = chain.invoke({"text": text})

print(result)

chain.get_graph().print_ascii()