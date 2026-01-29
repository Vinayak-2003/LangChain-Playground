from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()

model = ChatAnthropic(model="claude-3.5-sonnet-20241022", temperature=1.5, max_completion_tokens=10)
response = model.invoke("Write 5 line poem on cricket")

print(response.content)