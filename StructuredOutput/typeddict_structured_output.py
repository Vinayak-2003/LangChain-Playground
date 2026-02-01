from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional, Literal

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1-0528",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

class Review(TypedDict):
    key_themes: Annotated[str, "Key themes discussed in the review"]
    summary: Annotated[str, "A brief summary of the review"]
    sentiment: Annotated[str, "Return sentiment of the review either negative, neutral or positive"]
    sentiment_literal: Annotated[Literal["pos", "neg"], "Return sentiment of the review either negative, neutral or positive"]
    pros: Annotated[Optional[str], "List the pros mentioned in the review, if any"]
    cons: Annotated[Optional[str], "List the cons mentioned in the review, if any"]

structured_model = model.with_structured_output(Review)

prompt = """The hardware is great, but the software feels bloated. There are too many pre-installed apps that I can't remove. Also, the UI looks outdated compared to other brands. Hoping for a software update to fix this."""

structured_result = structured_model.invoke(prompt)
print(structured_result)

# unstructured_result = model.invoke(prompt)
# print(type(unstructured_result), unstructured_result)


