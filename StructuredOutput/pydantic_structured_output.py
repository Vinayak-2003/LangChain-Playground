from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Optional, Literal

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1-0528",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

class ReviewModel(BaseModel):
    key_themes: list[str] = Field(description="Key themes discussed in the review")
    summary: str = Field(description="A brief summary of the review")
    sentiment: str = Field(description="Return sentiment of the review either negative, neutral or positive")
    sentiment_literal: Literal["pos", "neg"] = Field(description="Return sentiment of the review either negative, neutral or positive")
    pros: Optional[str] = Field(None, description="List the pros mentioned in the review, if any")
    cons: Optional[str] = Field(None, description="List the cons mentioned in the review, if any")

structured_model = model.with_structured_output(ReviewModel)

prompt = """The hardware is great, but the software feels bloated. There are too many pre-installed apps that I can't remove. Also, the UI looks outdated compared to other brands. Hoping for a software update to fix this."""

structured_result = structured_model.invoke(prompt)
print(structured_result)