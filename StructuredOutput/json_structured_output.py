from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1-0528",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

json_schema = {
    "title": "Review",
    "description": "schema for a product review",
    "type": "object",
    "properties": {
        "key_themes": {
            "type": "array",
            "items": {
                "type": "string"
            },
            "description": "List of key themes mentioned in the review in a list format"
        },
        "summary": {
            "type": "string",
            "description": "A concise summary of the review"
        },
        "sentiment": {
            "type": "string",
            "enum": ["pos", "neg", "neut"],
            "description": "The overall sentiment of the review positive, negatice or neutral"
        },
        "pros": {
            "type": ["array", "null"],
            "items": {
                "type": "string"
            },
            "description": "List of pros mentioned in the review in a list format"
        },
        "cons": {
            "type": ["array", "null"],
            "items": {
                "type": "string"
            },
            "description": "List of cons mentioned in the review in a list format"
        }
    },
    "required": ["summary", "sentiment"]
}

structured_model = model.with_structured_output(json_schema)

prompt = """The hardware is great, but the software feels bloated. There are too many pre-installed apps that I can't remove. Also, the UI looks outdated compared to other brands. Hoping for a software update to fix this."""

structured_result = structured_model.invoke(prompt)
print(structured_result)
