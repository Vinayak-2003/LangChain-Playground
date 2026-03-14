from langchain_core.tools import tool

@tool
def multiply(a: int, b: int) -> int:
    """Multiplies two numbers"""
    return a*b

result = multiply.invoke({"a": 4, "b": 5})
print(result)


print(multiply.name)
print(multiply.description)
print(multiply.args)

print(multiply.args_schema.model_json_schema())
"""
{
    "description": "Multiplies two numbers",
    "properties": {
        "a": {
            "title": "A",
            "type": "integer"
        },
        "b": {
            "title": "B",
            "type": "integer"
        }
    },
    "required": [
        "a",
        "b"
    ],
    "title": "multiply",
    "type": "object"
}
"""

