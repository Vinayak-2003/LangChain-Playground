from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

class MultiplyArgs(BaseModel):
    a: int = Field(..., description="The first number to multiply")
    b: int = Field(..., description="The second number to multiply")

def multiply_func(a, b):
    return a * b

multiply_tool = StructuredTool.from_function(
    func=multiply_func,
    name="multiply",
    description="Multiplies two numbers",
    args_schema=MultiplyArgs
)

result = multiply_tool.invoke({"a": 7, "b": 6})
print(result)

print(multiply_tool.name)
print(multiply_tool.description)
print(multiply_tool.args_schema)
"""
42
multiply
Multiplies two numbers
<class '__main__.MultiplyArgs'>
"""

