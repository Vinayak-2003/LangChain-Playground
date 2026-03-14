from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type

class MultiplyArgs(BaseModel):
    a: int = Field(..., description="The first number to multiply")
    b: int = Field(..., description="The second number to multiply")


class MultiplyTool(BaseTool):
    name: str = "multiply"
    description: str = "Multiplies two numbers"

    args_schema: Type[BaseModel] = MultiplyArgs

    def _run(self, a: int, b: int) -> int:
        return a * b
    

multiply_tool = MultiplyTool()
result = multiply_tool.invoke({"a": 3, "b": 8})
print(result)

print(multiply_tool.name)
print(multiply_tool.description)
print(multiply_tool.args_schema)
"""
24
multiply
Multiplies two numbers
<class '__main__.MultiplyArgs'>
"""

