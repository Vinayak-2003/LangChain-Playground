from langchain_core.tools import tool

@tool
def add(a: int, b: int) -> int:
    """Adds two numbers"""
    return a+b

@tool
def subtract(a: int, b: int) -> int:
    """Subtracts two numbers"""
    return a-b

@tool
def multiply(a: int, b: int) -> int:
    """Multiplies two numbers"""
    return a*b

@tool
def divide(a: int, b: int) -> int:
    """Divides two numbers"""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a/b

# creating toolkit class to group all the math tools together
class MathToolkit:
    def get_tools(self):
        return [add, subtract, multiply, divide]
    

math_toolkit = MathToolkit()
tools = math_toolkit.get_tools()

for mathTool in tools:
    result = mathTool.invoke({"a": 10, "b": 5})
    print(f"{mathTool.name} result: {result}")
    print(f"{mathTool.name} description: {mathTool.description}")
    print(f"{mathTool.name} args: {mathTool.args}")
    print("\n")


"""
add result: 15
add description: Adds two numbers
add args: {'a': {'title': 'A', 'type': 'integer'}, 'b': {'title': 'B', 'type': 'integer'}}


subtract result: 5
subtract description: Subtracts two numbers
subtract args: {'a': {'title': 'A', 'type': 'integer'}, 'b': {'title': 'B', 'type': 'integer'}}


multiply result: 50
multiply description: Multiplies two numbers
multiply args: {'a': {'title': 'A', 'type': 'integer'}, 'b': {'title': 'B', 'type': 'integer'}}


divide result: 2.0
divide description: Divides two numbers
divide args: {'a': {'title': 'A', 'type': 'integer'}, 'b': {'title': 'B', 'type': 'integer'}}
"""