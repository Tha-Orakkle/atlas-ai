import ast
import operator

calculate_tool = {
    "type": "function",
    "name": "calculate",
    "description": (
        "Evaluate a basic arithmetic expression. "
        "Supports addition, subtraction, multiplication, "
        "division, exponentiation, decimals, and parentheses."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "A basic arithmetic expression to evaluate."
            },
        },
        "required": ["expression"],
        "additionalProperties": False,
    },
    "strict": True
}

OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos
}


def evaluate(node: ast.AST) -> int | float:
    """
    Recursively evaluates an AST node representing a mathematical expression.
    Args:
        - node: An AST node representing a mathematical expression.
    Returns:
        - The result of the evaluated expression as an integer or float.
    Raises:
        - ValueError: If the node contains unsupported operations or types.
    """

    if isinstance(node, ast.Constant):
        if not isinstance(node.value, (int, float)):
            raise ValueError("Only numbers are allowed.")

        return node.value

    if isinstance(node, (ast.BinOp, ast.UnaryOp)):
        operation = OPERATORS.get(type(node.op))
        if not operation:
            raise ValueError("Unsupported operator.")

    if isinstance(node, ast.BinOp):
        left = evaluate(node.left)
        right = evaluate(node.right)

        return operation(left, right)

    if isinstance(node, ast.UnaryOp):
        operand = evaluate(node.operand)
        return operation(operand)

    raise ValueError("Unsupported expression.")


def calculate(expression: str) -> dict[str, int | float | str]:
    """
    Performs basic arithmetic operations including addition, subtraction,
    multiplication, division, exponentiation, parentheses, positive,
    and negative.
    Args:
        - expression(str): a string of operation. e.g '3 * 4',
          '4 ** 2)', '(2 * 3 + 4) - 3'.
    Returns:
        - dict: a dictionary containing the result of the calculation
          or an error message.
    """

    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        return {"error": f"Invalid mathematical expression: {exc}"}

    try:
        result = evaluate(tree.body)
    except ValueError as exc:
        return {"error": f"Unable to perform operation: {exc}"}

    return {"result": result}
