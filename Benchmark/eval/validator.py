import ast

def is_ast_valid(code_str: str) -> bool:
    """Check if the generated code is syntactically valid Python."""
    try:
        ast.parse(code_str)
        return True
    except SyntaxError:
        return False