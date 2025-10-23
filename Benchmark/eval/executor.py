import contextlib
import io
import traceback

def passes_all_tests(code_str: str, test_code: str, timeout: float = 2.0) -> bool:
    """
    Executes the generated code and runs the provided test cases.
    Returns True if all tests pass, False otherwise.
    """
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            exec_env = {}
            exec(code_str, exec_env)
            exec(test_code, exec_env)
        return True
    except Exception:
        return False