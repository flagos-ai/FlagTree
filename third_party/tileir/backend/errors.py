from typing import Optional

from triton.errors import TritonError
from triton.runtime.errors import BackendCompilerError


class TileirasError(BackendCompilerError):

    def __init__(self, error_message: Optional[str] = None):
        self.error_message = error_message

    def __str__(self) -> str:
        error_message = self.error_message or ""
        return f"tileiras error: {error_message}"


class HitFallback(TritonError):

    def __init__(self, required, name):
        self.required = required
        self.name = name

    def __str__(self) -> str:
        return f"HitFallback: {self.name}, Required: {self.required}."

    def __reduce__(self):
        # this is necessary to make CompilationError picklable
        return (type(self), (self.required, self.name))
