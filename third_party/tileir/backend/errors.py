from triton.errors import TritonError


class HitFallback(TritonError):

    def __init__(self, required, name):
        self.required = required
        self.name = name

    def __str__(self) -> str:
        return f"HitFallback: {self.name}, Required: {self.required}."

    def __reduce__(self):
        # this is necessary to make CompilationError picklable
        return (type(self), (self.required, self.name))
