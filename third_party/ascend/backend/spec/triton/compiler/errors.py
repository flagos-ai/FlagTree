import importlib.util
import sys
from typing import Optional
from triton.compiler.errors import TritonError

class MLIRCompilationError(TritonError):
    def __init__(self, stage_name: Optional[str], message: Optional[str] = None):
        self.stage_name = stage_name
        self.message = f"\n" \
            f"{self.format_line_delim('[ERROR][Triton][BEG]')}" \
            f"[{self.stage_name}] encounters error:\n" \
            f"{self.filter_message(message)}" \
            f"{self.format_line_delim('[ERROR][Triton][END]')}"
    def __str__(self):
        return self.message
    def filter_message(self, message):
        # Content starting from "Stack dump without symbol names" means nothing to the users
        return message.split("Stack dump without symbol names")[0]
    def format_line_delim(self, keyword):
        return f"///------------------{keyword}------------------\n"
