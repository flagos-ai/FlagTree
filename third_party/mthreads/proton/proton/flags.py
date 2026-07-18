# Copyright 2026 FlagOS Contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Centralized, process-local flags with a minimal interface (no environment variables).

Usage:
    from triton.profiler.flags import flags

    # Toggle
    flags.profiling_on = True
    flags.instrumentation_on = False

    # Check
    if flags.command_line:
            ...
"""
from dataclasses import dataclass


@dataclass
class ProfilerFlags:
    # Whether profiling is enabled. Default is False.
    profiling_on: bool = False
    # Whether instrumentation is enabled. Default is False.
    instrumentation_on: bool = False
    # Whether the script is run from the command line. Default is False.
    command_line: bool = False


flags = ProfilerFlags()
