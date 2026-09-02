# Copyright 2018-2020 Philippe Tillet
# Copyright 2020-2022 OpenAI
# Copyright 2025-     FlagOS Contributors
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

from triton.experimental.gluon.language._core import builtin

__all__ = ["arrive", "wait"]


@builtin
def arrive(_semantic=None):
    """
    Signals that the cluster has arrived at a cluster barrier, used to synchronize execution of CTAs within the same cluster.
    """
    _semantic.builder.create_amd_cluster_arrive()


@builtin
def wait(_semantic=None):
    """
    Wait on a cluster barrier to be arrived by all CTAs within the same cluster.
    Arrive and wait operations must come in pairs. Waiting before arriving or arriving more than once
    without a corresponding wait will result in undefined behavior.
    """
    _semantic.builder.create_amd_cluster_wait()
