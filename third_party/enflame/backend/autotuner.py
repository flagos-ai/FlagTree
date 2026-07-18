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

from triton.testing import do_bench, do_bench_cudagraph
from triton.runtime.autotuner import Autotuner
from triton.runtime.errors import OutOfResources


class TritonGCUAutotuner(Autotuner):

    def _bench(self, *args, config, **meta):
        from triton.compiler.errors import CompileTimeAssertionFailure

        # check for conflicts, i.e. meta-parameters both provided
        # as kwargs and by the autotuner
        conflicts = meta.keys() & config.kwargs.keys()
        if conflicts:
            raise ValueError(f"Conflicting meta-parameters: {', '.join(conflicts)}."
                             " Make sure that you don't re-define auto-tuned symbols.")
        # augment meta-parameters with tunable ones
        current = dict(meta, **config.all_kwargs())
        full_nargs = {**self.nargs, **current}

        def kernel_call():
            if config.pre_hook:
                config.pre_hook(full_nargs)
            self.pre_hook(args)
            try:
                self.fn.run(
                    *args,
                    **current,
                )
            except Exception as e:
                try:
                    self.post_hook(args, exception=e)
                finally:
                    # Throw exception raised by `self.fn.run`
                    raise

            self.post_hook(args, exception=None)

        try:
            if self.use_cuda_graph:
                import torch
                with torch.cuda.stream(torch.cuda.Stream()):
                    bench_res = do_bench_cudagraph(kernel_call, rep=self.num_reps, return_mode="median")
                return bench_res
            return do_bench(kernel_call, warmup=self.num_warmups, rep=self.num_reps, quantiles=(0.5, 0.2, 0.8))
        except (OutOfResources, CompileTimeAssertionFailure):
            return float("inf") if self.use_cuda_graph else [float("inf"), float("inf"), float("inf")]


Autotuner._bench = TritonGCUAutotuner._bench
