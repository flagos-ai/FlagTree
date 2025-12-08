import subprocess
import os
import textwrap
import inspect
import tempfile
from io import StringIO
import contextlib
import ast
import pandas as pd
import triton.runtime as runtime


def flagtree_do_bench(fn, warmup=10, rep=5, quantiles=None, return_mode="mean"):
    assert return_mode in ["mean", "min", "max", "sum"]
    bench = FlagtreeBench(current_fn=fn, warmup=warmup, rep=rep, quantiles=quantiles, return_mode=return_mode)
    bench.do_bench()
    return bench.results[return_mode]


'''
    IndentedBuffer Referred to
    https://github.com/flagos-ai/FlagGems/blob/master/src/flag_gems/utils/code_utils.py::IndentedBuffer
'''


class IndentedBuffer:
    tabwidth = 4

    def __init__(self, initial_indent=0):
        self._lines = []
        self._indent = initial_indent

    def getvalue(self) -> str:
        buf = StringIO()
        for line in self._lines:
            assert isinstance(line, str)
            buf.write(line)
            buf.write("\n")
        return buf.getvalue()

    def clear(self):
        self._lines.clear()

    def __bool__(self):
        return bool(self._lines)

    def prefix(self):
        return " " * (self._indent * self.tabwidth)

    def newline(self):
        self.writeline("\n")

    def writeline(self, line):
        if line.strip():
            self._lines.append(f"{self.prefix()}{line}")
        else:
            self._lines.append("")

    def tpl(self, format_str, **kwargs):
        assert isinstance(format_str, str), "format_str must be string of type."
        format_str = format_str.format(**kwargs)
        lines = format_str.strip().splitlines()
        for line in lines:
            line = line.replace("\t", " " * self.tabwidth)
            self.writeline(line)

    def writelines(self, lines):
        for line in lines:
            self.writeline(line)

    def writemultiline(self, s):
        self.writelines(s.splitlines())

    def indent(self, offset=1):

        @contextlib.contextmanager
        def ctx():
            self._indent += offset
            try:
                yield
            finally:
                self._indent -= offset

        return ctx()


'''
    FlagtreeBench using ncu to measure performance
'''


class FlagtreeBench:

    def __init__(self, current_fn, warmup=10, rep=5, quantiles=None, return_mode="mean", metrics='gpu__time_duration'):
        if FlagtreeBench.check_ncu():
            self._current_fn = current_fn
            self.metrics = metrics
            self.warmup = warmup
            self.rep = rep
            self.quantiles = quantiles
            self.return_mode = return_mode
            self.triton_funcs = []
            self._get_package_path()
            self._create_temp_file()

    @staticmethod
    def check_ncu():
        cmd = ["ncu", "--query-metrics"]
        try:
            subprocess.run(cmd, capture_output=True, check=True)
            print("[INFO]: ncu check successfully")
            return True
        except Exception as err_msg:
            print(f"\033[31m[Error] The inability to invoke ncu on this machine"
                  f"might be due to issues such as the absence of ncu, "
                  f"lack of permissions, or a version that is too low. Specifically \n{err_msg}\033[0m")
            return False

    @staticmethod
    def gather_triton_jit_kernel(mod):
        '''
            attrs temporarily adds specialized support to the kernel of flag_gems.
            About flag_gems see https://github.com/flagos-ai/FlagGems
        '''
        if FlagtreeBench.is_from_sitepackages(mod):
            return set()

        kernels = set()
        attrs = ['AnonymousLibTunerImpl', 'LibEntry', 'JITFunction']
        for node in dir(mod):
            if node.startswith('__'):
                continue
            obj = getattr(mod, node)
            if hasattr(obj, '__class__') and obj.__class__.__name__ in attrs:
                kernels.add(node)
        return kernels

    @staticmethod
    def is_from_sitepackages(mod):
        return 'site-packages' in mod.__file__

    def _get_current_function_used_mod(self, _fn=None):
        _fn = _fn or self._current_fn
        func_global_dict = _fn.__globals__
        source = inspect.getsource(_fn)
        tree = ast.parse(source)
        modules = set()
        calls = set()
        deps_path = set()
        triton_jit_kernels = set()

        class Visitor(ast.NodeVisitor):

            def visit_Attribute(self, node):
                if isinstance(node.value, ast.Name):
                    mod_name = node.value.id
                    mod_instance = func_global_dict[mod_name]
                    if hasattr(mod_instance, '__file__'):
                        mod_dir_path = os.path.dirname(os.path.dirname(mod_instance.__file__))
                        deps_path.add(mod_dir_path)
                    modules.add(mod_name)
                self.generic_visit(node)

            def visit_Call(self, node):
                if isinstance(node.func, ast.Name):
                    fun_name = node.func.id
                    func_instance = func_global_dict[fun_name]
                    mod_instance = __import__(func_instance.__module__)
                    triton_jit_kernels.update(FlagtreeBench.gather_triton_jit_kernel(mod_instance))
                    if hasattr(mod_instance, '__file__'):
                        mod_dir_path = os.path.dirname(mod_instance.__file__)
                        deps_path.add(mod_dir_path)
                    calls.add((fun_name, mod_instance.__name__))

                elif isinstance(node.func, ast.Attribute):
                    fun_name = node.func.attr
                    if isinstance(node.func.value, ast.Name):
                        mod = node.func.value.id
                        mod_instance = func_global_dict[mod]
                        triton_jit_kernels.update(FlagtreeBench.gather_triton_jit_kernel(mod_instance))
                self.generic_visit(node)

        Visitor().visit(tree)
        return (calls, modules, deps_path)

    def _get_package_path(self):
        self.user_package_path = os.environ.get('BENCH_MODULE_PATH', '')

    def _create_temp_file(self):
        self.python_exec = tempfile.NamedTemporaryFile(delete=False, suffix=".py")
        self.python_exec.close()

        self.out_csv = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        self.out_csv.close()

    def _write_script(self, script):
        with open(self.python_exec.name, 'w+') as f:
            f.write(script)

    def _exec(self):
        runtime.driver.active.clear_cache(self.bench_cache)
        cmd = [
            "ncu",
            "--metrics",
            self.metrics,
            "--csv",
            "--log-file",
            self.out_csv.name,
            "python3",
            self.python_exec.name,
        ]
        print(f"[INFO]: ncu running on {self.python_exec.name}")
        subprocess.run(cmd, check=True)
        self._pure_csv_log()

    def _pure_csv_log(self):
        FILTER_PREFIXES = ["==PROF=", "==ERROR=", "==WARNING="]
        with open(self.out_csv.name, 'r') as csv_f:
            lines = csv_f.readlines()
        new_lines = [line for line in lines if not any(line.startswith(prefix) for prefix in FILTER_PREFIXES)]
        with open(self.out_csv.name, "w") as csv_f:
            csv_f.writelines(new_lines)

    def _get_index(self):
        indexs = ['avg', 'max', 'min', 'sum']
        patterns = "at::|std::"
        index_dict = dict.fromkeys(indexs, 0)
        df = pd.read_csv(self.out_csv.name)
        metric_values = df[~df["Kernel Name"].str.contains(patterns, regex=True)][["Metric Name", "Metric Value"]]
        for _, row in metric_values.iterrows():
            metric_name = str(row['Metric Name']).split('.')[-1]
            gpu_time = float(row['Metric Value']) / 1e6
            index_dict[metric_name] += gpu_time
        index_dict['mean'] = index_dict['avg']
        return index_dict

    def _gen_import_and_path(self, script_code: IndentedBuffer, path_mode='insert'):
        calls, modules, deps_path = self._get_current_function_used_mod()
        sys_path_action_str = '0, '
        if path_mode == 'insert':
            script_code.writeline('import torch')
            script_code.writeline('import os')
            script_code.writeline('import sys')
        else:
            sys_path_action_str = ''
        if self.user_package_path != '':
            script_code.writeline(f"sys.path.{path_mode}({sys_path_action_str}'{self.user_package_path}')")
        for path in deps_path:
            if not os.path.isdir(path):
                path = os.path.dirname(path)
            script_code.writeline(f"sys.path.{path_mode}({sys_path_action_str}'{path}')")
        if path_mode == 'insert':
            for mod in modules:
                script_code.writeline(f'import {mod}')
            for call, mod in calls:
                script_code.writeline(f"from {mod} import {call}")

    def _generate_script(self, _fn=None):
        _fn = _fn or self._current_fn
        fn_src_code_string = textwrap.dedent(inspect.getsource(_fn))
        script_code = IndentedBuffer()
        self._gen_import_and_path(script_code, path_mode='insert')

        script_code.writeline(fn_src_code_string)
        script_code.writeline(f'{_fn.__name__}()')
        script_code.writeline("torch.cuda.synchronize()")

        self._gen_import_and_path(script_code, path_mode='remove')
        self.script = script_code.getvalue()
        self._write_script(self.script)

    def _pre_operation(self, _fn=None):
        '''
            Referred to triton.testing.do_bench
        '''
        _fn = _fn or self._current_fn
        di = runtime.driver.active.get_device_interface()
        _fn()
        di.synchronize()
        cache = runtime.driver.active.get_empty_cache_for_benchmark()

        # Estimate the runtime of the function
        start_event = di.Event(enable_timing=True)
        end_event = di.Event(enable_timing=True)
        start_event.record()
        for _ in range(5):
            runtime.driver.active.clear_cache(cache)
            _fn()
        end_event.record()
        di.synchronize()
        estimate_ms = start_event.elapsed_time(end_event) / 5

        # compute number of warmup and repeat
        n_warmup = max(1, int(self.warmup / estimate_ms))

        self.bench_cache = cache
        for _ in range(n_warmup):
            _fn()

    def do_bench(self):
        '''
            Measure the GPU kernel time of fn() using ncu.
            Generate a temporary Python file and then run it with 'ncu'.
        '''
        self.used_mods = self._get_current_function_used_mod()
        self._generate_script()
        self._pre_operation()
        self._exec()
        self.results = self._get_index()
