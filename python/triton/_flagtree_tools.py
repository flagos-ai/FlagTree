import subprocess
import os
import sys
import textwrap
import inspect
import tempfile
from io import StringIO
import contextlib
import ast
import types
import pandas as pd
import torch
import pickle
from dataclasses import dataclass
from typing import Callable, Any
import triton.runtime as runtime
'''
    Currently, the use of flagtree_do_bench is restricted, mainly including:

    1. This method can only be used to test the kernel running time of triton and torch;

    2. Arg fn, is either a direct single-test wrapper, such as
        (1) def test():
        (2). fn = op()

'''


def flagtree_do_bench(fn, warmup=10, rep=5, quantiles=None, return_mode="mean") -> float:
    assert return_mode in ["mean", "min", "max", "sum"]
    bench = FlagtreeBench(current_fn=fn, warmup=warmup, rep=rep, quantiles=quantiles, return_mode=return_mode)
    bench.do_bench()
    return bench.results[return_mode]


def get_cuda_impl(op_name):
    return torch._C._dispatch_get_registrations_for_dispatch_key("CUDA").get(op_name, None)


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


def function_warmup(_fn, warmup):
    '''
        Referred to triton.testing.do_bench
    '''
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
    n_warmup = max(1, int(warmup / estimate_ms))

    for _ in range(n_warmup):
        _fn()


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


@dataclass
class FuncAttrs:
    _globals: dict = None,
    is_lambda: bool = False
    is_torch_method: bool = False,
    source: str = ''
    ast_tree: any = None,
    functor_source: str = ''
    Argument_serialized: bool = False
    Argument_serialized_path: str = ''
    calls: list = None,
    modules: list = None,
    deps_path: list = None
    is_flaggems_functor: bool = False


class CodeGenerator:

    save_path: str = ''

    def gen_benchmark_python_code(self, _fn: Callable[..., Any] = None, Trait: FuncAttrs = None, save=True):
        script_code = IndentedBuffer()
        fn_src_code_string = textwrap.dedent(inspect.getsource(_fn))
        script_code = IndentedBuffer()
        CodeGenerator._gen_import_and_path(script_code, Trait, path_mode='insert')

        if Trait.is_lambda:
            CodeGenerator._gen_lambda_source_code(script_code, Trait)
        else:
            script_code.writeline(fn_src_code_string)
            script_code.writeline(f'{_fn.__name__}()')
        script_code.writeline("torch.cuda.synchronize()")

        CodeGenerator._gen_import_and_path(script_code, Trait, path_mode='remove')
        self.save_script_code(script_code)

    def save_script_code(self, script_code: IndentedBuffer):
        script = script_code.getvalue()
        python_temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".py")
        python_temp_file.close()
        with open(python_temp_file.name, 'w+') as f:
            f.write(script)
        self.save_path = python_temp_file.name

    @staticmethod
    def gen_load_args_kwargs_method(script_code: IndentedBuffer, Trait: FuncAttrs):
        script_code.writeline("def load_args_kwargs(filename):")
        with script_code.indent():
            script_code.writeline("import pickle")
            script_code.writeline("with open(filename, 'rb') as f:")
            with script_code.indent():
                script_code.writeline("data = pickle.load(f)")
                script_code.writeline("return data['args'], data['kwargs']")
        script_code.writeline(f"args, kwargs = load_args_kwargs('{Trait.Argument_serialized_path}')")
        if Trait.is_flaggems_functor:
            script_code.writeline("import flag_gems")
            script_code.writeline("with flag_gems.use_gems():")
            with script_code.indent():
                script_code.writeline(f"{Trait.functor_source}(*args, **kwargs)")
        else:
            script_code.writeline(f"{Trait.functor_source}(*args, **kwargs)")

    @staticmethod
    def _gen_import_and_path(script_code: IndentedBuffer, Trait: FuncAttrs, path_mode='insert'):
        sys_path_action_str = '0, '
        if path_mode == 'insert':
            script_code.writeline('import torch')
            script_code.writeline('import os')
            script_code.writeline('import sys')
        else:
            sys_path_action_str = ''
        user_package_path = os.environ.get('BENCH_MODULE_PATH', '')
        if user_package_path != '':
            script_code.writeline(f"sys.path.{path_mode}({sys_path_action_str}'{user_package_path}')")

        # create extra modules
        if Trait.is_lambda:
            return
        else:
            calls, modules, deps_path = Trait.calls, Trait.modules, Trait.deps_path
        for path in deps_path:
            if not os.path.isdir(path):
                path = os.path.dirname(path)
            script_code.writeline(f"sys.path.{path_mode}({sys_path_action_str}'{path}')")
        if path_mode == 'insert':
            for mod in modules:
                script_code.writeline(f'import {mod}')
            for call, mod in calls:
                script_code.writeline(f"from {mod} import {call}")

    @staticmethod
    def _gen_lambda_source_code(script_code: IndentedBuffer, Trait: FuncAttrs):
        if Trait.is_torch_method:
            CodeGenerator._gen_torch_using_lambda_code(script_code, Trait)
        else:
            CodeGenerator._gen_triton_code(script_code)

    @staticmethod
    def _gen_torch_using_lambda_code(script_code: IndentedBuffer, Trait: FuncAttrs):
        if Trait.Argument_serialized:
            CodeGenerator.gen_load_args_kwargs_method(script_code, Trait)
        else:
            script_code.writeline(f"{Trait.functor_source}()")

    @staticmethod
    def _gen_triton_code():
        ...


class FunctionExtractor:

    def __init__(self, fn: Callable[..., Any] = None):
        if fn.__name__ == "<lambda>":
            self.trait = self.analyse_lambda(fn)
        else:
            self.trait = self.analyse_general(fn)

    def args_serialization(self, *args, **kwargs):
        packed_datas = tempfile.NamedTemporaryFile(delete=False, suffix=".pkl")
        with open(packed_datas.name, "wb") as f:
            pickle.dump({"args": args, "kwargs": kwargs}, f)
        return packed_datas.name

    def analyse_lambda(self, fn):
        is_lambda = True
        if not hasattr(fn, '__closure__'):
            return

        def handlecase_single(closure_package, fn):
            ...

        def handlecase_both(closure_package, fn):
            ...

        def handlecase_all(closure_package, fn):
            source = inspect.getsource(fn)
            args, kwargs = closure_package[0].cell_contents, closure_package[1].cell_contents
            serialized_path = self.args_serialization(*args, **kwargs)
            functor = closure_package[-1].cell_contents

            # Case 1: method_descriptor → Tensor-level method
            # e.g., torch.Tensor.addmm
            if inspect.ismethoddescriptor(functor) or hasattr(functor, "__objclass__"):
                functor_mod_name = functor.__objclass__.__module__.replace("torch._C", "torch")
                functor_source = f"{functor_mod_name}.{functor.__name__}"

            # Case 2: bound method (x.addmm)
            if inspect.ismethod(functor):
                cls = functor.__self__.__class__
                functor_source = f"{cls.__module__}.{cls.__qualname__}.{functor.__name__}"

            if isinstance(functor, (types.BuiltinFunctionType, types.BuiltinMethodType)):
                functor_source = f"torch.{functor.__name__}"

            is_torch_method = 'torch' in functor_source
            return FuncAttrs(_globals=fn.__globals__, source=inspect.getsource(fn), is_torch_method=is_torch_method,
                             Argument_serialized=True, Argument_serialized_path=serialized_path,
                             functor_source=functor_source, ast_tree=ast.parse(textwrap.dedent(source)),
                             is_lambda=is_lambda, is_flaggems_functor=FunctionExtractor.is_flaggems_operator())

        casehandlers = [handlecase_single, handlecase_both, handlecase_all]
        casehandlers_mapping = {}
        for case_idx, handler in enumerate(casehandlers):
            casehandlers_mapping[case_idx] = handler

        handle_closure = lambda _type, _closure_package, _fn: casehandlers_mapping[_type](_closure_package, _fn)
        closure_package = fn.__closure__
        closure_package_len = len(closure_package) - 1

        return handle_closure(closure_package_len, closure_package, fn)

    def analyse_general(self, fn):
        source = inspect.getsource(fn)
        ast_tree = ast.parse(textwrap.dedent(source))
        calls, modules, deps_path = self._get_current_function_used_mod(fn, ast_tree)
        return FuncAttrs(
            source=source,
            ast_tree=ast_tree,
            is_lambda=False,
            _globals=fn.__globals__,
            calls=calls,
            modules=modules,
            deps_path=deps_path,
            is_flaggems_functor=FunctionExtractor.is_flaggems_operator(),
        )

    def _get_current_function_used_mod(self, _fn=None, ast_tree=None):
        func_global_dict = _fn.__globals__
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
                    if fun_name in func_global_dict:
                        func_instance = func_global_dict[fun_name]
                        mod_instance = __import__(func_instance.__module__)
                        triton_jit_kernels.update(FunctionExtractor.gather_triton_jit_kernel(mod_instance))
                        if hasattr(mod_instance, '__file__'):
                            mod_dir_path = os.path.dirname(mod_instance.__file__)
                            deps_path.add(mod_dir_path)
                        calls.add((fun_name, mod_instance.__name__))

                elif isinstance(node.func, ast.Attribute):
                    fun_name = node.func.attr
                    if isinstance(node.func.value, ast.Name):
                        mod = node.func.value.id
                        mod_instance = func_global_dict[mod]
                        triton_jit_kernels.update(FunctionExtractor.gather_triton_jit_kernel(mod_instance))
                self.generic_visit(node)

        Visitor().visit(ast_tree)
        return (calls, modules, deps_path)

    @staticmethod
    def gather_triton_jit_kernel(mod):
        '''
            attrs temporarily adds specialized support to the kernel of flag_gems.
            About flag_gems see https://github.com/flagos-ai/FlagGems
        '''
        if FunctionExtractor.is_from_sitepackages(mod):
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

    @staticmethod
    def is_flaggems_operator():
        try:
            import flag_gems
            with flag_gems.use_gems():
                return False
        except Exception:
            return True


'''
    FlagtreeBench using ncu to measure performance
'''


class FlagtreeBench:

    def __init__(self, current_fn, warmup=10, rep=5, quantiles=None, return_mode="mean", metrics='gpu__time_duration'):
        if check_ncu():
            self._current_fn = current_fn
            self.metrics = metrics
            self.warmup = warmup
            self.rep = rep
            self.quantiles = quantiles
            self.return_mode = return_mode
            self._create_temp_file()

    def _create_temp_file(self):
        self.out_csv = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        self.out_csv.close()

    def run_code_script(self):
        path = self.code_instance.save_path
        cmd = [
            "ncu",
            "--metrics",
            self.metrics,
            "--csv",
            "--log-file",
            self.out_csv.name,
            sys.executable,
            path,
        ]
        subprocess.Popen(cmd, text=True).communicate()
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
        patterns = "at::|std::|void"
        index_dict = dict.fromkeys(indexs, 0)
        df = pd.read_csv(self.out_csv.name)
        if self.fn_trait.is_torch_method and not self.fn_trait.is_flaggems_functor:
            metric_values = df[df["Kernel Name"].str.contains(patterns, regex=True)][["Metric Name", "Metric Value"]]
        else:
            metric_values = df[~df["Kernel Name"].str.contains(patterns, regex=True)][["Metric Name", "Metric Value"]]
        for _, row in metric_values.iterrows():
            metric_name = str(row['Metric Name']).split('.')[-1]
            gpu_time = float(row['Metric Value']) / 1e6
            index_dict[metric_name] += gpu_time
        index_dict['mean'] = index_dict['avg']
        return index_dict

    def do_bench(self) -> float:
        '''
            Measure the GPU kernel time of fn() using ncu.
            Generate a temporary Python file and then run it with 'ncu'.
        '''
        self.fn_trait = FunctionExtractor(self._current_fn).trait
        self.code_instance = CodeGenerator()
        self.code_instance.gen_benchmark_python_code(_fn=self._current_fn, Trait=self.fn_trait)
        function_warmup(self._current_fn, self.warmup)
        self.run_code_script()
        self.results = self._get_index()
