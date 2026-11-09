"""Compile a Triton JIT kernel to Ascend ttadapter MLIR without launching it.

Example:
    python3 third_party/ascend/unittest/Conversion/General/DynamicCVPipeline/ttadapter/compile_ttadapter.py \
        third_party/ascend/unittest/DynamicCVPipeline_ut/test_acf01_mlir_no_loop_independent_cv.py:acf01_tc01_no_loop_cv \
        --signature '{"a_ptr":"*fp16","b_ptr":"*fp16","c_ptr":"*fp16","out_ptr":"*fp16","M":"i32","N":"i32","K":"i32","stride_am":"i32","stride_ak":"i32","stride_bk":"i32","stride_c":"i32","stride_out":"i32"}' \
        --constants '{"BLOCK_SIZE_N":64,"BLOCK_SIZE_K":32}' \
        --output /tmp/acf01.ttadapter.mlir

From this directory, check the Mojo kernels and print a summary table:
    python3.11 compile_ttadapter.py --batch dynamic_cv_cases_github_master.txt \
        --root /path/to/mojo_opset

For a different Mojo checkout, pass ``--root /path/to/mojo_opset``. This
overrides the batch file's ``[defaults].root`` without editing the case list.
Pass ``--arch Ascend910_9589`` to compile without querying an NPU target.
The sibling ``dynamic_cv_cases_github_master.txt`` supports both the GitHub
master layout and pinned commit 73f472fa6ed9fefc5fea2d5963dc1595fae905e6.
``--validate-sources --cube-cores 40`` checks its test-derived specializations
without loading PyTorch, Triton, or an NPU driver; it does not produce MLIR.

Batch cases with ``test`` use a source reader for the supported pytest/kernel
pair. The test function is parsed, never called. Cases without ``test`` can
still provide signature, constants and options explicitly. A batch file may
also define [sources] and [tests] alias tables to keep each case short. Use
``cases = [{kernel = "alias:name", test = "test_alias", ...}, ...]`` before
the tables for one case per line, or the equivalent ``[[cases]]`` form.
"""

import argparse
import dataclasses
import importlib
import importlib.util
import inspect
import io
import json
import re
import sys
import tomllib
import traceback
import types
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path


def _split_npu_options(option_cls, compile_options):
    """Split options into constructor args and post-init fields."""
    signature = inspect.signature(option_cls)
    constructor_names = {
        name
        for name, parameter in signature.parameters.items()
        if name != "self" and parameter.kind in {
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        }
    }
    field_names = {field.name for field in dataclasses.fields(option_cls)}
    constructor_options = {}
    post_init_options = {}
    ignored_options = {}
    for name, value in compile_options.items():
        if name in constructor_names:
            constructor_options[name] = value
        elif name in field_names:
            post_init_options[name] = value
        else:
            ignored_options[name] = value
    return constructor_options, post_init_options, ignored_options


def _resolve_triton_constant_values(constants):
    resolved = dict(constants or {})
    dtype_constants = {
        "tl.float32": "float32",
        "tl.float16": "float16",
        "tl.bfloat16": "bfloat16",
        "tl.int32": "int32",
        "tl.int64": "int64",
    }
    needed = {value for value in resolved.values() if isinstance(value, str) and value in dtype_constants}
    if needed:
        import triton.language as tl
        for name, value in list(resolved.items()):
            if isinstance(value, str) and value in dtype_constants:
                resolved[name] = getattr(tl, dtype_constants[value])
    return resolved


def compile_ttadapter(kernel, signature, constants=None, *, arch="Ascend910_9589", enable_dynamic_cv_pipeline=True,
                      output_path=None, return_metadata=False, **compile_options):
    """Return the ttadapter MLIR for a ``@triton.jit`` kernel.

    ``signature`` maps runtime argument names to Triton types (for example,
    ``{"x_ptr": "*fp16", "n": "i32"}``). ``constants`` maps constexpr argument
    names to their compile-time values. No tensors, NPU allocation, or kernel
    launch are needed. If provided, ``output_path`` receives the same MLIR text.
    Additional ``compile_options`` are forwarded to ``NPUOptions`` so they can
    match the original launch. Set ``return_metadata=True`` to also receive the
    compiler metadata, including whether DynamicCVPipeline fell back.
    Compilation errors propagate to the caller.
    """
    from triton.compiler.compiler import ASTSource
    from triton.compiler.code_generator import ast_to_ttir
    from triton._C.libtriton import ir
    from triton._C.libtriton.ascend import ir as ascend_ir
    from triton.backends.ascend.compiler import NPUOptions, make_ttir, min_dot_size, ttir_to_linalg

    constructor_options, post_init_options, ignored_options = _split_npu_options(NPUOptions, compile_options)
    options = NPUOptions(arch=arch, enable_dynamic_cv_pipeline=enable_dynamic_cv_pipeline, **constructor_options)
    for name, value in post_init_options.items():
        try:
            setattr(options, name, value)
        except dataclasses.FrozenInstanceError:
            object.__setattr__(options, name, value)
    if ignored_options:
        setattr(options, "_ignored_compile_options", ignored_options)
    source = ASTSource(kernel, signature, _resolve_triton_constant_values(constants))
    context = ir.context()
    ir.load_dialects(context)
    ascend_ir.load_dialects(context)

    ttir = ast_to_ttir(kernel, source, context, options, {"min_dot_size": min_dot_size(None)}, {})
    metadata = dict(options.__dict__)
    ttir = make_ttir(ttir, metadata, options)
    mlir = ttir_to_linalg(ttir, metadata, options, named_ops=True)

    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(mlir, encoding="utf-8")
    return (mlir, metadata) if return_metadata else mlir


def dynamic_cv_status(mlir, metadata, requested):
    """Summarize whether DynamicCVPipeline produced scope ops without fallback."""
    scope_count = len(re.findall(r"(?m)^\s*(?:%[^\n=]+\s*=\s*)?scope\.scope\b", mlir))
    supported_arch = metadata.get("compile_on_910_95") is True
    no_fallback = metadata.get("enable_dynamic_cv_pipeline") is True
    return {
        "scope_count": scope_count,
        "cube_scope_count": len(re.findall(r"#hivm\.tcore_type<CUBE>", mlir)),
        "vector_scope_count": len(re.findall(r"#hivm\.tcore_type<VECTOR>", mlir)),
        "pass_applied": bool(requested and supported_arch and no_fallback and scope_count),
        "fell_back": bool(requested and supported_arch and not no_fallback),
    }


@contextmanager
def _compile_only_npu_driver():
    """Expose A5 target metadata while Mojo kernel modules are imported."""
    import triton

    driver_config = triton.runtime.driver
    previous_driver = getattr(driver_config, "_active", None)
    target = types.SimpleNamespace(backend="npu", arch="Ascend950PR_9579", warp_size=0)
    properties = {"num_aicore": 40, "num_vectorcore": 80}
    compile_only_driver = types.SimpleNamespace(
        get_current_target=lambda: target,
        utils=types.SimpleNamespace(get_device_properties=lambda *_args, **_kwargs: properties),
    )
    driver_config.set_active(compile_only_driver)
    try:
        yield
    finally:
        driver_config._active = previous_driver


def _load_mojo_module(path, package_parts, package_root):
    """Load one Mojo source file without running package ``__init__`` files.

    Mojo's top-level package detects an attached accelerator and selects a
    backend while it is imported. Compile-only CI deliberately has no NPU, so
    importing the package normally selects ``meta_device`` before the A5 kernel
    module can be reached. Namespace packages preserve relative and absolute
    imports used by the kernel source without triggering that runtime probe.
    """
    package_path = package_root
    parent_name = None
    for index, part in enumerate(package_parts):
        package_path /= part
        package_name = ".".join(package_parts[:index + 1])
        package = sys.modules.get(package_name)
        if package is None:
            package = types.ModuleType(package_name)
            package.__file__ = str(package_path / "__init__.py")
            package.__package__ = package_name
            package.__path__ = [str(package_path)]
            package.__spec__ = importlib.util.spec_from_loader(package_name, loader=None, is_package=True)
            sys.modules[package_name] = package
            if parent_name is not None:
                setattr(sys.modules[parent_name], part, package)
        parent_name = package_name

    module_name = ".".join([*package_parts, path.stem])
    module = sys.modules.get(module_name)
    if module is not None:
        return module
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    if parent_name is not None:
        setattr(sys.modules[parent_name], path.stem, module)
    try:
        with _compile_only_npu_driver():
            spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        if parent_name is not None:
            parent = sys.modules.get(parent_name)
            if parent is not None and getattr(parent, path.stem, None) is module:
                delattr(parent, path.stem)
        raise
    return module


def _load_kernel(reference):
    module_name, separator, attribute = reference.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("kernel must be MODULE:NAME or FILE.py:NAME")

    if module_name.endswith(".py"):
        path = Path(module_name).resolve()
        package_parts = []
        parent = path.parent
        while (parent / "__init__.py").is_file():
            package_parts.insert(0, parent.name)
            parent = parent.parent
        if package_parts:
            if package_parts[0] == "mojo_opset":
                module = _load_mojo_module(path, package_parts, parent)
            else:
                # Import under the real package name so relative imports in
                # ordinary package modules continue to work.
                if str(parent) not in sys.path:
                    sys.path.insert(0, str(parent))
                module = importlib.import_module(".".join([*package_parts, path.stem]))
        else:
            spec = importlib.util.spec_from_file_location(path.stem, path)
            if spec is None or spec.loader is None:
                raise ImportError(f"cannot import {path}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
    else:
        module = importlib.import_module(module_name)
    kernel = getattr(module, attribute)
    return _jit_kernel_object(kernel)


def _json_object(value):
    try:
        result = json.loads(value)
    except json.JSONDecodeError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    if not isinstance(result, dict):
        raise argparse.ArgumentTypeError("expected a JSON object")
    return result


def _alias_value(table, value, table_name):
    """Resolve a TOML alias table entry while keeping normal paths valid."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{table_name} reference must be a string")
    if not isinstance(table, dict):
        raise ValueError(f"[{table_name}] must be a TOML table")
    resolved = table.get(value, value)
    if not isinstance(resolved, str):
        raise ValueError(f"[{table_name}].{value} must be a string")
    return resolved


def _resolve_kernel_reference(reference, sources):
    """Resolve either FILE.py:NAME or SOURCE_ALIAS:NAME."""
    module_name, separator, kernel_name = reference.partition(":")
    if not separator or not module_name or not kernel_name:
        raise ValueError("kernel must be MODULE:NAME, FILE.py:NAME, or SOURCE_ALIAS:NAME")
    module_name = _alias_value(sources, module_name, "sources")
    return f"{module_name}:{kernel_name}", module_name, kernel_name


def _candidate_roots(root):
    """Accept either the checkout root, its parent, or the inner package root."""
    roots = [Path(root)]
    if root.name == "mojo_opset":
        roots.append(root.parent)
    package_root = root / "mojo_opset"
    if package_root.is_dir():
        roots.append(package_root)
    deduped = []
    seen = set()
    for item in roots:
        normalized = item.resolve() if item.exists() else item
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(item)
    return deduped


def _resolve_existing_file(root, requested, alternatives=()):
    """Resolve a source or test file across supported Mojo checkout layouts."""
    requested = Path(requested).expanduser()
    if requested.is_absolute():
        candidates = [requested]
    else:
        requested_paths = [requested, *(Path(item) for item in alternatives)]
        candidates = []
        for candidate_root in _candidate_roots(root):
            for item in requested_paths:
                candidates.append(candidate_root / item)
                if item.parts and item.parts[0] == "mojo_opset":
                    candidates.append(candidate_root / Path(*item.parts[1:]))
    deduped = []
    seen = set()
    for candidate in candidates:
        normalized = candidate.resolve() if candidate.exists() else candidate
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(candidate)
        if candidate.is_file():
            return candidate
    tried = "\n  ".join(str(path) for path in deduped)
    raise FileNotFoundError(f"{root} does not contain {requested}; tried:\n  {tried}")


def _source_alternatives(module_path):
    text = Path(module_path).as_posix()
    pairs = (
        ("mojo_opset/kernels/npu_a5_triton/", "mojo_opset/backends/ttx/kernels/npu/a5/"),
        ("mojo_opset/backends/ttx/kernels/npu/a5/", "mojo_opset/kernels/npu_a5_triton/"),
    )
    for prefix, replacement in pairs:
        if text.startswith(prefix):
            return [replacement + text[len(prefix):]]
    return []


def _test_alternatives(test_file, kernel_name, module_path):
    """Return equivalent pytest files in the old and GitHub master layouts."""
    text = Path(test_file).as_posix()
    source = Path(module_path).as_posix()
    if text == "tests/functions/test_flex_attention.py":
        return ["mojo_opset/tests/accuracy/functions/test_flex_attention.py"]
    if text == "tests/functions/test_paged_attention.py":
        return ["mojo_opset/tests/accuracy/operators/test_attention.py"]
    if text == "tests/functions/test_sdpa.py":
        return ["mojo_opset/tests/accuracy/operators/test_attention.py"]
    if text == "tests/functions/test_attention.py":
        if source.endswith("/swa.py") and kernel_name == "_sdpa_infer_kernel":
            return [
                "mojo_opset/tests/accuracy/operators/test_attention.py",
                "mojo_opset/tests/accuracy/functions/test_attention.py"
            ]
        return [
            "mojo_opset/tests/accuracy/functions/test_attention.py",
            "mojo_opset/tests/accuracy/operators/test_attention.py"
        ]
    if text == "mojo_opset/tests/accuracy/functions/test_flex_attention.py":
        return ["tests/functions/test_flex_attention.py"]
    if text == "mojo_opset/tests/accuracy/functions/test_attention.py":
        return ["tests/functions/test_attention.py"]
    if text == "mojo_opset/tests/accuracy/operators/test_attention.py":
        if kernel_name in {
                "paged_decode_kernel", "paged_decode_fd_kernel", "paged_prefill_kernel",
                "paged_prefill_page_aggregation_kernel", "_swa_paged_decode_kernel", "_swa_paged_prefill_kernel",
                "_swa_paged_prefill_aggregation_kernel"
        }:
            return ["tests/functions/test_paged_attention.py"]
        if source.endswith("/sdpa.py") or kernel_name in {
                "kernel_sdpa_fwd", "kernel_sdpa_bwd_qkv", "kernel_sdpa_bwd_q"
        }:
            return ["tests/functions/test_sdpa.py", "tests/functions/test_attention.py"]
        return ["tests/functions/test_attention.py"]
    return []


def _normalize_mojo_test_case(kernel_path, kernel_name, test_name, config_id):
    """Normalize logical cases when a batch file is used with the other layout."""
    old_layout = kernel_path.parent.name == "a5" and kernel_path.parent.parent.name == "npu"
    master_layout = kernel_path.parent.name == "npu_a5_triton" and kernel_path.parent.parent.name == "kernels"
    if old_layout:
        old_tests = {
            "test_business": "test_flex_attention",
            "test_swa": "test_swa_function",
        }
        if test_name == "test_paged":
            if kernel_name in {"paged_decode_kernel", "paged_decode_fd_kernel"}:
                test_name = "test_paged_decode_gqa"
            elif kernel_name in {"paged_prefill_kernel", "paged_prefill_page_aggregation_kernel"}:
                test_name = "test_paged_prefill_gqa"
            elif kernel_name == "_swa_paged_decode_kernel":
                test_name = "test_paged_decode_swa"
            elif kernel_name in {"_swa_paged_prefill_kernel", "_swa_paged_prefill_aggregation_kernel"}:
                test_name = "test_paged_prefill_swa"
        else:
            test_name = old_tests.get(test_name, test_name)
        if kernel_path.name == "swa.py" and kernel_name == "_sdpa_infer_kernel":
            config_id = {"bf16": "M_BF16"}.get(config_id, config_id)
        elif kernel_path.name == "swa.py" and kernel_name in {
                "_swa_fwd_kernel", "_swa_bwd_dkdv_kernel", "_swa_bwd_dq_kernel"
        }:
            config_id = {"1024": "M_BF16_1024"}.get(config_id, config_id)
    elif master_layout:
        master_tests = {
            "test_flex_attention": "test_business",
            "test_swa_function": "test_swa",
            "test_paged_decode_gqa": "test_paged",
            "test_paged_prefill_gqa": "test_paged",
            "test_paged_decode_swa": "test_paged",
            "test_paged_prefill_swa": "test_paged",
        }
        test_name = master_tests.get(test_name, test_name)
        if kernel_path.name == "swa.py" and kernel_name == "_sdpa_infer_kernel":
            config_id = {"M_BF16": "bf16"}.get(config_id, config_id)
        elif kernel_path.name == "swa.py" and kernel_name in {
                "_swa_fwd_kernel", "_swa_bwd_dkdv_kernel", "_swa_bwd_dq_kernel"
        }:
            config_id = {"M_BF16_1024": "1024"}.get(config_id, config_id)
    return test_name, config_id


def _mojo_checkout_root(root, path):
    """Return the outer Mojo checkout root for helper code that reads package files."""
    path = Path(path).resolve()
    parts = path.parts
    for index, part in enumerate(parts):
        if part != "mojo_opset" or index + 1 >= len(parts):
            continue
        if parts[index + 1] == "mojo_opset":
            return Path(*parts[:index + 1]) if index else Path("/")
        if parts[index + 1] in {"backends", "core", "tests", "kernels"}:
            return Path(*parts[:index]) if index else Path("/")
    return Path(root)


def _jit_kernel_object(kernel):
    """Return the inner Triton JIT object from optional autotune wrappers."""
    seen = set()
    current = kernel
    while True:
        if hasattr(current, "arg_names") and hasattr(current, "cache_key") and hasattr(current, "__name__"):
            return current
        if id(current) in seen or not hasattr(current, "fn"):
            return kernel
        seen.add(id(current))
        current = current.fn


def _run_batch(config_path, output_dir=None, verbose=False, root_override=None, arch_override=None,
               validate_sources=False, cube_cores=None):
    """Compile TOML cases independently and print only their Dynamic CV status."""
    config_path = config_path.resolve()
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    if output_dir is None:
        output_dir = config_path.parent / "ttadapter_outputs"
    else:
        output_dir = output_dir.expanduser()
        if not output_dir.is_absolute():
            output_dir = (Path.cwd() / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale_path in output_dir.glob("*.ttadapter.mlir"):
        stale_path.unlink()
    for stale_path in (output_dir / "summary.tsv", output_dir / "diagnostics.log", output_dir / "diagnostics.tsv"):
        if stale_path.exists():
            stale_path.unlink()

    defaults = config.get("defaults", {})
    cases = config.get("cases")
    if not isinstance(defaults, dict) or not isinstance(cases, list) or not cases:
        raise ValueError("batch file needs a nonempty cases array and an optional [defaults] table")

    root = Path(root_override if root_override is not None else defaults.get("root", config_path.parent)).expanduser()
    if not root.is_absolute():
        root = (config_path.parent / root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"batch source root does not exist: {root}")
    default_options = defaults.get("options", {})
    if not isinstance(default_options, dict):
        raise ValueError("[defaults.options] must be a TOML table")
    if cube_cores is not None and cube_cores <= 0:
        raise ValueError("--cube-cores must be positive")
    sources = config.get("sources", {})
    tests = config.get("tests", {})
    if not isinstance(sources, dict) or not all(
            isinstance(key, str) and isinstance(value, str) for key, value in sources.items()):
        raise ValueError("[sources] must map aliases to source paths")
    if not isinstance(tests, dict) or not all(
            isinstance(key, str) and isinstance(value, str) for key, value in tests.items()):
        raise ValueError("[tests] must map aliases to test references")

    rows = []
    diagnostics = []
    diagnostic_rows = []
    for index, case in enumerate(cases, 1):
        name = case.get("name") if isinstance(case, dict) else None
        name = name or f"case_{index}"
        diagnostic = io.StringIO()
        try:
            if not isinstance(case, dict):
                raise ValueError("case must be a TOML table")
            reference = case.get("kernel", defaults.get("kernel"))
            if not isinstance(reference, str):
                raise ValueError("kernel must be MODULE:NAME, FILE.py:NAME, or SOURCE_ALIAS:NAME")
            reference, module_path, kernel_name = _resolve_kernel_reference(reference, sources)
            if module_path.endswith(".py") and not Path(module_path).is_absolute():
                reference = f"{root / module_path}:{kernel_name}"

            if "name" not in case:
                name = kernel_name

            source_options = {}
            test_reference = case.get("test", defaults.get("test"))
            if test_reference is not None:
                from ttadapter_test_source import infer_mojo_case

                test_reference = _alias_value(tests, test_reference, "tests")
                test_file, separator, test_name = test_reference.partition("::")
                if not separator or not test_file.endswith(".py") or not module_path.endswith(".py"):
                    raise ValueError("test must be FILE.py::TEST_NAME and kernel must be FILE.py:NAME")
                test_path = _resolve_existing_file(root, test_file,
                                                   _test_alternatives(test_file, kernel_name, module_path))
                kernel_path = _resolve_existing_file(root, module_path, _source_alternatives(module_path))
                reference = f"{kernel_path}:{kernel_name}"
                source_root = _mojo_checkout_root(root, kernel_path)
                test_name, config_id = _normalize_mojo_test_case(kernel_path, kernel_name, test_name,
                                                                 case.get("config", defaults.get("config")))
                with redirect_stdout(diagnostic), redirect_stderr(diagnostic):
                    signature, constants, source_options = infer_mojo_case(source_root, test_path, test_name,
                                                                           kernel_path, kernel_name, config_id,
                                                                           case.get("layout",
                                                                                    defaults.get("layout")), cube_cores)
            else:
                signature = case["signature"]
                constants = case.get("constants", {})
            case_options = case.get("options", {})
            if not all(isinstance(item, dict) for item in (signature, constants, case_options)):
                raise ValueError("signature, constants and options must be TOML tables")
            if validate_sources:
                rows.append((str(name), "True"))
                continue
            options = {**default_options, **source_options, **case_options}
            requested = options.pop("enable_dynamic_cv_pipeline", True)
            arch = arch_override or case.get("arch", defaults.get("arch", "Ascend910_9589"))
            safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name))
            output_path = output_dir / f"{index:03d}_{safe_name}.ttadapter.mlir"

            with redirect_stdout(diagnostic), redirect_stderr(diagnostic):
                if arch == "auto":
                    import triton
                    arch = triton.runtime.driver.active.get_current_target().arch
                kernel = _load_kernel(reference)
                mlir, metadata = compile_ttadapter(kernel, signature, constants, arch=arch,
                                                   enable_dynamic_cv_pipeline=requested, output_path=output_path,
                                                   return_metadata=True, **options)
            status = dynamic_cv_status(mlir, metadata, requested)
            applied = status["pass_applied"]
            rows.append((str(name), str(applied)))
            if not applied:
                reason = "compiler fell back" if status["fell_back"] else "no verified Dynamic CV scopes"
                diagnostics.append((name, reason, diagnostic.getvalue()))
                diagnostic_rows.append((str(name), str(applied), reason))
        except Exception as error:
            rows.append((str(name), "ERROR"))
            reason = f"{type(error).__name__}: {error}"
            details = diagnostic.getvalue()
            if details and not details.endswith("\n"):
                details += "\n"
            details += traceback.format_exc()
            diagnostics.append((name, reason, details))
            diagnostic_rows.append((str(name), "ERROR", reason))

    name_width = max(len("Operator"), *(len(name) for name, _ in rows))
    status_header = "Source extraction" if validate_sources else "DynamicCVPipeline applied"
    print(f"{'Operator':<{name_width}} | {status_header}")
    print(f"{'-' * name_width}-+-{'-' * len(status_header)}")
    for name, status in rows:
        print(f"{name:<{name_width}} | {status}")
    directory_label = "validation output dir" if validate_sources else "ttadapter output dir"
    print(f"\n{directory_label}: {output_dir.resolve()}")

    summary_path = output_dir / "summary.tsv"
    summary_path.write_text(f"Operator\t{status_header}\n" + "".join(f"{name}\t{status}\n" for name, status in rows),
                            encoding="utf-8")

    diagnostics_path = output_dir / "diagnostics.log"
    diagnostics_tsv_path = output_dir / "diagnostics.tsv"
    if diagnostics:
        diagnostics_tsv_path.write_text(
            f"Operator\t{status_header}\treason\n" +
            "".join(f"{name}\t{status}\t{reason}\n" for name, status, reason in diagnostic_rows), encoding="utf-8")
        diagnostics_text = []
        for name, reason, diagnostic in diagnostics:
            diagnostics_text.append(f"{name}: {reason}\n")
            if diagnostic:
                diagnostics_text.append(diagnostic)
                if not diagnostic.endswith("\n"):
                    diagnostics_text.append("\n")
            diagnostics_text.append("\n")
        diagnostics_path.write_text("".join(diagnostics_text), encoding="utf-8")
        print(f"diagnostics: {diagnostics_path.resolve()}")
        print(f"diagnostics tsv: {diagnostics_tsv_path.resolve()}")

    if verbose:
        for name, reason, diagnostic in diagnostics:
            print(f"\n{name}: {reason}", file=sys.stderr)
            if diagnostic:
                print(diagnostic, file=sys.stderr, end="")
    return 0 if all(status == "True" for _, status in rows) else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kernel", nargs="?", help="Python module or .py file and kernel name, separated by ':'")
    parser.add_argument("--batch", type=Path, help="TOML text file containing cases = [...] or [[cases]]")
    parser.add_argument("--output-dir", type=Path, help="save batch ttadapter files in this directory")
    parser.add_argument("--root", type=Path, help="override [defaults].root for batch source and test paths")
    parser.add_argument("--validate-sources", action="store_true",
                        help="check source-derived batch cases without importing or compiling kernels")
    parser.add_argument("--cube-cores", type=int,
                        help="cube core count for source validation without an NPU (paged decode FD)")
    parser.add_argument("--verbose", action="store_true", help="print batch compilation errors")
    parser.add_argument("--case", type=Path, help="JSON file with signature, constants and compile options")
    parser.add_argument("--signature", type=_json_object,
                        help='runtime argument types as JSON, e.g. \'{"x_ptr":"*fp16"}\'')
    parser.add_argument("--constants", type=_json_object, help="constexpr values as JSON")
    parser.add_argument("--arch", help="Ascend target architecture, or 'auto' to query the NPU")
    parser.add_argument("--options", type=_json_object,
                        help="other NPU compile options as JSON, e.g. '{\"num_warps\":16}'")
    parser.add_argument("--output", type=Path, help="path for the .ttadapter.mlir file")
    parser.add_argument("--require-scope", action="store_true",
                        help="exit with status 1 if the generated MLIR has no scope.scope op")
    parser.add_argument("--require-dynamic-cv", action="store_true",
                        help="exit with status 1 unless DynamicCVPipeline produced scope ops without fallback")
    args = parser.parse_args()

    if args.batch is not None:
        if (args.kernel or args.case or args.signature is not None or args.constants is not None
                or args.options is not None or args.output or args.require_scope or args.require_dynamic_cv):
            parser.error("--batch reads kernels and their compile parameters from the text file")
        return _run_batch(args.batch, args.output_dir, args.verbose, args.root, args.arch, args.validate_sources,
                          args.cube_cores)
    if args.kernel is None:
        parser.error("provide a kernel or --batch FILE.txt")
    if (args.output_dir is not None or args.root is not None or args.validate_sources or args.cube_cores is not None):
        parser.error("--output-dir, --root, --validate-sources and --cube-cores require --batch")

    case = _json_object(args.case.read_text(encoding="utf-8")) if args.case else {}
    signature = args.signature if args.signature is not None else case.get("signature")
    if not isinstance(signature, dict):
        parser.error("provide --signature or a --case file containing 'signature'")
    constants = args.constants if args.constants is not None else case.get("constants", {})
    options = dict(case.get("options", {}))
    options.update(args.options or {})
    arch = args.arch or case.get("arch") or options.pop("arch", None) or "Ascend910_9589"
    if arch == "auto":
        import triton
        arch = triton.runtime.driver.active.get_current_target().arch
    options["arch"] = arch
    options.setdefault("enable_dynamic_cv_pipeline", True)

    kernel = _load_kernel(args.kernel)
    output = args.output or Path(f"{kernel.__name__}.ttadapter.mlir")
    mlir, metadata = compile_ttadapter(kernel, signature, constants, output_path=output, return_metadata=True,
                                       **options)
    status = dynamic_cv_status(mlir, metadata, options["enable_dynamic_cv_pipeline"])
    print(f"ttadapter: {output.resolve()}")
    print(f"scope.scope ops: {status['scope_count']}")
    print(f"CUBE / VECTOR tcore_type markers: {status['cube_scope_count']} / {status['vector_scope_count']}")
    print(f"DynamicCVPipeline applied: {status['pass_applied']}")
    if status["fell_back"]:
        print("DynamicCVPipeline fell back to standard compilation")
    if args.require_scope and not status["scope_count"]:
        return 1
    if args.require_dynamic_cv and not status["pass_applied"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
