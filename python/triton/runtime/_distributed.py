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

import functools


class DistributedRtContext:
    _init_count = 0

    def __init__(self, _comm_ptr=None, _mem_ptr=None):
        if _comm_ptr is not None and _mem_ptr is not None:
            type(self)._init_count += 1
        self._comm_ptr = _comm_ptr
        self._mem_ptr = _mem_ptr
        self._registered_buffer = None

    def register_buffer(self, buffer) -> None:
        self._registered_buffer = buffer

    @property
    def registered_buffer(self):
        return getattr(self, "_registered_buffer", None)

    def get_packed_data(self):
        return int(self._mem_ptr), int(self._comm_ptr)

    @property
    def comm_ptr(self) -> int | None:
        """Communication runtime pointer."""
        return self._comm_ptr

    def _get_needed_params(self):
        return {"device_comm_ptr": self._comm_ptr, "device_mem_ptr": self._mem_ptr}

    @property
    def mem_ptr(self) -> int | None:
        """Distributed memory runtime pointer."""
        return self._mem_ptr

    @property
    def is_lite_mode(self) -> bool:
        import os
        user_action = os.getenv("FLAGTREE_LITE_DIST", "").strip().upper() in {
            "1",
            "ON",
            "TRUE",
        }
        inner_action = self._init_count == 1
        return inner_action and user_action

    def __getitem__(self, index=0):
        return list(self._get_needed_params().values())[index]

    def add_args_to_jitfunction(self, **kwargs):
        params = kwargs['params']  # list
        _kwargs = kwargs['kwargs']
        if (isinstance(params, list) and len(params) > 0):
            needed_params = self._get_needed_params()
            needed_params_size = len(needed_params)
            template_ele = params[0]
            KernelParam = type(template_ele)  # KernelParam type
            Parameter = type(template_ele._param)  # inspect.Parameter type

            dist_params = []
            for i, (name, val) in enumerate(needed_params.items()):
                _kwargs[name] = val
                param = Parameter(name, kind=template_ele._param.kind)
                dist_params.append(KernelParam(i, param, False, False))

            new_params = []
            for param in params:
                new_loc = param.num + needed_params_size
                #num: int, param: inspect.Parameter, do_not_specialize: bool, do_not_specialize_on_alignment: bool
                new_params.append(
                    KernelParam(new_loc, param._param, param.do_not_specialize, param.do_not_specialize_on_alignment))
            new_params = dist_params + new_params
            params[:] = new_params


@functools.lru_cache(maxsize=256)
def _parse_node_buffer_bindings(encoded_bindings):
    roles = {"s": "source", "d": "destination"}
    bindings = []
    for encoded in encoded_bindings.split(","):
        try:
            role_code, ordinal_text, handle_text = encoded.split(":", 2)
            ordinal = int(ordinal_text)
            handle = int(handle_text)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"invalid node buffer binding {encoded!r}") from exc
        role = roles.get(role_code)
        if role is None or ordinal < 0:
            raise RuntimeError(f"invalid node buffer binding {encoded!r}")
        bindings.append((encoded, role, ordinal, handle))
    return tuple(bindings)


def _collect_distributed_contexts(values):
    return tuple({id(value): value for value in values if isinstance(value, DistributedRtContext)}.values())


def _collect_runtime_items(bound_args, specialization):
    return tuple(
        (name, value) for (name, value), spec in zip(bound_args.items(), specialization) if spec[0] != "constexpr")


def validate_node_buffer_bindings(
    contexts,
    runtime_args,
    encoded_bindings,
    kernel_name: str,
    arg_names=None,
) -> None:
    if not encoded_bindings:
        return

    contexts_by_handle = {int(ctx[0]): ctx for ctx in contexts}
    for encoded, role, ordinal, handle in _parse_node_buffer_bindings(encoded_bindings):
        if ordinal >= len(runtime_args):
            raise RuntimeError(f"kernel {kernel_name!r} has invalid node buffer binding "
                               f"{encoded!r}")

        ctx = contexts_by_handle.get(handle)
        if ctx is None:
            raise RuntimeError(f"kernel {kernel_name!r} did not receive the node context "
                               f"with mem handle 0x{handle:x}. Pass the same ctx returned by "
                               "ctx = tle.create_dist_tensor(buffer) to this launch")
        registered = ctx.registered_buffer
        if registered is None:
            raise RuntimeError(f"kernel {kernel_name!r} node context has no registered buffer. "
                               "Create it with ctx = tle.create_dist_tensor(buffer) instead of "
                               "constructing DistributedRtContext directly")

        value = runtime_args[ordinal]
        data_ptr = getattr(value, "data_ptr", None)
        if not isinstance(value, int) and not callable(data_ptr):
            raise TypeError(f"kernel {kernel_name!r} node local {role} buffer must be a tensor "
                            "with data_ptr() or an integer device pointer. Pass the same buffer "
                            "used in ctx = tle.create_dist_tensor(buffer)")
        actual = value if isinstance(value, int) else data_ptr()
        expected = registered.data_ptr()
        if actual == expected:
            continue

        name = (arg_names[ordinal]
                if arg_names is not None and ordinal < len(arg_names) else f"runtime ordinal {ordinal}")
        raise ValueError(f"kernel {kernel_name!r} node local {role} buffer argument "
                         f"{name!r} does not match its context buffer: expected "
                         f"0x{expected:x}, got 0x{actual:x}. Use ctx = "
                         "tle.create_dist_tensor(buffer) and launch with that same buffer/ctx "
                         "pair; a ctx created for a different buffer cannot be used")


def validate_node_launch_bindings(kernel, bound_args, specialization):
    """Validate node buffer bindings for one kernel launch."""
    encoded_bindings = getattr(kernel.metadata, "tle_node_buffer_bindings", "")
    if not encoded_bindings:
        return

    contexts = _collect_distributed_contexts(bound_args.values())
    runtime_items = _collect_runtime_items(bound_args, specialization)
    validate_node_buffer_bindings(
        contexts,
        tuple(value for _, value in runtime_items),
        encoded_bindings,
        kernel.name,
        tuple(name for name, _ in runtime_items),
    )
