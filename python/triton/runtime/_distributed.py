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


class DistributedRtContext:
    _instance = None
    _initialized = False
    _init_count = 0

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._comm_ptr = None
            cls._instance._mem_ptr = None
        return cls._instance

    def __init__(self, _comm_ptr=None, _mem_ptr=None):
        if _comm_ptr and _mem_ptr:
            type(self)._init_count += 1

        if self._initialized:
            return
        self._comm_ptr = _comm_ptr
        self._mem_ptr = _mem_ptr
        self._initialized = True

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
