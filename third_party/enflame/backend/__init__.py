#
# Copyright 2024 Enflame. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#import triton_gcu.triton.libdevice
try:
    import torch_gcu
except ImportError:
    pass

# append gcu backend and driver
#from triton.backends import Backend, backends
#from triton_gcu.triton.compiler import _GCUBackend
#from triton_gcu.triton.driver import _GCUDriver
#backends.clear()
#backends["gcu"] = Backend(_GCUBackend, _GCUDriver)
