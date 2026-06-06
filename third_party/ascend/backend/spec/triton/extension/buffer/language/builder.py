# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
"""
buffer-specific builder utilities for code generation.
"""

__all__ = [
    "create_builder_method_wrapper_with_buffer_builder",
    "attach_builder_methods_with_buffer_builder",
    "setup_unified_builder_with_buffer_builder",
]


def create_builder_method_wrapper_with_buffer_builder(main_builder, delegate_builder, method_name):
    """
    Create a wrapper that delegates a method call to another builder while
    synchronizing insertion points and locations.
    """
    delegate_method = getattr(delegate_builder, method_name)

    def wrapper(*args, **kwargs):
        saved_ip = main_builder.get_insertion_point()
        saved_loc = main_builder.get_loc()
        delegate_builder.restore_insertion_point(saved_ip)
        if saved_loc:
            delegate_builder.set_loc(saved_loc)
        result = delegate_method(*args, **kwargs)
        main_builder.restore_insertion_point(saved_ip)
        if saved_loc:
            main_builder.set_loc(saved_loc)
        return result

    wrapper.__name__ = method_name
    wrapper.__doc__ = getattr(delegate_method, '__doc__', None)
    return wrapper


def attach_builder_methods_with_buffer_builder(main_builder, delegate_builder, method_names):
    """Attach multiple methods from a delegate builder to the main builder."""
    for method_name in method_names:
        wrapper = create_builder_method_wrapper_with_buffer_builder(main_builder, delegate_builder, method_name)
        setattr(main_builder, method_name, wrapper)


def setup_unified_builder_with_buffer_builder(main_builder, buffer_builder):
    """Set up a unified builder interface by attaching methods from specialized builders."""
    main_builder._buffer_builder = buffer_builder
    buffer_methods = [
        'get_null_attr',
        'get_str_array_attr',
        'alloc',
        'to_buffer',
        'to_tensor',
        'subview',
    ]
    attach_builder_methods_with_buffer_builder(main_builder, buffer_builder, buffer_methods)
