<!--
 Copyright 2026 FlagOS Contributors

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in all
 copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 SOFTWARE.
 -->

triton.language
===============

.. currentmodule:: triton.language


Programming Model
-----------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    tensor
    tensor_descriptor
    program_id
    num_programs


Creation Ops
------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    arange
    cat
    full
    zeros
    zeros_like
    cast


Shape Manipulation Ops
----------------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    broadcast
    broadcast_to
    expand_dims
    interleave
    join
    permute
    ravel
    reshape
    split
    trans
    view


Linear Algebra Ops
------------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    dot
    dot_scaled


Memory/Pointer Ops
----------

.. autosummary::
    :toctree: generated
    :nosignatures:

    load
    store
    make_tensor_descriptor
    load_tensor_descriptor
    store_tensor_descriptor
    make_block_ptr
    advance


Indexing Ops
------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    flip
    where
    swizzle2d


Math Ops
--------

.. autosummary::
    :toctree: generated
    :nosignatures:

    abs
    cdiv
    ceil
    clamp
    cos
    div_rn
    erf
    exp
    exp2
    fdiv
    floor
    fma
    log
    log2
    maximum
    minimum
    rsqrt
    sigmoid
    sin
    softmax
    sqrt
    sqrt_rn
    umulhi


Reduction Ops
-------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    argmax
    argmin
    max
    min
    reduce
    sum
    xor_sum

Scan/Sort Ops
-------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    associative_scan
    cumprod
    cumsum
    histogram
    sort
    gather

Atomic Ops
----------

.. autosummary::
    :toctree: generated
    :nosignatures:

    atomic_add
    atomic_and
    atomic_cas
    atomic_max
    atomic_min
    atomic_or
    atomic_xchg
    atomic_xor

Random Number Generation
------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    randint4x
    randint
    rand
    randn


Iterators
-----------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    range
    static_range


Inline Assembly
-----------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    inline_asm_elementwise


Compiler Hint Ops
-----------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    assume
    debug_barrier
    max_constancy
    max_contiguous
    multiple_of


Debug Ops
-----------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    static_print
    static_assert
    device_print
    device_assert
