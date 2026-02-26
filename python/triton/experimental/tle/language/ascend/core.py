# Copyright (c) 2025  XCoreSigma Inc. All rights reserved.

from triton.language.core import (
    _unwrap_if_constexpr,
)

class layout:
    ASCEND = ['ND', 'NZ']

    def __init__(self, name):
        name = _unwrap_if_constexpr(name)
        self.name = name
        assert name in layout.ASCEND, name

    def __str__(self):
        return self.name

    def codegen_name(self):
        return self.name

    @property
    def cache_key_part(self) -> str:
        """See cache_key_part() in triton.cc."""
        return self.name

    def __repr__(self):
        """Output of repr needs to be an evaluatable expression"""
        return f'triton.language.{self.codegen_name()}'


ND = layout('ND')
NZ = layout('NZ')

class scope:
    ASCEND = ['UB', 'L1', 'L0A', 'L0B', 'L0C']

    def __init__(self, name):
        name = _unwrap_if_constexpr(name)
        self.name = name
        assert name in scope.ASCEND, name

    def __str__(self):
        return self.name

    def codegen_name(self):
        return self.name

    @property
    def cache_key_part(self) -> str:
        """See cache_key_part() in triton.cc."""
        return self.name

    def __repr__(self):
        """Output of repr needs to be an evaluatable expression"""
        return f'triton.language.{self.codegen_name()}'

UB = scope('UB')
L1 = scope('L1')
L0A = scope('L0A')
L0B = scope('L0B')
L0C = scope('L0C')
