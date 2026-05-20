"""
Triton GCU passes module.

This module provides type-safe wrapper functions for GCU MLIR passes
and common MLIR framework passes.
"""

from . import gcu300, gcu400, mlir

__all__ = ['gcu300', 'gcu400', 'mlir']
