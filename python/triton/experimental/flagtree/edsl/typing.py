from typing import Annotated

from mlir import ir


class InOut:

    def __class_getitem__(cls, desc: str) -> Annotated[ir.MemRefType, str]:
        return Annotated[ir.MemRefType, desc]


class Input:

    def __class_getitem__(cls, desc: str) -> Annotated[ir.MemRefType, str]:
        return Annotated[ir.MemRefType, desc]
