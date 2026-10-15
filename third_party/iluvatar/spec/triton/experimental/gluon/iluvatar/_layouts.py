from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from triton.language.core import _unwrap_if_constexpr

from triton.experimental.gluon.language._layouts import (
    BlockedLayout,
    DistributedLayout,
    DotOperandLayout,
    SwizzledSharedLayout,
)

__all__ = [
    "IluvatarBlockedLayout",
    "IluvatarDotOperandLayout",
    "IluvatarMMALayout",
    "IluvatarSwizzledSharedLayout",
]


@dataclass(frozen=True, eq=True)
class IluvatarMMALayout(DistributedLayout):
    """
    Represents a layout for Iluvatar TCU (tensor core unit) operations.

    Args:
        version (List[int]): [major, minor] version of the MMA instruction. The TCU is MMA v1 only,
            so major must be 1. Minor selects the operand swizzling variant and is 0 unless a
            compiler pass rewrote it.
        warps_per_cta (List[int]): The warp layout in the block. Each warp computes a 16x16 tile.
        instr_shape (List[int]): The shape in the form of (M, N, K) of the TCU instruction:
            [16, 16, 32] for int8 operands, [16, 16, 16] otherwise.
        cga_layout (Optional[List[List[int]]]): Bases describing CTA tiling.
    """
    version: List[int]
    warps_per_cta: List[int]
    instr_shape: List[int]
    cga_layout: List[List[int]] = field(default_factory=list)

    def __post_init__(self):
        super().__setattr__("version", _unwrap_if_constexpr(self.version))
        super().__setattr__("warps_per_cta", _unwrap_if_constexpr(self.warps_per_cta))
        super().__setattr__("instr_shape", _unwrap_if_constexpr(self.instr_shape))

        object.__setattr__(self, "cga_layout", self.cga_layout)
        self.verify()

    def _to_ir(self, builder):
        return builder.get_iluvatar_mma_layout(
            self.version,
            self.warps_per_cta,
            self.cga_layout,
            self.instr_shape,
        )

    def mangle(self) -> str:
        cga_layout = "_".join("~".join(map(str, vec)) for vec in self.cga_layout) if self.cga_layout else ""
        return f"IMMA_{self.version}_{self.warps_per_cta}_{self.instr_shape}_{cga_layout}_IMMA"

    def verify(self):
        assert len(self.version) == 2, "version must be in the [major, minor] form"
        assert self.version[0] == 1, f"the Iluvatar TCU is MMA v1 only, got major version {self.version[0]}"
        assert 0 <= self.version[1] <= 3, "minor version must be in the [0, 3] range"
        assert len(self.instr_shape) == 3, "instr_shape must follow the (M, N, K) format"
        assert list(self.instr_shape[0:2]) == [16,
                                               16], f"invalid intrinsic shape {self.instr_shape}, M and N must be 16"
        assert self.instr_shape[2] in [16, 32], f"invalid intrinsic shape {self.instr_shape}, K must be 16 or 32"

        rank = len(self.warps_per_cta)
        assert all(len(vec) == rank for vec in self.cga_layout), "cga_layout basis rank mismatch"

    def __hash__(self):
        return hash((tuple(self.version), tuple(self.warps_per_cta), tuple(self.instr_shape),
                     tuple(tuple(vec) for vec in self.cga_layout)))

    @property
    def rank(self):
        return len(self.warps_per_cta)


# The layouts below extend core layouts with Iluvatar-only fields. Their `__eq__`
# and `__hash__` treat an all-default instance as its core counterpart, so mixing
# them with core layouts in layout comparisons stays well-defined.
@dataclass(frozen=True, eq=False)
class IluvatarBlockedLayout(BlockedLayout):
    """
    A BlockedLayout describing an Iluvatar SME (streaming memory engine) access.

    Args:
        is_sme (bool): Whether this is an SME source layout.
        sme_mask (bool): Whether this layout carries an SME mask.
        sme_warps_per_cta (Optional[List[int]]): Warp distribution used by SME.
    """
    is_sme: bool = False
    sme_mask: bool = False
    sme_warps_per_cta: List[int] = field(default_factory=list)

    def __post_init__(self):
        super().__post_init__()
        super().__setattr__("is_sme", _unwrap_if_constexpr(self.is_sme))
        super().__setattr__("sme_mask", _unwrap_if_constexpr(self.sme_mask))
        super().__setattr__("sme_warps_per_cta", _unwrap_if_constexpr(self.sme_warps_per_cta))
        assert not self.sme_warps_per_cta or len(self.sme_warps_per_cta) == len(self.order)

    def _to_ir(self, builder):
        return builder.get_blocked_layout(
            self.size_per_thread,
            self.threads_per_warp,
            self.warps_per_cta,
            self.order,
            self.cga_layout,
            self.is_sme,
            self.sme_mask,
            self.sme_warps_per_cta,
        )

    def mangle(self) -> str:
        mangled = super().mangle()
        if not _sme_extra(self):
            return mangled
        sme_warps_per_cta = "_".join(map(str, self.sme_warps_per_cta))
        return mangled + f"SME{int(self.is_sme)}_{int(self.sme_mask)}_{sme_warps_per_cta}SME"

    def __eq__(self, other):
        if not isinstance(other, BlockedLayout):
            return NotImplemented
        return _base_eq(BlockedLayout, self, other) and _sme_extra(self) == _sme_extra(other)

    def __hash__(self):
        base = BlockedLayout.__hash__(self)
        extra = _sme_extra(self)
        return hash((base, ) + extra) if extra else base


@dataclass(frozen=True, eq=False)
class IluvatarDotOperandLayout(DotOperandLayout):
    """
    A DotOperandLayout whose operand is fed by the Iluvatar SME.

    Args:
        use_sme (int): SME operand selector. Defaults to 0.
        k_rotate (int): Iluvatar chain-dot K rotation. Defaults to 0.
    """
    use_sme: int = 0
    k_rotate: int = 0

    def __post_init__(self):
        super().__post_init__()
        super().__setattr__("use_sme", _unwrap_if_constexpr(self.use_sme))
        super().__setattr__("k_rotate", _unwrap_if_constexpr(self.k_rotate))

    def _to_ir(self, builder):
        return builder.get_dot_operand_layout(self.operand_index, self.parent._to_ir(builder), self.k_width,
                                              self.use_sme, self.k_rotate)

    def mangle(self) -> str:
        result = f"DO{self.operand_index}_{self.parent.mangle()}_{self.k_width}DO"
        if self.use_sme or self.k_rotate:
            result += f"SME{self.use_sme}_{self.k_rotate}SME"
        return result

    def __eq__(self, other):
        if not isinstance(other, DotOperandLayout):
            return NotImplemented
        return (_base_eq(DotOperandLayout, self, other) and self.use_sme == getattr(other, "use_sme", 0)
                and self.k_rotate == getattr(other, "k_rotate", 0))

    def __hash__(self):
        base = DotOperandLayout.__hash__(self)
        extra = ()
        if self.use_sme:
            extra += (self.use_sme, )
        if self.k_rotate:
            extra += (self.k_rotate, )
        return hash((base, ) + extra) if extra else base


@dataclass(frozen=True, eq=False)
class IluvatarSwizzledSharedLayout(SwizzledSharedLayout):
    """
    A SwizzledSharedLayout laid out for the Iluvatar TCU.

    Args:
        use_tcu (bool): Whether this is a TCU shared layout.
    """
    use_tcu: bool = False

    def __post_init__(self):
        super().__post_init__()
        super().__setattr__("use_tcu", _unwrap_if_constexpr(self.use_tcu))

    def _to_ir(self, builder):
        return builder.get_swizzled_shared_layout(
            self.vec,
            self.per_phase,
            self.max_phase,
            self.order,
            self.cga_layout,
            self.use_tcu,
        )

    def mangle(self) -> str:
        mangled = super().mangle()
        return mangled + "TCU" if self.use_tcu else mangled

    def __eq__(self, other):
        if not isinstance(other, SwizzledSharedLayout):
            return NotImplemented
        return _base_eq(SwizzledSharedLayout, self, other) and self.use_tcu == getattr(other, "use_tcu", False)

    def __hash__(self):
        base = SwizzledSharedLayout.__hash__(self)
        return hash((base, self.use_tcu)) if self.use_tcu else base


def _base_eq(base_cls, lhs, rhs):
    getter = base_cls.__dataclass_fields__
    return all(getattr(lhs, name) == getattr(rhs, name) for name in getter)


def _sme_extra(layout):
    is_sme = getattr(layout, "is_sme", False)
    sme_mask = getattr(layout, "sme_mask", False)
    sme_warps_per_cta = tuple(getattr(layout, "sme_warps_per_cta", ()) or ())
    if not (is_sme or sme_mask or sme_warps_per_cta):
        return ()
    return (is_sme, sme_mask, sme_warps_per_cta)
