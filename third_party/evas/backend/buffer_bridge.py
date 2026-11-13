"""Synchronous full-buffer bridge for automatic dual-die kernels.

Device allocation/copy stays in torch-lpu. This module owns descriptors,
alias-preserving argument bindings and temporary lifetimes. It never changes
the kernel's shape/stride arguments or guesses outputs from their names.
"""

from dataclasses import dataclass
import math
import sys

from .arg_access import ACCESS_VERSION


@dataclass(frozen=True)
class BufferDescriptor:
    tensor: object
    root: object
    device: object
    dtype: object
    shape: tuple
    strides: tuple
    itemsize: int
    offset_bytes: int
    root_bytes: int
    storage_key: tuple
    physical_ranges: tuple
    split_dim: int
    affinity: int

    @property
    def materialize(self):
        return self.split_dim != -1 or self.affinity != 0


@dataclass(frozen=True)
class BoundPointer:
    descriptor: BufferDescriptor
    address: int

    @property
    def device(self):
        return self.descriptor.device

    @property
    def dtype(self):
        return self.descriptor.dtype

    def data_ptr(self):
        return self.address

    def numel(self):
        return math.prod(self.descriptor.shape)

    def element_size(self):
        return self.descriptor.itemsize


def _placement(tensor):
    if not callable(getattr(tensor, "split_dim", None)) or not callable(getattr(tensor, "die_affinity", None)):
        raise ValueError("EVAS buffer has no verifiable allocation placement")
    return int(tensor.split_dim()), int(tensor.die_affinity())


def _check_bounds(shape, strides, itemsize, offset, extent):
    if len(shape) != len(strides) or any(n < 0 for n in shape):
        raise ValueError("EVAS invalid buffer shape/strides")
    if not all(shape):
        if not 0 <= offset <= extent:
            raise ValueError("EVAS empty view lies outside its base buffer")
        return
    low = offset + sum(min(0, (n - 1) * s) for n, s in zip(shape, strides)) * itemsize
    high = offset + sum(max(0, (n - 1) * s) for n, s in zip(shape, strides)) * itemsize + itemsize
    if low < 0 or high > extent:
        raise ValueError("EVAS view exceeds its described base buffer")


class TorchBufferRuntime:
    """Adapter using installed torch-lpu APIs, without SDK struct inspection."""

    def __init__(self):
        import torch
        import torch_lpu  # noqa: F401 -- registers the actual runtime APIs
        self.torch = torch

    def describe(self, argument):
        torch = self.torch
        wrapper_module = sys.modules.get("flag_gems.utils.tensor_wrapper")
        wrapper_type = getattr(wrapper_module, "StridedBuffer", ())
        is_wrapper = isinstance(argument, wrapper_type)
        tensor = argument._base if is_wrapper else argument
        if not isinstance(tensor, torch.Tensor):
            raise ValueError("EVAS materialization requires a Tensor or a known StridedBuffer")
        if tensor.device.type != "lpu":
            raise ValueError("EVAS automatic dual-die requires device tensors; CPU staging is unsupported")
        root, seen = tensor, set()
        while getattr(root, "_base", None) is not None:
            if id(root) in seen or not isinstance(root._base, torch.Tensor):
                raise ValueError("EVAS invalid tensor base chain")
            seen.add(id(root))
            root = root._base
        if root.device != tensor.device or root.layout != torch.strided:
            raise ValueError("EVAS requires a strided base on the same device")
        split, affinity = _placement(root)
        if split < -1 or split >= root.ndim or (split == -1 and affinity not in (0, 1)):
            raise ValueError("EVAS unsupported allocation placement")
        materialize = split != -1 or affinity != 0
        # A dense base can be copied without changing the kernel's offsets.
        # Views and StridedBuffer may be non-contiguous, including negative
        # strides, provided their full byte span fits in that base.
        if materialize and (not root.is_contiguous() or root.storage_offset() != 0):
            raise ValueError("EVAS materialization needs a contiguous, zero-offset base")
        itemsize = int(argument.element_size())
        offset = int(tensor.storage_offset()) * tensor.element_size()
        offset -= int(root.storage_offset()) * root.element_size()
        if is_wrapper:
            offset += int(argument.offset) * itemsize
        shape = tuple(int(n) for n in argument.shape)
        strides = tuple(int(s) for s in argument.stride())
        root_bytes = int(root.numel()) * root.element_size()
        if not materialize:
            # Native complete buffers retain their actual storage span, even
            # when their owning Tensor has non-contiguous geometry.
            root_bytes = root.untyped_storage().nbytes() - root.storage_offset() * root.element_size()
        _check_bounds(shape, strides, itemsize, offset, root_bytes)
        pieces = [root]
        if split >= 0:
            if root.shape[split] < 2:
                raise ValueError("EVAS runtime cannot describe shards with split-axis size < 2; "
                                 "use a complete single-die allocation")
            pieces = list(torch.ops.lpu.unzip_double_die_tensor(root))
            if len(pieces) != 2:
                raise ValueError("EVAS expected two physical shards")
            split_at = (root.shape[split] + 1) // 2
            for die, piece in enumerate(pieces):
                expected = list(root.shape)
                expected[split] = split_at if die == 0 else root.shape[split] - split_at
                if (tuple(piece.shape) != tuple(expected) or piece.dtype != root.dtype or piece.device != root.device
                        or _placement(piece) != (-1, die) or not piece.is_contiguous()):
                    raise ValueError("EVAS physical shards disagree with logical base geometry")
        ranges = []
        identities = []
        for piece in pieces:
            storage = piece.untyped_storage()
            start, size = int(storage.data_ptr()), int(storage.nbytes())
            if piece.numel() and (not start or size < piece.numel() * piece.element_size()):
                raise ValueError("EVAS invalid allocation extent")
            identities.append((start, size) if size else ("empty", id(root)))
            if size:
                ranges.append((start, start + size))
        device_index = root.device.index
        if device_index is None:
            raise ValueError("EVAS tensor descriptor requires an explicit device index")
        # Allocation ranges identify aliasing, not affinity: affinity was
        # obtained separately through the runtime placement API.
        key = (device_index, tuple(identities))
        return BufferDescriptor(argument, root, root.device, argument.dtype, shape, strides, itemsize, offset,
                                root_bytes, key, tuple(ranges), split, affinity)

    def allocate(self, descriptor):
        torch = self.torch
        with torch.lpu.device(descriptor.device), torch.lpu.SingleDieMode(die_affinity=0):
            result = torch.ops.lpu.empty_with_format(tuple(descriptor.root.shape), dtype=descriptor.root.dtype,
                                                     device=descriptor.device, split_dim=None, die_affinity=0)
        if (_placement(result) != (-1, 0) or not result.is_contiguous() or result.shape != descriptor.root.shape
                or result.dtype != descriptor.root.dtype):
            raise RuntimeError("EVAS runtime did not allocate the requested complete die-0 buffer")
        return result

    def copy(self, destination, source):
        # Copy is handled by torch-lpu; no host round trip and no global mode
        # change during tensor data movement.
        with self.torch.no_grad():
            destination.copy_(source, non_blocking=False)

    def allocate_local(self, descriptor, elements, die):
        torch = self.torch
        with torch.lpu.device(descriptor.device), torch.lpu.SingleDieMode(die_affinity=die):
            result = torch.ops.lpu.empty_with_format((elements, ), dtype=descriptor.root.dtype,
                                                     device=descriptor.device, split_dim=None, die_affinity=die)
        if (_placement(result) != (-1, die) or not result.is_contiguous() or result.numel() != elements
                or result.dtype != descriptor.root.dtype):
            raise RuntimeError("EVAS runtime did not allocate the requested local window")
        return result

    def flat_view(self, tensor, begin, elements):
        return tensor.view(-1).narrow(0, begin, elements)

    def borrow_local(self, descriptor, begin, end, die):
        """Reuse an already resident linear window when its geometry is known."""
        if descriptor.split_dim == -1 and descriptor.affinity == die:
            return self.flat_view(descriptor.root, begin, end - begin)
        if descriptor.split_dim == 0:
            root = descriptor.root
            split = ((root.shape[0] + 1) // 2) * math.prod(root.shape[1:])
            low, high = (0, split) if die == 0 else (split, root.numel())
            if low <= begin < end <= high:
                piece = self.torch.ops.lpu.unzip_double_die_tensor(root)[die]
                return self.flat_view(piece, begin - low, end - begin)
        return None

    def synchronize(self, device):
        self.torch.lpu.synchronize(device)


# If the runtime cannot establish completion after an error, do not let Python
# return potentially in-flight storage to the allocator. Explicit recovery can
# release it once the same device successfully synchronizes.
_quarantined = []


def recover_buffer_temporaries():
    from .local_shards import recover_local_temporaries
    recover_local_temporaries()
    for plan in list(_quarantined):
        plan.runtime.synchronize(plan.device)
        plan.temporaries.clear()
        for group in plan.groups.values():
            group.pop("temporary", None)
        plan.bindings = list(plan.arguments)
        _quarantined.remove(plan)


class AutoBufferPlan:

    def __init__(self, signature, arguments, contract, runtime=None):
        self.arguments = tuple(arguments)
        self.bindings = list(arguments)
        self.runtime = runtime
        self.device = None
        self.groups = {}
        self.temporaries = []
        self.active = False
        self.descriptors = []
        raw_pointers = False
        if len(signature) != len(arguments):
            raise ValueError("EVAS signature and argument count disagree")
        roles = None
        if contract is not None:
            if (contract.get("version") != ACCESS_VERSION or len(contract.get("roles", ())) != len(arguments)):
                raise ValueError("EVAS incompatible argument access metadata; recompile the kernel")
            if any(role not in ("r", "w", "rw", "none") for role in contract["roles"]):
                raise ValueError("EVAS invalid argument access role")
            if contract.get("supported"):
                roles = contract["roles"]
        for position, ((_, ty), argument) in enumerate(zip(signature.items(), arguments)):
            if not ty.startswith("*") or argument is None:
                continue
            if isinstance(argument, int):
                raw_pointers |= argument != 0
                continue
            if self.runtime is None:
                self.runtime = TorchBufferRuntime()
            descriptor = self.runtime.describe(argument)
            self.descriptors.append(descriptor)
            if self.device is not None and self.device != descriptor.device:
                raise ValueError("EVAS materialization cannot span multiple devices")
            self.device = descriptor.device
            group = self.groups.setdefault(descriptor.storage_key,
                                           {"descriptor": descriptor, "positions": [], "write": False})
            previous = group["descriptor"]
            # Two independently shaped roots for the same split allocation do
            # not establish a unique logical storage order. Reject that case.
            if descriptor.materialize or previous.materialize:
                if (tuple(descriptor.root.shape), tuple(descriptor.root.stride()), descriptor.root.dtype,
                        descriptor.split_dim,
                        descriptor.affinity) != (tuple(previous.root.shape), tuple(previous.root.stride()),
                                                 previous.root.dtype, previous.split_dim, previous.affinity):
                    raise ValueError("EVAS aliased roots have incompatible materialization layouts")
            group["positions"].append((position, descriptor))
            group["write"] |= roles is not None and "w" in roles[position]
        staged = [g for g in self.groups.values() if g["descriptor"].materialize]
        if staged and roles is None:
            reason = "missing metadata" if contract is None else ", ".join(contract.get("reasons", ()))
            raise ValueError("EVAS cannot materialize arguments without verified access roles: " + reason)
        if staged and raw_pointers:
            raise ValueError("EVAS cannot prove raw-pointer aliasing while materializing tensors")
        groups = list(self.groups.values())
        for i, left in enumerate(groups):
            a = left["descriptor"]
            for right in groups[i + 1:]:
                b = right["descriptor"]
                if not (a.materialize or b.materialize):
                    continue
                if any(max(x0, y0) < min(x1, y1) for x0, x1 in a.physical_ranges for y0, y1 in b.physical_ranges):
                    raise ValueError("EVAS overlapping allocations need a common logical base descriptor")

    def __enter__(self):
        if not any(g["descriptor"].materialize for g in self.groups.values()):
            return self
        self.active = True
        try:
            self.runtime.synchronize(self.device)
            for group in self.groups.values():
                descriptor = group["descriptor"]
                if not descriptor.materialize:
                    continue
                temporary = self.runtime.allocate(descriptor)
                self.temporaries.append(temporary)
                group["temporary"] = temporary
                # A store may be masked or conditional. Preserve untouched
                # elements, even for a write-only argument, absent coverage proof.
                self.runtime.copy(temporary, descriptor.root)
                for position, view in group["positions"]:
                    self.bindings[position] = BoundPointer(view, int(temporary.data_ptr()) + view.offset_bytes)
            self.runtime.synchronize(self.device)
            return self
        except BaseException:
            self._release()
            raise

    def _release(self):
        if not self.active:
            return
        try:
            self.runtime.synchronize(self.device)
        except BaseException:
            if self not in _quarantined:
                _quarantined.append(self)
            raise
        finally:
            self.active = False
        self.temporaries.clear()
        for group in self.groups.values():
            group.pop("temporary", None)
        self.bindings = list(self.arguments)

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            if self.active and exc_type is None:
                self.runtime.synchronize(self.device)
                for group in self.groups.values():
                    if group["write"] and "temporary" in group:
                        self.runtime.copy(group["descriptor"].root, group["temporary"])
        finally:
            self._release()
        return False
