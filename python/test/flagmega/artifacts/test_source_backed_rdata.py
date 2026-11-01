# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT

import json
from dataclasses import replace

import numpy as np
import pytest
import torch
from safetensors.numpy import save_file

from triton.flagmega import ir as fm
from triton.flagmega.artifacts import pack_rdata, verify_rdata, write_artifact, load_artifact
from triton.flagmega.errors import ArtifactError
from triton.flagmega.importer import DirectoryCheckpoint
from triton.flagmega.passes.constants import freeze_constant_islands
from triton.flagmega.passes.tir.bufferize.planner import BufferizationOptions
from triton.flagmega.passes.tir.bufferize.policy import NttBufferizationPolicy


def _fixture(tmp_path, *, axis=0, cyclic=False, reshape=False):
    root = tmp_path / "checkpoint"
    root.mkdir()
    (root / "config.json").write_text("{}")
    shapes = ((2, 6), (4, 6)) if axis == 0 else ((6, 2), (6, 4))
    values = {f"w{i}": np.arange(np.prod(shape), dtype=np.float32).reshape(shape) + i * 100
              for i, shape in enumerate(shapes)}
    save_file(values, root / "model.safetensors")
    checkpoint = DirectoryCheckpoint(root)
    placement = fm.Placement((2, 2), "cx", "cb")
    builder = fm.IRBuilder(dialect="tir", stage="selected_tir",
                           metadata={"launch_contract": {"grid_mesh": placement.to_data()}})
    weights = [builder.weight(key, fm.tensor_type("float32", value.shape),
                              source="checkpoint", key=key, id=key) for key, value in values.items()]
    shape = (6, 6)
    joined = builder.call("tensors.concat", weights, fm.tensor_type("float32", shape),
                          attrs={"axis": axis}, id="joined")
    if reshape:
        shape = (2, 18)
        joined = builder.call("tensors.reshape", (joined,), fm.tensor_type("float32", shape),
                              attrs={"shape": shape}, id="view")
    policies = (fm.SBP.split_contiguous((0,)),
                fm.SBP.split_block_cyclic((1,), block_size=2) if cyclic else fm.SBP.split_contiguous((1,)))
    distributed = fm.DistributedType(fm.tensor_type("float32", shape), policies, placement)
    output = builder.call("distributed.sharded_view", (joined,), distributed,
                          attrs={"new_type": distributed}, id="output")
    builder.function("main", (), (output,))
    module = freeze_constant_islands(builder.build(entry="main"))
    module = NttBufferizationPolicy(replace(BufferizationOptions.generic(alignment=4),
                                           rdata_chip_sharding=True)).bufferize(module)
    return replace(module, stage="bufferized_tir"), checkpoint, np.concatenate(tuple(values.values()), axis=axis).reshape(shape)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("axis,cyclic,reshape", [(0, False, False), (1, False, False),
                                               (0, True, False), (0, False, True)])
def test_source_snapshots_load_exact_dense_chip_layout(tmp_path, axis, cyclic, reshape, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA source transfer requires a GPU")
    from triton.flagmega.runtime.source_rdata import load_source_rdata
    from triton.flagmega.artifacts.rdata import _extract_local_shard
    module, checkpoint, source = _fixture(tmp_path, axis=axis, cyclic=cyclic, reshape=reshape)
    assets = tmp_path / "assets"
    index = pack_rdata(module, checkpoint, assets)
    assert index["digest_kind"] == "sha256-source-layout/v1"
    assert index["programs"]
    # The artifact survives source removal and subsequent edits independently.
    (checkpoint.root / "model.safetensors").unlink()
    buffer = next(b for b in fm.verify_buffer_plan(module).buffers if b.storage == "rdata")
    for chip in range(2):
        image = load_source_rdata(assets, index, module, chip=chip, device=device).cpu().numpy()
        for block in range(2):
            descriptor = fm.local_shard_descriptor(buffer.distributed_type, (chip, block))
            expected = _extract_local_shard(source, descriptor, buffer.component_shape)
            offset = buffer.offset + block * buffer.component_stride_bytes
            actual = image[offset:offset + buffer.nbytes].view(np.float32).reshape(expected.shape)
            np.testing.assert_array_equal(actual, expected)
    verify_rdata(assets, module=module)


def test_source_recipe_never_materializes_a_global_concat(tmp_path, monkeypatch):
    import triton.flagmega.artifacts.rdata as rdata
    module, checkpoint, _ = _fixture(tmp_path)
    def forbidden(*args, **kwargs):
        pytest.fail("source layout physically materialized its byte-preserving concat")
    monkeypatch.setattr(rdata, "materialize_numpy_recipe", forbidden)
    index = pack_rdata(module, checkpoint, tmp_path / "assets")
    assert index["digest_kind"] == "sha256-source-layout/v1"


def test_source_snapshot_corruption_is_detected(tmp_path):
    module, checkpoint, _ = _fixture(tmp_path)
    assets = tmp_path / "assets"
    index = pack_rdata(module, checkpoint, assets)
    source = assets / index["sources"][0]["path"]
    with source.open("r+b") as stream:
        stream.seek(-1, 2)
        stream.write(b"\xff")
    with pytest.raises(ArtifactError, match="source.*hash"):
        verify_rdata(assets, module=module)


def test_source_layout_manifest_roundtrip(tmp_path):
    module, checkpoint, _ = _fixture(tmp_path)
    root = write_artifact(module, tmp_path / "artifact", target="test", checkpoint=checkpoint)
    manifest, loaded = load_artifact(root)
    assert loaded.semantic_hash == module.semantic_hash
    assert manifest["rdata"]["digest_kind"] == "sha256-source-layout/v1"
    assert any(section["kind"] == "rdata.source/v1" for section in manifest["sections"])


@pytest.mark.parametrize("tamper", ["shape", "destination", "source_bounds", "numeric"])
def test_source_geometry_is_verified_against_recipe_and_plan(tmp_path, tamper):
    from triton.flagmega.artifacts.source_layout import content_digest
    module, checkpoint, _ = _fixture(tmp_path)
    assets = tmp_path / "assets"
    index = pack_rdata(module, checkpoint, assets)
    program = index["programs"][0]
    if tamper == "shape":
        program["shape"] = [1, 36, 4]
    elif tamper == "destination":
        program["ranges"][0][0] += 1
    elif tamper == "source_bounds":
        program["operands"][0]["offset"] = index["sources"][0]["nbytes"]
    else:
        recipe = module.constant_recipes[0]
        nodes = tuple(replace(node, op="tensors.permute", attrs={"axes": (1, 0)})
                      if node.op == "tensors.concat" else node for node in recipe.nodes)
        module = replace(module, constant_recipes=(replace(recipe, nodes=nodes),))
    index["sha256"] = content_digest(index)
    (assets / "rdata.index.json").write_text(json.dumps(index))
    with pytest.raises(ArtifactError):
        verify_rdata(assets, module=module, verify_image_bytes=False)


def test_snapshot_clone_resolves_only_identical_exposed_overlay_inode(tmp_path):
    from triton.flagmega.artifacts.rdata import _overlay_clone_source
    root = tmp_path / "overlay"
    upper = tmp_path / "upper"
    root.mkdir()
    upper.mkdir()
    source = root / "weight.bin"
    backing = upper / "weight.bin"
    source.write_bytes(b"verified source")
    backing.hardlink_to(source)
    mountinfo = f"1 0 0:1 / {root} rw - overlay overlay rw,upperdir={upper}\n"
    assert _overlay_clone_source(source, backing.stat().st_dev, mountinfo=mountinfo) == backing
    backing.unlink()
    backing.write_bytes(b"different source")
    assert _overlay_clone_source(source, backing.stat().st_dev, mountinfo=mountinfo) is None


def test_snapshot_clone_uses_verified_bind_mount_alias(tmp_path):
    from triton.flagmega.artifacts.rdata import _bind_clone_destination
    filesystem = tmp_path / "filesystem"
    bind = tmp_path / "bind"
    (filesystem / "sub").mkdir(parents=True)
    bind.mkdir()
    source = filesystem / "source.bin"
    source.write_bytes(b"source")
    alias = filesystem / "sub" / "target.bin"
    alias.write_bytes(b"")
    destination = bind / "target.bin"
    destination.hardlink_to(alias)
    mountinfo = (f"1 0 0:1 / {filesystem} rw - xfs test rw\n"
                 f"2 1 0:1 /sub {bind} rw - xfs test rw\n")
    assert _bind_clone_destination(destination, source, mountinfo=mountinfo) == alias
    alias.unlink()
    alias.write_bytes(b"different inode")
    assert _bind_clone_destination(destination, source, mountinfo=mountinfo) is None


def test_mount_lookup_resolves_each_path_once_regardless_of_mount_count(tmp_path, monkeypatch):
    from triton.flagmega.artifacts.rdata import _bind_clone_destination

    source, destination = tmp_path / "source", tmp_path / "destination"
    source.write_bytes(b"source")
    destination.write_bytes(b"destination")
    mountinfo = "".join(f"{i} 0 0:1 / /unrelated/{i} rw - xfs test rw\n" for i in range(100))
    calls = []
    original = type(source).resolve

    def measure(path, *args, **kwargs):
        calls.append(path)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(type(source), "resolve", measure)
    assert _bind_clone_destination(destination, source, mountinfo=mountinfo) is None
    assert calls == [source, destination]


def test_source_snapshots_are_preserved_by_private_artifact_cache(tmp_path):
    module, checkpoint, _ = _fixture(tmp_path)
    first = pack_rdata(module, checkpoint, tmp_path / "first", cache_dir=tmp_path / "cache")
    second = pack_rdata(module, checkpoint, tmp_path / "second", cache_dir=tmp_path / "cache")
    assert first == second
    verify_rdata(tmp_path / "second", module=module)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("lanes,axes", [((2, 2), (0, 1)), ((2, 2), (1, 1))])
def test_permuted_vector_packing_transfers_without_cpu_reordering(tmp_path, device, lanes, axes):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA source transfer requires a GPU")
    from triton.flagmega.runtime.source_rdata import load_source_rdata
    from triton.flagmega.ir.ops.tensors.pack import pack_physical
    root = tmp_path / "checkpoint"
    root.mkdir()
    (root / "config.json").write_text("{}")
    value = np.arange(64, dtype=np.float32).reshape(8, 8)
    save_file({"weight": value}, root / "model.safetensors")
    checkpoint = DirectoryCheckpoint(root)
    class Graph(fm.Module):
        def forward(self):
            weight = self.weight("weight", fm.tensor_type("float32", (8, 8)), source="checkpoint", key="weight")
            packed = fm.F.tensors.pack(weight, lanes=lanes, axes=axes)
            permuted = fm.F.tensors.permute(packed, (1, 0))
            output = fm.F.tensors.pack(permuted, lanes=(2,), axes=(0,))
            self.function("main", (), (output,))
    module = freeze_constant_islands(Graph(dialect="tir", stage="selected_tir", entry="main").build())
    module = NttBufferizationPolicy(BufferizationOptions.generic(alignment=4)).bufferize(module)
    module = replace(module, stage="bufferized_tir", dialect="tir")
    expected = pack_physical(torch.from_numpy(value), 2, lanes, axes).permute(1, 0, 2, 3).contiguous()
    expected = pack_physical(expected, 2, (2,), (0,), lanes)
    assets = tmp_path / "assets"
    index = pack_rdata(module, checkpoint, assets)
    assert len(index["programs"][0]["transforms"]) == 3
    actual = load_source_rdata(assets, index, module, chip=0, device=device).cpu().numpy()
    np.testing.assert_array_equal(actual[:expected.numel() * 4], expected.view(torch.uint8).numpy().reshape(-1))
    verify_rdata(assets, module=module)


class _FakeDriverFunction:
    def __init__(self, calls, name, status=0, pointer=None):
        self.calls, self.name, self.status, self.pointer = calls, name, status, pointer
        self.argtypes = None

    def __call__(self, *args):
        self.calls.append((self.name, args))
        if self.pointer is not None:
            args[0]._obj.value = self.pointer
        return self.status


class _FakeDriver:
    POINTER = 0x7F00DEAD0000

    def __init__(self, names, *, register=0, get_pointer=0, unregister=0):
        self.calls = []
        register_name, pointer_name, unregister_name = names
        setattr(self, register_name, _FakeDriverFunction(self.calls, "register", register))
        setattr(self, pointer_name, _FakeDriverFunction(self.calls, "get_pointer", get_pointer, self.POINTER))
        setattr(self, unregister_name, _FakeDriverFunction(self.calls, "unregister", unregister))


_DRIVERS = {
    "hip": ("6.4", "libamdhip64.so",
            ("hipHostRegister", "hipHostGetDevicePointer", "hipHostUnregister"), 0x03),
    "cuda": (None, "libcuda.so.1",
             ("cuMemHostRegister_v2", "cuMemHostGetDevicePointer_v2", "cuMemHostUnregister"), 0x0B),
}


def _registered_file(tmp_path, monkeypatch, build, **statuses):
    from triton.flagmega.runtime import source_rdata
    hip_version, library, names, _ = _DRIVERS[build]
    driver = _FakeDriver(names, **statuses)
    opened = []
    monkeypatch.setattr(torch.version, "hip", hip_version, raising=False)
    monkeypatch.setattr(source_rdata.ctypes, "CDLL", lambda name: opened.append(name) or driver)
    path = tmp_path / "source.bin"
    path.write_bytes(bytes(range(64)))
    return source_rdata._RegisteredFile, path, driver, opened, library


@pytest.mark.parametrize("build", ["hip", "cuda"])
def test_source_pages_register_through_the_build_driver(tmp_path, monkeypatch, build):
    registered_file, path, driver, opened, library = _registered_file(tmp_path, monkeypatch, build)
    mapping = registered_file(path)
    assert opened == [library]
    assert mapping.device_pointer == driver.POINTER
    (register, (address, nbytes, flags)), (get_pointer, (_, pointer_address, pointer_flags)) = driver.calls
    assert (register, get_pointer) == ("register", "get_pointer")
    assert (address, nbytes, flags) == (mapping.address, 64, _DRIVERS[build][3])
    assert (pointer_address, pointer_flags) == (mapping.address, 0)
    mapping.close()
    assert driver.calls[-1] == ("unregister", (mapping.address,))
    assert mapping.array._mmap.closed


@pytest.mark.parametrize("build", ["hip", "cuda"])
def test_source_page_registration_failure_releases_the_mapping(tmp_path, monkeypatch, build):
    registered_file, path, driver, _, _ = _registered_file(tmp_path, monkeypatch, build, register=1)
    arrays = []
    monkeypatch.setattr(np, "memmap", lambda *args, _memmap=np.memmap, **kwargs:
                        arrays.append(_memmap(*args, **kwargs)) or arrays[-1])
    with pytest.raises(ArtifactError, match="cannot register"):
        registered_file(path)
    assert [name for name, _ in driver.calls] == ["register"]
    assert arrays[0]._mmap.closed


@pytest.mark.parametrize("build", ["hip", "cuda"])
def test_source_page_mapping_failure_unregisters(tmp_path, monkeypatch, build):
    registered_file, path, driver, _, _ = _registered_file(tmp_path, monkeypatch, build, get_pointer=1)
    arrays = []
    monkeypatch.setattr(np, "memmap", lambda *args, _memmap=np.memmap, **kwargs:
                        arrays.append(_memmap(*args, **kwargs)) or arrays[-1])
    with pytest.raises(ArtifactError, match="cannot map"):
        registered_file(path)
    assert [name for name, _ in driver.calls] == ["register", "get_pointer", "unregister"]
    assert arrays[0]._mmap.closed


@pytest.mark.parametrize("build", ["hip", "cuda"])
def test_source_page_unregister_failure_still_closes_the_mapping(tmp_path, monkeypatch, build):
    registered_file, path, driver, _, _ = _registered_file(tmp_path, monkeypatch, build, unregister=1)
    mapping = registered_file(path)
    with pytest.raises(ArtifactError, match="cannot unregister"):
        mapping.close()
    assert mapping.array._mmap.closed
