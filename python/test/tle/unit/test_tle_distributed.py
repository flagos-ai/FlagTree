# flagtree tle
import pytest

import triton.experimental.tle.language as tle
from triton.experimental.tle.language import (
    _infer_submesh_barrier_group,
    _mesh_to_cluster_dims,
    _normalize_remote_shard_id,
    _resolve_launch_axis,
    _is_cluster_submesh,
)
import triton.language.core as tlcore


class TestDeviceMesh:

    def test_device_mesh_shape_and_flatten(self):
        mesh = tle.device_mesh({
            "node": [("node_x", 2), ("node_y", 2)],
            "device": 4,
            "block_cluster": [("cluster_x", 2), ("cluster_y", 2)],
            "block": 4,
        })
        assert mesh.shape == (2, 2, 4, 2, 2, 4)
        assert mesh.ndim == 6
        assert mesh.size == 256

        flat = mesh.flatten()
        assert flat.shape == (256, )
        assert flat.dim_names == ("flat", )

    def test_device_mesh_slice_submesh(self):
        mesh = tle.device_mesh({
            "node": [("node_x", 2), ("node_y", 2)],
            "device": 4,
        })
        sub = mesh[1, :, 2]
        assert sub.shape == (2, )
        assert sub.dim_names == ("node_y", )
        assert sub.size == 2

    def test_device_mesh_invalid_topology(self):
        with pytest.raises(TypeError):
            tle.device_mesh({"node": "2"})
        with pytest.raises(ValueError):
            tle.device_mesh({"node": []})
        with pytest.raises(ValueError):
            tle.device_mesh({"node": [("x", 0)]})


class TestShardingSpec:

    def test_sharding_spec_states(self):
        mesh = tle.device_mesh({
            "device": 4,
            "cluster": [("cluster_x", 2), ("cluster_y", 2)],
            "block": 4,
        })
        spec = tle.sharding(mesh, split=[["cluster_x", "cluster_y"], "device"], partial=["block"])
        assert spec.axis_state("cluster_x") == "S"
        assert spec.axis_state("cluster_y") == "S"
        assert spec.axis_state("device") == "S"
        assert spec.axis_state("block") == "P"
        assert spec.broadcast == tuple()

    def test_sharding_rejects_overlap(self):
        mesh = tle.device_mesh({"device": 4, "block": 4})
        with pytest.raises(ValueError):
            tle.sharding(mesh, split=["device"], partial=["device"])

    def test_make_sharded_tensor(self):
        mesh = tle.device_mesh({"device": 4, "block": 4})
        spec = tle.sharding(mesh, split=["device", "block"], partial=[])
        st = tle.make_sharded_tensor("x_ptr", sharding=spec, shape=[4, 8])
        assert st.handle == "x_ptr"
        assert st.shape == (4, 8)
        assert st.sharding == spec

    def test_reshard_deferred(self):
        mesh = tle.device_mesh({"device": 4})
        spec = tle.sharding(mesh, split=["device"], partial=[])
        st = tle.make_sharded_tensor("x_ptr", sharding=spec, shape=[8])
        with pytest.raises(NotImplementedError):
            tle.reshard(st, spec)


class TestRemoteShardId:

    def test_normalize_remote_shard_id_scalar(self):
        assert _normalize_remote_shard_id(3, None) == 3
        with pytest.raises(ValueError):
            _normalize_remote_shard_id(-1, None)

    def test_normalize_remote_shard_id_tuple(self):
        mesh = tle.device_mesh({"cluster": [("x", 2), ("y", 4)]})
        assert _normalize_remote_shard_id((1, 3), mesh) == 7
        with pytest.raises(ValueError):
            _normalize_remote_shard_id((2, 0), mesh)
        with pytest.raises(ValueError):
            _normalize_remote_shard_id((1, 3), None)

    def test_m3_entrypoints_are_builtins(self):
        assert tlcore.is_builtin(tle.remote)
        assert tlcore.is_builtin(tle.distributed_barrier)
        assert tlcore.is_builtin(tle.shard_id)

    def test_remote_buffered_tensor_attach_metadata(self):

        class buffered_tensor:

            def __init__(self):
                self.handle = object()
                self.type = object()

        buf = buffered_tensor()
        remote_buf = tle.remote(buf, 1, _semantic=_FakeSemantic())

        assert remote_buf is not buf
        shard_id = getattr(remote_buf.type, "_tle_remote_shard_id", getattr(remote_buf, "_tle_remote_shard_id", None))
        scope = getattr(remote_buf.type, "_tle_remote_scope", getattr(remote_buf, "_tle_remote_scope", None))
        assert shard_id == 1
        assert scope is None
        assert not hasattr(buf, "_tle_remote_shard_id")
        assert not hasattr(buf.type, "_tle_remote_shard_id")
        assert not hasattr(buf.type, "_tle_remote_scope")

    def test_remote_buffered_tensor_rejects_duplicate_annotation(self):

        class buffered_tensor:

            def __init__(self):
                self.handle = object()
                self.type = object()

        remote_buf = tle.remote(buffered_tensor(), 0, _semantic=_FakeSemantic())
        with pytest.raises(ValueError, match="cannot be applied twice"):
            tle.remote(remote_buf, 1, _semantic=_FakeSemantic())

    def test_remote_buffered_tensor_validates_shard_id_early(self):

        class buffered_tensor:

            def __init__(self):
                self.handle = object()
                self.type = object()

        buf = buffered_tensor()
        with pytest.raises(ValueError, match="shard_id must be >= 0"):
            tle.remote(buf, -1, _semantic=_FakeSemantic())
        with pytest.raises(ValueError, match="tuple shard_id requires scope"):
            tle.remote(buf, (0, 0), _semantic=_FakeSemantic())


class TestClusterDims:

    def test_mesh_to_cluster_dims_prefers_cluster_axes(self):
        mesh = tle.device_mesh({
            "node": [("node_x", 2), ("node_y", 2)],
            "device": 4,
            "block_cluster": [("cluster_x", 2), ("cluster_y", 1)],
            "block": 8,
        })
        assert _mesh_to_cluster_dims(mesh) == (2, 1, 1)

    def test_mesh_to_cluster_dims_fallback_to_block_axes(self):
        mesh = tle.device_mesh({"device": 4, "block": [("block_x", 2), ("block_y", 2)]})
        assert _mesh_to_cluster_dims(mesh) == (2, 2, 1)

    def test_mesh_to_cluster_dims_submesh_keeps_launch_view(self):
        mesh = tle.device_mesh({"block_cluster": [("cluster_x", 2), ("cluster_y", 2)]})
        sub = mesh[0, :]
        assert sub.shape == (2, )
        assert _mesh_to_cluster_dims(sub) == (2, 2, 1)

    def test_mesh_to_cluster_dims_uses_topology_level_with_custom_axis_names(self):
        mesh = tle.device_mesh(
            tle.MeshConfig(
                node=[("rack", 2)],
                device=[("gpu", 4)],
                block_cluster=[("cluster_x", 2)],
            ))
        assert _mesh_to_cluster_dims(mesh) == (2, 1, 1)

    def test_resolve_launch_axis(self):
        mesh = tle.device_mesh({"block_cluster": [("cluster_x", 2), ("cluster_y", 2)]})
        assert _resolve_launch_axis(mesh, "cluster_x") == 0
        assert _resolve_launch_axis(mesh, "cluster_y") == 1
        assert _resolve_launch_axis(mesh, 0) == 0
        assert _resolve_launch_axis(mesh, -1) == 1
        with pytest.raises(ValueError):
            _resolve_launch_axis(mesh, "missing_axis")
        with pytest.raises(IndexError):
            _resolve_launch_axis(mesh, 2)


class _FakeOptions:

    def __init__(self):
        self.num_ctas = 1
        self.cluster_dims = (1, 1, 1)
        self.launch_cooperative_grid = False


class _FakeBuilder:

    def __init__(self):
        self.options = _FakeOptions()
        self.distributed_barrier_calls = 0
        self.distributed_barrier_group_args = []
        self.barrier_calls = 0

    def create_distributed_barrier(self, *args):
        self.distributed_barrier_calls += 1
        self.distributed_barrier_group_args.append(tuple(args))

    def create_barrier(self):
        self.barrier_calls += 1


class _LegacyDistributedBarrierBuilder(_FakeBuilder):

    def create_distributed_barrier(self):
        self.distributed_barrier_calls += 1
        self.distributed_barrier_group_args.append(tuple())


class _FakeSemantic:

    def __init__(self, builder=None):
        self.builder = _FakeBuilder() if builder is None else builder


class _FakeDevicePtr:

    def __init__(self, handle="comm"):
        self.handle = handle


class _LegacyBarrierSemantic:

    def __init__(self):
        self.builder = _LegacyDistributedBarrierBuilder()


class TestShardId:

    @pytest.mark.parametrize("axis", ("device", "node"))
    def test_rank_axis_requires_device_dptr(self, axis):
        mesh = tle.device_mesh({"node": 2, "device": 4})
        semantic = _FakeSemantic()
        with pytest.raises(ValueError, match=rf"device_dptr is required for axis '{axis}'"):
            tle.shard_id(mesh, axis, _semantic=semantic)


class TestDistributedBarrierScope:

    @pytest.mark.require_tle("distributed_barrier")
    @pytest.mark.parametrize(
        ("space", "mesh", "required_level"),
        (("inter", tle.device_mesh({"device": 2}), "node"), ("world", tle.device_mesh({"device": 2}), "node"),
         ("device", tle.device_mesh({"node": 2}), "device")),
    )
    def test_flagcx_barrier_requires_matching_mesh_axis(self, space, mesh, required_level):
        with pytest.raises(ValueError, match=rf"requires mesh to define a '{required_level}' topology axis"):
            tle.distributed_barrier(
                mesh=mesh,
                device_dptr=_FakeDevicePtr(),
                space=space,
                _semantic=_FakeSemantic(),
            )

    def test_flagcx_barrier_requires_mesh(self):
        with pytest.raises(ValueError, match="mesh is required"):
            tle.distributed_barrier(
                device_dptr=_FakeDevicePtr(),
                space="device",
                _semantic=_FakeSemantic(),
            )

    def test_flagcx_barrier_requires_device_pointer(self):
        mesh = tle.device_mesh({"node": 2})
        with pytest.raises(ValueError, match="device_dptr is required"):
            tle.distributed_barrier(mesh=mesh, space="world", _semantic=_FakeSemantic())

    def test_distributed_barrier_full_cluster_mesh(self):
        mesh = tle.device_mesh({"block_cluster": [("cluster_x", 2), ("cluster_y", 1)]})
        semantic = _FakeSemantic()
        tle.distributed_barrier(mesh=mesh, _semantic=semantic)
        assert semantic.builder.options.cluster_dims == (2, 1, 1)
        assert semantic.builder.distributed_barrier_calls == 1
        assert semantic.builder.distributed_barrier_group_args == [tuple()]

    @pytest.mark.require_tle("distributed_barrier")
    def test_distributed_barrier_submesh_emits_group_descriptor(self):
        mesh = tle.device_mesh({"block_cluster": [("cluster_x", 2), ("cluster_y", 2)]})
        sub = mesh[0, :]
        semantic = _FakeSemantic()
        tle.distributed_barrier(mesh=sub, _semantic=semantic)
        assert semantic.builder.distributed_barrier_calls == 1
        args = semantic.builder.distributed_barrier_group_args[0]
        assert args[0] == "submesh"
        assert args[1] == [2]
        assert args[2] == [1]
        assert args[3] == [0, 1]

    def test_distributed_barrier_submesh_requires_group_aware_builder(self):
        mesh = tle.device_mesh({"block_cluster": [("cluster_x", 2), ("cluster_y", 2)]})
        sub = mesh[0, :]
        semantic = _LegacyBarrierSemantic()
        with pytest.raises(NotImplementedError, match="requires rebuilt TLE extension"):
            tle.distributed_barrier(mesh=sub, _semantic=semantic)

    @pytest.mark.require_tle("distributed_barrier")
    def test_distributed_barrier_grid_mesh_enables_coop_launch(self):
        mesh = tle.device_mesh({"block": [("block_x", 4)]})
        semantic = _FakeSemantic()
        tle.distributed_barrier(mesh=mesh, _semantic=semantic)
        assert semantic.builder.options.launch_cooperative_grid is True
        assert semantic.builder.options.cluster_dims == (1, 1, 1)
        assert semantic.builder.distributed_barrier_calls == 1
        assert semantic.builder.distributed_barrier_group_args == [("grid", [], [], [])]

    def test_distributed_barrier_grid_mesh_rejects_cluster_launch(self):
        mesh = tle.device_mesh({"block": [("block_x", 4)]})
        semantic = _FakeSemantic()
        semantic.builder.options.cluster_dims = (2, 1, 1)
        with pytest.raises(ValueError, match="requires cluster_dims=\\(1, 1, 1\\)"):
            tle.distributed_barrier(mesh=mesh, _semantic=semantic)

    @pytest.mark.require_tle("distributed_barrier")
    def test_distributed_barrier_auto_can_pick_cluster_or_grid(self):
        mesh = tle.device_mesh({
            "block_cluster": [("cluster_x", 2)],
            "block": [("block_x", 4)],
        })

        semantic_cluster = _FakeSemantic()
        tle.distributed_barrier(mesh=mesh[:, 0], _semantic=semantic_cluster)
        assert semantic_cluster.builder.distributed_barrier_group_args == [tuple()]

        semantic_grid = _FakeSemantic()
        tle.distributed_barrier(mesh=mesh[0, :], _semantic=semantic_grid)
        assert semantic_grid.builder.distributed_barrier_group_args == [("grid", [], [], [])]

    def test_infer_submesh_barrier_group(self):
        mesh = tle.device_mesh({"block_cluster": [("cluster_x", 2), ("cluster_y", 2)]})
        sub = mesh[0, :]
        group = _infer_submesh_barrier_group(sub, (2, 2, 1))
        assert group is not None
        assert group.kind == "submesh"
        assert group.rank == 1
        assert group.shape == (2, )
        assert group.axes == (1, )
        assert group.mask == (0, 1)

    def test_infer_submesh_barrier_group_full_mesh_returns_none(self):
        mesh = tle.device_mesh({"block_cluster": [("cluster_x", 2), ("cluster_y", 2)]})
        assert _is_cluster_submesh(mesh) is None

    def test_is_cluster_submesh_returns_analysis(self):
        mesh = tle.device_mesh({"block_cluster": [("cluster_x", 2), ("cluster_y", 2)]})
        analysis = _is_cluster_submesh(mesh[0, :])
        assert analysis is not None
        assert analysis.cluster_axes == (0, 1)
        assert analysis.cluster_members_by_outer_coord == {(): {(0, 0), (0, 1)}}

    def test_infer_hierarchical_submesh_uses_cluster_local_mask(self):
        mesh = tle.device_mesh({
            "node": [("rack", 2)],
            "device": [("gpu", 4)],
            "block_cluster": [("cluster_x", 2)],
        })
        group = _infer_submesh_barrier_group(mesh[1, 3, 1:2], (2, 1, 1))

        assert group.shape == (1, )
        assert group.axes == (0, )
        assert group.mask == (1, )

    def test_infer_hierarchical_submesh_reuses_common_mask_across_outer_coords(self):
        mesh = tle.device_mesh({
            "node": [("rack", 2)],
            "device": [("gpu", 4)],
            "block_cluster": [("cluster_x", 2)],
        })
        group = _infer_submesh_barrier_group(mesh[:, :, 1:2], (2, 1, 1))

        assert group.shape == (1, )
        assert group.axes == (0, )
        assert group.mask == (1, )

    def test_infer_hierarchical_submesh_with_device_and_block_axes(self):
        mesh = tle.device_mesh({
            "device": [
                ("device_x", 2),
                ("device_y", 2),
            ],
            "block_cluster": [
                ("cluster_x", 2),
                ("cluster_y", 2),
            ],
            "block": [
                ("block_x", 4),
                ("block_y", 4),
            ],
        })

        cluster_x_submesh = mesh[:, :, 1, :, :, :]
        cluster_y_submesh = mesh[:, :, :, 1, :, :]
        cluster_x_group = _infer_submesh_barrier_group(cluster_x_submesh, (2, 2, 1))
        cluster_y_group = _infer_submesh_barrier_group(cluster_y_submesh, (2, 2, 1))

        assert cluster_x_group.shape == (2, )
        assert cluster_x_group.axes == (1, )
        assert cluster_x_group.mask == (2, 3)
        assert cluster_y_group.shape == (2, )
        assert cluster_y_group.axes == (0, )
        assert cluster_y_group.mask == (1, 3)

    def test_infer_submesh_rejects_different_masks_across_outer_coords(self):
        mesh = tle.device_mesh(
            None,
            _shape=(2, ),
            _dim_names=("cluster_x", ),
            _physical_ids=(0, 3),
            _launch_shape=(2, 2),
            _launch_dim_names=("rack", "cluster_x"),
        )

        with pytest.raises(ValueError, match="one mask for different cluster member selections"):
            _infer_submesh_barrier_group(mesh, (2, 1, 1))
