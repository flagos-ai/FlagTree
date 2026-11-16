"""A unit logical dimension must contribute coordinate zero to group IDs."""
import triton
import re
import pytest
import math
import triton.language as tl
import triton.experimental.tle.language as tle
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

DOMAIN = tle.device_mesh({'block': [('block_y', 33), ('block_x', 1)]})
GROUP = tl.constexpr(DOMAIN.axis_group(('block_y',), group_shape=(33,)))


@triton.jit
def unit_axis(data, output, SELECTED: tl.constexpr):
    owner = tl.program_id(0)
    lane = tl.arange(0, 32)
    tl.store(data + owner * 32 + lane, owner * 32 + lane)
    if SELECTED:
        tle.distributed_barrier(GROUP, protocol='flags32', launch_grid=(33, 1, 1))
    else:
        tle.distributed_barrier(GROUP)
    value = tl.load(data + ((owner + 1) % 33) * 32 + lane)
    tl.store(output + owner * 32 + lane, value)


def compile_unit_axis(selected):
    return triton.compile(ASTSource(unit_axis, {'data': '*i32', 'output': '*i32'},
                          constexprs={'SELECTED': selected}),
                          target=GPUTarget('cuda', 90, 32),
                          options={'num_warps': 8, 'ptx_version': 87})


def first_counter_address(llir, owner, total=33):
    """Evaluate the actual native address at the first release operation."""
    definitions = dict(re.findall(r'^\s*(%[\w.]+) = (.*)$', llir, re.M))
    signature = next(line for line in llir.splitlines() if line.startswith('define '))
    arguments = re.findall(r'%[\w.]+', signature.split('{')[0])
    scratch = arguments[-2]
    values = {scratch: 0}

    def evaluate(name):
        if not name.startswith('%'):
            return int(name)
        if name in values:
            return values[name]
        expression = definitions[name]
        register = re.search(r'@llvm.nvvm.read.ptx.sreg.([\w.]+)\(', expression)
        if register:
            result = {'ctaid.x': owner, 'ctaid.y': 0, 'ctaid.z': 0,
                      'nctaid.x': total, 'nctaid.y': 1, 'nctaid.z': 1,
                      'tid.x': 0}[register.group(1)]
        elif match := re.match(r'(add|sub|mul|shl|lshr|and|or|xor|urem|udiv) (?:nuw |nsw |exact )*i\d+ ([%\w.-]+), ([%\w.-]+)', expression):
            op, left, right = match.groups(); a, b = evaluate(left), evaluate(right)
            result = {'add': lambda: a+b, 'sub': lambda: a-b, 'mul': lambda: a*b, 'shl': lambda: a<<b,
                      'lshr': lambda: a>>b, 'and': lambda: a&b,
                      'or': lambda: a|b, 'xor': lambda: a^b,
                      'urem': lambda: a%b, 'udiv': lambda: a//b}[op]()
        elif match := re.match(r'(?:sext|zext) (?:nneg )?i\d+ ([%\w.-]+) to i\d+', expression):
            result = evaluate(match.group(1))
        elif match := re.match(r'getelementptr (?:inbounds )?i8, ptr addrspace\(1\) (%[\w.]+), i\d+ ([%\w.-]+)', expression):
            result = evaluate(match.group(1)) + evaluate(match.group(2))
        else:
            raise AssertionError(f'unhandled native counter address expression: {expression}')
        values[name] = result
        return result

    release = next(line for line in llir.splitlines()
                   if 'asm sideeffect' in line and ('atom.add.release' in line or 'st.release.gpu' in line))
    address = re.search(r'\(ptr addrspace\(1\) (%[\w.]+),', release).group(1)
    return evaluate(address)


@pytest.mark.parametrize('selected', [False, True])
def test_unit_dimension_has_no_group_displacement(selected):
    compiled = compile_unit_axis(selected)
    for owner in range(33):
        assert first_counter_address(compiled.asm['llir'], owner) == (4 * owner if selected else 0)


@triton.jit
def axis_neighbor(data, output, MESH: tl.constexpr, COUNT: tl.constexpr, SELECTED: tl.constexpr):
    owner = tl.program_id(0)
    lane = tl.arange(0, 32)
    tl.store(data + owner * 32 + lane, owner * 32 + lane)
    if SELECTED:
        tle.distributed_barrier(MESH, protocol='flags32', launch_grid=(COUNT, 1, 1))
    else:
        tle.distributed_barrier(MESH)
    tl.store(output + owner * 32 + lane, tl.load(data + owner * 32 + lane))


@pytest.mark.parametrize('selected', [False, True])
@pytest.mark.parametrize('shape,axis,extent', [((1, 33), 1, 33), ((2, 1, 3), 0, 2),
                         ((2, 1, 3), 2, 3), ((2, 3, 1), 1, 3), ((8, 16), 0, 8)])
def test_axis_neighbors_actual_native_addresses(shape, axis, extent, selected):
    names = [f'block_{i}' for i in range(len(shape))]
    mesh = tle.device_mesh({'block': list(zip(names, shape))}).axis_group((names[axis],), group_shape=(extent,))
    total = math.prod(shape)
    k = triton.compile(ASTSource(axis_neighbor, {'data': '*i32', 'output': '*i32'},
                       constexprs={'MESH': mesh, 'COUNT': total, 'SELECTED': selected}),
                       target=GPUTarget('cuda', 90, 32), options={'num_warps': 8, 'ptx_version': 87})
    import itertools
    group_bytes = ((extent * 4 + 127) // 128) * 128 if selected else 4
    for owner, coords in enumerate(itertools.product(*(range(n) for n in shape))):
        group = 0
        for dim, (coord, size) in enumerate(zip(coords, shape)):
            group_extent = extent if dim == axis else 1
            group = group * (size // group_extent) + coord // group_extent
        expected = group * group_bytes + (4 * (coords[axis] % extent) if selected else 0)
        assert first_counter_address(k.asm['llir'], owner, total) == expected
