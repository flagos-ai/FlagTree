# tle.dsa.ascend.sync_block_all 接口文档

# 1. 硬件背景

当不同核之间操作同一块全局内存且可能存在读后写、写后读以及写后写等数据依赖问题时，通过调用该函数来插入同步语句来避免上述数据依赖时可能出现的数据读写错误问题。

# 2. 接口说明

<table>
  <tr>
    <td>Plain Text<br>def sync_block_all(mode, event_id, _builder=None):</td>
  </tr>
</table>

## 2.1 入参

<table>
  <tr>
    <td>参数名</td>
    <td>类型</td>
    <td>必需</td>
    <td>说明</td>
  </tr>
  <tr>
    <td>mode</td>
    <td>str</td>
    <td>是</td>
    <td>同步的模式 ，可选字符串:all_cube/all_vector/all/all_sub_vector。&lt;br&gt;all_cube：同步所有cube核&lt;br&gt;all_vector: 同步所有vector核&lt;br&gt;all: 同步所有cube核和vector核&lt;br&gt;all_sub_vector：Vector子块间同步</td>
  </tr>
  <tr>
    <td>event_id</td>
    <td>int</td>
    <td>是</td>
    <td>标记id。 范围是[0,15]</td>
  </tr>
</table>

## 2.2 返回值

无

# 3. 约束

- mode可选字符串:all_cube/all_vector/all/all_sub_vector

- event_id范围是[0,15]

# 4. 示例

<table>
  <tr>
    <td>Plain Text<br>import os<br>import pytest<br>import triton<br>import triton.language as tl<br>import triton.language.extra.cann.extension as al<br>from triton.compiler.compiler import ASTSource<br>from triton.compiler.code_generator import ast_to_ttir<br>from triton._C.libtriton import ir<br>from triton._C.libtriton.ascend import ir as ascend_ir<br><br>os.environ[&quot;TORCH_DEVICE_BACKEND_AUTOLOAD&quot;] = &quot;0&quot;<br><br><br>class Options:<br>    num_warps = 4<br>    num_stages = 3<br>    num_ctas = 1<br>    cluster_dims = (1, 1, 1)<br>    enable_fp_fusion = True<br>    debug = False<br><br><br>def compile_kernel(kernel, signature, constants):<br>    &quot;&quot;&quot;Helper to compile a kernel to MLIR.&quot;&quot;&quot;<br>    src = ASTSource(kernel, signature, constants)<br>    context = ir.context()<br>    ir.load_dialects(context)<br>    ascend_ir.load_dialects(context)<br>    module = ast_to_ttir(kernel, src, context, Options(), {}, {})<br>    return str(module)<br><br>@triton.jit<br>def test_sync_block_all():<br>    al.sync_block_all(&quot;all_cube&quot;, 8)<br>    al.sync_block_all(&quot;all_vector&quot;, 9)<br>    al.sync_block_all(&quot;all&quot;, 10)<br>    al.sync_block_all(&quot;all_sub_vector&quot;, 11)<br><br>if __name__ == &quot;__main__&quot;:<br>    print(&quot;=&quot; * 60)<br>    print(&quot;Test 1: test_sync_block_all&quot;)<br>    print(&quot;=&quot; * 60)<br>    mlir = compile_kernel(test_sync_block_all, {}, {})<br>    print(f&quot;✅ Generated MLIR ({len(mlir)} chars):\n&quot;)<br>    print(mlir)</td>
  </tr>
</table>

输出：

<table>
  <tr>
    <td>Plain Text<br>module {<br>  tt.func public @test_sync_block_all() attributes {noinline = false} {<br>    hivm.hir.sync_block[&lt;ALL_CUBE&gt;, 8 : index] tcube_pipe = &lt;PIPE_ALL&gt; loc(#loc1)<br>    hivm.hir.sync_block[&lt;ALL_VECTOR&gt;, 9 : index] tvector_pipe = &lt;PIPE_ALL&gt; loc(#loc2)<br>    hivm.hir.sync_block[&lt;ALL&gt;, 10 : index] tcube_pipe = &lt;PIPE_ALL&gt; tvector_pipe = &lt;PIPE_ALL&gt; loc(#loc3)<br>    hivm.hir.sync_block[&lt;ALL_SUB_VECTOR&gt;, 11 : index] tvector_pipe = &lt;PIPE_ALL&gt; loc(#loc4)<br>    tt.return loc(#loc5)<br>  } loc(#loc)<br>} loc(#loc)<br>#loc = loc(&quot;/home/linxin/triton-test/sync_block_all.py&quot;:37:0)<br>#loc1 = loc(&quot;/home/linxin/triton-test/sync_block_all.py&quot;:38:34)<br>#loc2 = loc(&quot;/home/linxin/triton-test/sync_block_all.py&quot;:39:36)<br>#loc3 = loc(&quot;/home/linxin/triton-test/sync_block_all.py&quot;:40:29)<br>#loc4 = loc(&quot;/home/linxin/triton-test/sync_block_all.py&quot;:41:40)<br>#loc5 = loc(&quot;/home/linxin/triton-test/sync_block_all.py&quot;:41:4)</td>
  </tr>
</table>
