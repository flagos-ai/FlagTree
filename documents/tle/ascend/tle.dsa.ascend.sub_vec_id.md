# tle.dsa.ascend.sub_vec_id

## 1. 硬件背景

昇腾硬件AIC与AIV核数配比不同（1:N），Triton编程抽象屏蔽了Cube核与Vector核的硬件细节，因此，Triton算子开发者无法控制如何切分数据在N个Vector核间并行处理，由编译器通过AutoSubTiling Pass自动实现。

sub_vec_id编程接口返回N个Vector核的sub id，允许算子开发者根据vector核sub id决定每个核处理哪些数据。

## 2. 接口说明

<table>
  <tr>
    <td>Python<br>def sub_vec_id() -&gt; i16</td>
  </tr>
</table>

- 返回值：返回范围为 [0, N) 的 Sub Vector ID，算子开发者可根据该ID决定N个并行Vector核中每个核处理的数据分片

- 入参：无

## 3. 约束说明

仅在AIC和AIV核混合使用场景中有效，不可在纯Cube类算子或者纯Vector类算子中使用，否则会触发编译报错。

## 4. 用例示例

<table>
  <tr>
    <td>Python<br><br>import os<br><br>import triton<br><br>import triton.language as tl<br><br>import triton.language.extra.cann.extension as al<br><br>from triton.compiler.compiler import ASTSource<br><br>from triton.compiler.code_generator import ast_to_ttir<br><br>from triton._C.libtriton import ir, buffer_ir<br><br>from triton._C.libtriton.ascend import ir as ascend_ir<br><br>os.environ[&quot;TORCH_DEVICE_BACKEND_AUTOLOAD&quot;] = &quot;0&quot;<br><br>class Options:<br><br>    num_warps = 4<br><br>    num_stages = 3<br><br>    num_ctas = 1<br><br>    cluster_dims = (1, 1, 1)<br><br>    enable_fp_fusion = True<br><br>    debug = False<br><br>    arch = &quot;Ascend910_95&quot;<br><br>def compile_kernel(kernel, signature, constants):<br><br>    &quot;&quot;&quot;Helper to compile a kernel to MLIR.&quot;&quot;&quot;<br><br>    src = ASTSource(kernel, signature, constants)<br><br>    context = ir.context()<br><br>    ir.load_dialects(context)<br><br>    buffer_ir.load_dialects(context)<br><br>    ascend_ir.load_dialects(context)<br><br>    module = ast_to_ttir(kernel, src, context, Options(), {}, {})<br><br>    return str(module)<br><br>@triton.jit<br><br>def verify_sub_vec_id_kernel(<br><br>    out_ptr,<br><br>    N: tl.constexpr,<br><br>):<br><br>    with al.scope(core_mode=&quot;vector&quot;):<br><br>        sub_id = al.sub_vec_id()<br><br>        <br><br>        offs = sub_id * N + tl.arange(0, N)<br><br>        out_ptrs = out_ptr + offs<br><br>        <br><br>        tl.store(out_ptrs, sub_id.to(tl.int32))<br><br>def test_sub_vec_id_1to2():<br><br>    print(&quot;=&quot; * 60)<br><br>    print(&quot;Test: Verify sub_vec_id (Simplified)&quot;)<br><br>    print(&quot;=&quot; * 60)<br><br>    <br><br>    mlir = compile_kernel(<br><br>        kernel=verify_sub_vec_id_kernel,<br><br>        signature={&quot;out_ptr&quot;: &quot;*i32&quot;},<br><br>        constants={&quot;N&quot;: 8},<br><br>    )<br><br>    <br><br>    print(f&quot;✅ Generated MLIR ({len(mlir)} chars):\n&quot;)<br><br>    print(mlir)<br><br># ============== Main ==============<br><br>if __name__ == &quot;__main__&quot;:<br><br>    test_sub_vec_id_1to2()</td>
  </tr>
</table>

输出：

<table>
  <tr>
    <td>Plain Text<br>============================================================<br><br>Test: Verify sub_vec_id (Simplified)<br><br>============================================================<br><br>✅ Generated MLIR (1893 chars):<br><br>#loc = loc(&quot;/home/linxin/triton-test/sub_vec_id.py&quot;:35:0)<br><br>module attributes {hivm.disable_auto_tile_and_bind_subblock} {<br><br>  tt.func public @verify_sub_vec_id_kernel(%arg0: !tt.ptr&lt;i32&gt; loc(&quot;/home/linxin/triton-test/sub_vec_id.py&quot;:35:0)) attributes {noinline = false} {<br><br>    %0:3 = scope.scope : () -&gt; (i64, tensor&lt;8xi64&gt;, tensor&lt;8x!tt.ptr&lt;i32&gt;&gt;) {<br><br>      %1 = hivm.hir.get_sub_block_idx -&gt; i64 loc(#loc2)<br><br>      %c8_i32 = arith.constant 8 : i32 loc(#loc3)<br><br>      %c8_i64 = arith.constant 8 : i64 loc(#loc3)<br><br>      %2 = arith.muli %1, %c8_i64 : i64 loc(#loc3)<br><br>      %3 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor&lt;8xi32&gt; loc(#loc4)<br><br>      %4 = arith.extsi %3 : tensor&lt;8xi32&gt; to tensor&lt;8xi64&gt; loc(#loc5)<br><br>      %5 = tt.splat %2 : i64 -&gt; tensor&lt;8xi64&gt; loc(#loc5)<br><br>      %6 = arith.addi %5, %4 : tensor&lt;8xi64&gt; loc(#loc5)<br><br>      %7 = tt.splat %arg0 : !tt.ptr&lt;i32&gt; -&gt; tensor&lt;8x!tt.ptr&lt;i32&gt;&gt; loc(#loc6)<br><br>      %8 = tt.addptr %7, %6 : tensor&lt;8x!tt.ptr&lt;i32&gt;&gt;, tensor&lt;8xi64&gt; loc(#loc6)<br><br>      %9 = arith.trunci %1 : i64 to i32 loc(#loc7)<br><br>      %10 = tt.splat %9 : i32 -&gt; tensor&lt;8xi32&gt; loc(#loc8)<br><br>      tt.store %8, %10 : tensor&lt;8x!tt.ptr&lt;i32&gt;&gt; loc(#loc8)<br><br>      scope.return %1, %6, %8 : i64, tensor&lt;8xi64&gt;, tensor&lt;8x!tt.ptr&lt;i32&gt;&gt; loc(#loc8)<br><br>    } {hivm.tcore_type = #hivm.tcore_type&lt;VECTOR&gt;, noinline} loc(#loc1)<br><br>    tt.return loc(#loc9)<br><br>  } loc(#loc)<br><br>} loc(#loc)<br><br>#loc1 = loc(&quot;/home/linxin/triton-test/sub_vec_id.py&quot;:39:9)<br><br>#loc2 = loc(&quot;/home/linxin/triton-test/sub_vec_id.py&quot;:40:17)<br><br>#loc3 = loc(&quot;/home/linxin/triton-test/sub_vec_id.py&quot;:42:24)<br><br>#loc4 = loc(&quot;/home/linxin/triton-test/sub_vec_id.py&quot;:42:41)<br><br>#loc5 = loc(&quot;/home/linxin/triton-test/sub_vec_id.py&quot;:42:28)<br><br>#loc6 = loc(&quot;/home/linxin/triton-test/sub_vec_id.py&quot;:43:29)<br><br>#loc7 = loc(&quot;/home/linxin/triton-test/sub_vec_id.py&quot;:45:37)<br><br>#loc8 = loc(&quot;/home/linxin/triton-test/sub_vec_id.py&quot;:45:27)<br><br>#loc9 = loc(&quot;/home/linxin/triton-test/sub_vec_id.py&quot;:39:4)</td>
  </tr>
</table>
