# SPDX-License-Identifier: MIT
"""Shared text: module A acceptance items from docs/debugger_module_contract.md + third_party/Debugger/include/Debugger/README.md.

Referenced by test_module_a_*.py (do not duplicate the long form in each file — import MODULE_A_CHECKLIST).
"""

MODULE_A_CHECKLIST = """
模块 A 文档与公共契约验收摘要（对照用）

【边界 §2.1/§2.2】metadata JSON key 稳定；CUDA launcher 使用 hidden_arg 注入路径，
隐藏实参通过 prepare_kernel_launch(...) 产出并插入到 kernel 实参之前。

【§3.1】tl.debug_collect_start/end；marker 降到 TTIR（方言 flagtree_debug.*）；编排 flagtree_debug passes；
CompiledKernel.metadata 含 debug_enabled、debug_protocol_version、debug_record_level、debug_addr_level、
debug_export_mode、debug_kernel_id、debug_tracked_table 及实现侧 debug_launch_hidden_arg；
debug_enabled 为 False 时不应走 debug launcher 流程；单独开发期允许 mock control handle。

【§5】§5.1.1 mock 句柄经 prepare_launch_debug_ctrl/prepare_kernel_launch 进入 launch；
§5.2 A-1/A-2/A-3；§5.3 CTT-1（scope_id/负例）、CTT-3（hidden_arg 注入 ABI）。

【§6.1 A 侧可测】编译成功、metadata/TTIR/launch 实参可观测代理。

【third_party/Debugger/include/Debugger/README.md】debug_tracked_table 行与 TrackedOpTable 契约一致，A 仅透传不另造 schema。

【§7】可选：带 collect 的 kernel 编译+运行 smoke。
"""
