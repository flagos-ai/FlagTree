module attributes {flagtree.debug.record_level = 1 : i32} {
  func.func @statement_operand_capture(%a: f32, %b: f32) {
    "flagtree_debug.collect_begin"() {level = 1 : i32} : () -> ()
    %y = arith.addf %a, %b {
      flagtree.debug.triton_statement = "y = a + b"
    } : f32
    "flagtree_debug.collect_end"() : () -> ()
    func.return
  }
}
