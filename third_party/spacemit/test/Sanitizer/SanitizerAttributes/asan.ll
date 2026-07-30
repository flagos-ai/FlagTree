; RUN: %toolsdir/opt -load-pass-plugin %shlibdir/libSanitizerAttributes.so -passes=sanitizer-attributes -sanitizer-type=asan -S %s | FileCheck %s 

; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

declare ptr @malloc(i64)

define void @kernel(i64 %0, ptr %1, i32 %2, i32 %3, i32 %4, i32 %5, i32 %6, i32 %7, i32 %8) !dbg !3 {
  %10 = insertvalue { i64, ptr } poison, i64 %0, 0, !dbg !6
  %11 = insertvalue { i64, ptr } %10, ptr %1, 1, !dbg !6
  %12 = call ptr @malloc(i64 72)
  %13 = ptrtoint ptr %12 to i64
  %14 = add i64 %13, 63
  %15 = urem i64 %14, 64
  %16 = sub i64 %14, %15
  %17 = inttoptr i64 %16 to ptr
  %18 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } poison, ptr %12, 0
  %19 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %18, ptr %17, 1
  %20 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %19, i64 0, 2
  %21 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %20, i64 2, 3, 0
  %22 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %21, i64 1, 4, 0
  br label %23

23:                                               ; preds = %26, %9
  %24 = phi i64 [ %28, %26 ], [ 0, %9 ]
  %25 = icmp slt i64 %24, 2
  br i1 %25, label %26, label %29

26:                                               ; preds = %23
  %27 = getelementptr float, ptr %17, i64 %24
  store float 1.000000e+00, ptr %27, align 4
  %28 = add i64 %24, 1
  br label %23

29:                                               ; preds = %23
  %30 = mul i32 %6, 2, !dbg !7
  %31 = sext i32 %30 to i64, !dbg !8
  %32 = extractvalue { i64, ptr } %11, 1, !dbg !8
  %33 = load ptr, ptr %32, align 8, !dbg !8
  %34 = getelementptr ptr, ptr %32, i32 1, !dbg !8
  %35 = load ptr, ptr %34, align 8, !dbg !8
  %36 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } poison, ptr %33, 0, !dbg !8
  %37 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %36, ptr %35, 1, !dbg !8
  %38 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %37, i64 %31, 2, !dbg !8
  %39 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %38, i64 2, 3, 0, !dbg !8
  %40 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %39, i64 1, 4, 0, !dbg !8
  %41 = getelementptr float, ptr %17, i64 0, !dbg !9
  %42 = getelementptr float, ptr %35, i64 %31, !dbg !9
  call void @llvm.memcpy.p0.p0.i64(ptr %42, ptr %41, i64 8, i1 false), !dbg !9
  ret void, !dbg !6
}

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #0

attributes #0 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_Python, file: !1, producer: "MLIR", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly)
!1 = !DIFile(filename: "program.py", directory: "/path/to")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = distinct !DISubprogram(name: "kernel", linkageName: "kernel", scope: !1, file: !1, line: 9, type: !4, spFlags: DISPFlagDefinition, unit: !0)
!4 = !DISubroutineType(cc: DW_CC_normal, types: !5)
!5 = !{}
!6 = !DILocation(line: 9, scope: !3)
!7 = !DILocation(line: 12, column: 24, scope: !3)
!8 = !DILocation(line: 25, column: 26, scope: !3)
!9 = !DILocation(line: 25, column: 35, scope: !3)

; CHECK: ; Function Attrs: sanitize_address
; CHECK: define void @kernel