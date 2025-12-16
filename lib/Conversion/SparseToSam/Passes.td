//===-- Passes.td - Conversion pass definition file --------*- tablegen -*-===//
//
//===----------------------------------------------------------------------===//

#ifndef LIB_CONVERSION_LINALGTOSAM_TD_
#define LIB_CONVERSION_LINALGTOSAM_TD_

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// LinalgToSam
//===----------------------------------------------------------------------===//

def LinalgToSam : Pass<"convert-linalg-to-sam", "mlir::func::FuncOp"> {
  let summary = "Convert operations from linalg dialect to the sam dialect";
  let constructor = "mlir::createLinalgToSamPass(bool useUserInput, bool calculateHeuristic)";
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::func::FuncDialect",
    "mlir::linalg::LinalgDialect",
    "mlir::sam::SamDialect",
    "mlir::tensor::TensorDialect",
  ];
}

def ConvertFillOps : Pass<"convert-fill-ops", "mlir::func::FuncOp"> {
  let summary = "Convert tensor+fill ops to splats";
  let constructor = "mlir::createConvertFillOpsPass()";
  let dependentDialects = [
    // "mlir::arith::ArithDialect",
    "mlir::func::FuncDialect",
    "mlir::linalg::LinalgDialect",
    "mlir::sam::SamDialect",
    "mlir::tensor::TensorDialect",
  ];
}

def RemoveJoinerDuplicateStreams : Pass<"remove-joiner-duplicates", "mlir::func::FuncOp"> {
  let summary = "Remove duplicate streams from joiner";
  let constructor = "mlir::removeJoinerDuplicateStreams()";
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::func::FuncDialect",
    "mlir::linalg::LinalgDialect",
    "mlir::sam::SamDialect",
    "mlir::tensor::TensorDialect",
  ];
}

def InsertIterateLocate : Pass<"insert-iterate-locate", "mlir::func::FuncOp"> {
  let summary = "Remove intersections with a sparse and dense level and replace with a locate block";
  let constructor = "mlir::insertIterateLocate()";
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::func::FuncDialect",
    "mlir::linalg::LinalgDialect",
    "mlir::sam::SamDialect",
    "mlir::tensor::TensorDialect",
  ];
}

def FusionDispatchGroups:
    Pass<"fusion-grouping"> {
  let summary = "Groups ops to be fused";
  let constructor = "mlir::createFusionDispatchGroupsPass()";
  let dependentDialects = [
    "mlir::sam::SamDialect",
  ];
}

#endif // LIB_CONVERSION_LINALGTOSAM_TD_