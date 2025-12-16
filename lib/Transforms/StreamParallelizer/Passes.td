//===-- Passes.td - Conversion pass definition file --------*- tablegen -*-===//
//
//===----------------------------------------------------------------------===//

#ifndef LIB_TRANSFORMS_STREAMPARALLELIZER_TD_
#define LIB_TRANSFORMS_STREAMPARALLELIZER_TD_

include "mlir/Pass/PassBase.td"


def StreamParallelizer : Pass<"stream-parallelizer", "ModuleOp"> {
  let summary = "Parallelizes streams from stream level";
  let description = [{
    A pass that inserts fork/join pairs with parallel region.
  }];
}

def StreamParallelizerAnalysis : Pass<"ModuleOp"> {
  let summary = "Determines where to insert parallel region";
  let description = [{
    A pass that inserts no-op pairs to trigger parallel region.
  }];
}

def StreamParallelizerUnroll : Pass<"ModuleOp"> {
  let summary = "Determines where to insert parallel region";
  let description = [{
    A pass that inserts no-op pairs to trigger parallel region.
  }];
}


#endif // LIB_TRANSFORMS_STREAMPARALLELIZER_TD_