//===-- Passes.td - SparseAnnotate pass definition file ----*- tablegen -*-===//
//
//===----------------------------------------------------------------------===//

#ifndef LIB_TRANSFORMS_SPARSEANNOTATE_TD_
#define LIB_TRANSFORMS_SPARSEANNOTATE_TD_

include "mlir/Pass/PassBase.td"


def SparseAnnotate : Pass<"annotate-sparsity", "ModuleOp"> {
  let summary = "Drive Honeybee to annotate every un-annotated tensor's sparsity";
  let description = [{
    Walks the module, collects every distinct SSA tensor value that does not
    already carry a sparse_tensor.encoding attribute, emits a Honeybee prop
    file (P_Tensor + TensorSparseAnnotation goal) for each, invokes the
    Honeybee CLI on each in sequence, and merges the per-tensor JSON results
    into a single annotations.json under the output directory.
  }];
}


#endif // LIB_TRANSFORMS_SPARSEANNOTATE_TD_
