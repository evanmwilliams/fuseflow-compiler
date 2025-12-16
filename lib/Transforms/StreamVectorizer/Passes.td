//===-- Passes.td - Conversion pass definition file --------*- tablegen -*-===//
//
//===----------------------------------------------------------------------===//

#ifndef LIB_TRANSFORMS_STREAMVECTORIZER_TD_
#define LIB_TRANSFORMS_STREAMVECTORIZER_TD_

include "mlir/Pass/PassBase.td"


def StreamVectorizer : Pass<"stream-vectorizer", "ModuleOp"> {
// def StreamVectorizer : Pass<"stream-vectorizer"> {
  let summary = "Automatically generate sparse tensor code from sparse tensor types";
  let description = [{
    A pass that converts stream values after sam conversion into vector/block streams.
  }];
}


#endif // LIB_TRANSFORMS_STREAMVECTORIZER_TD_