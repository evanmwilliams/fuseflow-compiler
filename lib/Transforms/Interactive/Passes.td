//===-- Passes.td - Interactive pass definition file -------*- tablegen -*-===//
//
//===----------------------------------------------------------------------===//

#ifndef LIB_TRANSFORMS_INTERACTIVE_TD_
#define LIB_TRANSFORMS_INTERACTIVE_TD_

include "mlir/Pass/PassBase.td"


def Interactive : Pass<"interactive", "ModuleOp"> {
  let summary = "Walk each op and register it for Honeybee program emission";
  let description = [{
    Skeleton pass that iterates over every operation in the module, assigns a
    stable index, and prints the op name. Foundation for the later pass that
    will emit per-op Honeybee program (.hb.toml) files.
  }];
}


#endif // LIB_TRANSFORMS_INTERACTIVE_TD_
