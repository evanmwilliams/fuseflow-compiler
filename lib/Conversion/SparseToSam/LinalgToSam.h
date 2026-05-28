//===- LinalgToSam.h - Utils to convert from the linalg dialect ----------===//
//
//===----------------------------------------------------------------------===//

#ifndef LIB_CONVERSION_LINALGTOSAM_H_
#define LIB_CONVERSION_LINALGTOSAM_H_

#include <memory>

#include "mlir/Pass/Pass.h"
//#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"
//  #include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"
#include <llvm/ADT/TypeSwitch.h>
// #include <mlir/Dialect/Linalg/IR/LinalgInterfaces.h.inc>
#include <mlir/Dialect/Math/IR/MathOps.h.inc>


#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir {
    struct LinalgToSamPipelineOptions
            : public PassPipelineOptions<LinalgToSamPipelineOptions> {
        PassOptions::Option<bool> useUserInput{
                *this, "user-input",
                llvm::cl::desc(
                        "Determine whether user-input should be required for dataflow orderings (from topological sort)."),
                llvm::cl::init(false)};
        PassOptions::Option<bool> getHeuristic{
                *this, "calculate-heuristic",
                llvm::cl::desc(
                        "Calculate the number of flops and bytes accessed and print out both."),
                llvm::cl::init(false)};
        PassOptions::Option<std::string> honeybee{
                *this, "honeybee",
                llvm::cl::desc("Path to the Honeybee CLI binary. When set with library and output-dir, "
                               "loop-order selection is driven by Honeybee."),
                llvm::cl::init("")};
        PassOptions::Option<std::string> library{
                *this, "library",
                llvm::cl::desc("Path to the dataflow_order.hblib.toml library file."),
                llvm::cl::init("")};
        PassOptions::Option<std::string> outputDir{
                *this, "output-dir",
                llvm::cl::desc("Directory for per-op Honeybee session subdirs and loop_order.json files."),
                llvm::cl::init("")};
    };

    /// Create a pass to convert Linalg operations to the SAMML dialect.
    std::unique_ptr<mlir::Pass> createLinalgToSamPass(bool useUserInput, bool calculateHeuristic,
                                                      llvm::StringRef honeybee = "",
                                                      llvm::StringRef library = "",
                                                      llvm::StringRef outputDir = "");

    std::unique_ptr<mlir::Pass> createFusionDispatchGroupsPass();

    std::unique_ptr<mlir::Pass> createConvertFillOpsPass();

    std::unique_ptr<mlir::Pass> createRemoveJoinerDuplicateStreamsPass();

    std::unique_ptr<mlir::Pass> createInsertIterateLocatePass();
} // namespace mlir



#endif // LIB_CONVERSION_LINALGTOSAM_H_
