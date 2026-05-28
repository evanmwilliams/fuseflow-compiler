#ifndef LIB_TRANSFORMS_INTERACTIVE_PASSES_H_
#define LIB_TRANSFORMS_INTERACTIVE_PASSES_H_

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace sam {

std::unique_ptr<Pass> createInteractivePass();

#define GEN_PASS_DECL
#include "lib/Transforms/Interactive/Passes.h.inc"

void registerInteractivePipeline();

} // namespace sam
} // namespace mlir

#endif // LIB_TRANSFORMS_INTERACTIVE_PASSES_H_
