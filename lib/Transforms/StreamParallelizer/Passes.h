#ifndef LIB_TRANSFORMS_STREAMPARALLELIZER_PASSES_H_
#define LIB_TRANSFORMS_STREAMPARALLELIZER_PASSES_H_

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace sam {

std::unique_ptr<Pass> createStreamParallelizer(unsigned stream_level,
                                               unsigned par_factor);

std::unique_ptr<Pass> createStreamParallelizerAnalysis(unsigned stream_level);

std::unique_ptr<Pass> createStreamParallelizerUnroll(unsigned par_factor);

#define GEN_PASS_DECL
#include "lib/Transforms/StreamParallelizer/Passes.h.inc"

struct StreamParallelizerPipelineOptions
    : public PassPipelineOptions<StreamParallelizerPipelineOptions> {
  PassOptions::Option<int64_t> stream_level{
      *this, "stream-level",
      llvm::cl::desc("Set the stream level/s to parallelize."),
      llvm::cl::init(0)};
  PassOptions::Option<int64_t> par_factor{
      *this, "par-factor", llvm::cl::desc("Set the parallel factor."),
      llvm::cl::init(1)};
};

void registerStreamParallelizerPipeline();

} // namespace sam
} // namespace mlir

#endif // LIB_TRANSFORMS_STREAMPARALLELIZER_PASSES_H_