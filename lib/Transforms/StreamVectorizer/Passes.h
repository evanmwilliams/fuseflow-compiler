#ifndef LIB_TRANSFORMS_STREAMVECTORIZER_PASSES_H_
#define LIB_TRANSFORMS_STREAMVECTORIZER_PASSES_H_

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace sam {

std::unique_ptr<Pass> createStreamVectorizer(unsigned stream_shape,
                                             bool enableBlockSparse);

#define GEN_PASS_DECL
#include "lib/Transforms/StreamVectorizer/Passes.h.inc"

struct StreamVectorizerPipelineOptions
    : public PassPipelineOptions<StreamVectorizerPipelineOptions> {
  PassOptions::Option<int32_t> streamShape{
      *this, "stream-shape",
      llvm::cl::desc(
          "Set the vector/block size (use 0 to use default scalar stream)."),
      llvm::cl::init(0)};
  PassOptions::Option<bool> enableBlockSparse{
      *this, "enable-block-sparse",
      llvm::cl::desc(
          "Set to block sparse mode (use false to use default vector stream)."),
      llvm::cl::init(false)};
};

void registerStreamVectorizerPipeline();

} // namespace sam
} // namespace mlir

#endif // LIB_TRANSFORMS_STREAMVECTORIZER_PASSES_H_