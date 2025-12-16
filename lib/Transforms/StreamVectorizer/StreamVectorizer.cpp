#include "lib/Transforms/StreamVectorizer/Passes.h"
#include <memory>

#include "lib/Dialect/SAM/SamOps.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Pass/PassManager.h" // from @llvm-project
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LogicalResult.h" // from @llvm-project

#include "llvm/ADT/TypeSwitch.h" // from @llvm-project

namespace mlir {
namespace sam {
#define GEN_PASS_DEF_STREAMVECTORIZER
#include "lib/Transforms/StreamVectorizer/Passes.h.inc"

struct StreamVectorizer : public impl::StreamVectorizerBase<StreamVectorizer> {
  using StreamVectorizerBase::StreamVectorizerBase;

  StreamVectorizer(unsigned shape, bool blockSparse)
      : streamShape(shape), enableBlockSparse(blockSparse) {}

  void runOnOperation() override;

private:
  unsigned streamShape;
  bool enableBlockSparse;
};

LogicalResult processArrayVal(sam::SamArrayVal op, unsigned streamShape,
                              bool enableBlockSparse) {
  unsigned size = enableBlockSparse ? 2 : 1;
  SmallVector<int64_t> shapeLst(size, streamShape);
  op.setStreamShape(ArrayRef(shapeLst));
  return success();
}

void StreamVectorizer::runOnOperation() {
  auto moduleOp = getOperation();
  moduleOp.walk([&](Operation *op) {
    llvm::TypeSwitch<Operation *, LogicalResult>(op).Case<sam::SamArrayVal>(
        [&](auto op) {
          return processArrayVal(op, streamShape, enableBlockSparse);
        });
  });
}

std::unique_ptr<mlir::Pass> createStreamVectorizer(unsigned streamShape,
                                                   bool enableBlockSparse) {
  return std::make_unique<mlir::sam::StreamVectorizer>(streamShape,
                                                       enableBlockSparse);
}

void registerStreamVectorizerPipeline() {
  PassPipelineRegistration<StreamVectorizerPipelineOptions>(
      "stream-vectorizer", "The stream vectorizer pipeline",
      [&](OpPassManager &pm, const StreamVectorizerPipelineOptions &options) {
        pm.addPass(createStreamVectorizer(options.streamShape,
                                          options.enableBlockSparse));
      });
}

} // namespace sam
} // namespace mlir
