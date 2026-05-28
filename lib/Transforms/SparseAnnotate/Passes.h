#ifndef LIB_TRANSFORMS_SPARSEANNOTATE_PASSES_H_
#define LIB_TRANSFORMS_SPARSEANNOTATE_PASSES_H_

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassOptions.h"
#include <memory>
#include <string>

namespace mlir {
namespace sam {

std::unique_ptr<Pass> createSparseAnnotatePass(std::string library,
                                               std::string honeybee,
                                               std::string outputDir);

#define GEN_PASS_DECL
#include "lib/Transforms/SparseAnnotate/Passes.h.inc"

struct SparseAnnotatePipelineOptions
    : public PassPipelineOptions<SparseAnnotatePipelineOptions> {
  PassOptions::Option<std::string> library{
      *this, "library",
      llvm::cl::desc("Path to the sparse_anno.hblib.toml library file."),
      llvm::cl::init("")};
  PassOptions::Option<std::string> honeybee{
      *this, "honeybee",
      llvm::cl::desc("Path to the Honeybee CLI binary."),
      llvm::cl::init("honeybee")};
  PassOptions::Option<std::string> outputDir{
      *this, "output-dir",
      llvm::cl::desc("Directory for prop files, per-tensor result subdirs, "
                     "and the merged annotations.json."),
      llvm::cl::init("sparse_annotations")};
};

void registerSparseAnnotatePipeline();

} // namespace sam
} // namespace mlir

#endif // LIB_TRANSFORMS_SPARSEANNOTATE_PASSES_H_
