#include "lib/Transforms/Interactive/Passes.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include "lib/Transforms/Interactive/toml.hpp"

#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace mlir {
namespace sam {
#define GEN_PASS_DEF_INTERACTIVE
#include "lib/Transforms/Interactive/Passes.h.inc"

namespace {

std::string moduleStemFromLoc(Operation *op) {
  if (auto flc = llvm::dyn_cast<FileLineColLoc>(op->getLoc())) {
    return llvm::sys::path::stem(flc.getFilename().str()).str();
  }
  return "module";
}

std::vector<std::string> identityLoopNames(unsigned n) {
  std::vector<std::string> names;
  names.reserve(n);
  for (unsigned i = 0; i < n; ++i)
    names.push_back("i" + std::to_string(i));
  return names;
}

std::string formatLoopOrder(const std::vector<std::string> &order) {
  std::ostringstream oss;
  for (size_t i = 0; i < order.size(); ++i) {
    if (i)
      oss << ' ';
    oss << order[i];
  }
  return oss.str();
}

std::vector<std::vector<std::string>>
allPermutations(std::vector<std::string> names) {
  std::sort(names.begin(), names.end());
  std::vector<std::vector<std::string>> perms;
  do {
    perms.push_back(names);
  } while (std::next_permutation(names.begin(), names.end()));
  return perms;
}

struct TensorDesc {
  std::string name;
  int numIndices;
  std::string indices;
};

std::vector<TensorDesc> collectTensorDescs(linalg::LinalgOp linalgOp) {
  llvm::SmallVector<AffineMap> maps = linalgOp.getIndexingMapsArray();
  std::vector<TensorDesc> descs;

  auto operands = linalgOp->getOperands();
  unsigned k = 0;
  for (unsigned i = 0, e = operands.size(); i < e; ++i) {
    Value operand = operands[i];
    auto shaped = llvm::dyn_cast<ShapedType>(operand.getType());
    if (!shaped || !shaped.hasRank())
      continue;

    TensorDesc d;
    d.name = "t" + std::to_string(k);
    d.numIndices = static_cast<int>(shaped.getRank());

    std::vector<std::string> idxNames;
    if (i < maps.size()) {
      for (AffineExpr expr : maps[i].getResults()) {
        if (auto dim = llvm::dyn_cast<AffineDimExpr>(expr)) {
          idxNames.push_back("i" + std::to_string(dim.getPosition()));
        } else {
          idxNames.push_back("?");
        }
      }
    }
    if (idxNames.empty()) {
      // Fallback: no indexing map — name dims i0..iN-1.
      for (int64_t d2 = 0; d2 < shaped.getRank(); ++d2)
        idxNames.push_back("i" + std::to_string(d2));
    }
    d.indices = formatLoopOrder(idxNames);

    descs.push_back(std::move(d));
    ++k;
  }
  return descs;
}

// Build a Honeybee goal TOML document for a single op.
toml::table buildOpDoc(const std::string &programPath, unsigned numLoops,
                       const std::vector<TensorDesc> &tensors,
                       const std::vector<std::vector<std::string>> &orders) {
  toml::table doc;
  toml::array &prop =
      *doc.insert("Prop", toml::array{}).first->second.as_array();

  auto add_entry = [&](std::string name, toml::table args) {
    toml::table entry;
    entry.insert("name", std::move(name));
    entry.insert("args", std::move(args));
    prop.push_back(std::move(entry));
  };

  {
    toml::table args;
    args.insert("path", programPath);
    args.insert("num_loops", static_cast<int>(numLoops));
    args.insert("num_tensors", static_cast<int>(tensors.size()));
    add_entry("P_MlirProgram", std::move(args));
  }

  for (size_t i = 0; i < tensors.size(); ++i) {
    const TensorDesc &t = tensors[i];
    toml::table args;
    args.insert("name", t.name);
    args.insert("path", programPath);
    args.insert("tensor_order", static_cast<int>(i));
    args.insert("num_indices", t.numIndices);
    args.insert("indices", t.indices);
    add_entry("P_TensorInfo", std::move(args));
  }

  for (const auto &order : orders) {
    toml::table args;
    args.insert("path", programPath);
    args.insert("order", formatLoopOrder(order));
    add_entry("P_LoopOrderOption", std::move(args));
  }

  toml::table goal;
  goal.insert("name", "FuseFlowSchedule");
  goal.insert("args", toml::table{});
  doc.insert("Goal", std::move(goal));

  return doc;
}

struct Interactive : public impl::InteractiveBase<Interactive> {
  using InteractiveBase::InteractiveBase;

  void runOnOperation() override;
};

void Interactive::runOnOperation() {
  ModuleOp moduleOp = getOperation();
  std::string stem = moduleStemFromLoc(moduleOp.getOperation());

  llvm::outs() << "=== Interactive: emitting per-op goal files ===\n";

  unsigned opIdx = 0;
  moduleOp.walk([&](linalg::LinalgOp linalgOp) {
    unsigned numLoops = linalgOp.getNumLoops();
    std::string programPath = stem + "_op" + std::to_string(opIdx);
    std::string fileName = programPath + ".hb.toml";

    auto names = identityLoopNames(numLoops);
    auto orders = allPermutations(names);
    auto tensors = collectTensorDescs(linalgOp);

    toml::table doc = buildOpDoc(programPath, numLoops, tensors, orders);

    std::ofstream out(fileName);
    if (!out) {
      llvm::errs() << "failed to open '" << fileName << "' for writing\n";
      signalPassFailure();
      return WalkResult::interrupt();
    }
    out << doc;
    out.close();

    llvm::outs() << "[op" << opIdx << "] "
                 << linalgOp->getName().getStringRef() << " -> " << fileName
                 << " (num_loops=" << numLoops
                 << ", num_tensors=" << tensors.size()
                 << ", orders=" << orders.size() << ")\n";
    ++opIdx;
    return WalkResult::advance();
  });

  llvm::outs() << "=== Emitted " << opIdx << " goal files ===\n";
}

} // namespace

std::unique_ptr<mlir::Pass> createInteractivePass() {
  return std::make_unique<Interactive>();
}

void registerInteractivePipeline() {
  PassPipelineRegistration<>(
      "interactive",
      "Emit per-op Honeybee goal files (.hb.toml) for each linalg op",
      [&](OpPassManager &pm) { pm.addPass(createInteractivePass()); });
}

} // namespace sam
} // namespace mlir
