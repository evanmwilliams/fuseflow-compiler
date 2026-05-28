#include "lib/Transforms/SparseAnnotate/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SparseTensor/IR/Enums.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"

#include "lib/Transforms/Interactive/toml.hpp"

#include <fstream>
#include <string>
#include <utility>
#include <vector>

namespace mlir {
namespace sam {
#define GEN_PASS_DEF_SPARSEANNOTATE
#include "lib/Transforms/SparseAnnotate/Passes.h.inc"

namespace {

struct TensorEntry {
  Value value;
  std::string name;        // SSA name without leading '%' (e.g. "arga", "0")
  unsigned rank;
  std::string indicesStr;  // "i0 i1 ..."
};

std::string buildIndicesString(unsigned rank) {
  std::string out;
  for (unsigned i = 0; i < rank; ++i) {
    if (i)
      out.push_back(' ');
    out.append("i").append(std::to_string(i));
  }
  return out;
}

// Pull a Value's printed SSA name (e.g. "%arga") and strip the leading '%'.
std::string getSsaName(Value v, AsmState &asmState) {
  std::string buf;
  llvm::raw_string_ostream os(buf);
  v.printAsOperand(os, asmState);
  llvm::StringRef name(buf);
  if (name.starts_with("%"))
    name = name.drop_front();
  return name.str();
}

// Collect every distinct SSA tensor value reachable from the module that does
// not already carry a sparse_tensor.encoding attribute.
std::vector<TensorEntry> collectUnannotatedTensors(ModuleOp moduleOp) {
  AsmState asmState(moduleOp);
  llvm::DenseSet<Value> seen;
  std::vector<TensorEntry> entries;

  auto record = [&](Value v) {
    auto rtt = llvm::dyn_cast<RankedTensorType>(v.getType());
    if (!rtt)
      return;
    if (sparse_tensor::getSparseTensorEncoding(v.getType()))
      return;
    if (!seen.insert(v).second)
      return;
    TensorEntry e;
    e.value = v;
    e.name = getSsaName(v, asmState);
    e.rank = static_cast<unsigned>(rtt.getRank());
    e.indicesStr = buildIndicesString(e.rank);
    entries.push_back(std::move(e));
  };

  // Only annotate function arguments. Op result types are derived from
  // outs operands via DestinationStyleOpInterface and are propagated after
  // the apply step, so they don't need separate PBN sessions.
  moduleOp.walk([&](func::FuncOp func) {
    for (Value arg : func.getArguments())
      record(arg);
  });

  return entries;
}

// Build the Honeybee prop file (P_Tensor + TensorSparseAnnotation goal) for a
// single tensor.
toml::table buildTensorDoc(const TensorEntry &entry) {
  toml::table doc;
  toml::array &prop =
      *doc.insert("Prop", toml::array{}).first->second.as_array();

  toml::table propEntry;
  propEntry.insert("name", "P_Tensor");
  toml::table args;
  args.insert("name", entry.name);
  args.insert("num_indices", static_cast<int>(entry.rank));
  args.insert("indices", entry.indicesStr);
  propEntry.insert("args", std::move(args));
  prop.push_back(std::move(propEntry));

  toml::table goal;
  goal.insert("name", "TensorSparseAnnotation");
  goal.insert("args", toml::table{});
  doc.insert("Goal", std::move(goal));

  return doc;
}

LogicalResult writeTomlFile(const std::string &path, const toml::table &doc) {
  std::ofstream out(path);
  if (!out) {
    llvm::errs() << "failed to open '" << path << "' for writing\n";
    return failure();
  }
  out << doc;
  return success();
}

LogicalResult runProgram(llvm::StringRef bin,
                         llvm::ArrayRef<llvm::StringRef> argv,
                         llvm::StringRef cwd, llvm::StringRef tag) {
  // ExecuteAndWait has no cwd parameter, so chdir for the duration of the
  // call. Restore afterward so subsequent tensors get the same baseline.
  llvm::SmallString<256> savedCwd;
  if (auto ec = llvm::sys::fs::current_path(savedCwd)) {
    llvm::errs() << "failed to get cwd: " << ec.message() << "\n";
    return failure();
  }
  if (auto ec = llvm::sys::fs::set_current_path(cwd)) {
    llvm::errs() << "failed to chdir to '" << cwd << "': " << ec.message()
                 << "\n";
    return failure();
  }

  std::string err;
  int rc = llvm::sys::ExecuteAndWait(bin, argv,
                                     /*Env=*/std::nullopt,
                                     /*Redirects=*/{},
                                     /*SecondsToWait=*/0,
                                     /*MemoryLimit=*/0, &err);

  (void)llvm::sys::fs::set_current_path(savedCwd);

  if (rc != 0) {
    llvm::errs() << tag << " returned " << rc;
    if (!err.empty())
      llvm::errs() << ": " << err;
    llvm::errs() << "\n";
    return failure();
  }
  return success();
}

LogicalResult invokeHoneybee(llvm::StringRef honeybeeBin,
                             llvm::StringRef libraryPath,
                             llvm::StringRef progPath,
                             llvm::StringRef transcriptPath,
                             llvm::StringRef cwd) {
  auto resolved = llvm::sys::findProgramByName(honeybeeBin);
  if (!resolved) {
    llvm::errs() << "honeybee binary not found on PATH: " << honeybeeBin
                 << "\n";
    return failure();
  }

  llvm::SmallVector<llvm::StringRef> argv;
  argv.push_back(*resolved);
  argv.push_back("interact");
  argv.push_back("--library");
  argv.push_back(libraryPath);
  argv.push_back("--out");
  argv.push_back(transcriptPath);
  argv.push_back(progPath);

  return runProgram(*resolved, argv, cwd, "honeybee");
}

// The transcript Honeybee writes is a Python program that, when executed,
// dumps `annotation.json` under `output/<NNN-step>/` relative to cwd.
LogicalResult executeTranscript(llvm::StringRef transcriptPath,
                                llvm::StringRef cwd) {
  auto python = llvm::sys::findProgramByName("python3");
  if (!python)
    python = llvm::sys::findProgramByName("python");
  if (!python) {
    llvm::errs() << "python3/python not found on PATH\n";
    return failure();
  }

  llvm::SmallVector<llvm::StringRef> argv;
  argv.push_back(*python);
  argv.push_back(transcriptPath);

  return runProgram(*python, argv, cwd, "python");
}

// Walk `<cwd>/output/<NNN-step>/annotation.json` to locate the JSON the
// transcript wrote. There should be exactly one.
LogicalResult findAnnotationJson(llvm::StringRef sessionDir,
                                 llvm::SmallString<256> &out) {
  llvm::SmallString<256> outputDir(sessionDir);
  llvm::sys::path::append(outputDir, "output");

  std::error_code ec;
  for (llvm::sys::fs::directory_iterator dir(outputDir, ec), end;
       dir != end && !ec; dir.increment(ec)) {
    llvm::SmallString<256> candidate(dir->path());
    llvm::sys::path::append(candidate, "annotation.json");
    if (llvm::sys::fs::exists(candidate)) {
      out = candidate;
      return success();
    }
  }
  llvm::errs() << "no annotation.json found under '" << outputDir << "'\n";
  return failure();
}

// Build a sparse_tensor encoding from the JSON entry. Returns null encoding
// when the tensor is dense (caller should leave the type alone).
sparse_tensor::SparseTensorEncodingAttr
buildEncoding(MLIRContext *ctx, const TensorEntry &entry,
              const llvm::json::Object &annotation) {
  bool isSparse = annotation.getBoolean("sparse").value_or(false);
  if (!isSparse)
    return {};

  const auto *dims = annotation.getObject("dimensions");
  if (!dims)
    return {};

  llvm::SmallVector<llvm::StringRef> idxNames;
  llvm::StringRef(entry.indicesStr).split(idxNames, ' ');

  llvm::SmallVector<sparse_tensor::LevelType> lvlTypes;
  lvlTypes.reserve(idxNames.size());
  for (llvm::StringRef idx : idxNames) {
    bool dimSparse = dims->getBoolean(idx).value_or(false);
    lvlTypes.push_back(dimSparse ? sparse_tensor::LevelFormat::Compressed
                                 : sparse_tensor::LevelFormat::Dense);
  }

  return sparse_tensor::SparseTensorEncodingAttr::get(ctx, lvlTypes);
}

// Apply the merged annotations back onto the IR: rewrite function arg
// tensor types, propagate result types through destination-style ops, and
// patch up func signatures.
void applyAnnotations(ModuleOp moduleOp,
                      llvm::ArrayRef<TensorEntry> tensors,
                      const llvm::json::Object &merged) {
  MLIRContext *ctx = moduleOp.getContext();

  // 1. Update function argument types per the JSON annotations.
  for (const auto &entry : tensors) {
    const auto *parsed = merged.get(entry.name);
    if (!parsed)
      continue;
    const auto *obj = parsed->getAsObject();
    if (!obj)
      continue;

    auto encoding = buildEncoding(ctx, entry, *obj);
    if (!encoding)
      continue; // dense — leave type untouched

    auto blockArg = llvm::dyn_cast<BlockArgument>(entry.value);
    if (!blockArg)
      continue;
    auto oldType = llvm::cast<RankedTensorType>(blockArg.getType());
    blockArg.setType(RankedTensorType::get(
        oldType.getShape(), oldType.getElementType(), encoding));
  }

  // 2. Propagate destination-style ops' init operand types to their results.
  //    For linalg ops, the result type must equal the outs operand type, so
  //    sparse-encoded init operands force result types to match.
  moduleOp.walk([](DestinationStyleOpInterface dpsOp) {
    auto inits = dpsOp.getDpsInits();
    auto results = dpsOp->getResults();
    if (inits.size() != results.size())
      return;
    for (auto [init, result] : llvm::zip(inits, results))
      result.setType(init.getType());
  });

  // 3. Rebuild every func's signature from current block arg + return types.
  moduleOp.walk([&](func::FuncOp func) {
    llvm::SmallVector<Type> argTypes;
    for (auto arg : func.getArguments())
      argTypes.push_back(arg.getType());

    llvm::SmallVector<Type> resultTypes(func.getResultTypes());
    func.walk([&](func::ReturnOp ret) {
      resultTypes.assign(ret.getOperandTypes().begin(),
                         ret.getOperandTypes().end());
    });

    func.setFunctionType(FunctionType::get(ctx, argTypes, resultTypes));
  });
}

LogicalResult readAnnotationJson(llvm::StringRef path,
                                 llvm::json::Value &outValue) {
  auto buf = llvm::MemoryBuffer::getFile(path);
  if (!buf) {
    llvm::errs() << "failed to read '" << path
                 << "': " << buf.getError().message() << "\n";
    return failure();
  }
  auto parsed = llvm::json::parse((*buf)->getBuffer());
  if (!parsed) {
    llvm::errs() << "failed to parse JSON in '" << path
                 << "': " << llvm::toString(parsed.takeError()) << "\n";
    return failure();
  }
  outValue = std::move(*parsed);
  return success();
}

struct SparseAnnotate : public impl::SparseAnnotateBase<SparseAnnotate> {
  SparseAnnotate() = default;
  SparseAnnotate(std::string library, std::string honeybee,
                 std::string outputDir)
      : library(std::move(library)), honeybee(std::move(honeybee)),
        outputDir(std::move(outputDir)) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<sparse_tensor::SparseTensorDialect, func::FuncDialect>();
  }

  void runOnOperation() override;

  std::string library;
  std::string honeybee;
  std::string outputDir;
};

void SparseAnnotate::runOnOperation() {
  ModuleOp moduleOp = getOperation();

  if (library.empty()) {
    llvm::errs() << "--annotate-sparsity requires library=<path>\n";
    return signalPassFailure();
  }

  if (auto ec = llvm::sys::fs::create_directories(outputDir)) {
    llvm::errs() << "failed to create output dir '" << outputDir
                 << "': " << ec.message() << "\n";
    return signalPassFailure();
  }

  auto tensors = collectUnannotatedTensors(moduleOp);
  llvm::outs() << "=== SparseAnnotate: " << tensors.size()
               << " un-annotated tensor(s) ===\n";

  llvm::json::Object merged;

  for (const auto &entry : tensors) {
    llvm::SmallString<256> sessionDir(outputDir);
    llvm::sys::path::append(sessionDir, entry.name);
    if (auto ec = llvm::sys::fs::create_directories(sessionDir)) {
      llvm::errs() << "failed to create session dir '" << sessionDir
                   << "': " << ec.message() << "\n";
      return signalPassFailure();
    }

    llvm::SmallString<256> progPath(sessionDir);
    llvm::sys::path::append(progPath, "prog.hb.toml");

    llvm::SmallString<256> transcriptPath(sessionDir);
    llvm::sys::path::append(transcriptPath, "session.py");

    toml::table doc = buildTensorDoc(entry);
    if (failed(writeTomlFile(progPath.str().str(), doc)))
      return signalPassFailure();

    llvm::outs() << "\n=============================================\n"
                 << "Annotating tensor %" << entry.name << " (rank "
                 << entry.rank << ", indices: " << entry.indicesStr << ")\n"
                 << "  prog: " << progPath << "\n"
                 << "=============================================\n";

    if (failed(invokeHoneybee(honeybee, library, progPath, transcriptPath,
                              sessionDir)))
      return signalPassFailure();

    llvm::outs() << "    executing transcript...\n";
    if (failed(executeTranscript(transcriptPath, sessionDir)))
      return signalPassFailure();

    llvm::SmallString<256> jsonPath;
    if (failed(findAnnotationJson(sessionDir, jsonPath)))
      return signalPassFailure();

    llvm::json::Value parsed(nullptr);
    if (failed(readAnnotationJson(jsonPath, parsed)))
      return signalPassFailure();

    merged[entry.name] = std::move(parsed);
  }

  llvm::SmallString<256> mergedPath(outputDir);
  llvm::sys::path::append(mergedPath, "annotations.json");

  std::error_code ec;
  llvm::raw_fd_ostream os(mergedPath.str(), ec);
  if (ec) {
    llvm::errs() << "failed to open '" << mergedPath
                 << "': " << ec.message() << "\n";
    return signalPassFailure();
  }
  os << llvm::formatv("{0:2}", llvm::json::Value(llvm::json::Object(merged)))
     << "\n";

  llvm::outs() << "=== Wrote " << mergedPath << " ===\n";

  // Apply annotations to the module in place. mlir-opt's stdout will print
  // the annotated form; we also drop a copy at <outputDir>/annotated.mlir.
  applyAnnotations(moduleOp, tensors, merged);

  llvm::SmallString<256> annotatedPath(outputDir);
  llvm::sys::path::append(annotatedPath, "annotated.mlir");

  std::error_code mlirEc;
  llvm::raw_fd_ostream mlirOs(annotatedPath.str(), mlirEc);
  if (mlirEc) {
    llvm::errs() << "failed to open '" << annotatedPath
                 << "': " << mlirEc.message() << "\n";
    return signalPassFailure();
  }
  moduleOp.print(mlirOs);
  mlirOs << "\n";

  llvm::outs() << "=== Wrote annotated MLIR to " << annotatedPath << " ===\n";
}

} // namespace

std::unique_ptr<mlir::Pass> createSparseAnnotatePass(std::string library,
                                                     std::string honeybee,
                                                     std::string outputDir) {
  return std::make_unique<SparseAnnotate>(std::move(library), std::move(honeybee),
                                          std::move(outputDir));
}

void registerSparseAnnotatePipeline() {
  PassPipelineRegistration<SparseAnnotatePipelineOptions>(
      "annotate-sparsity",
      "Drive Honeybee to annotate every un-annotated tensor's sparsity",
      [](OpPassManager &pm, const SparseAnnotatePipelineOptions &options) {
        pm.addPass(createSparseAnnotatePass(options.library, options.honeybee,
                                            options.outputDir));
      });
}

} // namespace sam
} // namespace mlir
