//===- LinalgToStandard.cpp - conversion from Linalg to STeP dialect ------===//
//
//===----------------------------------------------------------------------===//

#include "lib/Conversion/SparseToSam/LinalgToSam.h"

#include <iostream>
#include <memory>

#include "lib/Dialect/SAM/SamDialect.h"
#include "lib/Dialect/SAM/SamOps.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/Index/IR/IndexAttrs.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
// #include "mlir/lib/IR/AffineExprDetail.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/FormatVariadic.h"

#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensorType.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"

namespace mlir {
#define GEN_PASS_DEF_INSERTITERATELOCATE

#include "lib/Conversion/SparseToSam/Passes.h.inc"

} // namespace mlir

using namespace mlir;
using namespace mlir::linalg;

struct InsertIterateLocate : public OpRewritePattern<mlir::sam::SamJoiner> {
public:
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(mlir::sam::SamJoiner joinerOp, PatternRewriter &rewriter) const override {
        auto in_crds = joinerOp.getInputCrd();
        auto in_refs = joinerOp.getInputRef();
        auto out_crd = joinerOp.getOutputCrd();
        auto out_refs = joinerOp.getOutputRefs();
        auto joinerType = joinerOp.getJoinerOp();

        if (joinerType != mlir::sam::JoinerType::Intersect) {
            return failure();
        }

        // Assume two inputs for now, one sparse and one dense
        if (in_crds.size() != 2) {
            return failure();
        }

        int sparseInputIdx = -1;
        int denseInputIdx = -1;

        for (auto refIter: llvm::enumerate(in_refs)) {
            const auto &ref = refIter.value();
            int idx = (int) refIter.index();
            mlir::sam::SamStreamEncodingAttr enc = mlir::sam::getSamStreamEncoding(ref.getType());
            auto format = enc.getFormat().str();
            if (format == "compressed") {
                sparseInputIdx = idx;
            } else if (format == "dense") {
                denseInputIdx = idx;
            }
        }

        // Needs one sparse level and one dense level
        if (sparseInputIdx == -1 || denseInputIdx == -1) {
            return failure();
        }

        auto denseLs = in_refs[denseInputIdx].getDefiningOp<mlir::sam::SamFiberLookup>();
        auto sparseLs = in_refs[sparseInputIdx].getDefiningOp<mlir::sam::SamFiberLookup>();

        // Sanity check, input should be a level scanner
        if (!denseLs) {
            return failure();
        }

        // Hardcoding order of outputs to be dense stream first
//        llvm::outs() << "Type: " << out_refs[denseInputIdx].getType() << "\n";
//        llvm::outs() << "Type: " << out_refs[sparseInputIdx].getType() << "\n";
//        llvm::outs() << "Type: " << out_crd.getType() << "\n";
        auto locateOp = rewriter.create<mlir::sam::SamLocate>(joinerOp->getLoc(), out_refs[denseInputIdx].getType(),
                                                              out_refs[sparseInputIdx].getType(), out_crd.getType(),
                                                              sparseLs.getOutputCrd(), sparseLs.getOutputRef());

        rewriter.replaceAllUsesWith(out_refs[denseInputIdx], locateOp.getOutputRef1());
        rewriter.replaceAllUsesWith(out_refs[sparseInputIdx], locateOp.getOutputRef2());
        rewriter.replaceAllUsesWith(out_crd, locateOp.getOutputCrd());

        return success();
    }
};

struct InsertIterateLocatePass
        : public mlir::impl::InsertIterateLocateBase<InsertIterateLocatePass> {
    void runOnOperation() override;
};

void InsertIterateLocatePass::runOnOperation() {
    MLIRContext &context = getContext();
    ConversionTarget target(context);

    // Call remove joiner pass twice, with a CSE pass in between to remove duplicates
    RewritePatternSet patterns(&context);
    patterns.add<InsertIterateLocate>(&context);
    // Run remove joiner duplicates pass before calling CSE
    (void) applyPatternsGreedily(getOperation(), std::move(patterns));
}

std::unique_ptr<mlir::Pass> mlir::createInsertIterateLocatePass() {
    return std::make_unique<InsertIterateLocatePass>();
}
