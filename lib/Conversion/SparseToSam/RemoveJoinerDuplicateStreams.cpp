//===- LinalgToStandard.cpp - conversion from Linalg to STeP dialect ------===//
//
//===----------------------------------------------------------------------===//

#include "lib/Conversion/SparseToSam/LinalgToSam.h"

#include <iostream>
#include <memory>

#include "lib/Dialect/SAM/SamDialect.h"
#include "lib/Dialect/SAM/SamOps.h"
#include "mlir/Analysis/SliceAnalysis.h"
/* #include "mlir/Dialect/Func/Transforms/OneToNFuncConversions.h" */
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
#define GEN_PASS_DEF_REMOVEJOINERDUPLICATESTREAMS

#include "lib/Conversion/SparseToSam/Passes.h.inc"

} // namespace mlir

using namespace mlir;
using namespace mlir::linalg;

struct RemoveJoinerDuplicateAfterRoot : public OpRewritePattern<mlir::sam::SamJoiner> {
public:
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(mlir::sam::SamJoiner joinerOp, PatternRewriter &rewriter) const override {
        auto in_crds = joinerOp.getInputCrd();
        auto in_refs = joinerOp.getInputRef();
        auto out_crd = joinerOp.getOutputCrd();
        auto out_refs = joinerOp.getOutputRefs();

        // for (auto ref : out_refs)
        //    llvm::DenseMap<mlir::OpResult, unsigned int> resultMapping;
        std::map<unsigned int, unsigned int> resultMapping;

        llvm::DenseSet<mlir::Value> seen;

        llvm::SmallVector<mlir::Value> finalCrds;
        llvm::SmallVector<mlir::Value> finalRefs;
        llvm::SmallVector<mlir::Type> finalOutRefs;
        for (auto crdRefPairIter: llvm::enumerate(llvm::zip(in_crds, in_refs, out_refs))) {
            auto crdRefPair = crdRefPairIter.value();
            unsigned int index = crdRefPairIter.index();
            if (seen.insert(std::get<1>(crdRefPair)).second) {
                finalCrds.push_back(std::get<0>(crdRefPair));
                finalRefs.push_back(std::get<1>(crdRefPair));
                finalOutRefs.push_back(std::get<2>(crdRefPair).getType());
            }
            auto it = llvm::find(finalCrds, std::get<0>(crdRefPair));
            if (it != finalRefs.end()) {
                unsigned int itIndex = it - finalCrds.begin();
                //            resultMapping[std::get<2>(crdRefPair)] = itIndex;
                resultMapping[index] = itIndex;
            }
        }

//        llvm::outs() << "Joiner: " << joinerOp << "\n";
        if (finalCrds.size() != in_crds.size()) {
            if (finalCrds.size() == 1) {
                for (auto out_ref: joinerOp.getOutputRefs()) {
                    rewriter.replaceAllUsesWith(out_ref, finalRefs.front());
                }
                rewriter.replaceAllUsesWith(joinerOp.getOutputCrd(), finalCrds.front());
                rewriter.eraseOp(joinerOp);
                return success();
            }

            // Create new joiner op to replace old one with the appropriate number of refs and crds
            auto joiner = rewriter.create<sam::SamJoiner>(joinerOp->getLoc(), out_crd.getType(), finalOutRefs,
                                                          finalCrds, finalRefs, joinerOp.getJoinerOp());
            // Reroute output references and crd to new joiner
            for (auto out_refIter: llvm::enumerate(joinerOp.getOutputRefs())) {
                auto out_ref = out_refIter.value();
                unsigned int index = out_refIter.index();
                rewriter.replaceAllUsesWith(out_ref, joiner.getOutputRefs()[resultMapping.at(index)]);
            }
            rewriter.replaceAllUsesWith(joinerOp.getOutputCrd(), joiner.getOutputCrd());
//            llvm::outs() << "JOINER: " << joiner << "\n";
            rewriter.eraseOp(joinerOp);
            return success();
        } else {
            return success();
        }
    }
};

struct RemoveJoinerDuplicateStreamsPass
        : public mlir::impl::RemoveJoinerDuplicateStreamsBase<RemoveJoinerDuplicateStreamsPass> {
    void runOnOperation() override;
};

void RemoveJoinerDuplicateStreamsPass::runOnOperation() {
    MLIRContext &context = getContext();
    ConversionTarget target(context);

    // Call remove joiner pass twice, with a CSE pass in between to remove duplicates
    RewritePatternSet patterns(&context);
    patterns.add<RemoveJoinerDuplicateAfterRoot>(&context);
    // Run remove joiner duplicates pass before calling CSE
    (void) applyPatternsGreedily(getOperation(), std::move(patterns));

    // Using dynamic pass manager to invoke CSE and canonicalizer pass before applying remove joiner duplicates pass again
    mlir::OpPassManager dynamicPM(getOperation()->getName());
    dynamicPM.addPass(mlir::createCanonicalizerPass());
    dynamicPM.addPass(mlir::createCSEPass());
    if (failed(runPipeline(dynamicPM, getOperation()))) {
        signalPassFailure();
    }

    // Call remove joiner duplicates pass a second time with new rewrite pattern set since first one was consumed
    RewritePatternSet patterns1(&context);
    patterns1.add<RemoveJoinerDuplicateAfterRoot>(&context);
    (void) applyPatternsGreedily(getOperation(), std::move(patterns1));
}

std::unique_ptr<mlir::Pass> mlir::createRemoveJoinerDuplicateStreamsPass() {
    return std::make_unique<RemoveJoinerDuplicateStreamsPass>();
}
