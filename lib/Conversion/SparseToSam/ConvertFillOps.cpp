//===- LinalgToStandard.cpp - conversion from Linalg to STeP dialect ------===//
//
//===----------------------------------------------------------------------===//

#include "lib/Conversion/SparseToSam/LinalgToSam.h"

#include <memory>

#include "lib/Dialect/SAM/SamDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Index/IR/IndexAttrs.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
//#include "mlir/lib/IR/AffineExprDetail.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"

#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTFILLOPS

#include "lib/Conversion/SparseToSam/Passes.h.inc"

} // namespace mlir

using namespace mlir;
using namespace mlir::linalg;

struct ConvertFillOpsPattern : public OpRewritePattern<linalg::FillOp> {
public:
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(linalg::FillOp fillOp,
                                  PatternRewriter &rewriter) const override {
        ShapedType outType = (ShapedType) fillOp.output().getType();
        TypedAttr constAttr;
        if (auto constOp =
                dyn_cast<arith::ConstantOp>(fillOp.value().getDefiningOp())) {
            constAttr = constOp.getValue();
        } else {
            return failure();
        }
        // TODO: Change to inputs
        DenseElementsAttr splat = DenseElementsAttr::get(outType, constAttr);
        auto newConst = rewriter.create<arith::ConstantOp>(
                fillOp.getLoc(), fillOp.output().getType(), splat);
        fillOp.getOutputs()[0].replaceAllUsesWith(newConst.getResult());
        // llvm::outs() << "Prev: " << fillOp << "\n";
        // llvm::outs() << "Prev: " << *fillOp.value().getDefiningOp() << "\n";
        // llvm::outs() << "NEW CONST: " << newConst << "\n";
        rewriter.replaceOp(fillOp, newConst);
        // fillOp.output().getDefiningOp()->erase();
        return success();
    }
};

struct ConvertTransposeOpsPattern : public OpRewritePattern<linalg::GenericOp> {
public:
    using OpRewritePattern::OpRewritePattern;

    ConvertTransposeOpsPattern(MLIRContext *ctx, PatternBenefit benefit = 1,
                               int count = 0)
            : OpRewritePattern(ctx, benefit), counter(count) {
//        counter = count;
    }

    LogicalResult matchAndRewrite(linalg::GenericOp op,
                                  PatternRewriter &rewriter) const override {

        if (op.getNumDpsInputs() != 1) {
            return failure();
        }

        // Check if the operands and results have reversed shapes:
        if (op->getNumResults() != 1) {
            return failure(); // Expected one operand and one result
        }

        auto operandType = mlir::dyn_cast<RankedTensorType>(op->getOperand(0).getType());
        auto resultType = mlir::dyn_cast<RankedTensorType>(op->getResult(0).getType());
        auto outType = (ShapedType) resultType;

        if (!operandType || !resultType) {
            return failure(); // Operands or results are not RankedTensorType
        }

        // Get indexing maps to compare the two
        auto affineMaps = op.getIndexingMapsArray();
        auto inputMap = affineMaps[0];
        auto outputMap = affineMaps[1];

        // Check if the input map is the identify map, no dimension permutation
        if (!inputMap.isIdentity()) {
            return failure();
        }

        // Check if first indexing map's results are the reverse of second indexing map's results
        // indicating a transpose operation
        for (unsigned int i = 0; i < inputMap.getNumResults(); ++i) {
            if (inputMap.getResult(i) !=
                outputMap.getResult(outputMap.getNumResults() - 1 - i)) {
                return failure(); // Shapes not reversed
            }
        }

        // Get underlying storage attribute for the input constant if it exists, otherwise return failure
        TypedAttr constAttr;
        if (auto constOp =
                dyn_cast<arith::ConstantOp>(op->getOperand(0).getDefiningOp())) {
            constAttr = constOp.getValue();
        } else {
            return failure();
        }

        // Cast constant attribute to a dense resource element required by the ConstantOp create function
//        auto splat = llvm::dyn_cast<DenseResourceElementsAttr>(constAttr);
        std::string tensorName = "torch_tensor_";
        std::set<unsigned int> seenShape;
        bool sameShapes = true;
        for (auto shapeIter: llvm::enumerate(outType.getShape())) {
            unsigned int shape = shapeIter.value();
            auto index = shapeIter.index();
            tensorName += std::to_string(shape);
            if (index != outType.getShape().size() - 1) {
                tensorName += "_";
            }
            if (!seenShape.insert(shape).second) {
                sameShapes = false;
            }
        }
        tensorName += "_" + std::to_string(counter++);
        arith::ConstantOp newConst;
        std::vector<float> dummyData{static_cast<float>(counter)};
        ArrayRef<float> dummyDataArray = ArrayRef(dummyData);
        auto splat = DenseResourceElementsAttr::get(outType, tensorName,
                                                    UnmanagedAsmResourceBlob::allocateInferAlign(dummyDataArray));
        if (!splat) {
            return failure();
        }
        // Create new constant op to replace the linalg generic with the same output type
        newConst = rewriter.create<arith::ConstantOp>(
                op.getLoc(), op->getOpResult(0).getType(), splat);

        // Replace all uses of the original generic op to the constant op's result
        op.getOutputs()[0].replaceAllUsesWith(newConst.getResult());
        // Replace the original op with the new constant op
        rewriter.replaceOp(op, newConst);
        return success();
    }

private:
    mutable int counter;
};

struct RemoveUnnecessaryExpandShape : public OpRewritePattern<tensor::ExpandShapeOp> {
public:
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(tensor::ExpandShapeOp expandOp,
                                  PatternRewriter &rewriter) const override {
        auto t = expandOp.getOutputShape();
//        llvm::outs() << "First: " << t[0] << "\n";
//        llvm::outs() << "Second: " << t[1] << "\n";
        return success();
    }
};

struct ConvertFillOpsPass
        : public mlir::impl::ConvertFillOpsBase<ConvertFillOpsPass> {
    void runOnOperation() override;
};

void ConvertFillOpsPass::runOnOperation() {
    func::FuncOp func = getOperation();
    MLIRContext &context = getContext();
    RewritePatternSet patterns(&context);
    patterns.add<ConvertFillOpsPattern>(&context);

    // Add folding with reshape by expansion patterns
    ControlFusionFn defaultControlFn = [](OpOperand *fusedOperand) {
        return true;
    };
    patterns.add<ConvertTransposeOpsPattern>(&context);

//     Folds linalg.transpose (and linalg.generic ops) on constant values
    mlir::linalg::populateConstantFoldLinalgOperations(patterns, defaultControlFn);
    mlir::linalg::populateFoldReshapeOpsByCollapsingPatterns(patterns, defaultControlFn);

    (void) applyPatternsGreedily(getOperation(), std::move(patterns));
}

std::unique_ptr<mlir::Pass> mlir::createConvertFillOpsPass() {
    return std::make_unique<ConvertFillOpsPass>();
}