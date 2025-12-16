//===- LinalgToStandard.cpp - conversion from Linalg to STeP dialect ------===//
//
//===----------------------------------------------------------------------===//

#include "lib/Conversion/SparseToSam/LinalgToSam.h"

#include <memory>

#include "lib/Dialect/SAM/SamDialect.h"
#include "lib/Dialect/SAM/SamOps.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
// #include "mlir/Transforms/TopologicalSortUtils.h"
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

namespace mlir
{
#define GEN_PASS_DEF_FUSIONDISPATCHGROUPS

#include "lib/Conversion/SparseToSam/Passes.h.inc"

} // namespace mlir


using namespace mlir;
using namespace mlir::linalg;
using namespace mlir::sparse_tensor;

#include "mlir/IR/Builders.h"
// #include "mlir/IR/Function.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
// #include "mlir/Dialect/StandardOps/IR/Ops.h"

using namespace mlir;

struct FusionDispatchGroupsPass : public mlir::impl::FusionDispatchGroupsBase<FusionDispatchGroupsPass>
{
    void runOnOperation() override;
};

void FusionDispatchGroupsPass::runOnOperation()
{
    Operation *root = getOperation();
    SmallVector<func::FuncOp> funcs;
    if (auto mod = dyn_cast<ModuleOp>(root))
    {
        mod.walk([&](func::FuncOp f) { funcs.push_back(f); });
    }
    else if (auto f = dyn_cast<func::FuncOp>(root))
    {
        funcs.push_back(f);
    }
    OpBuilder builder(root->getContext());

    auto isDataMovement = [](Operation *op) -> bool
    {
        return isa<linalg::TransposeOp, tensor::CollapseShapeOp, tensor::ExpandShapeOp, tensor::InsertSliceOp,
                   tensor::ExtractSliceOp, tensor::ConcatOp>(op);
    };

    // Collect ops in topological order per block and form groups delimited by data movement.
    for (func::FuncOp func : funcs)
    {
        for (Block &block : func.getBody().getBlocks())
        {
            SmallVector<Operation *> group;
            auto flushGroup = [&](Operation *insertPt)
            {
                if (group.empty())
                    return;
                // Identify group outputs (values used outside the group).
                DenseSet<Operation *> groupSet(group.begin(), group.end());
                SmallVector<Value> groupOutputs;
                SmallVector<Type> groupOutputTypes;
                for (Operation *op : group)
                {
                    for (Value res : op->getResults())
                    {
                        bool escapes =
                            llvm::any_of(res.getUsers(), [&](Operation *user) { return !groupSet.contains(user); });
                        if (escapes)
                        {
                            groupOutputs.push_back(res);
                            groupOutputTypes.push_back(res.getType());
                        }
                    }
                }
                // Determine external values used inside the group (captures).
                DenseSet<Value> internalValues;
                for (Operation *op : group)
                {
                    for (Value v : op->getResults())
                        internalValues.insert(v);
                }
                SmallVector<Value> captures;
                for (Operation *op : group)
                {
                    for (Value iv : op->getOperands())
                    {
                        if (!internalValues.contains(iv))
                            captures.push_back(iv);
                    }
                }
                // Create dispatch region op with results for escaping values and captures.
                if (insertPt)
                    builder.setInsertionPoint(insertPt);
                else
                    builder.setInsertionPointToEnd(&block);
                auto disp =
                    builder.create<sam::SamDispatchGroup>((insertPt ? insertPt->getLoc() : func.getLoc()),
                                                          TypeRange(groupOutputTypes), // ArrayAttr(), // ArrayAttr(),
                                                          // StringAttr(),
                                                          ValueRange(captures));
                Region &reg = disp.getDispatchRegion();
                reg.push_back(new Block());
                Block &rb = reg.front();
                // Add block arguments for captures.
                for (Value cap : captures)
                    rb.addArgument(cap.getType(), disp.getLoc());
                builder.setInsertionPointToEnd(&rb);
                // Move ops in order into region.
                for (Operation *op : group)
                {
                    // First move the op into the region.
                    op->moveBefore(&rb, rb.end());
                    // Remap any operands (including nested region ops) that are captures to block arguments.
                    op->walk(
                        [&](Operation *inner)
                        {
                            for (OpOperand &operand : inner->getOpOperands())
                            {
                                Value ov = operand.get();
                                auto it = llvm::find(captures, ov);
                                if (it != captures.end())
                                {
                                    unsigned idx = std::distance(captures.begin(), it);
                                    operand.set(rb.getArgument(idx));
                                }
                            }
                        });
                }
                // Yield the outputs.
                builder.create<sam::SamYield>(disp.getLoc(), groupOutputs);
                // Replace external uses of outputs with dispatch results.
                auto resRange = disp.getResult();
                for (auto it : llvm::enumerate(groupOutputs))
                {
                    Value oldv = it.value();
                    Value newv = *(resRange.begin() + it.index());
                    oldv.replaceUsesWithIf(newv, [&](OpOperand &use)
                                           { return !reg.isAncestor(use.getOwner()->getParentRegion()); });
                }
                group.clear();
            };

            for (Operation &op : llvm::make_early_inc_range(block))
            {
                if (auto linalgOp = dyn_cast<linalg::LinalgOp>(&op))
                {
                    group.push_back(&op);
                    continue;
                }
                if (isDataMovement(&op))
                {
                    flushGroup(&op);
                    continue;
                }
            }
            // Flush at block end.
            flushGroup(block.empty() ? nullptr : &block.back());
        }
    }
}

std::unique_ptr<mlir::Pass> mlir::createFusionDispatchGroupsPass()
{
    return std::make_unique<FusionDispatchGroupsPass>();
}
