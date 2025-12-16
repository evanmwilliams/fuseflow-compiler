//===- LinalgToStandard.cpp - conversion from Linalg to STeP dialect ------===//
//
//===----------------------------------------------------------------------===//

#include "lib/Conversion/SparseToSam/LinalgToSam.h"
#include "lib/Conversion/SparseToSam/Table.h"

#include <memory>
#include <utility>

#include "FusedCIN.h"
#include "lib/Dialect/SAM/SamDialect.h"
#include "lib/Dialect/SAM/SamOps.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Index/IR/IndexAttrs.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "ConstructEinsum.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"

namespace mlir
{
#define GEN_PASS_DEF_LINALGTOSAM

#include "lib/Conversion/SparseToSam/Passes.h.inc"

    // #include "mlir/Analysis/TopologicalSortUtils.h"

} // namespace mlir

using namespace mlir;
using namespace mlir::linalg;
using namespace mlir::sparse_tensor;

class StageRewriter final : public IRRewriter
{
public:
    explicit StageRewriter(MLIRContext *context) : IRRewriter(context) {}

    template <typename OpTy, typename... Args>
    OpTy create(Location location, Args &&...args);

private:
    // std::vector<std::tuple<std::tuple<std::string, OperandRange>, Operation *>>
    std::vector<unsigned int> tensorNums;
    std::vector<std::tuple<std::tuple<mlir::OperationName, OperandRange>, Operation *>> stageCache;
    std::vector<std::tuple<std::tuple<mlir::OperationName, OperandRange, std::string>, Operation *>> lookupStageCache;
};

template <typename OpTy, typename... Args>
OpTy StageRewriter::create(Location location, Args &&...args)
{
    auto op = IRRewriter::create<OpTy>(location, std::forward<Args>(args)...);

    // TODO: REMOVE RETURN STATEMENT BELOW
    //    return op;
    OperandRange operands = op->getOperands();
    auto name = op->getName();

    auto gen = llvm::dyn_cast<sam::SamGenerator>(*op);
    auto lookup = llvm::dyn_cast<sam::SamFiberLookup>(*op);
    auto joiner = llvm::dyn_cast<sam::SamJoiner>(*op);

    if (gen || lookup)
    {
        sam::SamStreamEncodingAttr enc;
        if (gen)
        {
            enc = sam::getSamStreamEncoding(gen.getOutputRef().getType());
        }
        else if (lookup)
        {
            enc = sam::getSamStreamEncoding(lookup.getOutputRef().getType());
        }
        std::string tensorName = enc.getTensor().str();
        tensorName = tensorName.substr(0, tensorName.find('-'));
        auto attrBundle = std::make_tuple(name, op->getOperands(), tensorName);
        for (auto iter : lookupStageCache)
        {
            auto key = std::get<0>(iter);
            if (attrBundle == key)
            {
                const auto cachedOp = std::get<1>(iter);
                op->erase();
                return llvm::dyn_cast<OpTy>(cachedOp);
            }
        }
        lookupStageCache.push_back(std::make_tuple(std::make_tuple(name, op->getOperands(), tensorName), op));
        return op;
    }

    auto pair = std::make_tuple(op->getName(), operands);
    //    llvm::outs() << "Name: " << op->getName() << "\n";
    for (auto iter : stageCache)
    {
        auto key = std::get<0>(iter);
        if (pair == key)
        {
            const auto cachedOp = std::get<1>(iter);
            op->erase();
            return llvm::dyn_cast<OpTy>(cachedOp);
        }
    }
    stageCache.push_back(std::make_tuple(std::make_tuple(name, op->getOperands()), op));
    return op;
}


static sam::SamStreamEncodingAttr getStreamAttr(MLIRContext *context, const sam::StreamKind streamType,
                                                AffineExpr dimExpr, const std::string &tensorName = "",
                                                const std::string &format = "compressed", const unsigned int mode = 0)
{
    const unsigned pos = llvm::cast<AffineDimExpr>(dimExpr).getPosition();
    auto builder = mlir::Builder(context);
    const auto tensor = tensorName.substr(0, tensorName.find('_'));
    return sam::SamStreamEncodingAttr::get(
        context, sam::StreamKindAttr::get(context, streamType), builder.getStringAttr("d" + std::to_string(pos)),
        builder.getStringAttr(tensor), builder.getStringAttr(format), builder.getI64IntegerAttr(mode));
}

template <typename T>
static void sortByPartialOrder(const llvm::SmallVector<T> &gold_list, llvm::SmallVector<T> &sorted_sublist)
{
    // Create a map to store the indices of elements in the gold list
    llvm::DenseMap<T, unsigned int> element_map;
    for (unsigned int i = 0; i < gold_list.size(); ++i)
    {
        element_map[gold_list[i]] = i;
    }

    // Sort the copied sublist using a custom comparator
    llvm::sort(sorted_sublist.begin(), sorted_sublist.end(),
               [&element_map](const T a, const T b) -> bool { return element_map[a] < element_map[b]; });
}

template <typename T>
static void sortByPartialOrder(const std::vector<T> &gold_list, std::vector<T> &sorted_sublist)
{
    // Create a map to store the indices of elements in the gold list
    std::map<T, unsigned int> element_map;
    for (unsigned int i = 0; i < gold_list.size(); ++i)
    {
        element_map[gold_list[i]] = i;
    }

    // Sort the copied sublist using a custom comparator
    llvm::sort(sorted_sublist.begin(), sorted_sublist.end(),
               [&element_map](const T a, const T b) -> bool { return element_map[a] < element_map[b]; });
}

// Helper function to check if a variable should be ignored for a given tensor view
// because it's a concat dimension variable for a concat operation that hasn't occurred yet
static bool shouldIgnoreConcatDimVar(const IndexVar &var, const TensorView &tensorView,
                                     const std::shared_ptr<AnalysisScope> &scope)
{
    // Check each concat operation in the scope
    for (const auto &concatInfo : scope->concatOperations)
    {
        // Get the IndexVar for the concat dimension from the UniqueVar
        IndexVar concatDimIndexVar = concatInfo.concatDimUniqueVar.getLogicalIndexVar();

        // Check if this variable is the concat dimension variable
        if (concatDimIndexVar == var)
        {
            // Check if the current tensor view is one of the inputs to this concat
            for (const auto &inputView : concatInfo.inputViews)
            {
                if (inputView == tensorView)
                {
                    // This is an input to a concat operation and var is the concat dimension
                    // so we should ignore it
                    return true;
                }
            }

            // Also check if this tensor view is used before the concat output
            // (i.e., it's not the concat output itself)
            if (!(concatInfo.outputView == tensorView))
            {
                // Check if this tensor might be processed before the concat
                // by checking if it's one of the operands that feeds into the concat
                return true;
            }
        }
    }
    return false;
}

template <typename blockOpType>
double getScalarOperand(mlir::linalg::LinalgOp op)
{
    const auto yieldOp = llvm::cast<linalg::YieldOp>(op.getBlock()->getTerminator());
    const mlir::Value producerOutput = yieldOp->getOperand(0);
    auto secondOperand = producerOutput.getDefiningOp<blockOpType>()->getOperand(1);
    if (auto constOp = llvm::dyn_cast<arith::ConstantOp>(secondOperand.getDefiningOp()))
    {
        auto val = constOp.getValue();
        if (auto floatAttr = llvm::dyn_cast<FloatAttr>(val))
        {
            double scalar = floatAttr.getValueAsDouble();
            return scalar;
        }
    }
    else if (auto truncOp = llvm::dyn_cast<arith::TruncFOp>(secondOperand.getDefiningOp()))
    {
        auto truncOperand = truncOp->getOperand(0);
        if (auto constOpInput = llvm::dyn_cast<arith::ConstantOp>(truncOperand.getDefiningOp()))
        {
            auto val = constOpInput.getValue();
            if (auto floatAttr = llvm::dyn_cast<FloatAttr>(val))
            {
                double scalar = floatAttr.getValueAsDouble();
                return scalar;
            }
        }
    }
    return 0.0;
}

static LogicalResult isGenericMaxReduce(linalg::LinalgOp genericOp)
{
    const unsigned int numLoops = genericOp.getNumLoops();
    if (const unsigned int numParallelLoops = genericOp.getNumParallelLoops(); numLoops == numParallelLoops)
    {
        // Return if no reduction in current op
        return failure();
    }

    // TODO: Figure out how second output of yield op for max reduce is used
    // Producer of linalg.yield op is arith.maximumf
    {
        const auto yieldOp = llvm::cast<linalg::YieldOp>(genericOp.getBlock()->getTerminator());
        const mlir::Value producerOutput = yieldOp->getOperand(0);
        if (const Operation *producer = producerOutput.getDefiningOp<arith::MaximumFOp>(); !producer)
        {
            return failure();
        }
    }

    return success();
}

static LogicalResult isGenericReduce(linalg::LinalgOp genericOp)
{
    unsigned int numLoops = genericOp.getNumLoops();
    unsigned int numParallelLoops = genericOp.getNumParallelLoops();
    if (numLoops == numParallelLoops)
    {
        // Return if no reduction in current op
        return failure();
    }

    // TODO: Figure out how second output of yield op for max reduce is used
    // Producer of linalg.yield op is arith.maximumf
    {
        const auto yieldOp = llvm::cast<linalg::YieldOp>(genericOp.getBlock()->getTerminator());
        const mlir::Value producerOutput = yieldOp->getOperand(0);
        if (const Operation *producer = producerOutput.getDefiningOp<arith::AddFOp>(); !producer)
        {
            return failure();
        }
    }

    return success();
}

static LogicalResult isGenericUpcast(linalg::LinalgOp genericOp)
{
    unsigned int numLoops = genericOp.getNumLoops();
    unsigned int numParallelLoops = genericOp.getNumParallelLoops();
    if (numLoops != numParallelLoops)
    {
        // Return if no reduction in current op
        return failure();
    }

    // Producer of linalg.yield op is arith.extf
    {
        const auto yieldOp = llvm::cast<linalg::YieldOp>(genericOp.getBlock()->getTerminator());
        const mlir::Value producerOutput = yieldOp->getOperand(0);
        if (const Operation *producer = producerOutput.getDefiningOp<arith::ExtFOp>(); !producer)
        {
            return failure();
        }
    }

    return success();
}

static LogicalResult isGenericDowncast(linalg::LinalgOp genericOp)
{
    unsigned int numLoops = genericOp.getNumLoops();
    unsigned int numParallelLoops = genericOp.getNumParallelLoops();
    if (numLoops != numParallelLoops)
    {
        // Return if no reduction in current op
        return failure();
    }

    // Producer of linalg.yield op is arith.truncf
    {
        const auto yieldOp = llvm::cast<linalg::YieldOp>(genericOp.getBlock()->getTerminator());
        const mlir::Value producerOutput = yieldOp->getOperand(0);
        if (const Operation *producer = producerOutput.getDefiningOp<arith::TruncFOp>(); !producer)
        {
            return failure();
        }
    }

    return success();
}

static LogicalResult isGenericScalarAdd(linalg::LinalgOp genericOp)
{
    unsigned int numLoops = genericOp.getNumLoops();
    unsigned int numParallelLoops = genericOp.getNumParallelLoops();
    if (numLoops != numParallelLoops)
    {
        // Return if no reduction in current op
        return failure();
    }

    if (genericOp.getNumDpsInputs() != 1)
    {
        return failure();
    }

    // Producer of linalg.yield op is arith.addf
    {
        const auto yieldOp = llvm::cast<linalg::YieldOp>(genericOp.getBlock()->getTerminator());
        const mlir::Value producerOutput = yieldOp->getOperand(0);
        if (const Operation *producer = producerOutput.getDefiningOp<arith::AddFOp>(); !producer)
        {
            return failure();
        }
    }

    return success();
}

static LogicalResult isGenericScalarMul(linalg::LinalgOp genericOp)
{
    unsigned int numLoops = genericOp.getNumLoops();
    unsigned int numParallelLoops = genericOp.getNumParallelLoops();
    if (numLoops != numParallelLoops)
    {
        // Return if no reduction in current op
        return failure();
    }

    if (genericOp.getNumDpsInputs() != 1)
    {
        return failure();
    }

    // Producer of linalg.yield op is arith.mulf
    {
        const auto yieldOp = llvm::cast<linalg::YieldOp>(genericOp.getBlock()->getTerminator());
        const mlir::Value producerOutput = yieldOp->getOperand(0);
        if (const Operation *producer = producerOutput.getDefiningOp<arith::MulFOp>(); !producer)
        {
            return failure();
        }
    }

    return success();
}

static LogicalResult isGenericScalarDiv(linalg::LinalgOp genericOp)
{
    unsigned int numLoops = genericOp.getNumLoops();
    unsigned int numParallelLoops = genericOp.getNumParallelLoops();
    if (numLoops != numParallelLoops)
    {
        // Return if no reduction in current op
        return failure();
    }

    if (genericOp.getNumDpsInputs() != 1)
    {
        return failure();
    }

    // Producer of linalg.yield op is arith.difv
    {
        const auto yieldOp = llvm::cast<linalg::YieldOp>(genericOp.getBlock()->getTerminator());
        const mlir::Value producerOutput = yieldOp->getOperand(0);
        if (const Operation *producer = producerOutput.getDefiningOp<arith::DivFOp>(); !producer)
        {
            return failure();
        }
    }

    return success();
}

static LogicalResult isGenericRsqrt(linalg::LinalgOp genericOp)
{
    unsigned int numLoops = genericOp.getNumLoops();
    unsigned int numParallelLoops = genericOp.getNumParallelLoops();
    if (numLoops != numParallelLoops)
    {
        // Return if no reduction in current op
        return failure();
    }

    if (genericOp.getNumDpsInputs() != 1)
    {
        return failure();
    }

    // Producer of linalg.yield op is math.rsqrt
    {
        const auto yieldOp = llvm::cast<linalg::YieldOp>(genericOp.getBlock()->getTerminator());
        const mlir::Value producerOutput = yieldOp->getOperand(0);
        if (const Operation *producer = producerOutput.getDefiningOp<math::RsqrtOp>(); !producer)
        {
            return failure();
        }
    }

    return success();
}

static LogicalResult isGenericRelu(linalg::LinalgOp genericOp)
{
    const unsigned int numLoops = genericOp.getNumLoops();
    if (const unsigned int numParallelLoops = genericOp.getNumParallelLoops(); numLoops != numParallelLoops)
    {
        // Return if reduction in current op, relu should not contain reductions
        return failure();
    }

    const auto yieldOp = llvm::cast<linalg::YieldOp>(genericOp.getBlock()->getTerminator());
    mlir::Value producerOutput;
    Operation *producer;

    // Producer of linalg.yield op is arith.select
    {
        producerOutput = yieldOp->getOperand(0);
        producer = producerOutput.getDefiningOp<arith::SelectOp>();
        if (!producer)
        {
            return failure();
        }
    }

    // Producer of arith.select op is arith.cmpf
    {
        producerOutput = producer->getOperand(0);
        producer = producerOutput.getDefiningOp<arith::CmpFOp>();
        if (!producer)
        {
            return failure();
        }

        auto cmp = llvm::cast<arith::CmpFOp>(producer);
        const mlir::arith::CmpFPredicate predicate = cmp.getPredicateAttr().getValue();
        auto cmpConstOperand = cmp->getOperand(1).getDefiningOp<arith::ConstantOp>();
        // Check if op is greater than predicate with float value of 0
        if (const auto cmpConstAsDouble = llvm::cast<mlir::FloatAttr>(cmpConstOperand.getValue()).getValueAsDouble();
            predicate != mlir::arith::CmpFPredicate::UGT || cmpConstAsDouble != 0.0)
        {
            return failure();
        }
    }

    return success();
}

static LogicalResult isGenericMatmulOp(linalg::LinalgOp genericOp)
{
    // Check that all loops are not parallel
    const unsigned numLoops = genericOp.getNumLoops();
    if (const unsigned numParallelLoops = genericOp.getNumParallelLoops(); numLoops == numParallelLoops)
    {
        return failure();
    }

    // Work back from linalg.yield and check body of LinalgOp.
    // The LinalgOp should yield the result of an arith.add.
    const auto yieldOp = llvm::cast<linalg::YieldOp>(genericOp.getBlock()->getTerminator());
    Operation *producer;

    {
        const Value producerOutput = yieldOp->getOperand(0);
        producer = producerOutput.getDefiningOp<arith::AddFOp>();
        if (!producer || producer->getNumOperands() == 0)
        {
            return failure();
        }
    }

    {
        const mlir::Value producerOutput1 = producer->getOperand(0);
        const mlir::Value producerOutput2 = producer->getOperand(1);
        const Operation *producer1 = producerOutput1.getDefiningOp<arith::MulFOp>();
        if (const Operation *producer2 = producerOutput2.getDefiningOp<arith::MulFOp>();
            (!producer1 && !producer2) || producer->getNumOperands() == 0)
        {
            return failure();
        }
    }

    return success();
}

static LogicalResult isGenericElemMulOp(linalg::LinalgOp genericOp)
{
    // Check that all loops are parallel
    const unsigned numLoops = genericOp.getNumLoops();
    if (const unsigned numParallelLoops = genericOp.getNumParallelLoops(); numLoops != numParallelLoops)
    {
        return failure();
    }

    // Work back from linalg.yield and check body of genericOp.
    // The genericOp should yield the result of an arith.mulf.
    // Producer of linalg.yield op is input
    {
        const auto yieldOp = llvm::cast<linalg::YieldOp>(genericOp.getBlock()->getTerminator());
        const Value producerOutput = yieldOp->getOperand(0);
        if (Operation *producer = producerOutput.getDefiningOp<arith::MulFOp>();
            !producer || producer->getNumOperands() == 0)
        {
            return failure();
        }
    }

    return success();
}

static LogicalResult isGenericAddOp(linalg::LinalgOp genericOp)
{
    // Check that all loops are parallel
    const unsigned numLoops = genericOp.getNumLoops();
    if (const unsigned numParallelLoops = genericOp.getNumParallelLoops(); numLoops != numParallelLoops)
    {
        return failure();
    }

    // Work back from linalg.yield and check body of genericOp.
    // The genericOp should yield the result of an arith.mulf.
    // Producer of linalg.yield op is input
    {
        const auto yieldOp = llvm::cast<linalg::YieldOp>(genericOp.getBlock()->getTerminator());
        const Value producerOutput = yieldOp->getOperand(0);
        if (Operation *producer = producerOutput.getDefiningOp<arith::AddFOp>();
            !producer || producer->getNumOperands() == 0)
        {
            return failure();
        }
    }

    return success();
}

static LogicalResult isGenericSubOp(linalg::LinalgOp genericOp)
{
    // Check that all loops are parallel
    const unsigned numLoops = genericOp.getNumLoops();
    if (const unsigned numParallelLoops = genericOp.getNumParallelLoops(); numLoops != numParallelLoops)
    {
        return failure();
    }

    // Work back from linalg.yield and check body of genericOp.
    // The genericOp should yield the result of an arith.mulf.
    // Producer of linalg.yield op is input
    {
        const auto yieldOp = llvm::cast<linalg::YieldOp>(genericOp.getBlock()->getTerminator());
        const Value producerOutput = yieldOp->getOperand(0);
        if (Operation *producer = producerOutput.getDefiningOp<arith::SubFOp>();
            !producer || producer->getNumOperands() == 0)
        {
            return failure();
        }
    }

    return success();
}

static LogicalResult isGenericMultiplyOp(linalg::LinalgOp genericOp)
{
    // Check that all loops are parallel
    const unsigned numLoops = genericOp.getNumLoops();
    if (const unsigned numParallelLoops = genericOp.getNumParallelLoops(); numLoops != numParallelLoops)
    {
        // return failure();
    }

    // Work back from linalg.yield and check body of genericOp.
    // The genericOp should yield the result of an arith.mulf.
    // Producer of linalg.yield op is input
    {
        const auto firstOp = &genericOp.getBlock()->front();
        if (auto producer = llvm::dyn_cast<arith::MulFOp>(firstOp); !producer || producer->getNumOperands() == 0)
        {
            return failure();
        }
    }

    return success();
}

static LogicalResult isGenericDivOp(linalg::LinalgOp genericOp)
{
    // Check that all loops are parallel
    const unsigned numLoops = genericOp.getNumLoops();
    if (const unsigned numParallelLoops = genericOp.getNumParallelLoops(); numLoops != numParallelLoops)
    {
        // return failure();
    }

    // Work back from linalg.yield and check body of genericOp.
    // The genericOp should yield the result of an arith.divf.
    // Producer of linalg.yield op is input
    {
        const auto firstOp = &genericOp.getBlock()->front();
        if (auto producer = llvm::dyn_cast<arith::DivFOp>(firstOp); !producer || producer->getNumOperands() == 0)
        {
            return failure();
        }
    }

    return success();
}

static std::string getOpName(linalg::LinalgOp &linalgOp)
{
    Block &block = *linalgOp.getBlock();
    Operation &blockOp = block.front();
    const unsigned int numLoops = linalgOp.getNumLoops();
    const unsigned int numParallelLoops = linalgOp.getNumParallelLoops();
    if (isGenericRelu(linalgOp).succeeded())
    {
        return "relu";
    }
    else if (isGenericMaxReduce(linalgOp).succeeded())
    {
        return "maxReduce";
    }
    else if (isGenericReduce(linalgOp).succeeded() && isGenericMatmulOp(linalgOp).failed())
    {
        return "reduce";
    }
    else if (isGenericUpcast(linalgOp).succeeded())
    {
        return "upcast";
    }
    else if (isGenericDowncast(linalgOp).succeeded())
    {
        return "downcast";
    }
    else if (isGenericScalarAdd(linalgOp).succeeded())
    {
        return "scalarAdd";
    }
    else if (isGenericScalarMul(linalgOp).succeeded())
    {
        return "scalarMul";
    }
    else if (isGenericScalarDiv(linalgOp).succeeded())
    {
        return "scalarDiv";
    }
    else if (isGenericRsqrt(linalgOp).succeeded())
    {
        return "rsqrt";
    }
    return llvm::TypeSwitch<mlir::Operation &, std::string>(blockOp)
        .Case<mlir::arith::MulFOp>([&](auto) { return "mul"; })
        .Case<mlir::arith::DivFOp>([&](auto) { return "div"; })
        .Case<mlir::arith::AddFOp>(
            [&](auto)
            {
                if (numLoops == numParallelLoops)
                {
                    return "add";
                }
                return "";
            })
        .Case<mlir::arith::SubFOp>([&](auto) { return "sub"; })
        .Case<mlir::math::ExpOp>([&](auto) { return "exp"; });
}

void insertJoiners(Table &finalTable, const std::shared_ptr<AnalysisScope> &scope, StageRewriter &rewriter,
                   func::FuncOp op, const std::vector<IndexVar> &loopOrder)
{
    const auto context = op.getContext();
    std::vector<IndexVar> joinerVars;

    for (auto contractIt : llvm::enumerate(scope->contractionType))
    {
        auto &pair = contractIt.value();
        auto &var = pair.first;
        auto &joinBundle = pair.second;
        const unsigned int varIdx = contractIt.index();
        auto dim = mlir::getAffineDimExpr(var.getId(), context);
        std::string format = "dense";
        joinerVars.push_back(var);

        bool connected = false;
        std::optional<TensorView> matchedTensor;
        std::optional<TensorView> matchedTensor1;

        // If two joiners exist for the same level, check if output from one goes to the other
        // In the case where one exists, merge the tensor views from the first joiner to the joined views in the second
        // joiner This essentially forces all the outputs from the first joiner to the second joiner
        if (scope->contractionOrder.at(var).size() > 1)
        {
            for (const auto &view : joinBundle.at(scope->contractionOrder.at(var).back()))
            {
                auto finalView = view;
                // We always want to avoid moving indirection cells like Ptr cells pointing to another cell or Dup cells
                // which duplicate cells from another tensorview of the same tensor
                while (finalTable[finalView][var] &&
                       (finalTable[finalView][var]->getLabel().rfind("Dup", 0) == 0 ||
                        finalTable[finalView][var]->getLabel().rfind("Ptr", 0) == 0))
                {
                    finalView = scope->cellIndirectionMap.at(var).at(finalView);
                }
                //                if (view.getValue().getDefiningOp<linalg::LinalgOp>() &&
                //                (!scope->innerCrds.count(var))) {
                //                    finalView = scope->varToView.at(
                //                            scope->equalVars.at(view.getUniqueVars().front()).getId());
                //                }
                for (const auto &view2 : joinBundle.at(scope->contractionOrder.at(var).front()))
                {
                    auto finalView2 = view2;
                    if (view2.getValue().getDefiningOp<linalg::LinalgOp>() && (!scope->innerCrds.count(var)))
                    {
                        finalView2 = scope->varToView.at(scope->equalVars.at(view2.getUniqueVars().front()).getId());
                    }
                    if (finalView.getName() == finalView2.getName())
                    {
                        connected = true;
                        matchedTensor = finalView2;
                        matchedTensor1 = finalView;
                        break;
                    }
                }
                if (connected)
                {
                    break;
                }
            }
        }

        std::set<TensorView> otherTensors;
        for (auto joinTypeIter : llvm::enumerate(scope->contractionOrder.at(var)))
        {
            auto joinType = joinTypeIter.value();
            unsigned int joinIdx = joinTypeIter.index();

            // If more than 1 joiner exists on that level and at least one output from the first joiner is tied to the
            // second joiner as an input, we force the second joiner to take all outputs from the first joiner
            if (connected)
            {
                scope->contractionType[var][joinType].erase(matchedTensor1.value());
                scope->contractionType[var][joinType].insert(matchedTensor.value());
                scope->contractionType[var][joinType].insert(otherTensors.begin(), otherTensors.end());
            }
            std::set<TensorView> tensorSet = scope->contractionType.at(var).at(joinType);
            sam::JoinerTypeAttr joinerType;
            std::vector<std::shared_ptr<Cell<std::pair<std::optional<Value>, Value>>>> cells;
            std::vector<std::shared_ptr<Cell<Value>>> valCells;
            std::vector<TensorView> joinerViews;
            const std::vector views(tensorSet.begin(), tensorSet.end());
            if (views.size() < 2)
            {
                continue;
            }

            if (varIdx == 0)
            {
                auto t = 0;
                llvm::DenseSet<mlir::Value> dupDetect;
                for (const auto &view : views)
                {
                    TensorView finalView = view;

                    // We always want to avoid moving indirection cells like Ptr cells pointing to another cell or Dup
                    // cells which duplicate cells from another tensorview of the same tensor
                    while (finalTable[finalView][var] &&
                           (finalTable[finalView][var]->getLabel().rfind("Dup", 0) == 0 ||
                            finalTable[finalView][var]->getLabel().rfind("Ptr", 0) == 0))
                    {
                        finalView = scope->cellIndirectionMap.at(var).at(finalView);
                    }
                    dupDetect.insert(finalView.getValue());
                }
                if (dupDetect.size() == 1)
                {
                    //                    continue;
                }
            }
            std::set<TensorView> seenViews;
            int count = 0;
            for (const auto &view : views)
            {

                if (!finalTable[view][var])
                {
                    std::cerr << "ERROR: Attempted to add a null cell to joiner (" << var << ")" << std::endl;
                    exit(1);
                }

                auto localUniqueVars = view.getUniqueVars();
                const auto indexVars =
                    llvm::map_to_vector(localUniqueVars, [=](const UniqueVar &a) { return a.getLogicalIndexVar(); });

                // To avoid duplicate views in a joiner view bundle
                bool added = false;
                TensorView finalView = view;

                // We always want to avoid moving indirection cells like Ptr cells pointing to another cell or Dup cells
                // which duplicate cells from another tensorview of the same tensor
                while (finalTable[finalView][var] &&
                       (finalTable[finalView][var]->getLabel().rfind("Dup", 0) == 0 ||
                        finalTable[finalView][var]->getLabel().rfind("Ptr", 0) == 0))
                {
                    finalView = scope->cellIndirectionMap.at(var).at(finalView);
                }

                // If current view is result of a linalg op and does not translate down to a sam spacc that would feed
                // into this joiner, then get the corresponding input tensor view with same index variable
                if ((finalView.getValue().getDefiningOp<linalg::LinalgOp>()) && (!scope->innerCrds.count(var)))
                {
                    // Map output tensor view to an input tensor view with the same index var
                    finalView = scope->varToView.at(scope->equalVars.at(view.getUniqueVars().front()).getId());
                    // Try to insert view in a set to avoid inserting duplicates which would lead to accessing a NULL
                    // cell as each cell corresponding to that view at the current index var gets moved to the joiner
                    if (seenViews.insert(finalView).second)
                    {
                        cells.push_back(std::move(finalTable[finalView][var]));
                        added = true;
                    }
                }
                else if (finalView.getValue().getDefiningOp<linalg::LinalgOp>() ||
                         finalView.getValue().getDefiningOp<tensor::ConcatOp>())
                {
                    // If the current view is the result of a linalg op and translates down to a sam spacc that is an
                    // input to the current joiner, then retrieve the appropriate tensor view with that spacc from
                    // scope->innerCrds as it's not guaranteed that this tensor view is the one with the spacc as it
                    // could be the outermost tensor view of a linalg result which contains the same index var We move
                    // the index cell for that tensor view containing the spacc along with the value cell as that will
                    // be used as the reference input to the joiner since the value for the tensor has already been
                    // materialized
                    if (seenViews.insert(finalView).second)
                    {
                        cells.push_back(std::move(finalTable[finalView][var]));
                        valCells.push_back(std::move(finalTable[finalView].getMutableValue()));
                        added = true;
                    }
                }
                else
                {
                    // For all other cases where the tensor view is already an input tensor view, we can simply move
                    // that original tensor view as it's already an input
                    if (seenViews.insert(finalView).second)
                    {
                        cells.push_back(std::move(finalTable[finalView][var]));
                        added = true;
                    }
                }
                // If a tensor view was added above meaning it was not a duplicate tensor view that we attempted to
                // insert, push that final tensor view whose cells got moved into the contraction map and joinerViews to
                // be used later for wiring the outputs to the joiner Once cell
                if (added)
                {
                    scope->contractionMap[var].insert(finalView);
                    joinerViews.push_back(finalView);
                }
            }


            // Create joiner attribute with the current join type which will be used in the joiner creation inside the
            // lambda
            joinerType = sam::JoinerTypeAttr::get(context, joinType);

            // Make sure that we have as many moved cells as joiner views associated with them
            assert(cells.size() == joinerViews.size());

            // Construct the lambda function for instantiating the joiner op, this will be encapsulated by a Once cell
            // wrapping the lambda with multiple outputs, which will then be encapsulated by a cell in our table with
            // a mapping lambda function to map which output goes where in each cell
            // One Once cell can be wrapped by multiple cells as long as the lambda function wiring which output belongs
            // to it exists and appropriately maps the correct output type that is required for a cell
            std::function lambda = [=, scope = scope, &rewriter](
                                       Table &table) -> std::vector<std::pair<std::optional<mlir::Value>, mlir::Value>>
            {
                std::vector<std::pair<std::optional<mlir::Value>, mlir::Value>> joinerPairs;

                const auto crdAttr = getStreamAttr(context, sam::StreamKind::Crd, dim, "", format);

                // We don't care about the format of this intermediate tensor so we default it
                std::string form;
                llvm::SmallVector<mlir::Value> input_crds;
                llvm::SmallVector<mlir::Value> input_refs;
                llvm::SmallVector<mlir::Type> output_refs;
                std::vector allViews = joinerViews;

                for (const auto &joinerView : llvm::zip(cells, allViews))
                {
                    const auto &joinerPair = std::get<0>(joinerView);
                    auto view = std::get<1>(joinerView);
                    auto indexVars = llvm::map_to_vector(
                        view.getUniqueVars(), [&](const UniqueVar &localVar)
                        { return mlir::getAffineDimExpr(localVar.getLogicalIndexVar().getId(), context); });
                    const unsigned int dimIndex = llvm::find(indexVars, dim) - indexVars.begin();
                    std::string tensorName = scope->tensorViewNames.at(view);
                    const auto refAttr = getStreamAttr(context, sam::StreamKind::Ref, dim, tensorName, form);
                    const auto pair = joinerPair->get(table);
                    const mlir::Value ref = pair.first.value();
                    const auto crd = pair.second;
                    output_refs.push_back(ref.getType());
                    input_refs.push_back(ref);
                    input_crds.push_back(crd);
                }
                auto output_crd = RankedTensorType::get({}, IndexType::get(context), crdAttr);
                auto joiner = rewriter.create<sam::SamJoiner>(op->getLoc(), output_crd, output_refs, input_crds,
                                                              input_refs, joinerType);
                for (auto out_ref : joiner.getOutputRefs())
                {
                    joinerPairs.emplace_back(out_ref, joiner.getOutputCrd());
                }
                scope->indexVarMap[dim][0] = joiner.getOutputRefs()[0];
                scope->indexVarMap[dim][1] = joiner.getOutputCrd();
                return joinerPairs;
            };
            std::vector<std::string> labels;
            std::string opLabel = "Joiner";
            for (const auto &cell : cells)
            {
                labels.emplace_back(cell->getLabel());
            }

            std::ostringstream ss;
            std::copy(labels.begin(), labels.end(), std::ostream_iterator<std::string>(ss, ", "));
            auto once = make_once(lambda);
            for (auto viewIter : llvm::enumerate(joinerViews))
            {
                const auto &view = viewIter.value();
                const unsigned int viewIdx = viewIter.index();
                // Conversion function to retrieve the corresponding crd, ref joiner bundle for the current view
                std::function convertFunc = [=](std::vector<std::pair<std::optional<mlir::Value>, mlir::Value>> bundles)
                    -> std::pair<std::optional<mlir::Value>, mlir::Value> { return bundles.at(viewIdx); };
                std::string inputLst = ss.str();
                // Removing extra delimiter and space, calling pop_back twice
                if (!inputLst.empty())
                {
                    inputLst.pop_back();
                    inputLst.pop_back();
                }
                std::string joinName = joinerType.getValue() == sam::JoinerType::Intersect ? "Intersect" : "Union";
                std::ostringstream varStr;
                varStr << var;
                // Gets joiner label with the cell labels and joiner name
                std::string joinLabel = joinName.append("_") + varStr.str().append("(") +
                    inputLst.append(")").append("[") + std::to_string(viewIdx).append("]");
                // Creates new unique_ptr cell of joiner once structure with the appropriate conversion function to
                // retrieve the bundle for the current view
                auto newCell =
                    make_unique_cell<std::pair<std::optional<mlir::Value>, mlir::Value>>(once, convertFunc, joinLabel);

                // Input views for input tensors expect sam reference streams whereas intermediate view results expect a
                // sam value stream Updates val lambda for current view to retrieve the joiner output corresponding to
                // the filtered value stream
                if (view.getValue().getDefiningOp<linalg::LinalgOp>() ||
                    view.getValue().getDefiningOp<tensor::ConcatOp>())
                {
                    std::function convertVal =
                        [=](std::vector<std::pair<std::optional<mlir::Value>, mlir::Value>> bundles)
                    { return bundles.at(viewIdx).first.value(); };
                    auto newValCell = make_unique_cell<mlir::Value>(once, convertVal, joinLabel);
                    finalTable[view].setValue(std::move(newValCell));
                }
                // Updates the original cell that was replaced with the joiner block to now store the cell corresponding
                // to the joiner op
                if (matchedTensor.has_value() && !(view == matchedTensor.value()))
                {
                    otherTensors.insert(view);
                    auto name1 = view.getName();
                }

                // Set the appropriate table cell at the current var to own the newly created cell
                finalTable[view][var] = std::move(newCell);
            }
        }
    }
}

LogicalResult fuseOpsInDispatchGroup(func::FuncOp op,
                                     llvm::DenseMap<mlir::Value, std::shared_ptr<FusedCIN>> &metastageMap,
                                     const std::shared_ptr<AnalysisScope> &scope, StageRewriter &rewriter,
                                     bool calculateHeuristic)
{
    MLIRContext *context = op.getContext();
    rewriter.setInsertionPoint(&op.getBlocks().front().getOperations().front());

    // Retrieve tensor view for returned tensor by calling makeInterface on its FusedCIN
    mlir::Operation *returnOperation = &op.getBlocks().front().getOperations().back();
    const TensorView returnedTensorView =
        metastageMap.at(cast<func::ReturnOp>(*returnOperation).getOperand(0))->makeInterface();

    //    const TensorView returnedTensorView1 =
    //            metastageMap.at(cast<func::ReturnOp>(*returnOperation).getOperand(1))->makeInterface();
    //    const TensorView returnedTensorView2 =
    //            metastageMap.at(cast<func::ReturnOp>(*returnOperation).getOperand(2))->makeInterface();

    SmallVector<Value> tensors;

    llvm::DenseMap<mlir::Value, std::string> tensorNames;

    unsigned int outCount = 0;
    // Get all input tensors
    for (auto &operation : op.getBlocks().front().getOperations())
    {
        SmallVector<Value> operands;
        if (isa<func::ReturnOp>(operation))
        {
            break;
        }
        if (isa<arith::ConstantOp>(operation))
        {
            tensors.push_back(operation.getResult(0));
            continue;
        }
        if (isa<tensor::EmptyOp>(operation))
        {
            continue;
        }
        unsigned int numInputOperands = 0;
        // unsigned int numInputOperands = operation.getNumOperands() - operation.getNumResults();
        if (auto concatOp = llvm::dyn_cast<tensor::ConcatOp>(operation))
        {
            numInputOperands = concatOp.getInputs().size();
        }
        else if (auto linalgOp = llvm::dyn_cast<linalg::LinalgOp>(operation))
        {
            numInputOperands = linalgOp.getDpsInputOperands().size();
        }
        for (unsigned int i = 0; i < numInputOperands; i++)
        {
            auto genericOperand = operation.getOperand(i).getDefiningOp<LinalgOp>();
            auto concatOperand = operation.getOperand(i).getDefiningOp<tensor::ConcatOp>();
            if (!genericOperand && !concatOperand)
            {
                tensors.push_back(operation.getOpOperand(i).get());
            }
            else
            {
                const auto &operand = operation.getOpOperand(i).get();
                if (!tensorNames.contains(operand))
                {
                    tensorNames[operand] = "tOut" + std::to_string(outCount++);
                }
                for (auto viewIter : llvm::enumerate(metastageMap.at(operand)->getTensorViews()))
                {
                    const auto &view = viewIter.value();
                    const unsigned int viewIdx = viewIter.index();
                    scope->tensorViewNames[view] = tensorNames[operand] + "-" + std::to_string(viewIdx);
                    view.setName(scope->tensorViewNames[view]);
                }
            }
        }
    }
    tensorNames[returnedTensorView.getValue()] = "tOut" + std::to_string(outCount);
    scope->tensorViewNames[returnedTensorView] = tensorNames[returnedTensorView.getValue()] + "-" + std::to_string(0);
    returnedTensorView.setName(scope->tensorViewNames[returnedTensorView]);

    llvm::DenseSet<mlir::Value> filter;
    llvm::SmallVector<mlir::Value> allTensors;
    for (auto tens : tensors)
    {
        if (!filter.contains(tens))
        {
            filter.insert(tens);
            allTensors.push_back(tens);
        }
    }
    tensors = std::move(allTensors);

    llvm::DenseMap<mlir::Value, unsigned int> tensorIds;

    // Assign each unique mlir::Value representing input tensors or constant tensors a unique ID
    unsigned int uniqueId = 0;
    for (auto tensor : tensors)
    {
        unsigned int tensorId = 0;
        if (!tensorIds.contains(tensor))
        {
            tensorId = uniqueId++;
            tensorIds.insert(std::pair(tensor, tensorId));
            tensorNames.insert(std::pair(tensor, "t" + std::to_string(tensorId)));
            if (!metastageMap.contains(tensor))
            {
                continue;
            }
            for (auto viewIter : llvm::enumerate(metastageMap.at(tensor)->getTensorViews()))
            {
                const auto &view = viewIter.value();
                const unsigned int viewIdx = viewIter.index();
                scope->tensorViewNames[view] = tensorNames[tensor] + "-" + std::to_string(viewIdx);
                view.setName(scope->tensorViewNames[view]);
            }
        }
        else
        {
            tensorId = tensorIds[tensor];
        }
    }
    scope->tensorIds = tensorIds;
    scope->tensorNames = tensorNames;

    // Get contraction map
    returnedTensorView.getContractions();

    // Construct lowering table where each cell in the table stores lambdas representing pipelines for computing SAM
    // reference/coord pairs for each logical index var for each tensor view as well as how to compute every value for
    // every tensor view in the program including the output/intermediate tensors
    std::vector<IndexVar> loopOrder = scope->getLoopOrder();
    // std::cout << "Loop Order: " << std::endl;
    // for (auto loop : loopOrder)
    // {
    // std::cout << loop << std::endl;
    // }
    scope->effLoopOrder = loopOrder;

    // Get loops in the scope of every tensor, this allows us to ignore reduction variables from different paths
    auto red = returnedTensorView.getScopedLoops();

    // Construct table populating cells defining how to compute sam components related to every tensor
    auto t = returnedTensorView.constructTable(loopOrder);
    //    t = returnedTensorView1.constructTable(loopOrder);
    //    t = returnedTensorView2.constructTable(loopOrder);

    // Print table before adding joiners
    t.dump(scope, loopOrder);
    // Insert joiners by modifying affected cells
    insertJoiners(t, scope, rewriter, op, loopOrder);
    // Print table after adding joiners
    t.dump(scope, loopOrder);

    CSEVisitor cseVisitor;
    scope->einsumMap.at(returnedTensorView)->accept(cseVisitor);

    SoftmaxPatternMatcher softmaxPatternMatcher(scope);
    scope->einsumMap.at(returnedTensorView)->accept(softmaxPatternMatcher);

    // Update the AST in the scope if pattern was matched
    if (softmaxPatternMatcher.patternMatched)
    {
        scope->einsumMap.at(returnedTensorView) = softmaxPatternMatcher.optimizedRoot;
    }

    LayernormPatternMatcher layernorm_pattern_matcher(scope);
    scope->einsumMap.at(returnedTensorView)->accept(layernorm_pattern_matcher);

    // Update the AST in the scope if pattern was matched
    if (layernorm_pattern_matcher.patternMatched)
    {
        scope->einsumMap.at(returnedTensorView) = layernorm_pattern_matcher.optimizedRoot;
    }

    GetExprVisitor exprVisitor(returnedTensorView);
    scope->einsumMap.at(returnedTensorView)->accept(exprVisitor);
    std::cout << std::endl;

    EinsumTreeVisitor einsumVisitor;

    scope->einsumMap.at(returnedTensorView)->accept(einsumVisitor);
    einsumVisitor.printTree();

    auto finalStreamVal = t.getValue(returnedTensorView);

    if (calculateHeuristic)
    {
        CSEVisitor cseVisitor;
        scope->einsumMap.at(returnedTensorView)->accept(cseVisitor);

        SoftmaxPatternMatcher softmaxPatternMatcher(scope);
        scope->einsumMap.at(returnedTensorView)->accept(softmaxPatternMatcher);

        // Update the AST in the scope if pattern was matched
        if (softmaxPatternMatcher.patternMatched)
        {
            scope->einsumMap.at(returnedTensorView) = softmaxPatternMatcher.optimizedRoot;
        }

        LayernormPatternMatcher layernormPatternMatcher(scope);
        scope->einsumMap.at(returnedTensorView)->accept(layernormPatternMatcher);

        // Update the AST in the scope if pattern was matched
        if (layernormPatternMatcher.patternMatched)
        {
            scope->einsumMap.at(returnedTensorView) = layernormPatternMatcher.optimizedRoot;
        }

        // Print out full fused expression
        GetExprVisitor exprVisitor(returnedTensorView);
        scope->einsumMap.at(returnedTensorView)->accept(exprVisitor);
        std::cout << std::endl;

        // Collect all index variables for all tensors in fused expression along with their level format
        CollectIndicesVisitor indexCollectVisitor;
        scope->einsumMap.at(returnedTensorView)->accept(indexCollectVisitor);

        auto indexData = indexCollectVisitor.indexData;

        // for (auto &elem: indexData) {
        //     std::cout << "Var: " << indexVocab[elem.first.getId()] << std::endl;
        //     std::cout << "OG Var: " << elem.first << std::endl;
        //     for (auto &tensor: elem.second) {
        //         std::cout << "Tensor: " << tensor.first << std::endl;
        //         std::cout << "Format: " << tensor.second << std::endl;
        //     }
        // }

        // for (auto &elem: indexData) {
        //     std::cout << "Var: " << indexVocab[elem.first.getId()] << std::endl;
        //     std::cout << "OG Var: " << elem.first << std::endl;
        //     for (auto &tensor: elem.second) {
        //         std::cout << "Tensor: " << tensor.first << std::endl;
        //         std::cout << "Format: " << tensor.second << std::endl;
        //     }
        // }

        // If we only want to estimate the number of ops and bytes accessed, we prompt user input for
        // necessary dimensions of all index variables or an estimate of the number of coordinates based
        // on an estimated of percent nnz given by the user
        // For each dense level, user inputs the size of the dimension
        // For each sparse level, user inputs a tuple with the dimension along with a percentage nnz
        // For merged levels (intersect, union), user inputs a tuple with the dimension along with the percentage nnz
        std::map<IndexVar, double> indexVarDims;
        std::map<IndexVar, double> indexVarSparsity;
        for (auto &elem : indexData)
        {
            auto varStr = indexVocab[elem.first.getId()];

            std::cout << "Var: " << varStr << std::endl;

            if (elem.second.size() > 1 || (elem.second.size() == 1 && elem.second.front().second == "compressed"))
            {
                double dimCount = 0.0;
                double percentNnz = 0.0;

                if (elem.second.size() > 1)
                {
                    bool allDense = true;
                    for (auto &tensor : elem.second)
                    {
                        if (tensor.second == "compressed")
                        {
                            allDense = false;
                            break;
                        }
                    }
                    if (allDense)
                    {
                        double dimCount = 0.0;
                        std::cout << "Enter the dimension of " << varStr << ": ";
                        std::cin >> dimCount;
                        indexVarDims[elem.first] = dimCount;
                        continue;
                    }
                    std::ostringstream ss;
                    for (auto tensorIter : llvm::enumerate(elem.second))
                    {
                        const auto &tensor = tensorIter.value();
                        ss << tensor.first;
                        if (tensorIter.index() != elem.second.size() - 1)
                        {
                            ss << ", ";
                        }
                    }
                    std::cout << "Enter the size of " << varStr
                              << " and the estimated percent of non-intersected coords of tensors " << ss.str() << ": "
                              << std::endl;
                }
                else
                {
                    std::cout << "Enter the size of " << varStr
                              << " and the estimated percent of nnz coords: " << std::endl;
                }
                std::cout << "Dimension: ";
                std::cin >> dimCount;
                std::cout << "Percent nnz: ";
                std::cin >> percentNnz;
                double numCoords = dimCount * (percentNnz / 100.0);
                indexVarDims[elem.first] = numCoords;
                indexVarSparsity[elem.first] = percentNnz / 100.0;
            }
            else
            {
                double dimCount = 0.0;
                std::cout << "Enter the dimension of " << varStr << ": ";
                std::cin >> dimCount;
                indexVarDims[elem.first] = dimCount;
            }
        }
        for (auto &elem : indexVarDims)
        {
            std::cout << "Var: " << indexVocab[elem.first.getId()] << std::endl;
            std::cout << "Number of coords: " << elem.second << std::endl;
        }
        std::cout << std::endl;

        StatVisitor statVisitor(indexVarDims, indexVarSparsity);
        scope->einsumMap.at(returnedTensorView)->accept(statVisitor);

        // Print op count
        std::cout << "Op count: " << scope->einsumMap.at(returnedTensorView)->getOpsCount() << std::endl;
        // Print bytes count, multiply by 4 for single precision float
        // std::cout << "Bytes count: " << 4 * statVisitor.bytesCount << std::endl;
        std::cout << "Bytes count: " << scope->einsumMap.at(returnedTensorView)->getBytesCount() << std::endl;
        return success();
    }

    // Emit level writers for each logical index var in the output tensor
    for (const auto &var : returnedTensorView.getUniqueVars())
    {
        IndexVar tempVar = var.getLogicalIndexVar();
        rewriter.create<sam::SamFiberWrite>(op->getLoc(), t.at(returnedTensorView, tempVar).second,
                                            sam::WriteType::Crd);
    }

    // Emit level writer for the final output value
    rewriter.create<sam::SamFiberWrite>(op->getLoc(), finalStreamVal, sam::WriteType::Val);

    std::set<std::pair<std::string, std::vector<unsigned int>>> seenTensors;
    // Print out metadata information that is needed for end to end script
    std::cout << "\n// BEGIN GENERATED MLIR CODE" << std::endl;
    for (auto tensorPair : scope->modeOrders)
    {
        auto tensorType = mlir::dyn_cast<RankedTensorType>(tensorPair.first.getValue().getType());
        const std::string tensorName = scope->tensorViewNames[tensorPair.first];

        if (!seenTensors.insert(std::make_pair(tensorName, tensorPair.second)).second)
        {
            continue;
        }

        std::cout << "// " << scope->tensorViewNames[tensorPair.first] << ": ";
        std::cout << "dims(" << tensorPair.first.getUniqueVars().size() << "), ";
        std::cout << "vars(";
        for (auto varIter : llvm::enumerate(tensorPair.first.getUniqueVars()))
        {
            const auto &var = varIter.value();
            unsigned int idx = varIter.index();
            auto indexVar = var.getLogicalIndexVar();
            std::cout << indexVar;
            if (idx != tensorPair.first.getUniqueVars().size() - 1)
            {
                std::cout << ", ";
            }
        }
        std::vector<std::string> formats;
        std::vector<int> shape;
        std::cout << "), mode_order(";
        for (auto modeIter : llvm::enumerate(tensorPair.second))
        {
            auto mode = modeIter.value();
            unsigned int idx = modeIter.index();
            std::string format;
            if (const auto enc = getSparseTensorEncoding(tensorPair.first.getValue().getType()))
            {
                const auto lvlType = enc.getLvlTypes();
                format = toMLIRString(lvlType[idx]);
            }
            else
            {
                format = "dense";
            }
            formats.emplace_back(format);
            std::cout << mode;
            if (idx != tensorPair.second.size() - 1)
            {
                std::cout << ", ";
            }
        }
        std::cout << "), format(";
        for (const auto formatIter : llvm::enumerate(formats))
        {
            const std::string &format = formatIter.value();
            const unsigned int idx = formatIter.index();

            std::cout << format;
            if (idx != formats.size() - 1)
            {
                std::cout << ", ";
            }
        }

        std::cout << "), shape(";

        for (const auto dimIter : llvm::enumerate(tensorType.getShape()))
        {
            const auto &dim = dimIter.value();
            const unsigned int idx = dimIter.index();

            std::cout << dim;
            if (idx != tensorType.getShape().size() - 1)
            {
                std::cout << ", ";
            }
        }
        std::cout << ")";
        if (auto blockArg = llvm::dyn_cast<mlir::BlockArgument>(tensorPair.first.getValue()))
        {
            std::cout << ", BlockArg(" << blockArg.getArgNumber() << ")" << std::endl;
        }
        else
        {
            std::cout << std::endl;
        }
    }

    // Change function's output type to be the final stream value's type
    // Done to allow CSE pass to eliminate original ops that were lowered
    const auto types = op.getBody().getArgumentTypes();
    op.setType(mlir::FunctionType::get(context, types, finalStreamVal.getType()));
    rewriter.setInsertionPoint(&op.getBlocks().front().getOperations().back());
    const auto newReturn = rewriter.create<func::ReturnOp>(rewriter.getUnknownLoc(), finalStreamVal);
    rewriter.replaceOp(&op.getBlocks().front().getOperations().back(), newReturn);

    //    sortTopologically(op->getBlock());

    return success();
}

void constructInputLambdas(StageRewriter &rewriter, mlir::func::FuncOp funcOp, const Value &inputVal,
                           const std::shared_ptr<AnalysisScope> &scope,
                           llvm::DenseMap<Value, std::shared_ptr<FusedCIN>> &metastageMap)
{
    const auto input = llvm::dyn_cast<RankedTensorType>(inputVal.getType());
    const auto context = funcOp.getContext();
    if (!input)
    {
        return;
    }
    const unsigned int numLevels = input.getShape().size();
    const auto fusedCin = std::make_shared<FusedCIN>(scope);

    const Value finalInput = inputVal;

    fusedCin->tensorValue = inputVal;
    fusedCin->makeInterface = [=, &rewriter, scope = scope]() -> TensorView
    {
        std::vector<UniqueVar> uniqueVars;

        for (unsigned int i = 0; i < numLevels; i++)
        {
            auto newVar = scope->getNewUniqueVar(fusedCin);
            uniqueVars.push_back(newVar);
        }
        fusedCin->uniqueVars.push_back(uniqueVars);
        auto tensorView = TensorView(uniqueVars, inputVal);

        //        llvm::outs() << "Tensor Name: " << scope->tensorViewNames.at(tensorView) << "\n";
        for (unsigned int i = 0; i < uniqueVars.size() - 1; i++)
        {

            //            if (i > 1) {
            //                continue;
            //            }
            // Keeps original order of input tensors
            // FIXME: Comment this back in when you want a hardcoded partial order for testing
            scope->addPartialOrder(uniqueVars[i], uniqueVars[i + 1]);
            // if
            std::string format;
            const unsigned int varIndex = 0;
            if (const auto enc = getSparseTensorEncoding(inputVal.getType()))
            {
                const auto lvlType = enc.getLvlTypes();
                format = toMLIRString(lvlType[varIndex]);
            }
            else
            {
                format = "dense";
            }

            // if (format == "compressed" && scope->firstTime) {
            // llvm::outs() << "Found sparse: " << inputVal << "\n";
            // scope->addPartialOrder(uniqueVars[i+1], uniqueVars[i]);
            // scope->firstTime = false;
            // } else {
            // scope->addPartialOrder(uniqueVars[i], uniqueVars[i + 1]);
            // }
        }

        for (const auto &var : uniqueVars)
        {
            scope->varToView[var.getId()] = tensorView;
        }
        fusedCin->tensorViews.push_back(tensorView);

        std::function getScopedLoops = [=, scope = scope] { return std::set<IndexVar>(); };
        tensorView.setScopedLoopFunction(getScopedLoops);
        const std::function getContractions = [=, scope = scope]()
        {
            for (const auto &var : uniqueVars)
            {
                auto indexVar = var.getLogicalIndexVar();
                scope->equalVars[var] = var;
            }
        };
        tensorView.setContractionFunction(getContractions);

        auto constructTable = [=, scope = scope, &rewriter](const std::vector<IndexVar> &loopOrder) -> Table
        {
            Table graphTable;
            Column column;
            // const std::string tensorName = scope->tensorViewNames.at(tensorView);
            auto localUniqueVars = tensorView.getUniqueVars();
            const auto indexVars =
                llvm::map_to_vector(localUniqueVars, [=](const UniqueVar &a) { return a.getLogicalIndexVar(); });
            std::vector<IndexVar> returnedTensorIndex(indexVars.begin(), indexVars.end());
            auto tensorPair = std::make_pair(scope->tensorNames.at(tensorView.getValue()), returnedTensorIndex);
            if (scope->tableCache.count(tensorPair))
            {
                auto cachedTensor = scope->tableCache.at(tensorPair);
                for (const auto &loop : loopOrder)
                {
                    std::function cellFunc = [=, scope = scope](Table &table) { return table.at(cachedTensor, loop); };
                    std::string cellName = "Dup";
                    std::ostringstream ss;
                    ss << loop;
                    auto dupCellName =
                        cellName.append("(") + scope->tensorViewNames.at(cachedTensor) + ", " + ss.str() + ")";
                    scope->cellIndirectionMap[loop][tensorView] = cachedTensor;
                    auto dupCell =
                        make_unique_cell<std::pair<std::optional<mlir::Value>, mlir::Value>>(cellFunc, dupCellName);
                    column[loop] = std::move(dupCell);
                }
                std::function valCellFunc = [=, scope = scope](Table &table) { return table.getValue(cachedTensor); };
                std::string cellName = "Dup";
                auto dupCellName = cellName.append("(") + scope->tensorViewNames.at(cachedTensor) + ")";
                auto dupCell = make_unique_cell<mlir::Value>(valCellFunc, dupCellName);
                column.setValue(std::move(dupCell));
                graphTable[tensorView] = std::move(column);
                tensorView.setCompleted();
                return graphTable;
            }

            for (const auto varIter : llvm::enumerate(loopOrder))
            {
                const auto &var = varIter.value();
                if (const auto it = llvm::find(indexVars, var); it == indexVars.end())
                {
                    continue;
                }
                const unsigned int varIdx = varIter.index();
                const auto indexIt = llvm::find(indexVars, var);
                const unsigned int mode = indexIt - indexVars.begin();
                scope->modeOrders[tensorView].push_back(mode);
            }

            std::string tensorName = scope->tensorNames.at(inputVal);
            tensorName += "-";
            for (auto modeIter : llvm::enumerate(scope->modeOrders[tensorView]))
            {
                const unsigned int mode = modeIter.value();
                const unsigned int idx = modeIter.index();
                tensorName += std::to_string(mode);
                if (idx != scope->modeOrders[tensorView].size() - 1)
                {
                    tensorName += ",";
                }
            }
            scope->tensorViewNames[tensorView] = tensorName;

            for (const auto varIter : llvm::enumerate(loopOrder))
            {
                const auto var = varIter.value();
                const unsigned int varIdx = varIter.index();

                // If concat dim should be ignored for this input tensor view, add redirect cell
                if (shouldIgnoreConcatDimVar(var, tensorView, scope))
                {
                    if (varIdx > 0)
                    {
                        std::function redirectFunc = [=, &rewriter](Table &table)
                        { return table.at(tensorView, loopOrder[varIdx - 1]); };
                        std::string cellName = "Redirect";
                        std::ostringstream ss;
                        ss << loopOrder.at(varIdx - 1);
                        auto dupCellName =
                            cellName + "(" + scope->tensorViewNames.at(tensorView) + ", " + ss.str() + ")";
                        auto redirectCell = make_unique_cell<std::pair<std::optional<mlir::Value>, mlir::Value>>(
                            redirectFunc, dupCellName);
                        column[var] = std::move(redirectCell);
                    }
                    else
                    {
                        std::function genRoot =
                            [=, &rewriter](Table &table) -> std::pair<std::optional<mlir::Value>, mlir::Value>
                        {
                            constexpr unsigned int mode = 0;
                            auto dummyExpr = mlir::getAffineDimExpr(0, context);
                            const auto genRefAttr =
                                getStreamAttr(context, sam::StreamKind::Ref, dummyExpr, "", "", mode);
                            auto gen = rewriter.create<sam::SamGenerator>(
                                funcOp->getLoc(), RankedTensorType::get({}, IndexType::get(context), genRefAttr));
                            return std::make_pair(gen.getOutputRef(), gen.getOutputRef());
                        };
                        std::string rootName = "root";
                        auto redirectCell =
                            make_unique_cell<std::pair<std::optional<mlir::Value>, mlir::Value>>(genRoot, rootName);
                        column[var] = std::move(redirectCell);
                    }
                    continue;
                }

                if (const auto it = llvm::find(indexVars, var); it == indexVars.end())
                {
                    continue;
                }

                const auto indexIt = llvm::find(indexVars, var);
                const unsigned int mode = indexIt - indexVars.begin();
                //                scope->modeOrders[tensorView].push_back(mode);
                if (!scope->indexToView.count(var))
                {
                    scope->indexToView[var] = tensorView;
                }

                std::function cellFunc =
                    [=, &rewriter, scope = scope](Table &table) -> std::pair<std::optional<mlir::Value>, mlir::Value>
                {
                    std::string format;
                    const unsigned int varIndex = mode;
                    if (const auto enc = getSparseTensorEncoding(inputVal.getType()))
                    {
                        const auto lvlType = enc.getLvlTypes();
                        format = toMLIRString(lvlType[varIndex]);
                    }
                    else
                    {
                        format = "dense";
                    }
                    unsigned int dimId = var.getId();
                    const auto dimExpr = mlir::getAffineDimExpr(dimId, context);
                    auto dimIdAttr = IntegerAttr::get(IndexType::get(context), dimId);
                    const auto refAttr =
                        getStreamAttr(context, sam::StreamKind::Ref, dimExpr, tensorName, format, mode);
                    const auto crdAttr =
                        getStreamAttr(context, sam::StreamKind::Crd, dimExpr, tensorName, format, mode);

                    mlir::Value prevRef;
                    if (varIdx > 0)
                    {
                        const auto &prevVar = loopOrder[varIdx - 1];
                        const auto &prevCell = table.at(tensorView, prevVar);
                        prevRef = prevCell.first.value();
                    }
                    else
                    {
                        const auto genRefAttr =
                            getStreamAttr(context, sam::StreamKind::Ref, dimExpr, tensorName, "", mode);
                        auto gen = rewriter.create<sam::SamGenerator>(
                            funcOp->getLoc(), RankedTensorType::get({}, IndexType::get(context), genRefAttr));
                        prevRef = gen.getOutputRef();
                    }
                    auto lookup = rewriter.create<sam::SamFiberLookup>(
                        funcOp->getLoc(), RankedTensorType::get({}, IndexType::get(context), refAttr),
                        RankedTensorType::get({}, IndexType::get(context), crdAttr), prevRef, dimIdAttr,
                        IntegerAttr::get(IndexType::get(context), scope->tensorIds[inputVal]));
                    auto oldDim = mlir::getAffineDimExpr(var.getId(), context);
                    if (!scope->indexVarMap.contains(oldDim))
                    {
                        scope->indexVarMap[oldDim].push_back(lookup.getOutputRef());
                        scope->indexVarMap[oldDim].push_back(lookup.getOutputCrd());
                    }
                    return std::make_pair(std::make_optional(lookup.getOutputRef()), lookup.getOutputCrd());
                };
                std::string prevRefName;
                if (!(var == loopOrder.front()))
                {
                    std::ostringstream ss;
                    ss << loopOrder[varIdx - 1];
                    prevRefName = "Prev(" + scope->tensorViewNames.at(tensorView) + ", " + ss.str() + ")";
                }
                else
                {
                    prevRefName = "root";
                }
                std::string opLabel = "LScan";
                std::string label = opLabel.append("(") + prevRefName.append(")");
                auto cell =
                    make_unique_cell<std::pair<std::optional<mlir::Value>, mlir::Value>>(std::move(cellFunc), label);
                column[var] = std::move(cell);
            }
            std::function comp = [=, &rewriter, scope = scope](Table &table) -> mlir::Value
            {
                // return scope->tensorStreamMap[tensorView];
                const auto &refCrdPair = table.at(tensorView, loopOrder.back());
                // Ref input to arrayVal block set to last indexvar in loopOrder
                // Last indexVar is always a reference to the last local op loop order
                const auto ref = refCrdPair.first.value();
                // Set default value stream shape (scalar), can be changed by vectorize pass
                const SmallVector<int64_t> shape(1, 1);
                auto shapeLst = DenseI64ArrayAttr::get(context, ArrayRef(shape));

                const auto enc = mlir::sam::getSamStreamEncoding(ref.getType());

                const auto valAttr = getStreamAttr(context, sam::StreamKind::Val, mlir::getAffineDimExpr(0, context),
                                                   enc.getTensor().str(), "");
                // Instantiate arrayval block
                auto valRead = rewriter.create<sam::SamArrayVal>(
                    funcOp->getLoc(),
                    RankedTensorType::get({}, llvm::dyn_cast<RankedTensorType>(inputVal.getType()).getElementType(),
                                          valAttr),
                    ref, shapeLst);

                scope->regionOpToStream[inputVal] = valRead.getOutputVal();

                //                scope->effLoopOrder = loopOrder;

                return valRead.getOutputVal();
            };
            // TODO: Intermediate tensors don't have a tensor name assigned
            std::ostringstream ss;
            ss << loopOrder.back();
            std::string valCellLabel =
                "ValRead(" + scope->tensorNames.at(tensorView.getValue()) + ", " + ss.str() + ")";

            auto compCell = make_unique_cell<mlir::Value>(comp, valCellLabel);
            column.setValue(std::move(compCell));

            scope->tableCache[tensorPair] = tensorView;
            graphTable[tensorView] = std::move(column);
            return graphTable;
        };
        tensorView.setConstructTableFunction(constructTable);

        return tensorView;
    };

    metastageMap[finalInput] = fusedCin;
}

// Construct meta and comp lambda for output tensor in op and stash in map
void constructLambdas(StageRewriter &rewriter, mlir::Operation *operation, const std::shared_ptr<AnalysisScope> &scope,
                      llvm::DenseMap<Value, std::shared_ptr<FusedCIN>> &metastageMap,
                      llvm::SmallVector<mlir::Value> &tensorLst)
{
    linalg::LinalgOp linalgOp = llvm::dyn_cast<linalg::LinalgOp>(operation);
    tensor::ExpandShapeOp expandOp = llvm::dyn_cast<tensor::ExpandShapeOp>(operation);
    tensor::ConcatOp concatOp = llvm::dyn_cast<tensor::ConcatOp>(operation);

    if (expandOp)
    {
        const auto context = expandOp.getContext();
    }
    else if (concatOp)
    {
        const auto context = concatOp.getContext();
        const OperandRange inputs = concatOp.getInputs();
        const ResultRange outputs = concatOp.getODSResults(0);
        const uint64_t concatDim = concatOp.getDim();
        const Value result = outputs.front();
        const auto fusedCin = std::make_shared<FusedCIN>(scope, operation);
        fusedCin->tensorValue = result;

        fusedCin->makeInterface = [=, scope = scope, &metastageMap, &tensorLst, &rewriter]() -> TensorView
        {
            std::vector<TensorView> opTensors;
            std::vector<UniqueVar> outputVars;
            const auto outputType = llvm::dyn_cast<RankedTensorType>(result.getType());
            const unsigned int numLevels = outputType.getShape().size();

            // Create tensor views for all inputs
            for (auto operandIter : llvm::enumerate(inputs))
            {
                auto operand = operandIter.value();
                unsigned int operandIndex = operandIter.index();
                tensorLst.push_back(operand);
                opTensors.push_back(metastageMap.at(operand)->makeInterface());
            }

            // Create output variables
            for (unsigned int i = 0; i < numLevels; i++)
            {
                auto newVar = scope->getNewUniqueVar(fusedCin);
                outputVars.push_back(newVar);
            }

            // Register concat output variables into the global loop graph and partial order
            // so they appear in the global loop order and fusion table
            for (unsigned int i = 0; i < outputVars.size(); ++i)
            {
                scope->addNode(outputVars[i]);
                if (i + 1 < outputVars.size())
                {
                    scope->addPartialOrder(outputVars[i], outputVars[i + 1]);
                }
            }

            auto returnedTensorView = TensorView(outputVars, result);

            // Set up getScopedLoops function
            std::function getScopedLoops = [=, scope = scope]
            {
                std::set<IndexVar> loopSet;

                // For concat, all dimensions are preserved
                for (const auto &var : outputVars)
                {
                    returnedTensorView.insertLoop(var.getLogicalIndexVar());
                    loopSet.insert(var.getLogicalIndexVar());
                }

                // Propagate loops to input tensors
                for (const auto &tensor : opTensors)
                {
                    for (auto &loop : returnedTensorView.getLoops())
                    {
                        tensor.insertLoop(loop);
                    }
                    auto operandSet = tensor.getScopedLoops();
                    loopSet.insert(operandSet.begin(), operandSet.end());
                }

                return loopSet;
            };
            returnedTensorView.setScopedLoopFunction(getScopedLoops);

            // Set up constructTable function
            auto constructTable = [=, scope = scope, &rewriter](const std::vector<IndexVar> &loopOrder) -> Table
            {
                Table graphTable;
                Column column;

                // Track this concat operation in the scope now that we have the IndexVar mapping
                AnalysisScope::ConcatInfo concatInfo;
                concatInfo.outputView = returnedTensorView;
                concatInfo.concatDim = concatDim;
                if (concatDim < outputVars.size())
                {
                    concatInfo.concatDimUniqueVar = outputVars[concatDim];
                }
                concatInfo.inputViews = opTensors;
                scope->concatOperations.push_back(concatInfo);

                // Construct tables for all input tensors
                for (const auto &opTensor : opTensors)
                {
                    auto table = opTensor.constructTable(loopOrder);
                    graphTable.extend(table);
                }

                // Create AST node for concat operation
                // For simplicity, we'll treat concat as a special binary op
                // that concatenates tensors along a dimension
                // std::shared_ptr<ASTNode> concatNode;
                std::vector<std::shared_ptr<ASTNode>> inputNodes;
                for (const auto &opTensor : opTensors)
                {
                    if (scope->einsumMap.count(opTensor))
                    {
                        inputNodes.push_back(scope->einsumMap.at(opTensor));
                    }
                    else
                    {
                        inputNodes.push_back(std::make_shared<TensorNode>(opTensor));
                    }
                }

                // Create FuncOpNode for concat operation
                std::shared_ptr<ASTNode> concatNode =
                    std::make_shared<FuncOpNode>("concat", inputNodes, returnedTensorView,
                                                 static_cast<int>(concatDim) // Convert uint64_t to int for concatDim
                    );
                scope->einsumMap[returnedTensorView] = concatNode;

                // Handle mode orders
                for (const auto varIter : llvm::enumerate(loopOrder))
                {
                    const auto &var = varIter.value();
                    auto outputIndexVars =
                        llvm::map_to_vector(outputVars, [=](const UniqueVar &a) { return a.getLogicalIndexVar(); });

                    if (const auto it = llvm::find(outputIndexVars, var); it != outputIndexVars.end())
                    {
                        const unsigned int mode = it - outputIndexVars.begin();
                        scope->modeOrders[returnedTensorView].push_back(mode);
                    }
                }

                // Assign a stable tensor view name for the concat output using its mode order
                {
                    std::string tvName = scope->tensorNames.at(returnedTensorView.getValue());
                    tvName += "-";
                    for (auto moIter : llvm::enumerate(scope->modeOrders[returnedTensorView]))
                    {
                        const unsigned int mode = moIter.value();
                        const unsigned int idx = moIter.index();
                        tvName += std::to_string(mode);
                        if (idx != scope->modeOrders[returnedTensorView].size() - 1)
                        {
                            tvName += ",";
                        }
                    }
                    scope->tensorViewNames[returnedTensorView] = tvName;
                }

                // Build SamConcat once-cell to provide both output crds and value stream
                std::function concatFunc = [=, scope = scope,
                                            &rewriter](Table &table) -> std::tuple<mlir::Value, mlir::ResultRange>
                {
                    llvm::SmallVector<mlir::Value> in_crds;
                    llvm::SmallVector<mlir::Value> in_vals;

                    // Gather flattened input coord streams and value streams
                    for (const auto &tensorView : opTensors)
                    {
                        auto localUniqueVars = tensorView.getUniqueVars();
                        for (const auto &uvar : localUniqueVars)
                        {
                            auto iv = uvar.getLogicalIndexVar();
                            auto rc = table.at(tensorView, iv);
                            in_crds.push_back(rc.second);
                        }
                        in_vals.push_back(table.getValue(tensorView));
                    }

                    // Result types for out_crds (one per rank)
                    llvm::SmallVector<Type> resultTypes;
                    llvm::SmallVector<Type> outCrdTypes;
                    for (unsigned j = 0; j < numLevels; ++j)
                    {
                        auto dimExpr = mlir::getAffineDimExpr(outputVars[j].getLogicalIndexVar().getId(), context);
                        auto crdAttr = getStreamAttr(context, sam::StreamKind::Crd, dimExpr, "t_concat", "");
                        outCrdTypes.push_back(RankedTensorType::get({}, IndexType::get(context), crdAttr));
                        resultTypes.push_back(outCrdTypes.back());
                    }
                    // Result type for out_val
                    auto valAttr = getStreamAttr(context, sam::StreamKind::Val, mlir::getAffineDimExpr(0, context),
                                                 "t_concat", "");
                    auto outValType = RankedTensorType::get({}, outputType.getElementType(), valAttr);
                    resultTypes.push_back(outValType);

                    // Attributes
                    auto axisAttr = IntegerAttr::get(IndexType::get(context), concatDim);
                    auto rankAttr = IntegerAttr::get(IndexType::get(context), numLevels);
                    auto dimLenAttr = IntegerAttr::get(IndexType::get(context), outputType.getShape()[concatDim]);

                    auto concat = rewriter.create<sam::SamConcat>(concatOp->getLoc(), resultTypes, ValueRange(in_crds),
                                                                  ValueRange(in_vals), axisAttr, rankAttr, dimLenAttr);

                    return std::make_tuple(concat.getOutVals()[0], concat.getOutCrds());
                };

                auto once = make_once<std::tuple<mlir::Value, mlir::ResultRange>>(concatFunc);

                // Populate per-dimension coord cells from SamConcat outputs
                auto outputIndexVars =
                    llvm::map_to_vector(outputVars, [=](const UniqueVar &a) { return a.getLogicalIndexVar(); });

                for (const auto varIter : llvm::enumerate(loopOrder))
                {
                    const auto &var = varIter.value();
                    if (auto it = llvm::find(outputIndexVars, var); it != outputIndexVars.end())
                    {
                        const unsigned dimPos = it - outputIndexVars.begin();
                        std::function outRefCrdFunc = [=](std::tuple<mlir::Value, mlir::ResultRange> bundle) -> RefCrd
                        {
                            mlir::Value crd = std::get<1>(bundle)[dimPos];
                            // For now, only returns crd and value pair for last level since value can act as reference
                            // in downstream ops
                            if (dimPos == numLevels - 1)
                            {
                                // For the last level, return (val, crd) where 'val' acts as the reference
                                mlir::Value val = std::get<0>(bundle);
                                return std::make_pair(val, crd);
                            }
                            // For other levels, return (crd, crd)
                            return std::make_pair(std::nullopt, crd);
                        };
                        std::ostringstream ss;
                        ss << var;
                        std::string outLabel = "Concat[" + std::to_string(dimPos) + "](" + ss.str() + ")";
                        auto outputCell = make_unique_cell<std::pair<std::optional<mlir::Value>, mlir::Value>>(
                            once, outRefCrdFunc, outLabel);
                        column[var] = std::move(outputCell);
                    }
                }

                // Value cell from SamConcat out_val
                std::string valLabel = "Concat(";
                for (size_t i = 0; i < opTensors.size(); ++i)
                {
                    valLabel += scope->tensorNames[opTensors[i].getValue()];
                    if (i < opTensors.size() - 1)
                    {
                        valLabel += ", ";
                    }
                }
                valLabel += ")";

                std::function valFunc = [=](std::tuple<mlir::Value, mlir::ResultRange> bundle) -> mlir::Value
                { return std::get<0>(bundle); };
                auto compCell = make_unique_cell<mlir::Value>(once, valFunc, valLabel);
                column.setValue(std::move(compCell));

                graphTable[returnedTensorView] = std::move(column);

                // Ensure concat output view is included in scope->views for printing order
                if (llvm::find(scope->views, returnedTensorView) == scope->views.end())
                {
                    scope->views.push_back(returnedTensorView);
                }
                return graphTable;
            };
            returnedTensorView.setConstructTableFunction(constructTable);

            // Set up getContractions function
            std::function getContractions = [=, scope = scope]()
            {
                // Process contractions for input tensors
                for (size_t i = 0; i < opTensors.size(); ++i)
                {
                    opTensors[i].getContractions();
                }

                // For concat, dimensions other than concat dimension should be equivalent
                for (unsigned int dimIdx = 0; dimIdx < numLevels; ++dimIdx)
                {
                    if (dimIdx != concatDim)
                    {
                        // Mark variables at this dimension as equivalent across inputs
                        UniqueVar *firstVar = nullptr;
                        for (size_t inputIdx = 0; inputIdx < opTensors.size(); ++inputIdx)
                        {
                            auto &vars = opTensors[inputIdx].getUniqueVars();
                            if (dimIdx < vars.size())
                            {
                                if (!firstVar)
                                {
                                    firstVar = &vars[dimIdx];
                                    //                                    scope->equalVars[*firstVar] = *firstVar;
                                }
                                else
                                {
                                    scope->markEqual(*firstVar, vars[dimIdx]);
                                    //                                    scope->equalVars[vars[dimIdx]] =
                                    //                                    scope->equalVars[*firstVar];
                                }
                            }
                        }

                        // Map output variable to first input's equivalent variable
                        if (firstVar && dimIdx < outputVars.size())
                        {
                            scope->markEqual(outputVars[dimIdx], *firstVar);
                            //                            scope->equalVars[outputVars[dimIdx]] =
                            //                            scope->equalVars[*firstVar];
                        }
                    }
                }
            };
            returnedTensorView.setContractionFunction(getContractions);

            // Store variable to view mappings
            for (const auto &var : outputVars)
            {
                scope->varToView[var.getId()] = returnedTensorView;
            }

            fusedCin->uniqueVars.push_back(outputVars);
            fusedCin->tensorViews.push_back(returnedTensorView);

            return returnedTensorView;
        };

        metastageMap[result] = fusedCin;
    }
    else if (linalgOp)
    {
        const auto context = linalgOp.getContext();
        SmallVector<Value> inputs = linalgOp.getDpsInputs();
        const unsigned int numDpsInputs = linalgOp.getNumDpsInputs();
        const unsigned int numLoops = linalgOp.getNumLoops();
        const unsigned int numParallelLoops = linalgOp.getNumParallelLoops();
        ValueRange outputs = linalgOp.getDpsInits();
        SmallVector<mlir::AffineMap> indexingMaps = linalgOp.getIndexingMapsArray();
        const SmallVector<mlir::utils::IteratorType> iterators = linalgOp.getIteratorTypesArray();
        auto dpsInputOperands = linalgOp.getDpsInputOperands();
        SmallVector<Type> resultTypes = TypeRange(ValueRange(outputs));

        const unsigned int numDims = static_cast<const int>(indexingMaps[0].getNumDims());

        llvm::SmallVector<AffineExpr> dimExprs;

        for (unsigned int i = 0; i < numDims; i++)
        {
            dimExprs.push_back(getAffineDimExpr(i, context));
        }

        for (auto resultIter : llvm::enumerate(linalgOp->getOpResults()))
        {
            Value result = resultIter.value();
            const unsigned int resultIndex = resultIter.index();
            auto fusedCin = std::make_shared<FusedCIN>(scope, operation);
            fusedCin->tensorValue = result;
            fusedCin->makeInterface = [=, scope = scope, &metastageMap, &tensorLst, &rewriter]() -> TensorView
            {
                // std::set<mlir::AffineExpr> commonExprs;
                llvm::DenseMap<mlir::AffineExpr, UniqueVar> exprToUniqueVar;
                auto linalgOp = result.getDefiningOp<linalg::LinalgOp>();

                tensorLst.push_back(result);
                SmallVector<Value> tensors;
                std::vector<TensorView> opTensors;
                // std::vector<std::shared_ptr<FusedCIN>> interfaces;
                SmallVector<AffineMap> indexingMap;
                for (auto operandIter : llvm::enumerate(inputs))
                {
                    auto operand = operandIter.value();
                    unsigned int operandIndex = operandIter.index();
                    tensorLst.push_back(operand);
                    opTensors.push_back(metastageMap.at(operand)->makeInterface());
                    tensors.push_back(operand);
                    indexingMap.push_back(indexingMaps[operandIndex]);
                }
                indexingMap.push_back(indexingMaps[inputs.size() + resultIndex]);

                auto outputType = llvm::dyn_cast<RankedTensorType>(result.getType());
                unsigned int numLevels = outputType.getShape().size();

                std::vector<UniqueVar> outputVars;
                for (unsigned int i = 0; i < numLevels; i++)
                {
                    auto newVar = scope->getNewUniqueVar(fusedCin);
                    newVar.op = linalgOp;
                    outputVars.push_back(newVar);
                }

                for (auto dimIter : llvm::enumerate(dimExprs))
                {
                    unsigned int dimCount = 0;
                    mlir::AffineExpr dim = dimIter.value();
                    for (unsigned int i = 0; i < indexingMap.size(); i++)
                    {
                        // Iterate through indexing maps
                        const std::vector<UniqueVar> &vars = opTensors[i].getUniqueVars();
                        for (auto exprIter : llvm::enumerate(indexingMap[i].getResults()))
                        {
                            auto expr = exprIter.value();
                            unsigned int index = exprIter.index();
                            if (auto constDim = llvm::dyn_cast<mlir::AffineConstantExpr>(expr))
                            {
                                expr = dimExprs[index];
                            }
                            if (expr == dim)
                            {
                                if (!exprToUniqueVar.contains(expr))
                                {
                                    exprToUniqueVar[expr] = vars[index];
                                }
                            }
                        }
                    }
                }

                // llvm::outs() <<
                std::vector<UniqueVar> opLoops;
                for (auto dim : dimExprs)
                {
                    opLoops.push_back(exprToUniqueVar[dim]);
                }

                for (auto &view : opTensors)
                {
                    if (!scope->tensorLoopOrders.count(view))
                    {
                        scope->tensorLoopOrders[view] = opLoops;
                    }
                }

                auto returnedTensorView = TensorView(outputVars, result);

                std::function getScopedLoops = [=, scope = scope]
                {
                    std::set<IndexVar> redSet;

                    auto it = std::find(iterators.begin(), iterators.end(), utils::IteratorType::reduction);
                    std::optional<IndexVar> reduction;
                    if (it != iterators.end())
                    {
                        unsigned int reductionIndex = it - iterators.begin();
                        auto newDim = mlir::getAffineDimExpr(reductionIndex, context);
                        reduction =
                            exprToUniqueVar.at(mlir::getAffineDimExpr(reductionIndex, context)).getLogicalIndexVar();
                        scope->reducedVars.insert(reduction.value());
                    }

                    for (auto &loopDim : dimExprs)
                    {
                        const auto indexVar = exprToUniqueVar.at(loopDim).getLogicalIndexVar();
                        if (reduction.has_value() && reduction.value() == indexVar)
                        {
                            continue;
                        }
                        returnedTensorView.insertLoop(indexVar);
                    }
                    for (const auto &tensor : opTensors)
                    {
                        for (auto &loop : returnedTensorView.getLoops())
                        {
                            tensor.insertLoop(loop);
                        }
                        if (reduction.has_value())
                        {
                            tensor.insertLoop(reduction.value());
                        }
                        auto operandSet = tensor.getScopedLoops();
                        redSet.insert(operandSet.begin(), operandSet.end());
                    }

                    return redSet;
                };
                returnedTensorView.setScopedLoopFunction(getScopedLoops);

                auto constructTable = [=, scope = scope, &metastageMap,
                                       &rewriter](const std::vector<IndexVar> &loopOrder) -> Table
                {
                    auto linalgOp = result.getDefiningOp<linalg::LinalgOp>();
                    Table graphTable;
                    Column column;
                    for (const auto &opTensor : opTensors)
                    {
                        auto table = opTensor.constructTable(loopOrder);
                        graphTable.extend(table);
                    }

                    // Process binary ops
                    if (opTensors.size() > 1)
                    {
                        std::vector<std::shared_ptr<ASTNode>> inputNodes;
                        // Check if the map already contains tensor view in case it was an intermediate op result view
                        if (scope->einsumMap.count(opTensors.at(0)))
                        {
                            inputNodes.push_back(scope->einsumMap.at(opTensors.at(0)));
                        }
                        else
                        {
                            // Create new tensor node if it doesn't already exist
                            inputNodes.push_back(std::make_shared<TensorNode>(opTensors.at(0)));
                        }

                        // Same as before, check if the map already contains tensor view in case it was an intermediate
                        // op result view
                        if (scope->einsumMap.count(opTensors.at(1)))
                        {
                            inputNodes.push_back(scope->einsumMap.at(opTensors.at(1)));
                        }
                        else
                        {
                            inputNodes.push_back(std::make_shared<TensorNode>(opTensors.at(1)));
                        }

                        // Create new binary ast node
                        auto tensorA = inputNodes.at(0);
                        auto tensorB = inputNodes.at(1);
                        auto localUniqueVarOrder = opLoops;
                        std::vector<IndexVar> localLoopOrder;
                        for (const auto &uniqueVar : localUniqueVarOrder)
                        {
                            localLoopOrder.push_back(uniqueVar.getLogicalIndexVar());
                        }
                        sortByPartialOrder(loopOrder, localLoopOrder);
                        if (isGenericMatmulOp(linalgOp).succeeded() || isGenericScalarMul(linalgOp).succeeded() ||
                            isGenericElemMulOp(linalgOp).succeeded())
                        {
                            // If op is a matmul or hadamard product, create multiply binary op

                            // repeated code, refactor later
                            auto iterator_types = iterators;
                            std::optional<IndexVar> reductionVar;
                            auto it =
                                std::find(iterator_types.begin(), iterator_types.end(), utils::IteratorType::reduction);
                            if (it != iterator_types.end())
                            {
                                unsigned int reductionIndex = it - iterator_types.begin();
                                auto newDim = mlir::getAffineDimExpr(reductionIndex, context);
                                reductionVar = exprToUniqueVar.at(mlir::getAffineDimExpr(reductionIndex, context))
                                                   .getLogicalIndexVar();
                            }

                            auto multiply = std::make_shared<BinaryOpNode>(tensorA, tensorB, "*", returnedTensorView,
                                                                           localLoopOrder, reductionVar);
                            scope->einsumMap[returnedTensorView] = multiply;
                        }
                        else if (isGenericAddOp(linalgOp).succeeded())
                        {
                            // If op is an add, create add binary op
                            auto add = std::make_shared<BinaryOpNode>(tensorA, tensorB, "+", returnedTensorView,
                                                                      localLoopOrder);
                            scope->einsumMap[returnedTensorView] = add;
                        }
                        else if (isGenericDivOp(linalgOp).succeeded())
                        {
                            // If op is an div, create div binary op
                            auto div = std::make_shared<BinaryOpNode>(tensorA, tensorB, "/", returnedTensorView,
                                                                      localLoopOrder);
                            scope->einsumMap[returnedTensorView] = div;
                        }
                        else if (isGenericSubOp(linalgOp).succeeded())
                        {
                            // If op is a sub, create sub binary op
                            auto sub = std::make_shared<BinaryOpNode>(tensorA, tensorB, "-", returnedTensorView,
                                                                      localLoopOrder);
                            scope->einsumMap[returnedTensorView] = sub;
                        }
                    }
                    else
                    {
                        std::shared_ptr<ASTNode> tensor;
                        auto opName = getOpName(linalgOp);
                        if (scope->einsumMap.count(opTensors.at(0)))
                        {
                            tensor = scope->einsumMap.at(opTensors.at(0));
                        }
                        else
                        {
                            tensor = std::make_shared<TensorNode>(opTensors.at(0));
                        }
                        // Create new binary ast node
                        auto unary = std::make_shared<UnaryOpNode>(tensor, returnedTensorView, opName);
                        scope->einsumMap[returnedTensorView] = unary;
                    }

                    std::vector<IndexVar> returnedTensorIndex;
                    for (auto &var : returnedTensorView.getUniqueVars())
                    {
                        returnedTensorIndex.push_back(var.getLogicalIndexVar());
                    }
                    auto tensorPair =
                        std::make_pair(scope->tensorNames.at(returnedTensorView.getValue()), returnedTensorIndex);

                    if (scope->tableCache.count(tensorPair))
                    {
                        auto cachedTensor = scope->tableCache.at(tensorPair);
                        for (const auto &loop : loopOrder)
                        {
                            std::function cellFunc = [=, scope = scope](Table &table)
                            { return table.at(cachedTensor, loop); };
                            scope->cellIndirectionMap[loop][returnedTensorView] = cachedTensor;
                            std::string cellName = "Dup";
                            std::ostringstream ss;
                            ss << loop;
                            auto dupCellName =
                                cellName.append("(") + scope->tensorViewNames.at(cachedTensor) + ", " + ss.str() + ")";
                            auto dupCell = make_unique_cell<std::pair<std::optional<mlir::Value>, mlir::Value>>(
                                cellFunc, dupCellName);
                            column[loop] = std::move(dupCell);
                        }
                        std::function valCellFunc = [=, scope = scope](Table &table)
                        { return table.getValue(cachedTensor); };
                        std::string cellName = "Dup";
                        auto dupCellName = cellName.append("(") + scope->tensorViewNames.at(cachedTensor) + ")";
                        auto dupCell = make_unique_cell<mlir::Value>(valCellFunc, dupCellName);
                        column.setValue(std::move(dupCell));
                        graphTable[returnedTensorView] = std::move(column);
                        returnedTensorView.setCompleted();
                        return graphTable;
                    }
                    auto localUniqueVarOrder = opLoops;
                    std::vector<IndexVar> localLoopOrder;
                    for (const auto &uniqueVar : localUniqueVarOrder)
                    {
                        localLoopOrder.push_back(uniqueVar.getLogicalIndexVar());
                    }
                    sortByPartialOrder(loopOrder, localLoopOrder);

                    llvm::SmallVector<mlir::AffineExpr> reductionVars;
                    auto iterator_types = iterators;
                    std::optional<IndexVar> reductionVar;
                    auto it = std::find(iterator_types.begin(), iterator_types.end(), utils::IteratorType::reduction);
                    if (it != iterator_types.end())
                    {
                        unsigned int reductionIndex = it - iterator_types.begin();
                        auto newDim = mlir::getAffineDimExpr(reductionIndex, context);
                        reductionVar =
                            exprToUniqueVar.at(mlir::getAffineDimExpr(reductionIndex, context)).getLogicalIndexVar();
                        scope->reductionVars.push_back(reductionVar.value());
                        reductionVars.push_back(newDim);
                    }

                    for (auto varIter : llvm::enumerate(loopOrder))
                    {
                        const auto &var = varIter.value();
                        int varIdx = (int)varIter.index();

                        // Check if this variable should be ignored for the current tensor view
                        if (shouldIgnoreConcatDimVar(var, returnedTensorView, scope))
                        {
                            // Create a redirect cell to the previous dimension (cell above)
                            if (varIdx > 0)
                            {
                                std::function cellFunc = [=, scope = scope](Table &table)
                                { return table.at(returnedTensorView, loopOrder[varIdx - 1]); };
                                std::string cellName = "Redirect";
                                std::ostringstream ss;
                                ss << loopOrder[varIdx - 1];
                                auto dupCellName = cellName + "(" + scope->tensorViewNames.at(returnedTensorView) +
                                    ", " + ss.str() + ")";
                                auto dupCell = make_unique_cell<std::pair<std::optional<mlir::Value>, mlir::Value>>(
                                    cellFunc, dupCellName);
                                column[var] = std::move(dupCell);
                            }
                            else
                            {
                                // For the first dimension, synthesize a root reference
                                std::function genRoot =
                                    [=, &rewriter](Table &table) -> std::pair<std::optional<mlir::Value>, mlir::Value>
                                {
                                    constexpr unsigned int mode = 0;
                                    auto dummyExpr = mlir::getAffineDimExpr(0, context);
                                    const auto genRefAttr =
                                        getStreamAttr(context, sam::StreamKind::Ref, dummyExpr, "", "", mode);
                                    auto gen = rewriter.create<sam::SamGenerator>(
                                        linalgOp->getLoc(),
                                        RankedTensorType::get({}, IndexType::get(context), genRefAttr));
                                    return std::make_pair(gen.getOutputRef(), gen.getOutputRef());
                                };
                                std::string rootName = "root";
                                auto redirectCell =
                                    make_unique_cell<std::pair<std::optional<mlir::Value>, mlir::Value>>(genRoot,
                                                                                                         rootName);
                                column[var] = std::move(redirectCell);
                            }
                            continue;
                        }

                        const unsigned int currDimIdx = llvm::find(loopOrder, var) - loopOrder.begin();
                        const unsigned int lastDimIdx =
                            llvm::find(loopOrder, localLoopOrder.back()) - loopOrder.begin();

                        // If current loop var is after the last local op loop var in the
                        // global loop vars, then redirect to last local op loop var's cell
                        if (currDimIdx > lastDimIdx)
                        {
                            for (const auto &tensorView : opTensors)
                            {
                                if (tensorView.getCompleted())
                                {
                                    continue;
                                }
                                std::function cellFunc = [=, scope = scope](Table &table)
                                { return table.at(tensorView, loopOrder[varIdx - 1]); };
                                std::string cellName = "Redirect";
                                std::ostringstream ss;
                                ss << loopOrder[varIdx - 1];
                                //                            auto dupCellName = cellName.append("(") +
                                //                            scope->tensorNames.at(tensorView.getValue()) +
                                auto dupCellName = cellName.append("(") + scope->tensorViewNames.at(tensorView) + ", " +
                                    ss.str() + ")";
                                auto dupCell = make_unique_cell<std::pair<std::optional<mlir::Value>, mlir::Value>>(
                                    cellFunc, dupCellName);
                                graphTable[tensorView][var] = std::move(dupCell);
                            }
                            // TODO: Determine cell name for this case
                            std::function cellFunc = [=, scope = scope](Table &table)
                            { return table.at(returnedTensorView, loopOrder[varIdx - 1]); };
                            std::string cellName = "Redirect";
                            std::ostringstream ss;
                            ss << loopOrder[varIdx - 1];
                            //                        auto dupCellName = cellName + "(" +
                            //                        scope->tensorNames.at(returnedTensorView.getValue()) +
                            auto dupCellName =
                                cellName + "(" + scope->tensorViewNames.at(returnedTensorView) + ", " + ss.str() + ")";
                            auto dupCell = make_unique_cell<std::pair<std::optional<mlir::Value>, mlir::Value>>(
                                cellFunc, dupCellName);
                            column[var] = std::move(dupCell);
                            continue;
                        }

                        // Insert repeat cells
                        for (auto tensorViewIter : llvm::enumerate(opTensors))
                        {
                            const unsigned int tensorIdx = tensorViewIter.index();
                            const auto &tensorView = tensorViewIter.value();
                            if (tensorView.getCompleted())
                            {
                                continue;
                            }

                            // Skip if this is a concat dimension variable for this tensor view
                            if (shouldIgnoreConcatDimVar(var, tensorView, scope))
                            {
                                // Fill a redirect cell for this tensor view at current var
                                if (varIdx > 0)
                                {
                                    std::function redirectFunc = [=, &rewriter](Table &table)
                                    { return table.at(tensorView, loopOrder[varIdx - 1]); };
                                    std::string cellName = "Redirect";
                                    std::ostringstream ss;
                                    ss << loopOrder.at(varIdx - 1);
                                    auto dupCellName =
                                        cellName + "(" + scope->tensorViewNames.at(tensorView) + ", " + ss.str() + ")";
                                    auto redirectCell =
                                        make_unique_cell<std::pair<std::optional<mlir::Value>, mlir::Value>>(
                                            redirectFunc, dupCellName);
                                    graphTable[tensorView][var] = std::move(redirectCell);
                                }
                                else
                                {
                                    // Synthesize root for first dimension
                                    std::function genRoot =
                                        [=,
                                         &rewriter](Table &table) -> std::pair<std::optional<mlir::Value>, mlir::Value>
                                    {
                                        constexpr unsigned int mode = 0;
                                        auto dummyExpr = mlir::getAffineDimExpr(0, context);
                                        const auto genRefAttr =
                                            getStreamAttr(context, sam::StreamKind::Ref, dummyExpr, "", "", mode);
                                        auto gen = rewriter.create<sam::SamGenerator>(
                                            linalgOp->getLoc(),
                                            RankedTensorType::get({}, IndexType::get(context), genRefAttr));
                                        return std::make_pair(gen.getOutputRef(), gen.getOutputRef());
                                    };
                                    std::string rootName = "root";
                                    auto redirectCell =
                                        make_unique_cell<std::pair<std::optional<mlir::Value>, mlir::Value>>(genRoot,
                                                                                                             rootName);
                                    graphTable[tensorView][var] = std::move(redirectCell);
                                }
                                continue;
                            }
                            //                        std::string tensorName =
                            //                        scope->tensorNames.at(tensorView.getValue());
                            std::string tensorName = scope->tensorViewNames.at(tensorView);
                            auto tensorIndexVars = llvm::map_to_vector(
                                tensorView.getUniqueVars(), [&](const UniqueVar &a) { return a.getLogicalIndexVar(); });

                            // TODO: Remove extra repeats on intermediate output cells
                            if (const auto varIt = llvm::find(tensorIndexVars, var); varIt == tensorIndexVars.end())
                            {
                                // Check if this variable should be ignored because it's a concat dimension
                                if (shouldIgnoreConcatDimVar(var, tensorView, scope))
                                {
                                    continue;
                                }

                                // Patch for redirecting intermediate result output refCrd to output tensor in current
                                // op
                                if (tensorView.getValue().getDefiningOp<linalg::LinalgOp>())
                                {
                                    continue;
                                }
                                const auto dimExpr = mlir::getAffineDimExpr(var.getId(), context);
                                const auto refAttr =
                                    getStreamAttr(context, sam::StreamKind::Ref, dimExpr, tensorName, "");

                                const TensorView &repeatSigTensorView = opTensors[(tensorIdx + 1) % opTensors.size()];
                                // TODO: Temporary patch, need to test
                                bool found = true;
                                auto otherTensorUniqueVars = repeatSigTensorView.getUniqueVars();
                                auto otherTensorIndexVars = llvm::map_to_vector(
                                    otherTensorUniqueVars, [=](const UniqueVar &a) { return a.getLogicalIndexVar(); });
                                if (llvm::find(otherTensorIndexVars, var) == otherTensorIndexVars.end())
                                {
                                    found = false;
                                }
                                auto redVars = llvm::map_to_vector(scope->uniqueRedVars, [=](UniqueVar &a)
                                                                   { return a.getLogicalIndexVar(); });
                                auto loops = opTensors[0].getLoops();
                                std::vector<IndexVar> currentScopeLoopOrder(loops.begin(), loops.end());
                                sortByPartialOrder(loopOrder, currentScopeLoopOrder);

                                // If the current variable is the first in the loop order, emit a root for this cell


                                // If the current var is not in the current op's loop order, redirect cell to point to
                                // cell above it
                                if ((llvm::find(currentScopeLoopOrder, var) == currentScopeLoopOrder.end()))
                                {
                                    if (varIdx == 0)
                                    {
                                        std::function genRoot = [=, &rewriter](Table &table)
                                            -> std::pair<std::optional<mlir::Value>, mlir::Value>
                                        {
                                            constexpr unsigned int mode = 0;
                                            auto dummyExpr = mlir::getAffineDimExpr(0, context);
                                            const auto genRefAttr =
                                                getStreamAttr(context, sam::StreamKind::Ref, dummyExpr, "", "", mode);
                                            auto gen = rewriter.create<sam::SamGenerator>(
                                                linalgOp->getLoc(),
                                                RankedTensorType::get({}, IndexType::get(context), genRefAttr));
                                            return std::make_pair(gen.getOutputRef(), gen.getOutputRef());
                                        };
                                        std::string rootName = "root";
                                        auto redirectCell =
                                            make_unique_cell<std::pair<std::optional<mlir::Value>, mlir::Value>>(
                                                genRoot, rootName);
                                        graphTable[tensorView][var] = std::move(redirectCell);
                                        continue;
                                    }
                                    std::function redirectFunc = [=, &rewriter](Table &table)
                                    { return table.at(tensorView, loopOrder[varIdx - 1]); };
                                    std::string cellName = "Redirect";
                                    std::ostringstream ss;
                                    ss << loopOrder.at(varIdx - 1);
                                    auto dupCellName = cellName + "(" + scope->tensorNames.at(tensorView.getValue()) +
                                        ", " + ss.str() + ")";
                                    auto redirectCell =
                                        make_unique_cell<std::pair<std::optional<mlir::Value>, mlir::Value>>(
                                            redirectFunc, dupCellName);
                                    graphTable[tensorView][var] = std::move(redirectCell);
                                    // TODO: May need to add continue back
                                    continue;
                                }

                                // Func to instantiate repeat
                                std::function repeatFunc =
                                    [=, &rewriter,
                                     scope = scope](Table &table) -> std::pair<std::optional<mlir::Value>, mlir::Value>
                                {
                                    mlir::Value prevRef;
                                    if (!(var == loopOrder.front()))
                                    {
                                        const auto &prevCell = table.at(tensorView, loopOrder[varIdx - 1]);
                                        prevRef = prevCell.first.value();
                                    }
                                    else
                                    {
                                        // llvm::outs() << "Dim: " << dimExpr << "\n";
                                        // llvm::outs() << "TensorName: " << scope->tensorViewNames[tensorView] << "\n";
                                        const auto genRefAttr =
                                            getStreamAttr(context, sam::StreamKind::Ref, dimExpr, tensorName, "");
                                        // linalgOp->getParentOp()->dump();
                                        auto gen = rewriter.create<sam::SamGenerator>(
                                            linalgOp->getLoc(),
                                            RankedTensorType::get({}, IndexType::get(context), genRefAttr));
                                        prevRef = gen.getOutputRef();
                                    }


                                    RefCrd repeatSigCell;
                                    if (found)
                                    {
                                        repeatSigCell = table.at(repeatSigTensorView, var);
                                    }
                                    else
                                    {
                                        repeatSigCell = table.at(scope->indexToView[var], var);
                                    }
                                    mlir::Value repeatSig = repeatSigCell.first.value();

                                    auto repeat = rewriter.create<sam::SamRepeat>(
                                        linalgOp->getLoc(), RankedTensorType::get({}, IndexType::get(context), refAttr),
                                        prevRef, repeatSig);
                                    // TODO: No crd output from repeater
                                    // TODO: Make crd output of cell an std::optional
                                    // Duplicating ref output for now
                                    return std::make_pair(repeat.getOutputRef(), repeat.getOutputRef());
                                };
                                std::string prevRefName;
                                if (!(var == loopOrder.front()))
                                {
                                    std::ostringstream ss;
                                    ss << loopOrder[varIdx - 1];
                                    prevRefName =
                                        "Prev(" + scope->tensorNames.at(tensorView.getValue()) + ", " + ss.str() + ")";
                                }
                                else
                                {
                                    prevRefName = "root";
                                }
                                std::string repeatSigName = scope->tensorNames.at(repeatSigTensorView.getValue());
                                // TODO: Need to figure out how to extract the repeat signal
                                std::ostringstream varStr;
                                varStr << "_" << var;
                                std::string cellLabel = "Repeat" + varStr.str();
                                std::string repCellLabel =
                                    cellLabel.append("(") + prevRefName.append(", ") + repeatSigName.append(")");
                                auto repeatCell = make_unique_cell<std::pair<std::optional<mlir::Value>, mlir::Value>>(
                                    repeatFunc, repCellLabel);

                                // Patch column from input tensor at current var with repeat cell
                                graphTable[tensorView][var] = std::move(repeatCell);
                            }
                        }

                        auto outIndexVars =
                            llvm::map_to_vector(outputVars, [=](const UniqueVar &a) { return a.getLogicalIndexVar(); });

                        // Figure out output refs and crds from input tensors
                        // Use indexing maps to find where they're coming from
                        if (llvm::find(outIndexVars, var) != outIndexVars.end())
                        {
                            std::optional<TensorView> mappedTensor;
                            for (unsigned int idx = 0; idx < indexingMap.size() - 1; ++idx)
                            {
                                const auto &map = indexingMap[idx];
                                for (auto exprIter : llvm::enumerate(map.getResults()))
                                {
                                    auto expr = exprIter.value();
                                    const unsigned int exprId = exprIter.index();
                                    if (llvm::dyn_cast<mlir::AffineConstantExpr>(expr))
                                    {
                                        expr = mlir::getAffineDimExpr(exprId, context);
                                    }
                                    const auto &correspondingUniqueVar = exprToUniqueVar.at(expr);
                                    const auto &mappedIndexVar = correspondingUniqueVar.getLogicalIndexVar();
                                    if (mappedIndexVar == var)
                                    {
                                        mappedTensor = opTensors[idx];
                                        break;
                                    }
                                }
                            }
                            if (!mappedTensor.has_value())
                            {
                                std::cerr << "ERROR: Mapped tensor is empty in repeat" << std::endl;
                                exit(1);
                            }
                            std::function outFunc =
                                [=, scope = scope](Table &table) -> std::pair<std::optional<mlir::Value>, mlir::Value>
                            {
                                auto cellVal = table.at(mappedTensor.value(), var);
                                return cellVal;
                            };
                            std::ostringstream ss;
                            ss << var;
                            std::string outLabel = "Ptr(" + scope->tensorNames.at(mappedTensor.value().getValue()) +
                                ", " + ss.str().append(")");
                            scope->cellIndirectionMap[var][returnedTensorView] = mappedTensor.value();
                            auto outputCell = make_unique_cell<std::pair<std::optional<mlir::Value>, mlir::Value>>(
                                std::move(outFunc), outLabel);
                            column[var] = std::move(outputCell);
                            // return cellVal;
                        }
                        else
                        {
                            std::function dupFunc = [=, scope = scope](Table &table)
                            { return table.at(returnedTensorView, loopOrder[varIdx - 1]); };
                            std::string cellName = "Redirect";
                            if (varIdx > 0 && varIdx < loopOrder.size())
                            {
                                std::ostringstream ss;
                                ss << loopOrder[varIdx - 1];
                                auto dupCellName = cellName.append("(") +
                                    scope->tensorNames.at(returnedTensorView.getValue()) + ", " + ss.str().append(")");
                                auto dupCell = make_unique_cell<std::pair<std::optional<mlir::Value>, mlir::Value>>(
                                    std::move(dupFunc), dupCellName);
                                column[var] = std::move(dupCell);
                            }
                            else
                            {
                                std::function genRoot =
                                    [=, &rewriter](Table &table) -> std::pair<std::optional<mlir::Value>, mlir::Value>
                                {
                                    constexpr unsigned int mode = 0;
                                    auto dummyExpr = mlir::getAffineDimExpr(0, context);
                                    const auto genRefAttr =
                                        getStreamAttr(context, sam::StreamKind::Ref, dummyExpr, "", "", mode);
                                    auto gen = rewriter.create<sam::SamGenerator>(
                                        linalgOp->getLoc(),
                                        RankedTensorType::get({}, IndexType::get(context), genRefAttr));
                                    return std::make_pair(gen.getOutputRef(), gen.getOutputRef());
                                };
                                std::string rootName = "root";
                                auto redirectCell =
                                    make_unique_cell<std::pair<std::optional<mlir::Value>, mlir::Value>>(genRoot,
                                                                                                         rootName);
                                column[var] = std::move(redirectCell);
                            }
                        }
                    }

                    llvm::SmallVector<mlir::AffineExpr> opLoopOrder;
                    for (auto dim : dimExprs)
                    {
                        opLoopOrder.push_back(
                            mlir::getAffineDimExpr(exprToUniqueVar.at(dim).getLogicalIndexVar().getId(), context));
                    }
                    auto loopOrderIndex = loopOrder;
                    auto affineLoopOrder = llvm::map_to_vector(loopOrderIndex,
                                                               [&](const IndexVar &a)
                                                               {
                                                                   auto dim =
                                                                       mlir::getAffineDimExpr(a.getId(), context);
                                                                   // llvm::outs() << "Loop: " << dim << "\n";
                                                                   return dim;
                                                               });
                    sortByPartialOrder(affineLoopOrder, opLoopOrder);
                    std::vector<std::string> names;
                    for (auto input : linalgOp.getDpsInputs())
                    {
                        names.push_back(scope->tensorNames[input]);
                    }
                    std::ostringstream ss;
                    std::copy(names.begin(), names.end(), std::ostream_iterator<std::string>(ss, ", "));

                    std::string opName = getOpName(linalgOp);
                    std::string inputLst = ss.str();
                    // Removing last delimiter + extra space char
                    inputLst.pop_back();
                    inputLst.pop_back();
                    std::string valLabel = opName.append("(") + inputLst.append(")");
                    std::ostringstream varStr;
                    for (auto loopIter : llvm::enumerate(localLoopOrder))
                    {
                        const auto &loop = loopIter.value();
                        const unsigned int loopIdx = loopIter.index();
                        varStr << loop;
                        if (loopIdx != localLoopOrder.size() - 1)
                        {
                            varStr << ", ";
                        }
                    }
                    valLabel += varStr.str();
                    std::function comp = [=, scope = scope, &inputs, &rewriter](Table &table) -> mlir::Value
                    {
                        // llvm::outs() << result << "\n";
                        auto linalgOp = result.getDefiningOp<linalg::LinalgOp>();
                        llvm::SmallVector<mlir::Value> inputStreams;
                        for (unsigned int i = 0; i < inputs.size(); i++)
                        {
                            const auto &tensorView = opTensors.at(i);
                            auto localUniqueVars = tensorView.getUniqueVars();
                            const auto indexVars = llvm::map_to_vector(localUniqueVars, [=](const UniqueVar &a)
                                                                       { return a.getLogicalIndexVar(); });
                            std::vector<IndexVar> returnedTensorIndex(indexVars.begin(), indexVars.end());
                            auto tensorPair =
                                std::make_pair(scope->tensorNames.at(tensorView.getValue()), returnedTensorIndex);
                            if (scope->tableCache.count(tensorPair))
                            {
                                tensorView.setCompleted();
                            }
                            inputStreams.push_back(table.getValue(opTensors.at(i)));
                        }


                        // for (unsigned int i = 0; i < opTensors.size() - 1; i++) {

                        auto elemType = llvm::cast<RankedTensorType>(linalgOp->getResult(0).getType()).getElementType();
                        // Check if the sequence of block ops lowers to a unary op or custom
                        // reduction block first
                        if (isGenericMaxReduce(linalgOp).succeeded())
                        {
                            auto d = getAffineDimExpr(0, context);
                            auto valAttr = getStreamAttr(context, sam::StreamKind::Val, d, "t_reduce", "");
                            auto reduce_op = rewriter.create<sam::SamReduce>(
                                linalgOp->getLoc(), RankedTensorType::get({}, elemType, valAttr), inputStreams[0],
                                sam::ReduceTypeAttr::get(context, sam::ReduceType::Max));
                            auto lastLoop = localLoopOrder.back();
                            if (!(llvm::find(scope->effLoopOrder, lastLoop) == scope->effLoopOrder.end()))
                            {
                                const unsigned int idx =
                                    llvm::find(scope->effLoopOrder, lastLoop) - scope->effLoopOrder.begin();
                                scope->effLoopOrder.erase(scope->effLoopOrder.begin() + idx);
                            }
                            return reduce_op.getOutputVal();
                        }
                        if (isGenericRelu(linalgOp).succeeded())
                        {
                            // Setting max op for ReLU
                            auto opAttr = sam::OpTypeAttr::get(context, sam::OpType::Max);
                            auto d = getAffineDimExpr(0, context);
                            auto valAttr = getStreamAttr(context, sam::StreamKind::Val, d, "t_max", "");
                            llvm::SmallVector<Type> output_types;
                            output_types.push_back(RankedTensorType::get({}, elemType, valAttr));
                            auto alu_op =
                                rewriter.create<sam::SamALU>(linalgOp->getLoc(), output_types, inputStreams[0], opAttr,
                                                             mlir::FloatAttr::get(Float32Type::get(context), 0.0));
                            return alu_op.getOutputVal()[0];
                        }
                        if (isGenericRsqrt(linalgOp).succeeded())
                        {
                            auto opAttr = sam::OpTypeAttr::get(context, sam::OpType::RSqrt);
                            auto d = getAffineDimExpr(0, context);
                            auto valAttr = getStreamAttr(context, sam::StreamKind::Val, d, "t_rsqrt", "");
                            llvm::SmallVector<Type> output_types;
                            output_types.push_back(RankedTensorType::get({}, elemType, valAttr));
                            auto alu_op =
                                rewriter.create<sam::SamALU>(linalgOp->getLoc(), output_types, inputStreams[0], opAttr,
                                                             mlir::FloatAttr::get(Float32Type::get(context), 0.0));
                            return alu_op.getOutputVal()[0];
                        }
                        if (isGenericScalarAdd(linalgOp).succeeded())
                        {
                            auto opAttr = sam::OpTypeAttr::get(context, sam::OpType::ScalarAdd);
                            auto d = getAffineDimExpr(0, context);
                            auto valAttr = getStreamAttr(context, sam::StreamKind::Val, d, "t_scalaradd", "");
                            llvm::SmallVector<Type> output_types;
                            output_types.push_back(RankedTensorType::get({}, elemType, valAttr));
                            double scalar = getScalarOperand<arith::AddFOp>(linalgOp);
                            auto alu_op =
                                rewriter.create<sam::SamALU>(linalgOp->getLoc(), output_types, inputStreams[0], opAttr,
                                                             mlir::FloatAttr::get(Float32Type::get(context), scalar));
                            return alu_op.getOutputVal()[0];
                        }
                        if (isGenericScalarDiv(linalgOp).succeeded())
                        {
                            auto opAttr = sam::OpTypeAttr::get(context, sam::OpType::ScalarDiv);
                            auto d = getAffineDimExpr(0, context);
                            auto valAttr = getStreamAttr(context, sam::StreamKind::Val, d, "t_scalardiv", "");
                            llvm::SmallVector<Type> output_types;
                            double scalar = getScalarOperand<arith::DivFOp>(linalgOp);
                            output_types.push_back(RankedTensorType::get({}, elemType, valAttr));
                            auto alu_op =
                                rewriter.create<sam::SamALU>(linalgOp->getLoc(), output_types, inputStreams[0], opAttr,
                                                             mlir::FloatAttr::get(Float32Type::get(context), scalar));
                            return alu_op.getOutputVal()[0];
                        }
                        if (isGenericScalarMul(linalgOp).succeeded())
                        {
                            auto opAttr = sam::OpTypeAttr::get(context, sam::OpType::ScalarMul);
                            auto d = getAffineDimExpr(0, context);
                            auto valAttr = getStreamAttr(context, sam::StreamKind::Val, d, "t_scalarmul", "");
                            llvm::SmallVector<Type> output_types;
                            double scalar = getScalarOperand<arith::MulFOp>(linalgOp);
                            output_types.push_back(RankedTensorType::get({}, elemType, valAttr));
                            auto alu_op =
                                rewriter.create<sam::SamALU>(linalgOp->getLoc(), output_types, inputStreams[0], opAttr,
                                                             mlir::FloatAttr::get(Float32Type::get(context), scalar));
                            return alu_op.getOutputVal()[0];
                        }
                        if (isGenericReduce(linalgOp).succeeded() && isGenericMatmulOp(linalgOp).failed())
                        {
                            auto d = getAffineDimExpr(0, context);
                            auto valAttr = getStreamAttr(context, sam::StreamKind::Val, d, "t_reduce", "");
                            auto reduce_op = rewriter.create<sam::SamReduce>(
                                linalgOp->getLoc(), RankedTensorType::get({}, elemType, valAttr), inputStreams[0],
                                sam::ReduceTypeAttr::get(context, sam::ReduceType::AddSum));
                            return reduce_op.getOutputVal();
                        }
                        // Casting operations will just pass through input to output, since we don't care about type
                        if (isGenericUpcast(linalgOp).succeeded() || isGenericDowncast(linalgOp).succeeded())
                        {
                            return inputStreams[0];
                        }

                        Block &block = *linalgOp.getBlock();
                        bool yield = false;
                        for (auto blockOpIter : llvm::enumerate(block))
                        {
                            SmallVector<Value> value_lst;
                            SmallVector<Type> output_vals;
                            SmallVector<Value> input_vals;

                            sam::OpType opType;
                            Operation &blockOp = blockOpIter.value();
                            unsigned int blockOpNum = blockOpIter.index();

                            // Other blocks can be reasoned about at a basic block (bb) op
                            // granularity, so we can just loop through each bb op and
                            // instantiate
                            // appropriate SAMML nodes
                            llvm::TypeSwitch<Operation &, LogicalResult>(blockOp)
                                .Case<arith::MulFOp>(
                                    [&](auto)
                                    {
                                        if (blockOpNum == 0)
                                        {
                                            value_lst.push_back(linalgOp->getOperand(0));
                                            value_lst.push_back(linalgOp->getOperand(1));
                                        }
                                        else
                                        {
                                            value_lst.push_back(blockOp.getOpOperands()[0].get());
                                            value_lst.push_back(blockOp.getOpOperands()[1].get());
                                        }
                                        opType = sam::OpType::Mul;

                                        return success();
                                    })
                                .Case<arith::DivFOp>(
                                    [&](auto)
                                    {
                                        if (blockOpNum == 0)
                                        {
                                            value_lst.push_back(linalgOp->getOperand(0));
                                            value_lst.push_back(linalgOp->getOperand(1));
                                        }
                                        else
                                        {
                                            value_lst.push_back(blockOp.getOpOperands()[0].get());
                                            value_lst.push_back(blockOp.getOpOperands()[1].get());
                                        }
                                        opType = sam::OpType::Div;

                                        return success();
                                    })
                                .Case<arith::SubFOp>(
                                    [&](auto)
                                    {
                                        if (blockOpNum == 0)
                                        {
                                            value_lst.push_back(linalgOp->getOperand(0));
                                            value_lst.push_back(linalgOp->getOperand(1));
                                        }
                                        else
                                        {
                                            value_lst.push_back(blockOp.getOpOperands()[0].get());
                                            value_lst.push_back(blockOp.getOpOperands()[1].get());
                                        }
                                        opType = sam::OpType::Sub;

                                        return success();
                                    })
                                .Case<arith::AddFOp>(
                                    [&](auto)
                                    {
                                        if (blockOpNum == 0)
                                        {
                                            for (unsigned int i = 0; i < numDpsInputs; i++)
                                            {
                                                value_lst.push_back(linalgOp->getOperand(i));
                                            }
                                        }
                                        else
                                        {
                                            for (unsigned int i = 0; i < numDpsInputs; i++)
                                            {
                                                if (auto blockArg = llvm::dyn_cast<mlir::BlockArgument>(
                                                        blockOp.getOpOperand(i).get()))
                                                {
                                                    if (blockArg.getArgNumber() >= numDpsInputs)
                                                    {
                                                        continue;
                                                    }
                                                }
                                                value_lst.push_back(blockOp.getOpOperand(i).get());
                                            }
                                        }
                                        if (numLoops != numParallelLoops)
                                        {
                                            opType = sam::OpType::Reduce;
                                        }
                                        else
                                        {
                                            opType = sam::OpType::Add;
                                        }
                                        return success();
                                    })
                                .Case<math::ExpOp>(
                                    [&](auto)
                                    {
                                        if (blockOpNum == 0)
                                        {
                                            value_lst.push_back(linalgOp->getOperand(0));
                                        }
                                        else
                                        {
                                            value_lst.push_back(blockOp.getOperands()[0]);
                                        }
                                        opType = sam::OpType::Exp;
                                        return success();
                                    })
                                .Case<linalg::YieldOp>(
                                    [&](auto)
                                    {
                                        scope->regionOpToStream[linalgOp->getResult(0)] =
                                            scope->regionOpToStream[blockOp.getOperand(0)];
                                        yield = true;
                                        return success();
                                    });

                            if (yield)
                            {
                                return scope->regionOpToStream[blockOp.getOperand(0)];
                            }

                            auto d = mlir::getAffineDimExpr(0, context);
                            //                        table.dump(scope, loopOrder);
                            auto valAttr = getStreamAttr(context, sam::StreamKind::Val, d, "t_out", "");
                            output_vals.push_back(RankedTensorType::get({}, elemType, valAttr));

                            for (int i = 0; i < value_lst.size(); ++i)
                            {
                                if (blockOpNum == 0)
                                {
                                    input_vals.push_back(inputStreams[i]);
                                }
                                else
                                {
                                    input_vals.push_back(scope->regionOpToStream[value_lst[i]]);
                                }
                            }

                            auto opAttr = sam::OpTypeAttr::get(context, opType);
                            if (opType == sam::OpType::Reduce)
                            {
                                return input_vals.front();
                            }
                            else if (opType == sam::OpType::Mul || opType == sam::OpType::Div ||
                                     opType == sam::OpType::Add || opType == sam::OpType::Sub)
                            {
                                for (auto opOperandIter : llvm::enumerate(linalgOp.getDpsInputOperands()))
                                {
                                    auto opOperand = opOperandIter.value();
                                    unsigned int index = opOperandIter.index();
                                    if (const auto map = linalgOp.getMatchingIndexingMap(opOperand);
                                        llvm::dyn_cast<mlir::AffineConstantExpr>(map.getResults().back()))
                                    {
                                        AffineExpr missingDim = opLoopOrder.back();
                                        IndexVar dim = localLoopOrder.back();
                                        auto view = scope->indexToView.at(dim);

                                        auto corrIndexVar = dim;
                                        mlir::Value repeat_ref;
                                        // Make the repeat signal the most updated coordinate stream
                                        if (scope->seenInner.count(corrIndexVar))
                                        {
                                            repeat_ref =
                                                table.at(scope->seenInner.at(corrIndexVar), corrIndexVar).first.value();
                                        }
                                        else
                                        {
                                            repeat_ref = table.at(view, corrIndexVar).first.value();
                                        }
                                        auto refAttr =
                                            getStreamAttr(context, sam::StreamKind::Ref, missingDim, "t_repeat", "");
                                        auto repeater = rewriter.create<sam::SamRepeat>(
                                            linalgOp->getLoc(), RankedTensorType::get({}, elemType, refAttr),
                                            input_vals[index], repeat_ref);

                                        input_vals[index] = repeater.getOutputRef();
                                    }
                                }

                                auto op1IndexVars = llvm::map_to_vector(
                                    scope->tensorLoopOrders.at(opTensors[0]), [&](const UniqueVar &a)
                                    { return mlir::getAffineDimExpr(a.getLogicalIndexVar().getId(), context); });
                                auto op2IndexVars = llvm::map_to_vector(
                                    scope->tensorLoopOrders.at(opTensors[1]), [&](const UniqueVar &a)
                                    { return mlir::getAffineDimExpr(a.getLogicalIndexVar().getId(), context); });
                                sortByPartialOrder(affineLoopOrder, op1IndexVars);
                                sortByPartialOrder(affineLoopOrder, op2IndexVars);

                                auto input0DefiningOp = input_vals[0].getDefiningOp<sam::SamArrayVal>();
                                auto input1DefiningOp = input_vals[1].getDefiningOp<sam::SamArrayVal>();
                                if (input1DefiningOp &&
                                    input1DefiningOp.getInputRef().getDefiningOp<sam::SamFiberLookup>() &&
                                    !input_vals[0].getDefiningOp<sam::SamArrayVal>())
                                {
                                    const auto &loops = op2IndexVars;
                                    auto missingDim = loops.back();
                                    auto dimId = llvm::cast<AffineDimExpr>(missingDim).getPosition();
                                    auto corrIndexVar = IndexVar(dimId);
                                    //                                auto view = scope->indexToView.at(corrIndexVar);
                                    const auto &view = opTensors.at(1);
                                    mlir::Value repeat_ref = table.at(view, corrIndexVar).first.value();
                                    auto refAttr =
                                        getStreamAttr(context, sam::StreamKind::Ref, missingDim, "t_repeat", "");
                                    auto repeater = rewriter.create<sam::SamRepeat>(
                                        linalgOp->getLoc(), RankedTensorType::get({}, elemType, refAttr), input_vals[0],
                                        repeat_ref);

                                    input_vals[0] = repeater.getOutputRef();
                                }
                                else if (input0DefiningOp &&
                                         input0DefiningOp.getInputRef().getDefiningOp<sam::SamFiberLookup>() &&
                                         !input_vals[1].getDefiningOp<sam::SamArrayVal>() &&
                                         !input_vals[1].getDefiningOp<sam::SamRepeat>())
                                {
                                    const auto &loops = op1IndexVars;
                                    auto missingDim = loops.back();
                                    auto refAttr =
                                        getStreamAttr(context, sam::StreamKind::Ref, missingDim, "t_repeat", "");
                                    auto corrIndexVar = IndexVar(llvm::cast<AffineDimExpr>(missingDim).getPosition());
                                    //                                auto view = scope->indexToView.at(corrIndexVar);
                                    const auto &view = opTensors.at(0);
                                    mlir::Value repeat_ref = table.at(view, corrIndexVar).first.value();
                                    auto repeater = rewriter.create<sam::SamRepeat>(
                                        linalgOp->getLoc(), RankedTensorType::get({}, elemType, refAttr), input_vals[1],
                                        repeat_ref);

                                    input_vals[1] = repeater.getOutputRef();
                                }
                                else if (!input0DefiningOp && !input1DefiningOp)
                                {
                                    unsigned int op1Loc =
                                        llvm::find(loopOrder,
                                                   opTensors[0].getUniqueVars().back().getLogicalIndexVar()) -
                                        loopOrder.begin();
                                    unsigned int op2Loc =
                                        llvm::find(loopOrder,
                                                   opTensors[1].getUniqueVars().back().getLogicalIndexVar()) -
                                        loopOrder.begin();

                                    if (op1Loc > op2Loc)
                                    {
                                        const auto &loops = op1IndexVars;
                                        auto missingDim = loops.back();
                                        auto refAttr =
                                            getStreamAttr(context, sam::StreamKind::Ref, missingDim, "t_repeat", "");
                                        auto corrIndexVar =
                                            IndexVar(llvm::cast<AffineDimExpr>(missingDim).getPosition());
                                        //                                auto view =
                                        //                                scope->indexToView.at(corrIndexVar);
                                        const auto &view = opTensors.at(0);
                                        mlir::Value repeat_ref = table.at(view, corrIndexVar).first.value();
                                        auto repeater = rewriter.create<sam::SamRepeat>(
                                            linalgOp->getLoc(), RankedTensorType::get({}, elemType, refAttr),
                                            input_vals[1], repeat_ref);

                                        input_vals[1] = repeater.getOutputRef();
                                    }
                                    else if (op1Loc < op2Loc)
                                    {
                                        const auto &loops = op2IndexVars;
                                        auto missingDim = loops.back();
                                        auto dimId = llvm::cast<AffineDimExpr>(missingDim).getPosition();
                                        auto corrIndexVar = IndexVar(dimId);
                                        //                                auto view =
                                        //                                scope->indexToView.at(corrIndexVar);
                                        const auto &view = opTensors.at(1);
                                        mlir::Value repeat_ref = table.at(view, corrIndexVar).first.value();
                                        auto refAttr =
                                            getStreamAttr(context, sam::StreamKind::Ref, missingDim, "t_repeat", "");
                                        auto repeater = rewriter.create<sam::SamRepeat>(
                                            linalgOp->getLoc(), RankedTensorType::get({}, elemType, refAttr),
                                            input_vals[0], repeat_ref);

                                        input_vals[0] = repeater.getOutputRef();
                                    }
                                }

                                //                                llvm::outs() << "Op type: " << opType << "\n";
                                //                                std::cout << "Operand1: " <<
                                //                                table[opTensors[0]].getMutableValue()->getLabel()
                                //                                          << std::endl;
                                //                                std::cout << "Operand2: " <<
                                //                                table[opTensors[1]].getMutableValue()->getLabel()
                                //                                          << std::endl;
                                //                                for (auto &loop: opTensors[0].getUniqueVars()) {
                                //                                    std::cout << "Loop: " << loop.getLogicalIndexVar()
                                //                                    << std::endl;
                                //                                }
                                //                                std::cout << std::endl;
                                //                                for (auto &loop: opTensors[1].getUniqueVars()) {
                                //                                    std::cout << "Loop: " << loop.getLogicalIndexVar()
                                //                                    << std::endl;
                                //                                }
                                //                                std::cout << std::endl;
                                //                                for (auto &loop: op1IndexVars) {
                                //                                    llvm::outs() << "Loop: " << loop << "\n";
                                //                                }
                                //                                std::cout << std::endl;
                                //                                for (auto &loop: op2IndexVars) {
                                //                                    llvm::outs() << "Loop: " << loop << "\n";
                                //                                }

                                auto alu_op =
                                    rewriter.create<sam::SamALU>(linalgOp->getLoc(), output_vals, input_vals, opAttr,
                                                                 mlir::FloatAttr::get(Float32Type::get(context), 0.0));
                                scope->regionOpToStream[blockOp.getResult(0)] = alu_op.getOutputVal()[0];
                                scope->regionOpToStream[linalgOp->getResult(0)] = alu_op.getOutputVal()[0];
                                return alu_op.getOutputVal()[0];
                            }
                            else if (opType == sam::OpType::Exp)
                            {
                                auto alu_op =
                                    rewriter.create<sam::SamALU>(linalgOp->getLoc(), output_vals, input_vals, opAttr,
                                                                 mlir::FloatAttr::get(Float32Type::get(context), 0.0));
                                scope->regionOpToStream[blockOp.getResult(0)] = alu_op.getOutputVal()[0];
                                scope->regionOpToStream[linalgOp->getResult(0)] = alu_op.getOutputVal()[0];
                                return alu_op.getOutputVal()[0];
                            }
                        }
                        return nullptr;
                    };

                    Block &block = *linalgOp.getBlock();

                    if (reductionVars.empty() || isGenericMaxReduce(linalgOp).succeeded() ||
                        isGenericReduce(linalgOp).succeeded() && isGenericMatmulOp(linalgOp).failed())
                    {
                        auto compCell = make_unique_cell<mlir::Value>(comp, valLabel);
                        column.setValue(std::move(compCell));
                    }
                    else
                    {
                        for (auto blockOpIter : llvm::enumerate(block))
                        {
                            Operation &blockOp = blockOpIter.value();
                            if (unsigned int blockOpNum = blockOpIter.index(); blockOpNum == 0)
                            {
                                continue;
                            }
                            SmallVector<Value> value_lst;
                            SmallVector<Type> output_vals;
                            SmallVector<Value> input_vals;

                            sam::OpType opType;

                            llvm::TypeSwitch<Operation &, LogicalResult>(blockOp)
                                .Case<arith::AddFOp>(
                                    [&](auto)
                                    {
                                        for (unsigned int i = 0; i < numDpsInputs; i++)
                                        {
                                            if (auto blockArg =
                                                    llvm::dyn_cast<mlir::BlockArgument>(blockOp.getOpOperand(i).get()))
                                            {
                                                if (blockArg.getArgNumber() >= numDpsInputs)
                                                {
                                                    continue;
                                                }
                                            }
                                            value_lst.push_back(blockOp.getOpOperand(i).get());
                                        }
                                        if (numLoops > numParallelLoops)
                                        {
                                            opType = sam::OpType::Reduce;
                                        }
                                        else
                                        {
                                            opType = sam::OpType::Add;
                                        }
                                        return success();
                                    })
                                .Case<linalg::YieldOp>(
                                    [&](auto)
                                    {
                                        scope->regionOpToStream[linalgOp->getResult(0)] =
                                            scope->regionOpToStream[blockOp.getOperand(0)];
                                        return success();
                                    });

                            auto elemType =
                                llvm::cast<RankedTensorType>(linalgOp->getResult(0).getType()).getElementType();
                            auto d = mlir::getAffineDimExpr(0, context);
                            auto valAttr = getStreamAttr(context, sam::StreamKind::Val, d, "t_out", "");
                            output_vals.push_back(RankedTensorType::get({}, elemType, valAttr));

                            if (opType == sam::OpType::Reduce)
                            {
                                //                            std::cout << "Partial order: " << std::endl;
                                //                            for (auto loop : localLoopOrder)
                                //                            {
                                //                                std::cout << loop << std::endl;
                                //                            }
                                scope->effLoopOrder.clear();
                                const auto &loops = opTensors[0].getLoops();
                                for (const auto &loop : loopOrder)
                                {
                                    if (llvm::find(loops, loop) != loops.end() || loop == reductionVar.value())
                                    {
                                        // Ignore concat dimension variables for pre-concat operations
                                        if (shouldIgnoreConcatDimVar(loop, returnedTensorView, scope))
                                        {
                                            continue;
                                        }
                                        scope->effLoopOrder.push_back(loop);
                                    }
                                }
                                unsigned int num_dims = indexingMaps[0].getNumDims();
                                auto redIt = llvm::find(scope->effLoopOrder, reductionVar.value());
                                auto lastIt = llvm::find(scope->effLoopOrder, localLoopOrder.back());
                                unsigned int order =
                                    (lastIt - scope->effLoopOrder.begin()) - (redIt - scope->effLoopOrder.begin());

                                if (order == 0)
                                {
                                    std::function reduceFunc = [=, scope = scope,
                                                                &rewriter](Table &table) -> mlir::Value
                                    {
                                        const auto cont = rewriter.getContext();
                                        auto val = comp(table);
                                        auto reduce_op = rewriter.create<sam::SamReduce>(
                                            linalgOp->getLoc(), RankedTensorType::get({}, elemType, valAttr), val,
                                            sam::ReduceTypeAttr::get(cont, sam::ReduceType::Add));
                                        scope->regionOpToStream[linalgOp->getResult(0)] = reduce_op.getOutputVal();
                                        return reduce_op.getOutputVal();
                                    };

                                    auto lastLoop = localLoopOrder.back();
                                    const unsigned int idx =
                                        llvm::find(scope->effLoopOrder, lastLoop) - scope->effLoopOrder.begin();
                                    scope->effLoopOrder.erase(scope->effLoopOrder.begin() + idx);


                                    std::string reduceLabel = "Reduce(" + valLabel.append(")");
                                    auto reduceCell = make_unique_cell<mlir::Value>(reduceFunc, reduceLabel);
                                    column.setValue(std::move(reduceCell));
                                    break;
                                }
                                // TODO: Fix error with spacc, might need to stash metadata about
                                // coord streams in scope as a hack
                                else
                                {
                                    // TODO: Getting input to spacc from table instead of from local metadata
                                    std::vector<std::pair<std::optional<TensorView>, IndexVar>> input_outers;
                                    std::optional<std::pair<TensorView, IndexVar>> input_inner;

                                    const int finalDim = llvm::find(scope->effLoopOrder, localLoopOrder.back()) -
                                        scope->effLoopOrder.begin();
                                    // Iterate over the levels going into the spacc backwards
                                    for (int i = (int)finalDim; i >= (int)finalDim - (int)order; i--)
                                    {
                                        auto currVar = scope->effLoopOrder.at(i);
                                        // We first find the first tensorview that contains that index variable
                                        std::optional<TensorView> mappedTensor;
                                        for (unsigned int idx = 0; idx < indexingMap.size() - 1; ++idx)
                                        {
                                            const auto &map = indexingMap[idx];
                                            for (auto exprIter : llvm::enumerate(map.getResults()))
                                            {
                                                auto expr = exprIter.value();
                                                const unsigned int exprId = exprIter.index();
                                                if (llvm::dyn_cast<mlir::AffineConstantExpr>(expr))
                                                {
                                                    expr = mlir::getAffineDimExpr(exprId, context);
                                                }
                                                const auto &correspondingUniqueVar = exprToUniqueVar.at(expr);
                                                if (const auto &mappedIndexVar =
                                                        correspondingUniqueVar.getLogicalIndexVar();
                                                    mappedIndexVar == currVar)
                                                {
                                                    mappedTensor = opTensors.at(idx);
                                                    break;
                                                }
                                            }
                                            // Break out of outermost for loop
                                            if (mappedTensor.has_value())
                                            {
                                                break;
                                            }
                                        }
                                        if (!mappedTensor.has_value())
                                        {
                                            //                                        mappedTensor = returnedTensorView;
                                            // TODO: Remap to correct tensor, not intermediate output tensor
                                            for (auto op_result : linalgOp->getResults())
                                            {
                                                for (auto user : op_result.getUsers())
                                                {
                                                    if (auto linOp = llvm::dyn_cast<linalg::LinalgOp>(user))
                                                    {
                                                        for (auto opOperand : linOp->getOperands())
                                                        {
                                                            auto operand = opOperand;
                                                            // auto currTensorView =
                                                            // metastageMap.at(operand)->getTensorViews().front();
                                                            for (const auto &tensor :
                                                                 metastageMap.at(operand)->getTensorViews())
                                                            {
                                                                if (auto currIndexVars = llvm::map_to_vector(
                                                                        tensor.getUniqueVars(), [=](const UniqueVar &a)
                                                                        { return a.getLogicalIndexVar(); });
                                                                    llvm::find(currIndexVars, currVar) !=
                                                                        currIndexVars.end() &&
                                                                    !(tensor == returnedTensorView))
                                                                {
                                                                    mappedTensor = tensor;
                                                                    break;
                                                                }
                                                            }
                                                            if (mappedTensor.has_value())
                                                            {
                                                                break;
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                        if (i == finalDim)
                                        {
                                            // Inner crd is retrieved here
                                            input_inner = std::make_pair(mappedTensor.value(), currVar);
                                            scope->innerCrds[currVar].push_back(returnedTensorView);
                                            continue;
                                        }

                                        // If we still can't find current index level
                                        input_outers.emplace_back(mappedTensor, currVar);
                                    }


                                    const unsigned int idx = llvm::find(scope->effLoopOrder, reductionVar.value()) -
                                        scope->effLoopOrder.begin();
                                    scope->effLoopOrder.erase(scope->effLoopOrder.begin() + idx);

                                    if (!input_inner.has_value())
                                    {

                                        std::cerr << "ERROR: Input inner not populated" << std::endl;
                                        graphTable.dump(scope, loopOrder);
                                        exit(1);
                                    }

                                    std::function spaccFunc = [=, scope = scope, &rewriter](Table &table)
                                    {
                                        SmallVector<Type> output_outer_crds_type;
                                        SmallVector<Value> input_outer_crds;
                                        // auto currOpDimExprs = opOrders[&oper];
                                        const auto &currOpDimExprs = opLoopOrder;
                                        //                                  scope->indexToView[input_inner.value().second]
                                        //                                  = returnedTensorView;
                                        auto val = comp(table);

                                        // Retrieve the input inner coord from the table
                                        Value input_inner_crd =
                                            table.at(input_inner.value().first, input_inner.value().second).second;
                                        //                                    if (!input_inner.has_value()) {
                                        //                                        input_inner = scope
                                        //
                                        // Set output inner crd type to be the same as input inner crd type
                                        Type output_inner_crd_type = input_inner_crd.getType();

                                        for (const auto &input_outer : input_outers)
                                        {
                                            TensorView outerTensorView;
                                            if (input_outer.first.has_value())
                                            {
                                                outerTensorView = input_outer.first.value();
                                            }
                                            else
                                            {
                                                outerTensorView = scope->indexToView.at(input_outer.second);
                                                scope->indexToView[input_outer.second] = returnedTensorView;
                                            }
                                            auto dim = mlir::getAffineDimExpr(input_outer.second.getId(), context);
                                            const auto outer_crd = table.at(outerTensorView, input_outer.second).second;
                                            input_outer_crds.push_back(outer_crd);
                                            if (!(input_outer.second == reductionVar.value()))
                                            {
                                                output_outer_crds_type.push_back(outer_crd.getType());
                                            }
                                        }

                                        auto spacc_op = rewriter.create<sam::SamSpacc>(
                                            linalgOp->getLoc(), output_inner_crd_type,
                                            TypeRange(output_outer_crds_type), val.getType(),
                                            IntegerAttr::get(IndexType::get(context), order), input_inner_crd,
                                            ValueRange(input_outer_crds), val);

                                        scope->seenInner[input_inner->second] = returnedTensorView;

                                        scope->regionOpToStream[linalgOp->getResult(0)] = spacc_op.getOutputVal();

                                        scope->indexVarMap[mlir::getAffineDimExpr(input_inner.value().second.getId(),
                                                                                  context)][0] =
                                            llvm::cast<Value>(spacc_op.getOutputVal());
                                        scope->indexVarMap[mlir::getAffineDimExpr(input_inner.value().second.getId(),
                                                                                  context)][1] =
                                            llvm::cast<Value>(spacc_op.getOutputInnerCrd());
                                        auto dim = currOpDimExprs[currOpDimExprs.size() - 1];
                                        // llvm::outs() << "DIm: " << dim << "\n";

                                        const unsigned int dimId = llvm::cast<mlir::AffineDimExpr>(dim).getPosition();
                                        // table[returnedTensorView][IndexVar(dimId)] = std::move(newCell);
                                        for (unsigned int i = 0; i < spacc_op.getOutputOuterCrds().size(); ++i)
                                        {
                                            const unsigned int stride = (i + 1) == order ? i + 1 : i + 2;
                                            // scope->indexVarMap[currOpDimExprs[currOpDimExprs.size() - 1 - stride]]
                                            // .push_back(spacc_op.getOutputOuterCrds()[i]);
                                            // scope->indexVarMap[currOpDimExprs[currOpDimExprs.size() - 1 - stride]]
                                            // .push_back(spacc_op.getOutputOuterCrds()[i]);
                                        }
                                        return std::make_tuple(spacc_op.getOutputVal(), spacc_op.getOutputInnerCrd(),
                                                               spacc_op.getOutputOuterCrds());
                                    };

                                    std::function valFunc =
                                        [=](std::tuple<mlir::Value, mlir::Value, mlir::ResultRange> spaccBundle)
                                    { return std::get<0>(spaccBundle); };

                                    std::function pairFunc =
                                        [=](std::tuple<mlir::Value, mlir::Value, mlir::ResultRange> spaccBundle)
                                        -> std::pair<std::optional<mlir::Value>, mlir::Value>
                                    { return std::make_pair(std::get<0>(spaccBundle), std::get<1>(spaccBundle)); };
                                    auto once =
                                        make_once<std::tuple<mlir::Value, mlir::Value, mlir::ResultRange>>(spaccFunc);

                                    std::ostringstream inner_ss;
                                    inner_ss << reductionVar.value();
                                    std::string reduceLabel = "Spacc" + std::to_string(order).append("_") +
                                        inner_ss.str().append("(") + valLabel.append(")");
                                    auto newCell = make_unique_cell<std::pair<std::optional<mlir::Value>, mlir::Value>>(
                                        once, pairFunc, reduceLabel);
                                    column[input_inner.value().second] = std::move(newCell);

                                    // Overwrite output intermediate result outer coordinates with output of spacc
                                    for (auto crdIter : llvm::enumerate(input_outers))
                                    {
                                        unsigned int crdIdx = crdIter.index();
                                        const auto &pair = crdIter.value();
                                        if (pair.second == reductionVar.value())
                                        {
                                            continue;
                                        }
                                        std::function outerCrdFunc =
                                            [=](std::tuple<mlir::Value, mlir::Value, mlir::ResultRange> spaccBundle)
                                            -> std::pair<std::optional<mlir::Value>, mlir::Value>
                                        {
                                            mlir::Value crd = std::get<2>(spaccBundle)[crdIdx];
                                            mlir::Value ref = crd;
                                            if (crdIdx == 0)
                                            {
                                                ref = std::get<0>(spaccBundle);
                                            }
                                            // TODO: Might need a ref gen node here
                                            return std::make_pair(ref, crd);
                                        };
                                        std::string outLabel = reduceLabel;
                                        std::string outputCrdLabel =
                                            outLabel.append("[") + std::to_string(crdIdx).append("]");
                                        auto outputCrdCell =
                                            make_unique_cell<std::pair<std::optional<mlir::Value>, mlir::Value>>(
                                                once, outerCrdFunc, outputCrdLabel);
                                        column[pair.second] = std::move(outputCrdCell);
                                    }

                                    auto newValCell =
                                        make_unique_cell<mlir::Value>(once, valFunc, reduceLabel.append(".val()"));
                                    column.setValue(std::move(newValCell));
                                    break;
                                }
                            }
                        }
                    }

                    //                scope->columnCache[tensorPair] = column;

                    scope->tableCache[tensorPair] = returnedTensorView;

                    graphTable[returnedTensorView] = std::move(column);
                    return graphTable;
                };
                returnedTensorView.setConstructTableFunction(constructTable);

                for (const auto &var : outputVars)
                {
                    scope->varToView[var.getId()] = returnedTensorView;
                }
                opTensors.push_back(returnedTensorView);
                fusedCin->uniqueVars.push_back(outputVars);
                fusedCin->tensorViews.push_back(returnedTensorView);
                for (const auto &tensor : opTensors)
                {
                    // TODO: Assign unique names to tensorviews instead of mlir::Values in cases of multiple views
                    if (llvm::find(scope->views, tensor) == scope->views.end())
                    {
                        scope->views.push_back(tensor);
                    }
                }
                // interface->uniqueVars.push_back(
                //     metastageMap[output]->getTensorViews().back().getUniqueVars());
                // metastageMap[output]->getTensorViews().push_back(TensorView(outputVars));
                // std::function<std::vector<VarASTNode>(void)> getContractions;
                std::function<void(void)> getContractions;
                std::vector<std::pair<std::pair<VarASTNode, unsigned int>, std::pair<VarASTNode, unsigned int>>>
                    joinedPairs;
                std::vector<std::pair<std::pair<VarASTNode, unsigned int>, std::pair<VarASTNode, unsigned int>>>
                    equalPairs;
                for (auto dimIter : llvm::enumerate(dimExprs))
                {
                    unsigned int dimCount = 0;
                    mlir::AffineExpr dim = dimIter.value();
                    const unsigned int dimIndex = dimIter.index();
                    UniqueVar *uniqueVar = nullptr;
                    unsigned int tensorId;
                    for (unsigned int i = 0; i < indexingMap.size(); i++)
                    {
                        // Iterate through indexing
                        std::vector<UniqueVar> &vars = opTensors[i].getUniqueVars();
                        for (auto exprIter : llvm::enumerate(indexingMap[i].getResults()))
                        {
                            auto expr = exprIter.value();
                            const unsigned int index = exprIter.index();
                            if (auto constDim = llvm::dyn_cast<mlir::AffineConstantExpr>(expr))
                            {
                                expr = dimExprs[index];
                            }
                            if (expr == dim)
                            {
                                if (!exprToUniqueVar.contains(expr))
                                {
                                    exprToUniqueVar[expr] = vars[index];
                                }
                                if (!uniqueVar)
                                {
                                    uniqueVar = &vars[index];
                                    //                                uniqueVar->setView(opTensors[i]);
                                    tensorId = i;
                                }
                                else
                                {
                                    scope->markEqual(*uniqueVar, vars[index]);
                                    //                                vars[index].setView(opTensors[i]);
                                    if (i == indexingMap.size() - 1)
                                    {
                                        equalPairs.emplace_back(std::pair(VarASTNode(*uniqueVar), tensorId),
                                                                std::pair(VarASTNode(vars[index]), i));
                                    }
                                    else
                                    {
                                        joinedPairs.emplace_back(std::pair(VarASTNode(*uniqueVar), tensorId),
                                                                 std::pair(VarASTNode(vars[index]), i));
                                    }
                                }
                                dimCount++;
                            }
                        }
                    }
                }

                getContractions = [=, scope = scope]()
                {
                    std::map<IndexVar, std::vector<VarASTNode>> joiners;
                    std::vector<std::vector<VarASTNode>> nodes;
                    std::vector<VarASTNode> outNode;
                    for (unsigned int i = 0; i < opTensors.size() - 1; ++i)
                    {
                        opTensors[i].getContractions();
                    }
                    auto joinType = sam::JoinerType::Uninit;
                    // llvm::outs() << "LINALG OP: " << linalgOp << "\n";
                    // llvm::outs().flush();
                    // std::cout << returnedTensorView << std::endl;
                    if (isGenericMultiplyOp(linalgOp).succeeded())
                    {
                        joinType = sam::JoinerType::Intersect;
                    }
                    else if (isGenericAddOp(linalgOp).succeeded())
                    {
                        joinType = sam::JoinerType::Union;
                    }
                    auto node = JoinNode(joinType);

                    if (joinType != sam::JoinerType::Uninit)
                    {
                        for (const auto &[var1, var2] : joinedPairs)
                        {
                            //                        std::set<TensorView> tensorSet;
                            node.add_children(std::make_shared<VarASTNode>(var1.first));
                            auto uVar1 = var1.first.getLeaf().get();
                            auto uVar2 = var2.first.getLeaf().get();
                            auto eqVar1 = scope->equalVars[uVar1];
                            auto eqVar2 = scope->equalVars[uVar2];

                            auto indexVar = uVar1.getLogicalIndexVar();

                            auto tensorView1 = scope->varToView.at(eqVar1.getId());
                            auto tensorView2 = scope->varToView.at(eqVar2.getId());
                            auto ogTensorView1 = scope->varToView.at(uVar1.getId());
                            auto ogTensorView2 = scope->varToView.at(uVar2.getId());
                            auto indexVarsSV1 = llvm::map_to_vector(tensorView1.getUniqueVars(), [](UniqueVar &a)
                                                                    { return a.getLogicalIndexVar(); });
                            auto indexVarsSV2 = llvm::map_to_vector(tensorView2.getUniqueVars(), [](UniqueVar &a)
                                                                    { return a.getLogicalIndexVar(); });

                            std::vector<IndexVar> indexVars1(indexVarsSV1.begin(), indexVarsSV1.end());
                            std::vector<IndexVar> indexVars2(indexVarsSV2.begin(), indexVarsSV2.end());

                            auto tensorName1 = scope->tensorViewNames.at(tensorView1);
                            auto tensorName2 = scope->tensorViewNames.at(tensorView2);
                            auto ogTensorName1 = scope->tensorViewNames.at(ogTensorView1);
                            auto ogTensorName2 = scope->tensorViewNames.at(ogTensorView2);
                            std::string joinName = joinType == mlir::sam::JoinerType::Intersect ? "^" : "u";

                            //                            llvm::outs() << tensorName1 << " " << joinName << " " <<
                            //                            tensorName2 << "\n";

                            if (llvm::find(scope->contractionOrder[indexVar], joinType) ==
                                scope->contractionOrder[indexVar].end())
                            {
                                scope->contractionOrder[indexVar].push_back(joinType);
                            }

                            auto pair1 = std::make_pair(ogTensorName1.substr(0, tensorName1.find('-')), indexVars1);
                            auto pair2 = std::make_pair(ogTensorName2.substr(0, tensorName2.find('-')), indexVars2);

                            auto joinPair = std::make_pair(indexVar, joinType);

                            if (scope->tensorContractionSet[joinPair].insert(pair1).second)
                            {
                                scope->contractionType[indexVar][joinType].insert(ogTensorView1);
                            }
                            if (scope->tensorContractionSet[joinPair].insert(pair2).second)
                            {
                                scope->contractionType[indexVar][joinType].insert(ogTensorView2);
                            }
                            auto test = scope->contractionType;
                        }

                        // TODO: Print contraction map
                        //                    for (const auto& pair : scope->contractionType)
                        //                    {
                        //                        for (const auto& contract : pair.second)
                        //                        {
                        //                            for (const auto &view : contract.second)
                        //                            {
                        //                                std::cout << "View: " << view << std::endl;
                        //                            }
                        //                        }
                        //                    }
                    }
                    for (const auto &[var1, var2] : equalPairs)
                    {
                        auto uVar1 = var1.first.getLeaf().get();
                        auto uVar2 = var2.first.getLeaf().get();
                        scope->equalVars[uVar2] = scope->equalVars[uVar1];
                    }

                    // return outNode;
                };
                returnedTensorView.setContractionFunction(getContractions);

                bool inputFacingOp = false;
                bool outputFacingOp = false;
                for (auto operandIter : llvm::enumerate(linalgOp.getDpsInputOperands()))
                {
                    auto operand = operandIter.value()->get();
                    //                llvm::outs() << "Operand: " << operand << "\n";
                    unsigned int index = operandIter.index();
                    // if (llvm::dyn_cast<mlir::BlockArgument>(operand))
                    // {
                    if (operand.getDefiningOp<arith::ConstantOp>() || llvm::dyn_cast<mlir::BlockArgument>(operand))
                    {
                        inputFacingOp = true;
                        // TODO: REMOVE BREAK
                        break;
                        for (int i = 0; i < opTensors[index].getUniqueVars().size() - 1; i++)
                        {
                            //                        scope->addPartialOrder(opTensors[index].getUniqueVars()[i],
                            //                        opTensors[index].getUniqueVars()[i + 1]);
                        }
                    }
                }

                auto iterator_types = linalgOp.getIteratorTypesArray();

                llvm::SmallVector<mlir::AffineExpr> reductionVars;
                auto it = std::find(iterator_types.begin(), iterator_types.end(), utils::IteratorType::reduction);
                if (it != iterator_types.end())
                {
                    UniqueVar var = exprToUniqueVar[dimExprs[it - iterator_types.begin()]];
                    if (scope->ignoredDims.count(returnedTensorView.getValue()))
                    {
                        // TODO: Figure out if setting reduction variables across views with the same mode order to be
                        // equal
                        //                    scope->markEqual(var,
                        //                    scope->ignoredDims.at(returnedTensorView.getValue()));
                    }
                    else
                    {
                        scope->ignoredDims[returnedTensorView.getValue()] = var;
                    }
                }

                auto func = linalgOp->getParentOfType<func::FuncOp>();

                if (!llvm::dyn_cast<func::ReturnOp>(func.getBlocks().front().getOperations().back()).getOperand(0) ||
                    true)
                // returnedVal != linalgOp->getResult(0) || true)
                {
                    if (it != iterator_types.end())
                    {
                        mlir::AffineExpr reductionVar = dimExprs[it - iterator_types.begin()];
                        UniqueVar reduceUniqueVar = exprToUniqueVar[reductionVar];
                        scope->uniqueRedVars.push_back(reduceUniqueVar);
                        if (isGenericMatmulOp(linalgOp).succeeded())
                        {
                            /* scope->addPartialOrder(exprToUniqueVar[dimExprs[0]], reduceUniqueVar); */
                            /* scope->addPartialOrder(reduceUniqueVar, exprToUniqueVar[dimExprs[1]]); */
                            // llvm::outs() << "Linalg Op: " << linalgOp << "\n";
                            // std::cout << exprToUniqueVar[dimExprs[0]] << " -> " << reduceUniqueVar << " -> " <<
                            // exprToUniqueVar[dimExprs[2]] << std::endl;
                        }
                        else
                        {
                            for (auto exprIter : llvm::enumerate(dimExprs))
                            {
                                auto expr = exprIter.value();
                                unsigned int exprIdx = exprIter.index();
                                if (exprIdx == dimExprs.size() - 1)
                                {
                                    continue;
                                }
                                /* scope->addPartialOrder(exprToUniqueVar[dimExprs[exprIdx]],
                                 * exprToUniqueVar[dimExprs[exprIdx + 1]]); */
                                // if (exprIdx == )
                                if (expr == reductionVar)
                                {
                                    continue;
                                }
                                // TODO: Add back
                                //                            scope->addPartialOrder(exprToUniqueVar[expr],
                                //                            reduceUniqueVar);
                                // TODO: Hardcoded for kij order
                                //    scope->addPartialOrder(reduceUniqueVar, exprToUniqueVar[expr]);
                            }
                        }
                    }
                }

                //                Value returnedVal = linalgOp->getResult(0);
                //                for (auto &use: returnedVal.getUses()) {
                //                    if (Operation *user = use.getOwner(); llvm::isa<func::ReturnOp>(user)) {
                //                        outputFacingOp = true;
                //                        for (unsigned int i = 0;
                //                             i <
                //                             metastageMap[returnedVal]->getTensorViews().back().getUniqueVars().size()
                //                             - 1; i++) {
                //                            //                        scope->addPartialOrder(metastageMap[returnedVal]
                //                            //                        ->getTensorViews()
                //                            //                        .back()
                //                            //                        .getUniqueVars()[i],
                //                            //                        metastageMap[returnedVal]
                //                            //                        ->getTensorViews()
                //                            //                        .back()
                //                            //                        .getUniqueVars()[i + 1]);
                //                        }
                //                    }
                //                }

                // if (metastageMap.contains(linalgOp->getOpResult(0))) {

                //   llvm::outs() << "FOUND: " << outputs[0] << "ALready\n";
                //   llvm::outs() << linalgOp << "\n";
                //   return;
                // }

                //             inputFacingOp = true;
                // if (inputFacingOp || outputFacingOp)
                if (inputFacingOp)
                {
                    for (unsigned int i = 0; i < dimExprs.size() - 1; i++)
                    {
                        // llvm::outs() << "Num Views: " << interface->getTensorViews().size()
                        //              << "\n";
                        // std::cout << interface->getTensorViews().back() << std::endl;

                        // llvm::outs() << "Trying to push\n";
                        // for (auto op : scope->operands) {
                        // std::cout << "Operands: " << op << std::endl;
                        // }
                        // if (llvm::find(scope->operands, fusedCin->getTensorViews().back()) !=
                        // fusedCin->getTensorViews().end())
                        {
                            // llvm::outs() << "Pushing constraints\n";
                            // TODO: HARDCODED BASED ON USER SPECIFICATION
                            //          Currently fails for models that use matmul (inner-prod) by default
                            //    scope->addPartialOrder(exprToUniqueVar[dimExprs[i]],
                            //   exprToUniqueVar[dimExprs[i + 1]]);
                        }
                    }
                    scope->operands.push_back(fusedCin->getTensorViews().back());
                }
                for (mlir::AffineExpr dim : dimExprs)
                {
                    fusedCin->loopOrder.push_back(exprToUniqueVar[dim]);
                }
                for (auto dims : dimExprs)
                {
                    scope->addNode(exprToUniqueVar.at(dims));
                }
                // auto newTensorView =
                //     TensorView(interface->uniqueVars.back(), linalgOp->getOpResult(0));

                // std::vector<UniqueVar> newLoopOrder;
                // for (mlir::AffineExpr dim : dimExprs) {
                //   newLoopOrder.push_back(exprToUniqueVar[dim]);
                // }
                // Update loop order for operands of current op as well
                // TODO: Figure out loop order
                // for (auto val : inputs) {
                //   metastageMap[val]->getTensorViews().back().loopOrder = newLoopOrder;
                // }
                // newTensorView.loopOrder = newLoopOrder;

                // Get map from unique vars, which contains index var mappings, to mlir
                // Affine Exprs in case we need to manipulate the loop orders
                // newTensorView.exprToUniqueVar = exprToUniqueVar;

                return returnedTensorView;
            };
            metastageMap[result] = fusedCin;
        }
    }

    // TODO Fix to add interface for each output instead of hardcoding for one
}

class LinalgToSamPass : public mlir::impl::LinalgToSamBase<LinalgToSamPass>
{
    void runOnOperation() override;

    LogicalResult runOnFunction(func::FuncOp func);

private:
    bool useUserInput;
    bool calculateHeuristic;

public:
    explicit LinalgToSamPass(bool useUserInput, bool calculateHeuristic) :
        useUserInput(useUserInput), calculateHeuristic(calculateHeuristic)
    {
    }
};

LogicalResult LinalgToSamPass::runOnFunction(func::FuncOp func)
{
    MLIRContext &context = getContext();

    ConversionTarget target(context);
    // target.addIllegalDialect<linalg::LinalgDialect>();
    target.addLegalDialect<sam::SamDialect>();

    RewritePatternSet patterns(&context);

    return applyPatternsGreedily(func, std::move(patterns));
}

void LinalgToSamPass::runOnOperation()
{
    func::FuncOp func = getOperation();
    // IRRewriter rewriter(func.getContext());
    StageRewriter rewriter(func.getContext());
    llvm::DenseMap<mlir::Value, std::shared_ptr<FusedCIN>> metastageMap;
    llvm::SmallVector<mlir::Value> tensorLst;

    const auto scope = std::make_shared<AnalysisScope>(useUserInput);

    // Construct input lambdas for input arguments
    auto context = func.getContext();
    for (auto &blockArg : func.front().getArguments())
    {
        constructInputLambdas(rewriter, func, blockArg, scope, metastageMap);
    }

    for (Block &block : func)
    {
        for (Operation &op : block.getOperations())
        {
            if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op))
            {
                rewriter.setInsertionPoint(linalgOp);
                Operation *opPtr = &op;
                constructLambdas(rewriter, opPtr, scope, metastageMap, tensorLst);
            }
            else if (auto concatOp = llvm::dyn_cast<tensor::ConcatOp>(op))
            {
                rewriter.setInsertionPoint(concatOp);
                Operation *opPtr = &op;
                constructLambdas(rewriter, opPtr, scope, metastageMap, tensorLst);
            }
            else if (auto constOp = llvm::dyn_cast<arith::ConstantOp>(op))
            {
                Value constTensor = constOp.getResult();
                constructInputLambdas(rewriter, func, constTensor, scope, metastageMap);
            }
            else if (auto emptyOp = llvm::dyn_cast<tensor::EmptyOp>(op))
            {
                Value emptyTensor = emptyOp.getResult();
                constructInputLambdas(rewriter, func, emptyTensor, scope, metastageMap);
            }
        }
    }

    std::vector<IndexVar> resultIndexVars;
    std::vector<IndexVar> indexVars;
    llvm::SmallVector<mlir::AffineMap> indexingMaps;

    auto err = fuseOpsInDispatchGroup(func, metastageMap, scope, rewriter, calculateHeuristic);

    if (err.failed())
        return signalPassFailure();
}

std::unique_ptr<mlir::Pass> mlir::createLinalgToSamPass(bool useUserInput, bool calculateHeuristic)
{
    return std::make_unique<LinalgToSamPass>(useUserInput, calculateHeuristic);
}
