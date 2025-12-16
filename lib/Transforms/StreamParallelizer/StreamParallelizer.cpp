#include "lib/Transforms/StreamParallelizer/Passes.h"
#include <memory>
#include <iostream>

#include "lib/Dialect/SAM/SamOps.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Pass/PassManager.h" // from @llvm-project
#include "mlir/Pass/PassRegistry.h"

#include "mlir/IR/Block.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/TypeSwitch.h" // from @llvm-project

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir {
    namespace sam {
#define GEN_PASS_DEF_STREAMPARALLELIZER
#define GEN_PASS_DEF_STREAMPARALLELIZERANALYSIS
#define GEN_PASS_DEF_STREAMPARALLELIZERUNROLL

#include "lib/Transforms/StreamParallelizer/Passes.h.inc"

        struct StreamParallelizerAnalysis
                : public impl::StreamParallelizerAnalysisBase<StreamParallelizerAnalysis> {
            using StreamParallelizerAnalysisBase::StreamParallelizerAnalysisBase;

            StreamParallelizerAnalysis(unsigned level, unsigned par)
                    : stream_level(level), par_factor(par) {}

            void runOnOperation() override;

        private:
            unsigned stream_level{};
            unsigned par_factor{};
        };

        struct StreamParallelizer
                : public impl::StreamParallelizerBase<StreamParallelizer> {
            using StreamParallelizerBase::StreamParallelizerBase;

            StreamParallelizer(unsigned level, unsigned factor)
                    : stream_level(level), par_factor(factor) {}

            void runOnOperation() override;

        private:
            unsigned stream_level{};
            unsigned par_factor{};
        };

        struct StreamParallelizerUnroll
                : public impl::StreamParallelizerUnrollBase<StreamParallelizerUnroll> {
            using StreamParallelizerUnrollBase::StreamParallelizerUnrollBase;

            explicit StreamParallelizerUnroll(unsigned level, unsigned factor) : stream_level(level),
                                                                                 par_factor(factor) {}

            void runOnOperation() override;

        private:
            unsigned stream_level{};
            unsigned par_factor{};
        };

        void sortModuleOpWithTopologicalSort(Operation *op) {
            auto moduleOp = llvm::cast<ModuleOp>(op);

            // Iterate over each block in the module
            for (auto &block: moduleOp.getBodyRegion()) {
                llvm::SetVector<mlir::Operation *> opsToSort;

                for (auto &localOp: block.getOperations()) {
                    if (auto funcOp = llvm::dyn_cast<mlir::func::FuncOp>(localOp)) {
                        bool sorted = sortTopologically(&funcOp.getBody().front());
                        if (sorted) {
                            for (auto &oper: funcOp.getBody().front()) {
//                                llvm::outs() << "Operation: " << oper << "\n";
                            }
                        }
                    }
                }

                // Apply topological sort to each block

            }
        }

// Skeleton pass for inserting no ops as placeholders for parallel regions
// TODO: Expand to express different parallelization strategies
        void StreamParallelizerAnalysis::runOnOperation() {
            if (par_factor == 1) {
                return;
            }
            auto moduleOp = getOperation();
            sortModuleOpWithTopologicalSort(moduleOp);
            auto context = moduleOp->getContext();
            mlir::IRRewriter builder(context);
            int64_t index_var;
            bool end_region = false;
//            llvm::outs() << "Before par add\n";
//            moduleOp->dump();

            // Module walk to insert no-ops to indicate start and end of parallel region
            moduleOp->walk([&](Operation *oper) {
                llvm::TypeSwitch<Operation *, void>(oper)
                        .Case<sam::SamFiberLookup>([&](auto op) {
                            index_var = op.getIndexVarAttr().getInt();
                            if (index_var == stream_level) {
                                // If next node is a repeat, start parallelizer region right before
                                // it
                                // TODO: Need to generalize for ops on more than 2 tensors
                                if (isa<sam::SamRepeat, sam::SamGenerator>(oper->getNextNode())) {
                                    builder.setInsertionPointAfter(op);
                                } else if (isa<sam::SamFiberLookup>(oper->getNextNode())) {
                                    builder.setInsertionPointAfter(oper->getNextNode());
                                }
                                for (auto uses: oper->getResult(0).getUsers()) {
//                                    llvm::outs() << "Uses: " << *uses << "\n";
                                    if (isa<sam::SamJoiner>(uses)) {
                                        builder.setInsertionPointAfter(uses);
                                        break;
                                    }
                                }

                                builder.create<SamParallelBegin>(op.getLoc());
                            }
                        })
                        .Case<sam::SamFiberWrite>([&](auto op) {
                            if (!end_region && op.getWriteOp() == sam::WriteType::Crd) {
                                builder.setInsertionPoint(op);
                                builder.create<SamParallelEnd>(op.getLoc());
                                end_region = true;
                            }
                        });
            });
//            llvm::outs() << "\n";
//            llvm::outs() << "After par add\n";
//            moduleOp->dump();
        }

        void StreamParallelizer::runOnOperation() {
            if (par_factor == 1) {
                return;
            }
            auto moduleOp = getOperation();
            auto context = moduleOp.getContext();
            mlir::IRRewriter builder(context);

//            moduleOp.dump();

            sam::SamParallel parOp;
            bool in_region = false;
            SetVector<Operation *> opsToMove;
            moduleOp.walk([&](Operation *oper) {
                llvm::TypeSwitch<Operation *, void>(oper)
                        .Case<sam::SamParallelBegin>([&](auto op) {
                            builder.setInsertionPointAfter(op);
                            in_region = true;
                        })
                        .Case<sam::SamParallelEnd>([&](auto op) {
                            op.erase();
                            in_region = false;
                        });
                if (in_region) {
                    if (isa<sam::SamParallelBegin>(oper)) {
                        oper->erase();
                    } else {
                        opsToMove.insert(oper);
                    }
                }
            });

            SmallPtrSet<Operation *, 8> rangeSet;
            for (Operation *op: opsToMove) {
                rangeSet.insert(op);
            }

            // Find ops with dependencies outside of the parallel region
            SmallVector<Value, 8> results;
            SmallVector<Type, 8> results_types;
            for (Operation *op: llvm::make_early_inc_range(opsToMove)) {
                for (Value res: op->getResults()) {
                    for (Operation *user: res.getUsers()) {
                        if (rangeSet.count(user) == 0) {
                            results.push_back(res);
                            results_types.push_back(res.getType());
                            break;
                        }
                    }
                }
            }

            // Create parallel op instantiated with parallel factor
            parOp = builder.create<sam::SamParallel>(
                    opsToMove[0]->getLoc(), results_types,
                    IntegerAttr::get(IndexType::get(context), par_factor));
            parOp.getParallelBody().push_back(new mlir::Block);

            Block *newBlock = &parOp.getParallelBody().front();
            for (auto oper: opsToMove) {
                newBlock->getOperations().splice(newBlock->end(),
                                                 oper->getBlock()->getOperations(), oper);
            }

            SetVector<Value> region_dependencies;
            SetVector<Type> dependencies_type;
            DenseMap<Value, unsigned> dep_to_operand_idx;
            Region &par_region = parOp.getParallelBody();
            for (auto &op: par_region.getBlocks().begin()->getOperations()) {
                for (auto operand_it: llvm::enumerate(op.getOperands())) {
                    if (operand_it.value().getParentRegion() != &par_region &&
                        !operand_it.value().getDefiningOp<sam::SamGenerator>()) {

                        if (auto joiner = operand_it.value().getDefiningOp<sam::SamJoiner>()) {
//                            llvm::outs() << "Fiber: " << joiner << "\n";
                            SamStreamEncodingAttr enc = getSamStreamEncoding(joiner.getOutputCrd().getType());
                            std::string var = enc.getVar().str();
//                            llvm::outs() << "Joiner var: " << var << "\n";
//                            llvm::outs() << "Stream level: " << this->stream_level << "\n";
                            if (var != "d" + std::to_string(this->stream_level)) {
                                continue;
                            }
                        }
                        region_dependencies.insert(operand_it.value());
                        dependencies_type.insert(operand_it.value().getType());
                        dep_to_operand_idx[operand_it.value()] = operand_it.index();
                    }
                }
            }

            SmallVector<SamScatter> scatters;
            builder.setInsertionPoint(parOp);
            for (auto it: llvm::enumerate(dependencies_type)) {
                scatters.push_back(builder.create<SamScatter>(
                        parOp.getLoc(), it.value(), region_dependencies[it.index()],
                        IntegerAttr::get(IndexType::get(context), par_factor),
                        IntegerAttr::get(IndexType::get(context), stream_level)));
            }

            // Replace uses of results from before the parallel region, but only results
            // inside the region
            for (auto it: llvm::enumerate(region_dependencies)) {
                for (auto &use: llvm::make_early_inc_range(it.value().getUses())) {
                    if (use.getOwner()->getParentRegion() == &parOp.getParallelBody()) {
                        use.set(scatters[it.index()].getResult(0));
                    }
                }
            }

            // Create join ops
            // TODO: Might need a variadic type for the input
            builder.setInsertionPointAfter(parOp);
            SmallVector<SamGather> gathers;
            for (auto it: llvm::enumerate(results_types)) {
                SmallVector<Type> res_type{it.value()};
                SmallVector<Value> res_val{parOp.getResults()[it.index()]};
                gathers.push_back(builder.create<SamGather>(
                        parOp.getLoc(), res_type, res_val,
                        IntegerAttr::get(IndexType::get(context), par_factor),
                        IntegerAttr::get(IndexType::get(context), stream_level)));
            }

            // Remap results of ops that depend on ops inside the region, or need to be
            // updated
            for (auto it: llvm::enumerate(results)) {
                it.value().replaceAllUsesWith(gathers[it.index()].getResult());
            }

            // Set terminator inside parallel region to retun results with dependencies
            // outside of the region
            Operation *term = &parOp.getParallelBody().front().back();
            if (!results.empty()) {
                builder.setInsertionPointAfter(term);
                builder.create<sam::SamYield>(term->getLoc(), results);
            }
        }

        void StreamParallelizerUnroll::runOnOperation() {
            if (par_factor == 1) {
                return;
            }

            auto moduleOp = getOperation();

            auto context = moduleOp->getContext();
            mlir::IRRewriter builder(context);

            SmallVector<SmallVector<Value>> join_inputs(4);
            SmallVector<Type> join_types;

            // Module walk to insert no-ops to indicate start and end of parallel region
            moduleOp->walk([&](Operation *oper) {
                llvm::TypeSwitch<Operation *, void>(oper)
                        .Case<sam::SamParallel>([&](auto parOp) {
                            mlir::OpBuilder::InsertionGuard guard(builder);
                            builder.setInsertionPointAfter(parOp);
                            for (unsigned par = 0; par < par_factor; par++) {
                                SmallVector<Operation *> cloned_ops;

                                mlir::IRMapping mapping;
                                Operation *terminator;
                                for (auto &op: parOp.getParallelBody().begin()->getOperations()) {
                                    auto cloned = builder.clone(op);
                                    if (isa<sam::SamYield>(op)) {
                                        terminator = cloned;
                                    }
                                    cloned_ops.push_back(cloned);

                                    for (auto [old_result, new_result]:
                                            llvm::zip(op.getResults(), cloned->getResults())) {
                                        mapping.map(old_result, new_result);
                                    }
                                }

                                for (Operation *cloned: cloned_ops) {
                                    for (auto &opOper: cloned->getOpOperands()) {
                                        if (Value m = mapping.lookupOrNull(opOper.get())) {
                                            builder.modifyOpInPlace(cloned, [&]() { opOper.set(m); });
                                        }
                                    }
                                }

                                for (unsigned i = 0; i < cloned_ops.size() - 1; i++) {
                                    auto op = cloned_ops[i];
                                    for (auto operand_it: llvm::enumerate(op->getOperands())) {
                                        if (auto definingOp =
                                                operand_it.value().getDefiningOp<sam::SamScatter>()) {
                                            auto index_var = definingOp.getIndexVarAttr().getInt();
                                            if (index_var != stream_level) {
                                                continue;
                                            }
                                            op->setOperand(operand_it.index(), definingOp.getResult(par));
                                        }
                                    }
                                    builder.getInsertionBlock()->getOperations().splice(
                                            builder.getInsertionPoint(), op->getBlock()->getOperations(),
                                            op);
                                }
                                for (auto it: llvm::enumerate(parOp.getResult())) {
                                    it.value().replaceAllUsesWith(terminator->getOperand(it.index()));
//                                    moduleOp->dump();
//                                    if (llvm::find(join_inputs, it.index()) == join_inputs.end()) {
//                                        continue;
//                                    }
                                    join_inputs[it.index()].push_back(
                                            terminator->getOperand(it.index()));
                                }
                                terminator->erase();
                            }
                            parOp.erase();
//                            moduleOp->dump();
                        })
                        .Case<sam::SamScatter>([&](auto op) {
                            auto index_var = op.getIndexVarAttr().getInt();
                            if (index_var == stream_level) {
                                auto results = op.getResultTypes();
                                SmallVector res_types(par_factor, results[0]);
                                builder.setInsertionPoint(op);
                                auto new_fork = builder.create<SamScatter>(
                                        op.getLoc(), res_types, op.getInputStream(),
                                        IntegerAttr::get(IndexType::get(context), par_factor),
                                        IntegerAttr::get(IndexType::get(context), stream_level));
                                for (auto &use:
                                        llvm::make_early_inc_range(op.getResults().getUses())) {
                                    use.set(new_fork.getResults()[0]);
                                }
                                op.erase();
                            }
                        })
                        .Case<sam::SamGather>([&](auto op) {
                            auto res_type = op.getResult().getType();
                            auto index_var = op.getIndexVarAttr().getInt();
                            if (index_var == stream_level) {
                                builder.setInsertionPoint(op);
                                auto new_join = builder.create<SamGather>(
                                        op.getLoc(), res_type, join_inputs.front(),
                                        IntegerAttr::get(IndexType::get(context), par_factor),
                                        IntegerAttr::get(IndexType::get(context), stream_level));
                                join_inputs.erase(join_inputs.begin());

                                for (auto &use:
                                        llvm::make_early_inc_range(op.getResult().getUses())) {
                                    use.set(new_join.getResult());
                                }
                                op.erase();
                            }
                        });
            });
            // exit(0);
        }

        std::unique_ptr<mlir::Pass> createStreamParallelizer(unsigned streamLevel,
                                                             unsigned factor) {
            return std::make_unique<mlir::sam::StreamParallelizer>(streamLevel, factor);
        }

        std::unique_ptr<mlir::Pass>
        createStreamParallelizerUnroll(unsigned streamLevel, unsigned par_factor) {
            return std::make_unique<mlir::sam::StreamParallelizerUnroll>(streamLevel, par_factor);
        }

        std::unique_ptr<mlir::Pass>
        createStreamParallelizerAnalysis(unsigned streamLevel, unsigned par_factor) {
            return std::make_unique<mlir::sam::StreamParallelizerAnalysis>(streamLevel,
                                                                           par_factor);
        }

        void registerStreamParallelizerPipeline() {
            PassPipelineRegistration<StreamParallelizerPipelineOptions>(
                    "stream-parallelizer", "The stream parallelizer pipeline",
                    [&](OpPassManager &pm, const StreamParallelizerPipelineOptions &options) {
                        pm.addPass(createStreamParallelizerAnalysis(options.stream_level,
                                                                    options.par_factor));
                        pm.addPass(
                                createStreamParallelizer(options.stream_level, options.par_factor));
                        pm.addPass(createStreamParallelizerUnroll(options.stream_level, options.par_factor));
                    });
        }

    } // namespace sam
} // namespace mlir
