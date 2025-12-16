#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"

#include "lib/Conversion/SparseToSam/LinalgToSam.h"
#include "lib/Dialect/SAM/SamDialect.h"
#include "lib/Transforms/StreamParallelizer/Passes.h"
#include "lib/Transforms/StreamVectorizer/Passes.h"
#include "mlir/Transforms/Passes.h"

int main(int argc, char **argv)
{
    mlir::DialectRegistry registry;
    mlir::registerAllDialects(registry);
    registry.insert<mlir::sam::SamDialect, mlir::sparse_tensor::SparseTensorDialect>();

    mlir::sam::registerStreamVectorizerPipeline();
    mlir::sam::registerStreamParallelizerPipeline();

    mlir::PassPipelineRegistration<> pipeline1(
        "uninline-ops", "Uninlines ops into separate functions along with their inputs",
        [&](mlir::OpPassManager &pm) { pm.addPass(mlir::createFusionDispatchGroupsPass()); });

    mlir::PassPipelineRegistration<mlir::LinalgToSamPipelineOptions> pipeline(
        "linalg-to-sam", "Converts Linalg to SAMML",
        [&](mlir::OpPassManager &pm, const mlir::LinalgToSamPipelineOptions &options)
        {
            // pm.addPass(mlir::createFusionDispatchGroupsPass());
            pm.addNestedPass<mlir::func::FuncOp>(mlir::createConvertFillOpsPass());
            pm.addNestedPass<mlir::func::FuncOp>(
                mlir::createLinalgToSamPass(options.useUserInput, options.getHeuristic));

            pm.addPass(mlir::createCanonicalizerPass());
            pm.addPass(mlir::createCSEPass());
            pm.addNestedPass<mlir::func::FuncOp>(mlir::createRemoveJoinerDuplicateStreamsPass());
             pm.addNestedPass<mlir::func::FuncOp>(mlir::createInsertIterateLocatePass()); 
            pm.addPass(mlir::createCanonicalizerPass());
            pm.addPass(mlir::createCSEPass());
            //                                                                                  pm.addNes
        });
    mlir::registerAllPasses();

    return mlir::asMainReturnCode(mlir::MlirOptMain(argc, argv, "SAM Pass Driver", registry));
}
