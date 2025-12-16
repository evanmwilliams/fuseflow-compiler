
#include "lib/Target/ProtoEmitter/ProtoEmitter.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h" // from @llvm-project
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"

#include "lib/Dialect/SAM/SamDialect.h"

int main(int argc, char **argv) {
  mlir::sam::registerToProtoPass();

  return failed(mlir::mlirTranslateMain(argc, argv, "SAM Translation Tool"));
}