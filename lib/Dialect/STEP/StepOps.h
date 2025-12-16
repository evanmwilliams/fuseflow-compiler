#ifndef LIB_DIALECT_STEP_STEPOPS
#define LIB_DIALECT_STEP_STEPOPS

#include "lib/Dialect/STEP/StepDialect.h"
#include "lib/Dialect/STEP/StepTypes.h"
#include "mlir/include/mlir/IR/BuiltinOps.h"    // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Dialect.h"       // from @llvm-project

#define GET_OP_CLASSES
#include "lib/Dialect/STEP/StepOps.h.inc"

#endif /* LIB_DIALECT_STEP_STEPOPS */
