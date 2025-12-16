#ifndef LIB_DIALECT_SAM_SAMOPS
#define LIB_DIALECT_SAM_SAMOPS

#include "lib/Dialect/SAM/SamDialect.h"
#include "lib/Dialect/SAM/SamTypes.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/IR/BuiltinOps.h"   // from @llvm-project
#include "mlir/IR/BuiltinTypes.h" // from @llvm-project
//#include "mlir/include/mlir/IR/Dialect.h"      // from @llvm-project
#include "mlir/IR/Dialect.h"      // from @llvm-project

#define GET_OP_CLASSES
#include "lib/Dialect/SAM/SamOps.h.inc"

#endif /* LIB_DIALECT_SAM_SAMOPS */
