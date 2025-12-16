#include "lib/Dialect/STEP/StepDialect.h"
#include "lib/Dialect/STEP/StepOps.h"
#include "lib/Dialect/STEP/StepTypes.h"

#include "mlir/include/mlir/IR/Builders.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"

#include "lib/Dialect/STEP/StepDialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "lib/Dialect/STEP/StepTypes.cpp.inc"

#define GET_OP_CLASSES
#include "lib/Dialect/STEP/StepOps.cpp.inc"

#include "lib/Dialect/STEP/StepEnumDefs.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "lib/Dialect/STEP/StepAttrDefs.cpp.inc"

namespace mlir::step {

void StepDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "lib/Dialect/STEP/StepTypes.cpp.inc"
      >();

  addOperations<
#define GET_OP_LIST
#include "lib/Dialect/STEP/StepOps.cpp.inc"
      >();

  addAttributes<
#define GET_ATTRDEF_LIST
#include "lib/Dialect/STEP/StepAttrDefs.cpp.inc"
      >();
}

} // namespace mlir::step
