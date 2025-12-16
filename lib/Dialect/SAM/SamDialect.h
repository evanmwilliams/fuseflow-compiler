#ifndef LIB_DIALECT_SAM_SAMDIALECT
#define LIB_DIALECT_SAM_SAMDIALECT

#include "mlir/IR/DialectImplementation.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
// #include "mlir/Interfaces/CopyOpInterface.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

#include "lib/Dialect/SAM/SamDialect.h.inc"

#include "lib/Dialect/SAM/SamEnumDefs.h.inc"

#define GET_ATTRDEF_CLASSES
#include "lib/Dialect/SAM/SamAttrDefs.h.inc"

namespace mlir::sam
{
    SamStreamEncodingAttr getSamStreamEncoding(Type type);
}

#endif /* LIB_DIALECT_SAM_SAMDIALECT */
