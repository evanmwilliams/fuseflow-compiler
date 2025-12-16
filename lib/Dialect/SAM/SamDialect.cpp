#include "lib/Dialect/SAM/SamDialect.h"
#include "lib/Dialect/SAM/SamOps.h"
#include "lib/Dialect/SAM/SamTypes.h"

#include "mlir/IR/Builders.h"
#include "llvm/ADT/TypeSwitch.h"

#include "lib/Dialect/SAM/SamDialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "lib/Dialect/SAM/SamTypes.cpp.inc"

#define GET_OP_CLASSES
#include "lib/Dialect/SAM/SamOps.cpp.inc"

#include "lib/Dialect/SAM/SamEnumDefs.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "lib/Dialect/SAM/SamAttrDefs.cpp.inc"

namespace mlir::sam {

void SamDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "lib/Dialect/SAM/SamTypes.cpp.inc"
      >();

  addOperations<
#define GET_OP_LIST
#include "lib/Dialect/SAM/SamOps.cpp.inc"
      >();

  addAttributes<
#define GET_ATTRDEF_LIST
#include "lib/Dialect/SAM/SamAttrDefs.cpp.inc"
      >();
}

SamStreamEncodingAttr getSamStreamEncoding(Type type) {
  if (auto ttp = llvm::dyn_cast<RankedTensorType>(type))
    return llvm::dyn_cast_or_null<SamStreamEncodingAttr>(ttp.getEncoding());
  // if (auto mdtp = llvm::dyn_cast<StorageSpecifierType>(type))
  //   return mdtp.getEncoding();
  return nullptr;
}

} // namespace mlir::sam
