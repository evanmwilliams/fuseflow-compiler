#ifndef LIB_DIALECT_SAM_SAMDIALECT_TD
#define LIB_DIALECT_SAM_SAMDIALECT_TD

include "mlir/IR/DialectBase.td"

def Sam_Dialect : Dialect {
    let name = "sam";
    let summary = "A dialect for SAM Extended";
    let description = [{}];

    let cppNamespace = "::mlir::sam";
    let useDefaultTypePrinterParser = 1;
    let useDefaultAttributePrinterParser = 1;

  let dependentDialects = [
    // "mlir::sam::SamDialect",
    // "mlir::sparse_tensor::SparseTensorDialect",
  ];
}

#endif /* LIB_DIALECT_SAM_SAMDIALECT_TD */
