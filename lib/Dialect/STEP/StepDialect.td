#ifndef LIB_DIALECT_STEP_STEPDIALECT_TD
#define LIB_DIALECT_STEP_STEPDIALECT_TD

include "mlir/IR/DialectBase.td"

def Step_Dialect : Dialect {
    let name = "step";
    let summary = "A dialect for higher-order parallel patterns";
    let description = [{}];

    let cppNamespace = "::mlir::step";
    let useDefaultTypePrinterParser = 1;
    let useDefaultAttributePrinterParser = 1;

}

#endif /* LIB_DIALECT_STEP_STEPDIALECT_TD */
