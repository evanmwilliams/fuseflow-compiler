#ifndef LIB_DIALECT_SAM_SAMTYPES_TD
#define LIB_DIALECT_SAM_SAMTYPES_TD

include "SamDialect.td"
include "mlir/IR/AttrTypeBase.td"

class Sam_Type<string name, string typeMnemonic> : TypeDef<Sam_Dialect, name> {
    let mnemonic = typeMnemonic;
}

def Placeholder : Sam_Type<"Placeholder", "noop"> {
    let summary = "";
    let description = [{}];
}

#endif /* LIB_DIALECT_SAM_SAMTYPES_TD */
