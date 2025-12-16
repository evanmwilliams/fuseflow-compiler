#ifndef LIB_DIALECT_STEP_STEPTYPES_TD
#define LIB_DIALECT_STEP_STEPTYPES_TD

include "StepDialect.td"
include "mlir/IR/AttrTypeBase.td"

class Step_Type<string name, string typeMnemonic> : TypeDef<Step_Dialect, name> {
    let mnemonic = typeMnemonic;
}

def Placeholder : Step_Type<"Placeholder", "noop"> {
    let summary = "";
    let description = [{}];
}

#endif /* LIB_DIALECT_STEP_STEPTYPES_TD */
