#ifndef LIB_DIALECT_STEP_STEPENUM_TD
#define LIB_DIALECT_STEP_STEPENUM_TD

include "mlir/IR/EnumAttr.td"
include "StepDialect.td"

def ArgTypeKindEnum : I32EnumAttr<"ArgTypeKind", "stuff", [
    I32EnumAttrCase<"Value", 0>,
    I32EnumAttrCase<"Ref", 1>,
    I32EnumAttrCase<"MutRef", 2>
]>{
    let genSpecializedAttr = 0;
    let cppNamespace = Step_Dialect.cppNamespace;
}

def TensorKindEnum : I32EnumAttr<"TensorKind", "stuff", [
    I32EnumAttrCase<"Stream", 0>,
    I32EnumAttrCase<"Buffer", 1>,
]>{
    let genSpecializedAttr = 0;
    let cppNamespace = Step_Dialect.cppNamespace;
}



#endif /* LIB_DIALECT_STEP_STEPENUM_TD */