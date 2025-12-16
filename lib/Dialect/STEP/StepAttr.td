#ifndef LIB_DIALECT_STEP_STEPATTR_TD
#define LIB_DIALECT_STEP_STEPATTR_TD

include "StepDialect.td"
include "StepEnums.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"



class Step_Attr<string name, string attrMnemonic> : AttrDef<Step_Dialect, name, [TypedAttrInterface]> {
    let mnemonic = attrMnemonic;
}

def ArgTypeKindAttr
    : EnumAttr<Step_Dialect, ArgTypeKindEnum,
               "ArgTypeKind"> {
   let mnemonic = "arg_type";
   let assemblyFormat = "`<` $value `>`";
}

def TensorKindAttr
    : EnumAttr<Step_Dialect, TensorKindEnum,
               "TensorKind"> {
   let mnemonic = "kind";
   let assemblyFormat = "`<` $value `>`";
}

#endif /* LIB_DIALECT_STEP_STEPATTR_TD */
