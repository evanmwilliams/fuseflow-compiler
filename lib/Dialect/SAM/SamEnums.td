#ifndef LIB_DIALECT_SAM_SAMENUM_TD
#define LIB_DIALECT_SAM_SAMENUM_TD

include "mlir/IR/EnumAttr.td"
include "SamDialect.td"

def StreamKindEnum : I32EnumAttr<"StreamKind", "stuff", [
    I32EnumAttrCase<"Crd", 0>,
    I32EnumAttrCase<"Ref", 1>,
    I32EnumAttrCase<"Val", 2>
]>{
    let genSpecializedAttr = 0;
    let cppNamespace = Sam_Dialect.cppNamespace;
}

def JoinerTypeEnum : I32EnumAttr<"JoinerType", "stuff", [
    I32EnumAttrCase<"Intersect", 0>,
    I32EnumAttrCase<"Union", 1>,
    I32EnumAttrCase<"Uninit", 2>,
]>{
    let genSpecializedAttr = 0;
    let cppNamespace = Sam_Dialect.cppNamespace;
}

def WriteTypeEnum : I32EnumAttr<"WriteType", "stuff", [
    I32EnumAttrCase<"Crd", 0>,
    I32EnumAttrCase<"Val", 1>,
]>{
    let genSpecializedAttr = 0;
    let cppNamespace = Sam_Dialect.cppNamespace;
}

def ReduceTypeEnum : I32EnumAttr<"ReduceType", "stuff", [
    I32EnumAttrCase<"Add", 0>,
    I32EnumAttrCase<"Max", 1>,
    I32EnumAttrCase<"Min", 2>,
    I32EnumAttrCase<"AddSum", 3>,
]>{
    let genSpecializedAttr = 0;
    let cppNamespace = Sam_Dialect.cppNamespace;
}

def OpTypeEnum : I32EnumAttr<"OpType", "stuff", [
    I32EnumAttrCase<"Add", 0>,
    I32EnumAttrCase<"Sub", 1>,
    I32EnumAttrCase<"Mul", 2>,
    I32EnumAttrCase<"Div", 3>,
    I32EnumAttrCase<"Reduce", 4>,
    I32EnumAttrCase<"Exp", 5>,
    I32EnumAttrCase<"Sin", 6>,
    I32EnumAttrCase<"Cos", 7>,
    I32EnumAttrCase<"Max", 8>,
    I32EnumAttrCase<"MaxReduce", 9>,
    I32EnumAttrCase<"NoOp", 10>,
    I32EnumAttrCase<"ScalarAdd", 11>,
    I32EnumAttrCase<"ScalarMul", 12>,
    I32EnumAttrCase<"RSqrt", 13>,
    I32EnumAttrCase<"ScalarDiv", 14>,
    I32EnumAttrCase<"ElemMul", 15>,
]>{
    let genSpecializedAttr = 0;
    let cppNamespace = Sam_Dialect.cppNamespace;
}

// def AccumTypeEnum : I32EnumAttr<"AccumType", "stuff", [
//     I32EnumAttrCase<"Add", 0>,
//     I32EnumAttrCase<"Sub", 1>,
//     I32EnumAttrCase<"Mul", 2>,
//     I32EnumAttrCase<"Div", 3>,
// ]>{
//     let genSpecializedAttr = 0;
//     let cppNamespace = Sam_Dialect.cppNamespace;
// }

def TensorKindEnum : I32EnumAttr<"TensorKind", "stuff", [
    I32EnumAttrCase<"Stream", 0>,
    I32EnumAttrCase<"Buffer", 1>
]>{
    let genSpecializedAttr = 0;
    let cppNamespace = Sam_Dialect.cppNamespace;
}


#endif /* LIB_DIALECT_SAM_SAMENUM_TD */