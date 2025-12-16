#ifndef LIB_DIALECT_SAM_SAMATTR_TD
#define LIB_DIALECT_SAM_SAMATTR_TD

include "SamDialect.td"
include "SamEnums.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/TensorEncoding.td"



class Sam_Attr<string name, list<Trait> traits = []> : AttrDef<Sam_Dialect, name, traits>;

def StreamKindAttr
    : EnumAttr<Sam_Dialect, StreamKindEnum,
               "StreamKind"> {
   let mnemonic = "stream_kind";
   let assemblyFormat = "$value";
}

def JoinerTypeAttr
    : EnumAttr<Sam_Dialect, JoinerTypeEnum,
               "JoinerType"> {
   let mnemonic = "joiner_type";
   let assemblyFormat = "qualified($value)";
}

def WriteTypeAttr
    : EnumAttr<Sam_Dialect, WriteTypeEnum,
               "WriteType"> {
   let mnemonic = "write_type";
   let assemblyFormat = "qualified($value)";
}

def OpTypeAttr
    : EnumAttr<Sam_Dialect, OpTypeEnum,
               "OpType"> {
   let mnemonic = "op_type";
   let assemblyFormat = "qualified($value)";
}

def ReduceTypeAttr
    : EnumAttr<Sam_Dialect, ReduceTypeEnum,
               "ReduceType"> {
   let mnemonic = "reduce_type";
   let assemblyFormat = "qualified($value)";
}

def StreamEncodingAttr : Sam_Attr<"SamStreamEncoding"> {
    let mnemonic = "encoding";

    let parameters = (
        ins

        "sam::StreamKindAttr":$streamKind,
        "StringAttr":$var,
        "StringAttr":$tensor,
        "StringAttr":$format,
        // "IndexAttr":$mode
        // "IntegerType":$mode
        "IntegerAttr":$mode
    );

    let assemblyFormat = "`<` $streamKind `,` $var `,` $tensor `,` $format `,` qualified($mode) `>` ";
    // let hasCustomAssemblyFormat=false;
}

def TensorKindAttr
    : EnumAttr<Sam_Dialect, TensorKindEnum,
               "TensorKind"> {
   let mnemonic = "kind";
   let assemblyFormat = "`<` $value `>`";
}

#endif /* LIB_DIALECT_SAM_SAMATTR_TD */
