#ifndef LIB_DIALECT_STEP_STEPOPS_TD
#define LIB_DIALECT_STEP_STEPOPS_TD

include "mlir/IR/OpBase.td"

include "StepDialect.td"
include "StepTypes.td"

class StepOp<string mnemonic> : Op<Step_Dialect, mnemonic>;

// Range can be defined using tensor.generate

def StepMap : StepOp<"map"> {
    let arguments = (ins AnyRankedTensor:$input, FlatSymbolRefAttr:$func);
    let results = (outs AnyRankedTensor:$output);
    let assemblyFormat = "$func `,` $input `:` type($input) attr-dict `->` type($output)";
}

def StepAccum : StepOp<"accum"> {
    let arguments = (ins AnyRankedTensor:$input, FlatSymbolRefAttr:$foldOp);
    let results = (outs AnyRankedTensor:$output);
    let assemblyFormat = "$foldOp `,` $input `:` type($input) attr-dict `->` type($output)";
}

def StepBroadcast : StepOp<"bcast"> {
    let arguments = (ins AnyRankedTensor:$input);
    let results = (outs Variadic<AnyRankedTensor>:$outputs);
    let assemblyFormat = "$input `:` type($input) attr-dict `->` type($outputs)";
}

def StepZip : StepOp<"zip"> {
    let arguments = (ins Variadic<AnyRankedTensor>: $input);
    let results = (outs AnyRankedTensor:$output);
    let assemblyFormat = "$input `:` type($input) attr-dict `->` type($output)";
}

def StepProject : StepOp<"proj"> {
    let arguments = (ins AnyTuple:$input, IndexAttr:$ind);
    let results = (outs AnyType:$output);
    let assemblyFormat = "$input `:` type($input) `<` $ind `>` attr-dict `->` type($output)";
}

#endif /* LIB_DIALECT_STEP_STEPOPS_TD */
