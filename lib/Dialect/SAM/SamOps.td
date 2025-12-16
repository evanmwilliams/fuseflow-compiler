#ifndef LIB_DIALECT_SAM_SAMOPS_TD
#define LIB_DIALECT_SAM_SAMOPS_TD

include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"

include "SamDialect.td"
include "SamTypes.td"
include "SamAttr.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/TilingInterface.td"
include "mlir/Interfaces/ViewLikeInterface.td"


class SamOp<string mnemonic, list<Trait> traits = []> : Op<Sam_Dialect, mnemonic, traits>;

// Used to generate {0, D} input stream for initial fiberlookups
def SamGenerator : SamOp<"generator", [ Pure ]>
{
    let arguments = (ins);
    let results = (outs AnyRankedTensor : $output_ref);
    // let assemblyFormat = "attr-dict `->` type($output_ref)";
}

// TODO: Maybe move attribute from tensor to block
def SamFiberLookup : SamOp<"fiber_lookup", [ Pure ]>
{
    let arguments = (ins AnyRankedTensor : $input_ref, IndexAttr : $index_var, IndexAttr : $tensor_num);
    let results = (outs AnyRankedTensor : $output_ref, AnyRankedTensor : $output_crd);
    // let assemblyFormat = "$input_ref `:` type($input_ref) attr-dict `->` type($output_ref) `,` type($output_crd)";
}

def SamLocate : SamOp<"locate", [ Pure ]>
{
    let arguments = (ins AnyRankedTensor : $input_crd, AnyRankedTensor : $input_ref);
    let results = (outs AnyRankedTensor : $output_ref1, AnyRankedTensor : $output_ref2, AnyRankedTensor : $output_crd);
    // let assemblyFormat = "$input_crd `:` type($input_crd) `,` $input_ref `:` type($input_ref) attr-dict `->`
    // type($output_ref1) `,` type($output_ref2) `,` type($output_crd)";
}

def SamFiberWrite : SamOp<"fiber_write", []>
{
    let arguments = (ins AnyRankedTensor : $input_val, WriteTypeAttr : $writeOp);
    // let assemblyFormat = "$writeOp `,` $input_val `:` type($input_val) attr-dict";
}

def SamArrayVal : SamOp<"array_val", [ Pure ]>
{
    let arguments = (ins AnyRankedTensor : $input_ref, DenseI64ArrayAttr : $stream_shape);
    let results = (outs AnyRankedTensor : $output_val);
    // let assemblyFormat = "$input_ref `:` type($input_ref) attr-dict `->` type($output_val)";
}

def SamReduce : SamOp<"reduce", [ Pure ]>
{
    let arguments = (ins AnyRankedTensor : $input_val, ReduceTypeAttr : $reduceType);
    let results = (outs AnyRankedTensor : $output_val);
    // let assemblyFormat = "$input_val `:` type($input_val) attr-dict `->` type($output_val)";
}

// TODO: Add spacc order number
// TODO: Maybe combine with reduce
def SamSpacc : SamOp<"spacc", [ Pure ]>
{
    let arguments = (ins IndexAttr:$order, AnyRankedTensor:$input_inner_crd, Variadic<AnyRankedTensor>:$input_outer_crds, AnyRankedTensor:$input_val);
    let results = (outs AnyRankedTensor:$output_inner_crd, Variadic<AnyRankedTensor>:$output_outer_crds, AnyRankedTensor:$output_val);
    // let assemblyFormat = "$order `,` $input_inner_crd `:` type($input_inner_crd) `,` $input_outer_crds `:`
    // type($input_outer_crds) $input_val `:` type($input_val) attr-dict `->` type($output_inner_crd) `,`
    // type($output_outer_crds) `,` type($output_val)";
}

def SamScatter : SamOp<"fork", [ Pure ]>
{
    let arguments = (ins AnyRankedTensor : $input_stream, IndexAttr : $par_factor, IndexAttr : $index_var);
    let results = (outs Variadic<AnyRankedTensor> : $scattered_streams);
}

def SamGather : SamOp<"join", [ Pure ]>
{
    let arguments = (ins Variadic<AnyRankedTensor> : $scatter_streams, IndexAttr : $par_factor, IndexAttr : $index_var);
    let results = (outs AnyRankedTensor : $joined_stream);
}

def SamParallel : SamOp<"Parallel", [ Pure ]>
{
    // def SamParallel : SamOp<"Parallel", [SingleBlockImplicitTerminator<"SamYield">]> {
    let arguments = (ins IndexAttr : $par_factor);
    let regions = (region SizedRegion<1> : $parallel_body);
    let results = (outs Variadic<AnyRankedTensor> : $result);
}

def SamParallelBegin : SamOp<"par_begin", [ Pure ]>
{
    // No-op placeholder
}

def SamParallelEnd : SamOp<"par_end", [ Pure ]>
{
    // No-op placeholder
}

def SamDispatchGroup : SamOp<"fusion_region",
    [Pure, IsolatedFromAbove]> {
  let arguments = (ins Variadic<AnyRankedTensor>:$inputs);
  let regions   = (region SizedRegion<1>:$dispatch_region);
  let results   = (outs Variadic<AnyRankedTensor>:$result);
}

def SamOut : SamOp<"out", [ Pure ]>
{
    let arguments = (ins Variadic<AnyRankedTensor> : $out_streams);
    let results = (outs AnyTuple : $output_tuple);
}

// TODO: Still need to implement op in comal
def SamGenRef : SamOp<"gen_ref", [ Pure ]>
{
    let arguments = (ins AnyRankedTensor : $input_crd);
    let results = (outs AnyRankedTensor : $output_ref);
}

def SamYield : SamOp<"yield", [ Pure, ReturnLike, Terminator ]>, Arguments<(ins Variadic<AnyRankedTensor> : $values)>
{
    let summary = "Sam yield operation";
    //    let description = [ {
    //    `sam.yield` is a special terminator operation for blocks inside regions
    //    in `sam` parallel ops. It returns values to the immediately enclosing
    //    `sam` parallel op.
    //
    //    Example:
    //
    //    ```mlir
    //    sam.yield %f0, %f1 : f32, f32
    //    ```
    //    } ];
    //    let builders = [ OpBuilder<(ins), [ {/* nothing to do */} ]> ];
    //   let hasCustomAssemblyFormat = 1;
    //   let hasVerifier = 1;
}

def SamToTensorOp : SamOp<"to_tensor">
{
    let summary = "Convert a list of metadata streams (`tensor`) to a `tensor` it represents";
    let arguments = (ins Variadic<AnyRankedTensor> : $operand);
    let results = (outs AnyRankedTensor : $result);
}

def SamFromTensorOp : SamOp<"from_tensor">
{
    let summary = "Convert a `tensor` to a list of `tensor` to represent metadata streams";
    let arguments = (ins AnyRankedTensor : $operand);
    let results = (outs Variadic<AnyRankedTensor> : $result);
}

def SamRepeat : SamOp<"repeat", [ Pure ]>
{
    let arguments = (ins AnyRankedTensor : $input_ref, AnyRankedTensor : $repeat_ref);
    let results = (outs AnyRankedTensor : $output_ref);
    // let assemblyFormat = "$input_ref `:` type($input_ref) `,` $repeat_ref `:` type($repeat_ref) attr-dict `->`
    // type($output_ref)";
}

def SamALU : SamOp<"alu", [ Pure ]>
{
    let arguments = (ins Variadic<AnyRankedTensor> : $input_val, OpTypeAttr : $aluOp, OptionalAttr<F32Attr> : $scalar);
    let results = (outs Variadic<AnyRankedTensor> : $output_val);
}

def SamCrdDrop : SamOp<"crd_drop", [ Pure ]>
{
    let arguments = (ins AnyRankedTensor : $input_inner_crd, AnyRankedTensor : $input_outer_crd);
    let results = (outs AnyRankedTensor : $output_inner_crd, AnyRankedTensor : $output_outer_crd);
    // let assemblyFormat = "$input_inner_crd `:` type($input_inner_crd) `,` $input_outer_crd `:` type($input_outer_crd)
    // attr-dict `->` type($output_inner_crd) `,` type($output_outer_crd)";
}

def SamJoiner : SamOp<"joiner", [ Pure, AttrSizedOperandSegments ]>
{
    let arguments = (ins Variadic<AnyRankedTensor>:$input_crd, Variadic<AnyRankedTensor>:$input_ref, JoinerTypeAttr:$joinerOp);
    let results = (outs AnyRankedTensor : $output_crd, Variadic<AnyRankedTensor> : $output_refs);
    // let assemblyFormat = "$joinerOp `,` $input_crd `:` type($input_crd) $input_ref `:` type($input_ref) attr-dict
    // `->` type($output_crd) `,` type($output_refs)";
}

def SamCrdMasker : SamOp<"masker", [ Pure ]>
{
    let arguments = (ins AnyRankedTensor
                     : $input_inner_crd, AnyRankedTensor
                     : $input_outer_crd, AnyRankedTensor
                     : $input_ref);
    let results = (outs AnyRankedTensor
                   : $output_inner_crd, AnyRankedTensor
                   : $output_outer_crd, AnyRankedTensor
                   : $output_ref);
    // let assemblyFormat = "$input_inner_crd `:` type($input_inner_crd) `,` $input_outer_crd `:` type($input_outer_crd)
    // `,` $input_ref `:` type($input_ref) attr-dict `->` type($output_inner_crd) `,` type($output_outer_crd) `,`
    // type($output_ref)";
}

def SamConcat : SamOp<"concat",
    [Pure, AttrSizedOperandSegments, AttrSizedResultSegments]> {
  let summary = "Concatenate multiple tensor streams along an axis";
  let arguments = (ins
    // Flattened: [t0.d0, t0.d1, ..., t0.d{r-1},
    //             t1.d0, ..., t1.d{r-1}, ..., tm-1.d{r-1}]
    Variadic<AnyRankedTensor> : $in_crds,
    // One stream per flattened coord: non-last dims carry Ref streams; last dim carries Val stream
    Variadic<AnyRankedTensor> : $in_vals,
    IndexAttr                  : $axis,   // 0 <= axis < rank
    IndexAttr                  : $rank,   // common rank (compile-time)
    IndexAttr                  : $dim_len // dense length of concat dimension
  );
  let results = (outs
    // Result coord streams (exactly 'rank' outputs).
    Variadic<AnyRankedTensor> : $out_crds,
    // Result streams, one per level: non-last dims produce Ref streams; last produces Val stream.
    Variadic<AnyRankedTensor> : $out_vals
  );
}

#endif /* LIB_DIALECT_SAM_SAMOPS_TD */