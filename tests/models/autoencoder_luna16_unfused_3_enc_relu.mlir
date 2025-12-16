#map = affine_map<(d0, d1) -> (d0, d1)>
#SPARSE = #sparse_tensor.encoding<{
  map = (i,j) -> (i : compressed, j : compressed)
}>
// LUNA16 Unfused 3: Encoder ReLU only
// Input: (32, 512) -> Output: (32, 512)
module attributes {torch.debug_module_name = "LUNA16_Unfused_3_Enc_ReLU"} {
  ml_program.global private mutable @global_seed(dense<0> : tensor<i64>) : tensor<i64>
  func.func @forward(%input: tensor<32x512xf32, #SPARSE>) -> tensor<32x512xf32, #SPARSE> {
    %cst_0 = arith.constant 0.000000e+00 : f32

    // ReLU activation
    %enc_relu_out = tensor.empty() : tensor<32x512xf32, #SPARSE>
    %enc_relu = linalg.generic {
      indexing_maps = [#map, #map],
      iterator_types = ["parallel", "parallel"]
    } ins(%input : tensor<32x512xf32, #SPARSE>)
      outs(%enc_relu_out : tensor<32x512xf32, #SPARSE>) {
    ^bb0(%in: f32, %out: f32):
      %cmp = arith.cmpf ugt, %in, %cst_0 : f32
      %relu = arith.select %cmp, %in, %cst_0 : f32
      linalg.yield %relu : f32
    } -> tensor<32x512xf32, #SPARSE>

    return %enc_relu : tensor<32x512xf32, #SPARSE>
  }
}
