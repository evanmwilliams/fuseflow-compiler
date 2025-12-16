#map = affine_map<(d0, d1) -> (d0, d1)>
#SPARSE = #sparse_tensor.encoding<{
  map = (i,j) -> (i : compressed, j : compressed)
}>
// Unfused 3: Encoder ReLU only
// Input: (32, 256) -> Output: (32, 256)
module attributes {torch.debug_module_name = "Unfused_3_Enc_ReLU"} {
  ml_program.global private mutable @global_seed(dense<0> : tensor<i64>) : tensor<i64>
  func.func @forward(%input: tensor<32x256xf32, #SPARSE>) -> tensor<32x256xf32, #SPARSE> {
    %cst_0 = arith.constant 0.000000e+00 : f32

    %out = tensor.empty() : tensor<32x256xf32, #SPARSE>
    %result = linalg.generic {
      indexing_maps = [#map, #map],
      iterator_types = ["parallel", "parallel"]
    } ins(%input : tensor<32x256xf32, #SPARSE>)
      outs(%out : tensor<32x256xf32, #SPARSE>) {
    ^bb0(%in: f32, %out_val: f32):
      %cmp = arith.cmpf ugt, %in, %cst_0 : f32
      %relu = arith.select %cmp, %in, %cst_0 : f32
      linalg.yield %relu : f32
    } -> tensor<32x256xf32, #SPARSE>

    return %result : tensor<32x256xf32, #SPARSE>
  }
}
