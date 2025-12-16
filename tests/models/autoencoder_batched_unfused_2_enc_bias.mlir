#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1)>
#SPARSE = #sparse_tensor.encoding<{
  map = (i,j) -> (i : compressed, j : compressed)
}>
// Unfused 2: Encoder Bias Add only
// Input: (32, 256), Bias: (256,) -> Output: (32, 256)
module attributes {torch.debug_module_name = "Unfused_2_Enc_Bias"} {
  ml_program.global private mutable @global_seed(dense<0> : tensor<i64>) : tensor<i64>
  func.func @forward(%input: tensor<32x256xf32, #SPARSE>) -> tensor<32x256xf32, #SPARSE> {
    %cst_encoder_bias = arith.constant dense_resource<__elided__> : tensor<256xf32>

    %out = tensor.empty() : tensor<32x256xf32, #SPARSE>
    %result = linalg.generic {
      indexing_maps = [#map, #map1, #map],
      iterator_types = ["parallel", "parallel"]
    } ins(%input, %cst_encoder_bias : tensor<32x256xf32, #SPARSE>, tensor<256xf32>)
      outs(%out : tensor<32x256xf32, #SPARSE>) {
    ^bb0(%in: f32, %bias: f32, %out_val: f32):
      %sum = arith.addf %in, %bias : f32
      linalg.yield %sum : f32
    } -> tensor<32x256xf32, #SPARSE>

    return %result : tensor<32x256xf32, #SPARSE>
  }
}
