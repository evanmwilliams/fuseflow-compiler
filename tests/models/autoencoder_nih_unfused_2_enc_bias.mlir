#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1)>
#SPARSE = #sparse_tensor.encoding<{
  map = (i,j) -> (i : compressed, j : compressed)
}>
// NIH Unfused 2: Encoder Bias only
// Input: (32, 512), Bias: (512,) -> Output: (32, 512)
module attributes {torch.debug_module_name = "NIH_Unfused_2_Enc_Bias"} {
  ml_program.global private mutable @global_seed(dense<0> : tensor<i64>) : tensor<i64>
  func.func @forward(%input: tensor<32x512xf32, #SPARSE>) -> tensor<32x512xf32, #SPARSE> {
    %cst_encoder_bias = arith.constant dense_resource<__elided__> : tensor<512xf32>

    // Bias add (broadcast)
    %enc_bias_out = tensor.empty() : tensor<32x512xf32, #SPARSE>
    %enc_biased = linalg.generic {
      indexing_maps = [#map, #map1, #map],
      iterator_types = ["parallel", "parallel"]
    } ins(%input, %cst_encoder_bias : tensor<32x512xf32, #SPARSE>, tensor<512xf32>)
      outs(%enc_bias_out : tensor<32x512xf32, #SPARSE>) {
    ^bb0(%in: f32, %bias: f32, %out: f32):
      %sum = arith.addf %in, %bias : f32
      linalg.yield %sum : f32
    } -> tensor<32x512xf32, #SPARSE>

    return %enc_biased : tensor<32x512xf32, #SPARSE>
  }
}
