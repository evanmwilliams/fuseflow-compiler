#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1)>
#SPARSE = #sparse_tensor.encoding<{
  map = (i,j) -> (i : compressed, j : compressed)
}>
// NIH Unfused 5: Decoder Bias only
// Input: (32, 1048576), Bias: (1048576,) -> Output: (32, 1048576)
module attributes {torch.debug_module_name = "NIH_Unfused_5_Dec_Bias"} {
  ml_program.global private mutable @global_seed(dense<0> : tensor<i64>) : tensor<i64>
  func.func @forward(%input: tensor<32x1048576xf32, #SPARSE>) -> tensor<32x1048576xf32, #SPARSE> {
    %cst_decoder_bias = arith.constant dense_resource<__elided__> : tensor<1048576xf32>

    // Bias add (broadcast)
    %dec_bias_out = tensor.empty() : tensor<32x1048576xf32, #SPARSE>
    %dec_biased = linalg.generic {
      indexing_maps = [#map, #map1, #map],
      iterator_types = ["parallel", "parallel"]
    } ins(%input, %cst_decoder_bias : tensor<32x1048576xf32, #SPARSE>, tensor<1048576xf32>)
      outs(%dec_bias_out : tensor<32x1048576xf32, #SPARSE>) {
    ^bb0(%in: f32, %bias: f32, %out: f32):
      %sum = arith.addf %in, %bias : f32
      linalg.yield %sum : f32
    } -> tensor<32x1048576xf32, #SPARSE>

    return %dec_biased : tensor<32x1048576xf32, #SPARSE>
  }
}
