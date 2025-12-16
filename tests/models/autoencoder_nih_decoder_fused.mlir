#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d1)>
#SPARSE = #sparse_tensor.encoding<{
  map = (i,j) -> (i : compressed, j : compressed)
}>
// NIH Decoder Fused: SpMM + Bias
// Input: (32, 512), Weight: (512, 1048576), Bias: (1048576,) -> Output: (32, 1048576)
module attributes {torch.debug_module_name = "NIH_Decoder_Fused"} {
  ml_program.global private mutable @global_seed(dense<0> : tensor<i64>) : tensor<i64>
  func.func @forward(%input: tensor<32x512xf32, #SPARSE>) -> tensor<32x1048576xf32, #SPARSE> {
    %cst_0 = arith.constant 0.000000e+00 : f32
    %cst_decoder_weight = arith.constant dense_resource<__elided__> : tensor<512x1048576xf32, #SPARSE>
    %cst_decoder_bias = arith.constant dense_resource<__elided__> : tensor<1048576xf32>

    // SpMM: (32, 512) @ (512, 1048576) = (32, 1048576)
    %dec_out_empty = tensor.empty() : tensor<32x1048576xf32, #SPARSE>
    %dec_out_init = linalg.fill ins(%cst_0 : f32) outs(%dec_out_empty : tensor<32x1048576xf32, #SPARSE>) -> tensor<32x1048576xf32, #SPARSE>
    %dec_matmul = linalg.generic {
      indexing_maps = [#map2, #map3, #map4],
      iterator_types = ["parallel", "parallel", "reduction"]
    } ins(%input, %cst_decoder_weight : tensor<32x512xf32, #SPARSE>, tensor<512x1048576xf32, #SPARSE>)
      outs(%dec_out_init : tensor<32x1048576xf32, #SPARSE>) {
    ^bb0(%in_h: f32, %in_w: f32, %out: f32):
      %prod = arith.mulf %in_h, %in_w : f32
      %sum = arith.addf %out, %prod : f32
      linalg.yield %sum : f32
    } -> tensor<32x1048576xf32, #SPARSE>

    // Bias add (broadcast)
    %dec_bias_out = tensor.empty() : tensor<32x1048576xf32, #SPARSE>
    %dec_biased = linalg.generic {
      indexing_maps = [#map, #map1, #map],
      iterator_types = ["parallel", "parallel"]
    } ins(%dec_matmul, %cst_decoder_bias : tensor<32x1048576xf32, #SPARSE>, tensor<1048576xf32>)
      outs(%dec_bias_out : tensor<32x1048576xf32, #SPARSE>) {
    ^bb0(%in: f32, %bias: f32, %out: f32):
      %sum = arith.addf %in, %bias : f32
      linalg.yield %sum : f32
    } -> tensor<32x1048576xf32, #SPARSE>

    return %dec_biased : tensor<32x1048576xf32, #SPARSE>
  }
}
