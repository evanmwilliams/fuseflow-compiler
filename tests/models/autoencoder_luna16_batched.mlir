#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1, d0)>
#map2 = affine_map<(d0, d1) -> (d1)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map4 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map5 = affine_map<(d0, d1, d2) -> (d0, d1)>
#SPARSE = #sparse_tensor.encoding<{
  map = (i,j) -> (i : compressed, j : compressed)
}>
// LUNA16 CT Scan Batched Autoencoder: batch=32, 262144 (512x512 grayscale) -> 512 -> 262144
// Uses SpMM (Sparse Matrix-Matrix multiply) instead of SpMV
module attributes {torch.debug_module_name = "SparseAutoencoder_LUNA16_Batched"} {
  ml_program.global private mutable @global_seed(dense<0> : tensor<i64>) : tensor<i64>
  func.func @forward(%arg0: tensor<32x262144xf32, #SPARSE>) -> tensor<32x262144xf32, #SPARSE> {
    // Encoder bias (hidden_dim) - will be broadcast across batch
    %cst_encoder_bias = arith.constant dense_resource<__elided__> : tensor<512xf32>
    %cst_0 = arith.constant 0.000000e+00 : f32
    // Decoder weights (hidden_dim x input_dim)
    %cst_decoder_weight = arith.constant dense_resource<__elided__> : tensor<512x262144xf32, #SPARSE>
    // Decoder bias (input_dim) - will be broadcast across batch
    %cst_decoder_bias = arith.constant dense_resource<__elided__> : tensor<262144xf32>
    // Encoder weights (input_dim x hidden_dim)
    %cst_encoder_weight = arith.constant dense_resource<__elided__> : tensor<262144x512xf32, #SPARSE>

    // ===== ENCODER =====
    // Encoder: X @ W_enc -> (32, 262144) @ (262144, 512) = (32, 512)
    // Using linalg.generic for SpMM: C[i,j] = sum_k A[i,k] * B[k,j]
    %enc_out_empty = tensor.empty() : tensor<32x512xf32, #SPARSE>
    %enc_out_init = linalg.fill ins(%cst_0 : f32) outs(%enc_out_empty : tensor<32x512xf32, #SPARSE>) -> tensor<32x512xf32, #SPARSE>
    %enc_matmul = linalg.generic {
      indexing_maps = [#map3, #map4, #map5],
      iterator_types = ["parallel", "parallel", "reduction"]
    } ins(%arg0, %cst_encoder_weight : tensor<32x262144xf32, #SPARSE>, tensor<262144x512xf32, #SPARSE>)
      outs(%enc_out_init : tensor<32x512xf32, #SPARSE>) {
    ^bb0(%in_x: f32, %in_w: f32, %out: f32):
      %prod = arith.mulf %in_x, %in_w : f32
      %sum = arith.addf %out, %prod : f32
      linalg.yield %sum : f32
    } -> tensor<32x512xf32, #SPARSE>

    // Encoder: add bias (broadcast across batch dimension)
    %enc_bias_out = tensor.empty() : tensor<32x512xf32, #SPARSE>
    %enc_biased = linalg.generic {
      indexing_maps = [#map, #map2, #map],
      iterator_types = ["parallel", "parallel"]
    } ins(%enc_matmul, %cst_encoder_bias : tensor<32x512xf32, #SPARSE>, tensor<512xf32>)
      outs(%enc_bias_out : tensor<32x512xf32, #SPARSE>) {
    ^bb0(%in: f32, %bias: f32, %out: f32):
      %sum = arith.addf %in, %bias : f32
      linalg.yield %sum : f32
    } -> tensor<32x512xf32, #SPARSE>

    // Encoder: ReLU activation
    %enc_relu_out = tensor.empty() : tensor<32x512xf32, #SPARSE>
    %enc_relu = linalg.generic {
      indexing_maps = [#map, #map],
      iterator_types = ["parallel", "parallel"]
    } ins(%enc_biased : tensor<32x512xf32, #SPARSE>)
      outs(%enc_relu_out : tensor<32x512xf32, #SPARSE>) {
    ^bb0(%in: f32, %out: f32):
      %cmp = arith.cmpf ugt, %in, %cst_0 : f32
      %relu = arith.select %cmp, %in, %cst_0 : f32
      linalg.yield %relu : f32
    } -> tensor<32x512xf32, #SPARSE>

    // ===== DECODER =====
    // Decoder: H @ W_dec -> (32, 512) @ (512, 262144) = (32, 262144)
    // W_dec is already stored as (512 x 262144), no transpose needed
    %dec_out_empty = tensor.empty() : tensor<32x262144xf32, #SPARSE>
    %dec_out_init = linalg.fill ins(%cst_0 : f32) outs(%dec_out_empty : tensor<32x262144xf32, #SPARSE>) -> tensor<32x262144xf32, #SPARSE>
    %dec_matmul = linalg.generic {
      indexing_maps = [#map3, #map4, #map5],
      iterator_types = ["parallel", "parallel", "reduction"]
    } ins(%enc_relu, %cst_decoder_weight : tensor<32x512xf32, #SPARSE>, tensor<512x262144xf32, #SPARSE>)
      outs(%dec_out_init : tensor<32x262144xf32, #SPARSE>) {
    ^bb0(%in_h: f32, %in_w: f32, %out: f32):
      %prod = arith.mulf %in_h, %in_w : f32
      %sum = arith.addf %out, %prod : f32
      linalg.yield %sum : f32
    } -> tensor<32x262144xf32, #SPARSE>

    // Decoder: add bias (broadcast across batch dimension)
    %dec_bias_out = tensor.empty() : tensor<32x262144xf32, #SPARSE>
    %dec_biased = linalg.generic {
      indexing_maps = [#map, #map2, #map],
      iterator_types = ["parallel", "parallel"]
    } ins(%dec_matmul, %cst_decoder_bias : tensor<32x262144xf32, #SPARSE>, tensor<262144xf32>)
      outs(%dec_bias_out : tensor<32x262144xf32, #SPARSE>) {
    ^bb0(%in: f32, %bias: f32, %out: f32):
      %sum = arith.addf %in, %bias : f32
      linalg.yield %sum : f32
    } -> tensor<32x262144xf32, #SPARSE>

    return %dec_biased : tensor<32x262144xf32, #SPARSE>
  }
}
