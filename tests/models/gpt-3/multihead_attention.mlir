#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1, d0)>
#map2 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map4 = affine_map<(d0, d1, d2) -> (d2)>
#map5 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map6 = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>
#map7 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#map8 = affine_map<(d0, d1, d2, d4, d3) -> (d0, d1, d2, d3)>
#map9 = affine_map<(d0, d1, d2, d4, d3) -> (d0, d1, d3, d4)>
#map10 = affine_map<(d0, d1, d2, d4, d3) -> (d0, d1, d2, d4)>
#map11 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>
#map12 = affine_map<(d0, d1, d2, d3) -> (d2, d3)>
#map13 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map14 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>
#map15 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4)>
#map16 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>
#SPARSE = #sparse_tensor.encoding<{
  map = (i,j,k,l) -> (i : dense, j : dense, k : dense, l : dense)
}>
#SPARSE2 = #sparse_tensor.encoding<{
  map = (i,j) -> (i : compressed, j : compressed)
}>

module attributes {torch.debug_module_name = "MultiHeadAttention"} {
  ml_program.global private mutable @global_seed(dense<0> : tensor<i64>) : tensor<i64>
  func.func @forward(%arg1: tensor<1x12x1024x64xf32, #SPARSE>, %arg2: tensor<1x12x64x1024xf32, #SPARSE>, %arg3: tensor<1x12x1024x64xf32, #SPARSE>) -> tensor<1x12x1024x64xf32, #SPARSE> {
    %cst_7 = arith.constant 0.000000e+00 : f32
    %cst_8 = arith.constant 0xFF800000 : f32
    %c0_i64 = arith.constant 0 : i64
    %cst_9 = arith.constant 8.000000e+00 : f32
    %cst_10 = arith.constant dense_resource<torch_tensor_1024_1024_torch.float32> : tensor<1024x1024xf32, #SPARSE2>
    %7 = tensor.empty() : tensor<1x12x1024x64xf32, #SPARSE>
    %19 = tensor.empty() : tensor<1x12x64x1024xf32, #SPARSE>
    %21 = tensor.empty() : tensor<1x12x1024x1024xf32, #SPARSE>
    %22 = linalg.fill ins(%cst_7 : f32) outs(%21 : tensor<1x12x1024x1024xf32, #SPARSE>) -> tensor<1x12x1024x1024xf32, #SPARSE>
    %8 = tensor.empty() : tensor<1x12x1024x64xf32, #SPARSE>
    %23 = linalg.generic {indexing_maps = [#map8, #map9, #map10], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<1x12x1024x64xf32, #SPARSE>, tensor<1x12x64x1024xf32, #SPARSE>) outs(%22 : tensor<1x12x1024x1024xf32, #SPARSE>) {
    ^bb0(%in: f32, %in_12: f32, %out: f32):
      %43 = arith.mulf %in, %in_12 : f32
      %44 = arith.addf %43, %out : f32
      linalg.yield %44 : f32
    } -> tensor<1x12x1024x1024xf32, #SPARSE>
    %37 = linalg.generic {indexing_maps = [#map13, #map12, #map13], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%23, %cst_10 : tensor<1x12x1024x1024xf32, #SPARSE>, tensor<1024x1024xf32, #SPARSE2>) outs(%22 : tensor<1x12x1024x1024xf32, #SPARSE>) {
    ^bb0(%in: f32, %in_12: f32, %out: f32):
      %43 = arith.mulf %in, %in_12 : f32
      linalg.yield %43 : f32
    } -> tensor<1x12x1024x1024xf32, #SPARSE>
    %24 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%37 : tensor<1x12x1024x1024xf32, #SPARSE>) outs(%21 : tensor<1x12x1024x1024xf32, #SPARSE>) {
    ^bb0(%in: f32, %out: f32):
      %43 = arith.divf %in, %cst_9 : f32
      linalg.yield %43 : f32
    } -> tensor<1x12x1024x1024xf32, #SPARSE>
    %25 = tensor.empty() : tensor<1x12x1024x1xi64>
    %26 = linalg.fill ins(%c0_i64 : i64) outs(%25 : tensor<1x12x1024x1xi64>) -> tensor<1x12x1024x1xi64>
    %27 = tensor.empty() : tensor<1x12x1024x1xf32, #SPARSE>
    %28 = linalg.fill ins(%cst_8 : f32) outs(%27 : tensor<1x12x1024x1xf32, #SPARSE>) -> tensor<1x12x1024x1xf32, #SPARSE>
    %29:2 = linalg.generic {indexing_maps = [#map5, #map11, #map11], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%24 : tensor<1x12x1024x1024xf32, #SPARSE>) outs(%28, %26 : tensor<1x12x1024x1xf32, #SPARSE>, tensor<1x12x1024x1xi64>) {
    ^bb0(%in: f32, %out: f32, %out_12: i64):
      %43 = linalg.index 3 : index
      %44 = arith.index_cast %43 : index to i64
      %45 = arith.maximumf %in, %out : f32
      %46 = arith.cmpf ogt, %in, %out : f32
      %47 = arith.select %46, %44, %out_12 : i64
      linalg.yield %45, %47 : f32, i64
    } -> (tensor<1x12x1024x1xf32, #SPARSE>, tensor<1x12x1024x1xi64>)
    %30 = linalg.generic {indexing_maps = [#map5, #map11, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%24, %29#0 : tensor<1x12x1024x1024xf32, #SPARSE>, tensor<1x12x1024x1xf32, #SPARSE>) outs(%21 : tensor<1x12x1024x1024xf32, #SPARSE>) {
    ^bb0(%in: f32, %in_12: f32, %out: f32):
      %43 = arith.subf %in, %in_12 : f32
      linalg.yield %43 : f32
    } -> tensor<1x12x1024x1024xf32, #SPARSE>
    %31 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%30 : tensor<1x12x1024x1024xf32, #SPARSE>) outs(%21 : tensor<1x12x1024x1024xf32, #SPARSE>) {
    ^bb0(%in: f32, %out: f32):
      %43 = math.exp %in : f32
      linalg.yield %43 : f32
    } -> tensor<1x12x1024x1024xf32, #SPARSE>
    %32 = linalg.fill ins(%cst_7 : f32) outs(%27 : tensor<1x12x1024x1xf32, #SPARSE>) -> tensor<1x12x1024x1xf32, #SPARSE>
    %33 = linalg.generic {indexing_maps = [#map5, #map11], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%31 : tensor<1x12x1024x1024xf32, #SPARSE>) outs(%32 : tensor<1x12x1024x1xf32, #SPARSE>) {
    ^bb0(%in: f32, %out: f32):
      %43 = arith.addf %in, %out : f32
      linalg.yield %43 : f32
    } -> tensor<1x12x1024x1xf32, #SPARSE>
    %34 = linalg.generic {indexing_maps = [#map5, #map11, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%31, %33 : tensor<1x12x1024x1024xf32, #SPARSE>, tensor<1x12x1024x1xf32, #SPARSE>) outs(%21 : tensor<1x12x1024x1024xf32, #SPARSE>) {
    ^bb0(%in: f32, %in_12: f32, %out: f32):
      %43 = arith.divf %in, %in_12 : f32
      linalg.yield %43 : f32
    } -> tensor<1x12x1024x1024xf32, #SPARSE>
    %35 = tensor.empty() : tensor<1x12x1024x64xf32, #SPARSE>
    %36 = linalg.generic {indexing_maps = [#map14, #map15, #map16], iterator_types = ["parallel", "parallel", "parallel", "reduction", "parallel"]} ins(%34, %arg3 : tensor<1x12x1024x1024xf32, #SPARSE>, tensor<1x12x1024x64xf32, #SPARSE>) outs(%35 : tensor<1x12x1024x64xf32, #SPARSE>) {
    ^bb0(%in: f32, %in_12: f32, %out: f32):
      %43 = arith.mulf %in, %in_12 : f32
      %44 = arith.addf %43, %out : f32
      linalg.yield %44 : f32
    } -> tensor<1x12x1024x64xf32, #SPARSE>
    return %36 : tensor<1x12x1024x64xf32, #SPARSE>
  }
}