#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1, d0)>
#map2 = affine_map<(d0, d1) -> (d0, 0)>
#map3 = affine_map<(d0, d1) -> (d0, 0)>

#SPARSE = #sparse_tensor.encoding<{
  map = (i,j) -> (i : compressed, j : compressed)
}>
#SPARSE1D = #sparse_tensor.encoding<{
  map = (i) -> (i : compressed)
}>

// Unfused implementation of GCN function above from torch-mlir
// FIXME (owhsu): All functions commented out below are NOT working
func.func @matmul1 (%arg0 : tensor<1767x16xf32>, %argout : tensor<1767x16xf32>) -> tensor<1767x16xf32> {
  %cst_0 = arith.constant 0.000000e+00 : f32
  %2 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<1767x16xf32>) outs(%argout : tensor<1767x16xf32>) {
  ^bb0(%in: f32, %out: f32):
    %29 = arith.cmpf ugt, %in, %cst_0 : f32
    %30 = arith.select %29, %in, %cst_0 : f32
    linalg.yield %30 : f32
  } -> tensor<1767x16xf32>
  return %2 : tensor<1767x16xf32>
}