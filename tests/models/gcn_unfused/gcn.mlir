#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1, d0)>
#map2 = affine_map<(d0, d1) -> (d0, 0)>
#map3 = affine_map<(d0, d1) -> (d0, 0)>

// module attributes {torch.debug_module_name = "GCN"} {
//   ml_program.global private mutable @global_seed(dense<0> : tensor<i64>) : tensor<i64>
//   func.func @forward(%arg0: tensor<1767x50xf32>, %arg1: tensor<1767x1767xf32>) -> tensor<1767x121xf32> {
//     %cst = arith.constant dense_resource<__elided__> : tensor<16x50xf32>
//     %cst_0 = arith.constant 0.000000e+00 : f32
//     %c0_i64 = arith.constant 0 : i64
//     %cst_1 = arith.constant 0xFF800000 : f32
//     %cst_2 = arith.constant dense_resource<__elided__> : tensor<121x16xf32>
//     %arg1_clone = arith.constant dense_resource<__elided__> : tensor<1767x1767xf32>
    
//     %0 = tensor.empty() : tensor<1767x50xf32>
//     %1 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<1767x50xf32>) -> tensor<1767x50xf32>
    
//     %2 = linalg.matmul ins(%arg1, %arg0 : tensor<1767x1767xf32>, tensor<1767x50xf32>) outs(%1 : tensor<1767x50xf32>) -> tensor<1767x50xf32>
    
//     %3 = tensor.empty() : tensor<50x16xf32>
    
//     %4 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%cst : tensor<16x50xf32>) outs(%3 : tensor<50x16xf32>) {
//     ^bb0(%in: f32, %out: f32):
//       linalg.yield %in : f32
//     } -> tensor<50x16xf32>
    
//     %5 = tensor.empty() : tensor<1767x16xf32>
//     %6 = linalg.fill ins(%cst_0 : f32) outs(%5 : tensor<1767x16xf32>) -> tensor<1767x16xf32>
    
//     %7 = linalg.matmul ins(%2, %4 : tensor<1767x50xf32>, tensor<50x16xf32>) outs(%6 : tensor<1767x16xf32>) -> tensor<1767x16xf32>
    
//     %8 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%7 : tensor<1767x16xf32>) outs(%5 : tensor<1767x16xf32>) {
//     ^bb0(%in: f32, %out: f32):
//       %26 = arith.cmpf ugt, %in, %cst_0 : f32
//       %27 = arith.select %26, %in, %cst_0 : f32
//       linalg.yield %27 : f32
//     } -> tensor<1767x16xf32>
    
//     %9 = linalg.matmul ins(%arg1_clone, %8 : tensor<1767x1767xf32>, tensor<1767x16xf32>) outs(%6 : tensor<1767x16xf32>) -> tensor<1767x16xf32>
    
//     // %9 = linalg.matmul ins(%arg1, %8 : tensor<1767x1767xf32>, tensor<1767x16xf32>) outs(%6 : tensor<1767x16xf32>) -> tensor<1767x16xf32>
//     %10 = tensor.empty() : tensor<16x121xf32>
    
//     %11 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%cst_2 : tensor<121x16xf32>) outs(%10 : tensor<16x121xf32>) {
//     ^bb0(%in: f32, %out: f32):
//       linalg.yield %in : f32
//     } -> tensor<16x121xf32>
    
//     %12 = tensor.empty() : tensor<1767x121xf32>
//     %13 = linalg.fill ins(%cst_0 : f32) outs(%12 : tensor<1767x121xf32>) -> tensor<1767x121xf32>
    
//     %14 = linalg.matmul ins(%9, %11 : tensor<1767x16xf32>, tensor<16x121xf32>) outs(%13 : tensor<1767x121xf32>) -> tensor<1767x121xf32>
    
//     %15 = tensor.empty() : tensor<1767x1xi64>
//     %16 = linalg.fill ins(%c0_i64 : i64) outs(%15 : tensor<1767x1xi64>) -> tensor<1767x1xi64>
//     %17 = tensor.empty() : tensor<1767x1xf32>
//     %18 = linalg.fill ins(%cst_1 : f32) outs(%17 : tensor<1767x1xf32>) -> tensor<1767x1xf32>
    
//     %19:2 = linalg.generic {indexing_maps = [#map, #map2, #map2], iterator_types = ["parallel", "reduction"]} ins(%14 : tensor<1767x121xf32>) outs(%18, %16 : tensor<1767x1xf32>, tensor<1767x1xi64>) {
//     ^bb0(%in: f32, %out: f32, %out_3: i64):
//       %26 = linalg.index 1 : index
//       %27 = arith.index_cast %26 : index to i64
//       %28 = arith.maximumf %in, %out : f32
//       %29 = arith.cmpf ogt, %in, %out : f32
//       %30 = arith.select %29, %27, %out_3 : i64
//       linalg.yield %28, %30 : f32, i64
//     } -> (tensor<1767x1xf32>, tensor<1767x1xi64>)
    
//     // %expanded = tensor.expand_shape %19#0 [[0, 1]] output_shape [1767, 1] : tensor<1767x1xf32> into tensor<1767x1xf32>
//     %20 = linalg.generic {indexing_maps = [#map, #map3, #map], iterator_types = ["parallel", "parallel"]} ins(%14, %19#0 : tensor<1767x121xf32>, tensor<1767x1xf32>) outs(%12 : tensor<1767x121xf32>) {
//     ^bb0(%in: f32, %in_3: f32, %out: f32):
//       %26 = arith.subf %in, %in_3 : f32
//       linalg.yield %26 : f32
//     } -> tensor<1767x121xf32>
    
//     %21 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%20 : tensor<1767x121xf32>) outs(%12 : tensor<1767x121xf32>) {
//     ^bb0(%in: f32, %out: f32):
//       %26 = math.exp %in : f32
//       linalg.yield %26 : f32
//     } -> tensor<1767x121xf32>
    
//     %22 = tensor.empty() : tensor<1767x1xf32>
//     %23 = linalg.fill ins(%cst_0 : f32) outs(%22 : tensor<1767x1xf32>) -> tensor<1767x1xf32>
    
//     %24 = linalg.generic {indexing_maps = [#map, #map3], iterator_types = ["parallel", "reduction"]} ins(%21 : tensor<1767x121xf32>) outs(%23 : tensor<1767x1xf32>) {
//     ^bb0(%in: f32, %out: f32):
//       %26 = arith.addf %in, %out : f32
//       linalg.yield %26 : f32
//     } -> tensor<1767x1xf32>
    
//     %25 = linalg.generic {indexing_maps = [#map, #map3, #map], iterator_types = ["parallel", "parallel"]} ins(%21, %24 : tensor<1767x121xf32>, tensor<1767x1xf32>) outs(%12 : tensor<1767x121xf32>) {
//     ^bb0(%in: f32, %in_3: f32, %out: f32):
//       %26 = arith.divf %in, %in_3 : f32
//       linalg.yield %26 : f32
//     } -> tensor<1767x121xf32>
//     return %25 : tensor<1767x121xf32>
//   }
// }

// Unfused implementation of GCN function above from torch-mlir
// FIXME (owhsu): All functions commented out below are NOT working
func.func @matmul1 (%arg0 : tensor<1767x1767xf32>, %arg1 : tensor<1767x50xf32>, %argout : tensor<1767x50xf32>) -> tensor<1767x50xf32> {
  %2 = linalg.matmul ins(%arg0, %arg1 : tensor<1767x1767xf32>, tensor<1767x50xf32>) outs(%argout : tensor<1767x50xf32>) -> tensor<1767x50xf32>
  return %2 : tensor<1767x50xf32>
}

// func.func @transpose2(%arg0: tensor<16x50xf32>) -> tensor<50x16xf32> {
//   %tout = tensor.empty() : tensor<50x16xf32>
//   %0 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<16x50xf32>) outs(%tout : tensor<50x16xf32>) {
//     ^bb0(%in: f32, %out: f32):
//       linalg.yield %in : f32
//     } -> tensor<50x16xf32>
//   return %0 : tensor<50x16xf32>
// }

func.func @matmul3 (%arg0 : tensor<1767x50xf32>, %arg1 : tensor<50x16xf32>, %argout : tensor<1767x16xf32>) -> tensor<1767x16xf32> {
  %7 = linalg.matmul ins(%arg0, %arg1 : tensor<1767x50xf32>, tensor<50x16xf32>) outs(%argout : tensor<1767x16xf32>) -> tensor<1767x16xf32>
  return %7 : tensor<1767x16xf32>
}

// func.func @relu4 (%arg0: tensor<1767x16xf32>) ->  tensor<1767x16xf32> {
//   %cst_0 = arith.constant 0.000000e+00 : f32
//   %tout = tensor.empty() : tensor<1767x16xf32>
//   %0 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<1767x16xf32>) outs(%tout :  tensor<1767x16xf32>) {
//     ^bb0(%in: f32, %out: f32):
//       %26 = arith.cmpf ugt, %in, %cst_0 : f32
//       %27 = arith.select %26, %in, %cst_0 : f32
//       linalg.yield %27 : f32
//     } -> tensor<1767x16xf32>
//   return %0 : tensor<1767x16xf32>
// }

func.func @matmul5 (%arg0 : tensor<1767x1767xf32>, %arg1 : tensor<1767x16xf32>, %argout : tensor<1767x16xf32>) -> tensor<1767x16xf32> {
  %9 = linalg.matmul ins(%arg0, %arg1 : tensor<1767x1767xf32>, tensor<1767x16xf32>) outs(%argout : tensor<1767x16xf32>) -> tensor<1767x16xf32>
  return %9 : tensor<1767x16xf32>
}

// func.func @transpose6 (%arg0: tensor<121x16xf32> ) -> tensor<16x121xf32> {
//   %tout = tensor.empty() : tensor<16x121xf32>
//   %0 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<121x16xf32>) outs(%tout : tensor<16x121xf32>) {
//     ^bb0(%in: f32, %out: f32):
//       linalg.yield %in : f32
//     } -> tensor<16x121xf32>
//   return %0 : tensor<16x121xf32>
// }

func.func @matmul7 (%arg0 : tensor<1767x16xf32>, %arg1 : tensor<16x121xf32>, %argout : tensor<1767x121xf32>) -> tensor<1767x121xf32> {
  %14 = linalg.matmul ins(%arg0, %arg1 : tensor<1767x16xf32>, tensor<16x121xf32>) outs(%argout : tensor<1767x121xf32>) -> tensor<1767x121xf32>
  return %14 : tensor<1767x121xf32>
}

// func.func @reduce8 (%arg0 : tensor<1767x121xf32>) -> tensor<1767x1xf32> {
//   %cst_1 = arith.constant 0xFF800000 : f32
//   %cst_2 = arith.constant dense_resource<__elided__> : tensor<121x16xf32>
//   %c0_i64 = arith.constant 0 : i64

//   %t0 = tensor.empty() : tensor<1767x1xi64>
//   %tout0 = linalg.fill ins(%c0_i64 : i64) outs(%t0 : tensor<1767x1xi64>) -> tensor<1767x1xi64>
//   %t1 = tensor.empty() : tensor<1767x1xf32>
//   %tout1 = linalg.fill ins(%cst_1 : f32) outs(%t1 : tensor<1767x1xf32>) -> tensor<1767x1xf32>
  
//   %0:2 = linalg.generic {indexing_maps = [#map, #map2, #map2], iterator_types = ["parallel", "reduction"]} ins(%arg0 : tensor<1767x121xf32>) outs(%tout1, %tout0 : tensor<1767x1xf32>, tensor<1767x1xi64>) {
//   ^bb0(%in: f32, %out: f32, %out_3: i64):
//     %26 = linalg.index 1 : index
//     %27 = arith.index_cast %26 : index to i64
//     // reduce inputs floored to min f32
//     %28 = arith.maximumf %in, %out : f32
//     %29 = arith.cmpf ogt, %in, %out : f32
//     // sets second output to 1 if input was chosen
//     %30 = arith.select %29, %27, %out_3 : i64
//     linalg.yield %28, %30 : f32, i64
    
//   } -> (tensor<1767x1xf32>, tensor<1767x1xi64>)
//   return %0#0 : tensor<1767x1xf32>
// }

func.func @sub9 (%arg0 : tensor<1767x121xf32>, %arg1 : tensor<1767x1xf32>) -> tensor<1767x121xf32> {
  %tout = tensor.empty() : tensor<1767x121xf32>
  %0 = linalg.generic {indexing_maps = [#map, #map3, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : tensor<1767x121xf32>, tensor<1767x1xf32>) outs(%tout : tensor<1767x121xf32>) {
    ^bb0(%in: f32, %in_3: f32, %out: f32):
      %26 = arith.subf %in, %in_3 : f32
      linalg.yield %26 : f32
    } -> tensor<1767x121xf32>
  return %0 : tensor<1767x121xf32>
}

func.func @exp10 (%arg0 : tensor<1767x121xf32>) -> tensor<1767x121xf32> {
  %tout = tensor.empty() : tensor<1767x121xf32>
  %0 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<1767x121xf32>) outs(%tout : tensor<1767x121xf32>) {
    ^bb0(%in: f32, %out: f32):
      %26 = math.exp %in : f32
      linalg.yield %26 : f32
    } -> tensor<1767x121xf32>
  return %0 : tensor<1767x121xf32>
}

// func.func @reduce11 (%arg0 : tensor<1767x121xf32>) -> tensor<1767x1xf32> {
//   %tout = tensor.empty() : tensor<1767x1xf32>
//   %0 = linalg.generic {indexing_maps = [#map, #map3], iterator_types = ["parallel", "reduction"]} ins(%arg0 : tensor<1767x121xf32>) outs(%tout : tensor<1767x1xf32>) {
//     ^bb0(%in: f32, %out: f32):
//       %26 = arith.addf %in, %out : f32
//       linalg.yield %26 : f32
//     } -> tensor<1767x1xf32>
//   return %0 : tensor<1767x1xf32>
// }

func.func @div12 (%arg0 : tensor<1767x121xf32>, %arg1 : tensor<1767x1xf32>) ->  tensor<1767x121xf32> {
  %tout = tensor.empty() : tensor<1767x121xf32>
  %0 = linalg.generic {indexing_maps = [#map, #map3, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : tensor<1767x121xf32>, tensor<1767x1xf32>) outs(%tout : tensor<1767x121xf32>) {
    ^bb0(%in: f32, %in_3: f32, %out: f32):
      %26 = arith.divf %in, %in_3 : f32
      linalg.yield %26 : f32
    } -> tensor<1767x121xf32>
  return %0 : tensor<1767x121xf32>
}