// Output after linalg vectorize pass
// module attributes {torch.debug_module_name = "LayerNorm"} {
//   func.func @forward(%arg0: tensor<1x8x16x16xf32>) -> tensor<1x8x16x16xf32> {
//     %c0 = arith.constant 0 : index
//     %cst = arith.constant 0.000000e+00 : f32
//     %cst_0 = arith.constant dense<2.048000e+03> : vector<1xf32>
//     %cst_1 = arith.constant dense<0.000000e+00> : vector<1xf32>
//     %cst_2 = arith.constant dense<2.048000e+03> : vector<1x1x4x1xf32>
//     %cst_3 = arith.constant dense<0.000000e+00> : vector<1x1x1x4xf32>
//     %c1 = arith.constant 1 : index
//     %c4 = arith.constant 4 : index
//     %c8 = arith.constant 8 : index
//     %c16 = arith.constant 16 : index
//     %true = arith.constant true
//     %cst_4 = arith.constant 1.000000e-05 : f64
//     cf.assert %true, "mismatching contracting dimension"
//     cf.assert %true, "mismatching contracting dimension"
//     cf.assert %true, "mismatching contracting dimension"
//     cf.assert %true, "mismatching contracting dimension"
//     cf.assert %true, "mismatching contracting dimension"
//     cf.assert %true, "mismatching contracting dimension"
//     cf.assert %true, "mismatching contracting dimension"
//     cf.assert %true, "mismatching contracting dimension"
//     cf.assert %true, "mismatching contracting dimension"
//     %0 = linalg.init_tensor [1] : tensor<1xf32>
//     %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<1xf32>) -> tensor<1xf32>
//     %2 = vector.transfer_read %1[%c0], %cst {in_bounds = [true]} : tensor<1xf32>, vector<1xf32>
//     %3 = scf.for %arg1 = %c0 to %c8 step %c1 iter_args(%arg2 = %2) -> (vector<1xf32>) {
//       %20 = scf.for %arg3 = %c0 to %c16 step %c1 iter_args(%arg4 = %arg2) -> (vector<1xf32>) {
//         %21 = scf.for %arg5 = %c0 to %c16 step %c4 iter_args(%arg6 = %arg4) -> (vector<1xf32>) {
//           %22 = vector.transfer_read %arg0[%c0, %arg1, %arg3, %arg5], %cst {in_bounds = [true, true, true, true]} : tensor<1x8x16x16xf32>, vector<1x1x1x4xf32>
//           %23 = vector.multi_reduction <add>, %22 [1, 2, 3] : vector<1x1x1x4xf32> to vector<1xf32>
//           %24 = arith.addf %23, %arg6 : vector<1xf32>
//           scf.yield %24 : vector<1xf32>
//         }
//         scf.yield %21 : vector<1xf32>
//       }
//       scf.yield %20 : vector<1xf32>
//     }
//     %4 = arith.divf %3, %cst_0 : vector<1xf32>
//     %5 = linalg.fill ins(%cst : f32) outs(%0 : tensor<1xf32>) -> tensor<1xf32>
//     %6 = vector.broadcast %4 : vector<1xf32> to vector<1x1x4x1xf32>
//     %7 = vector.transpose %6, [3, 0, 1, 2] : vector<1x1x4x1xf32> to vector<1x1x1x4xf32>
//     %8 = vector.transfer_read %5[%c0], %cst {in_bounds = [true]} : tensor<1xf32>, vector<1xf32>
//     %9 = vector.extract %cst_1[0] : vector<1xf32>
//     %10 = scf.for %arg1 = %c0 to %c8 step %c1 iter_args(%arg2 = %8) -> (vector<1xf32>) {
//       %20 = scf.for %arg3 = %c0 to %c16 step %c1 iter_args(%arg4 = %arg2) -> (vector<1xf32>) {
//         %21 = scf.for %arg5 = %c0 to %c16 step %c4 iter_args(%arg6 = %arg4) -> (vector<1xf32>) {
//           %22 = vector.transfer_read %arg0[%c0, %arg1, %arg3, %arg5], %cst {in_bounds = [true, true, true, true]} : tensor<1x8x16x16xf32>, vector<1x1x1x4xf32>
//           %23 = arith.subf %22, %7 : vector<1x1x1x4xf32>
//           %24 = vector.extract %23[0, 0, 0] : vector<1x1x1x4xf32>
//           %25 = arith.mulf %24, %24 : vector<4xf32>
//           %26 = vector.reduction <add>, %25, %9 : vector<4xf32> into f32
//           %27 = vector.insert %26, %cst_1 [0] : f32 into vector<1xf32>
//           %28 = arith.addf %27, %arg6 : vector<1xf32>
//           scf.yield %28 : vector<1xf32>
//         }
//         scf.yield %21 : vector<1xf32>
//       }
//       scf.yield %20 : vector<1xf32>
//     }
//     %11 = linalg.init_tensor [1, 8, 16, 16] : tensor<1x8x16x16xf32>
//     %12 = vector.broadcast %10 : vector<1xf32> to vector<1x1x4x1xf32>
//     %13 = arith.divf %12, %cst_2 : vector<1x1x4x1xf32>
//     %14 = vector.transpose %13, [3, 0, 1, 2] : vector<1x1x4x1xf32> to vector<1x1x1x4xf32>
//     %15 = arith.truncf %cst_4 : f64 to f32
//     %16 = vector.broadcast %15 : f32 to vector<1x1x1x4xf32>
//     %17 = arith.addf %14, %16 : vector<1x1x1x4xf32>
//     %18 = math.rsqrt %17 : vector<1x1x1x4xf32>
//     %19 = scf.for %arg1 = %c0 to %c8 step %c1 iter_args(%arg2 = %11) -> (tensor<1x8x16x16xf32>) {
//       %20 = scf.for %arg3 = %c0 to %c16 step %c1 iter_args(%arg4 = %arg2) -> (tensor<1x8x16x16xf32>) {
//         %21 = scf.for %arg5 = %c0 to %c16 step %c4 iter_args(%arg6 = %arg4) -> (tensor<1x8x16x16xf32>) {
//           %22 = vector.transfer_read %arg0[%c0, %arg1, %arg3, %arg5], %cst {in_bounds = [true, true, true, true]} : tensor<1x8x16x16xf32>, vector<1x1x1x4xf32>
//           %23 = arith.subf %22, %7 : vector<1x1x1x4xf32>
//           %24 = arith.mulf %23, %18 : vector<1x1x1x4xf32>
//           %25 = arith.addf %24, %cst_3 : vector<1x1x1x4xf32>
//           %26 = vector.transfer_write %25, %arg6[%c0, %arg1, %arg3, %arg5] {in_bounds = [true, true, true, true]} : vector<1x1x1x4xf32>, tensor<1x8x16x16xf32>
//           scf.yield %26 : tensor<1x8x16x16xf32>
//         }
//         scf.yield %21 : tensor<1x8x16x16xf32>
//       }
//       scf.yield %20 : tensor<1x8x16x16xf32>
//     }
//     return %19 : tensor<1x8x16x16xf32>
//   }
// }

// This is a layernorm operation from the pytorch through torch-mlir
#map0 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0)>
#map2 = affine_map<(d0) -> (d0)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>
module attributes {torch.debug_module_name = "LayerNorm"} {
  func.func @forward(%arg0: tensor<1x8x16x16xf32>) -> tensor<1x8x16x16xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %c16_i64 = arith.constant 16 : i64
    %c8_i64 = arith.constant 8 : i64
    %cst_0 = arith.constant 1.000000e-05 : f64
    %cst_1 = arith.constant dense<1.000000e+00> : tensor<8x16x16xf32>
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<8x16x16xf32>
    %0 = arith.cmpi eq, %c8_i64, %c8_i64 : i64
    cf.assert %0, "mismatching contracting dimension"
    cf.assert %0, "mismatching contracting dimension"
    cf.assert %0, "mismatching contracting dimension"
    %1 = arith.cmpi eq, %c16_i64, %c16_i64 : i64
    cf.assert %1, "mismatching contracting dimension"
    cf.assert %1, "mismatching contracting dimension"
    cf.assert %1, "mismatching contracting dimension"
    cf.assert %1, "mismatching contracting dimension"
    cf.assert %1, "mismatching contracting dimension"
    cf.assert %1, "mismatching contracting dimension"
    %2 = arith.muli %c8_i64, %c16_i64 : i64
    %3 = arith.muli %2, %c16_i64 : i64
    %4 = arith.sitofp %3 : i64 to f32
    %5 = linalg.init_tensor [1] : tensor<1xf32>
    %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<1xf32>) -> tensor<1xf32>
    %7 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "reduction", "reduction", "reduction"]} ins(%arg0 : tensor<1x8x16x16xf32>) outs(%6 : tensor<1xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      %15 = arith.addf %arg2, %arg1 : f32
      linalg.yield %15 : f32
    } -> tensor<1xf32>
    %8 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%7 : tensor<1xf32>) outs(%5 : tensor<1xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      %15 = arith.divf %arg1, %4 : f32
      linalg.yield %15 : f32
    } -> tensor<1xf32>
    %9 = linalg.fill ins(%cst : f32) outs(%5 : tensor<1xf32>) -> tensor<1xf32>
    %10 = linalg.generic {indexing_maps = [#map0, #map1, #map1], iterator_types = ["parallel", "reduction", "reduction", "reduction"]} ins(%arg0, %8 : tensor<1x8x16x16xf32>, tensor<1xf32>) outs(%9 : tensor<1xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %15 = arith.subf %arg1, %arg2 : f32
      %16 = arith.mulf %15, %15 : f32
      %17 = arith.addf %arg3, %16 : f32
      linalg.yield %17 : f32
    } -> tensor<1xf32>
    %11 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%10 : tensor<1xf32>) outs(%5 : tensor<1xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      %15 = arith.divf %arg1, %4 : f32
      linalg.yield %15 : f32
    } -> tensor<1xf32>
    %12 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%11 : tensor<1xf32>) outs(%5 : tensor<1xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      %15 = arith.truncf %cst_0 : f64 to f32
      %16 = arith.addf %arg1, %15 : f32
      %17 = math.rsqrt %16 : f32
      linalg.yield %17 : f32
    } -> tensor<1xf32>
    %13 = linalg.init_tensor [1, 8, 16, 16] : tensor<1x8x16x16xf32>
    %14 = linalg.generic {indexing_maps = [#map0, #map1, #map1, #map3, #map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %8, %12, %cst_1, %cst_2 : tensor<1x8x16x16xf32>, tensor<1xf32>, tensor<1xf32>, tensor<8x16x16xf32>, tensor<8x16x16xf32>) outs(%13 : tensor<1x8x16x16xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32, %arg4: f32, %arg5: f32, %arg6: f32):
      %15 = arith.subf %arg1, %arg2 : f32
      %16 = arith.mulf %15, %arg3 : f32
      %17 = arith.mulf %16, %arg4 : f32
      %18 = arith.addf %17, %arg5 : f32
      linalg.yield %18 : f32
    } -> tensor<1x8x16x16xf32>
    return %14 : tensor<1x8x16x16xf32>
  }
}