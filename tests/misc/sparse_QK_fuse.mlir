#spmm = {
  indexing_maps = [
    affine_map<(i,j,k,l,m) -> (i,k,j,m)>, // A
    affine_map<(i,j,k,l,m) -> (i,l,j,m)>, // B
    affine_map<(i,j,k,l,m) -> (i,j,k,l)>  // X (out)
  ],
  iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"],
  doc = "X(i,j,k,l) += A(i,k,j,m) * B(i,l,j,m)"
}

#trait = {
  indexing_maps = [
    affine_map<(i,j,k,l,m) -> (i,k,j,m)>, // A
    affine_map<(i,j,k,l,m) -> (i,j,k,l)>  // X (out)
  ],
  iterator_types = ["parallel", "parallel", "parallel", "parallel"],
  doc = "X(i,j,k,l) += A(i,k,j,m) * B(i,l,j,m)"
}

// Test mat residual 
#CSF = #sparse_tensor.encoding<{
  // lvlTypes = [ "compressed", "compressed", "compressed", "compressed" ],
  // dimToLvl = affine_map<(i,j,k,l) -> (i,j,k,l)>
  map = (i,j,k,l) -> (i : compressed, j : compressed, k : compressed, l : compressed)
}>

func.func @kernel_spmm(%arga: tensor<5x5x5x5xf64, #CSF>,
                       %argb: tensor<5x5x5x5xf64, #CSF>,
                       %argx: tensor<5x5x5x5xf64, #CSF>) {
                      //  %argx: tensor<?x?xf64>) -> tensor<?x?xf64> {
    %t1 = tensor.empty() : tensor<5x5x5x5xf64, #CSF>
    %0 = linalg.generic #spmm
      ins(%arga, %argb: tensor<5x5x5x5xf64, #CSF>, tensor<5x5x5x5xf64, #CSF>)
      outs(%t1: tensor<5x5x5x5xf64, #CSF>) {
      ^bb(%a: f64, %b: f64, %x: f64):
        %0 = arith.mulf %a, %b : f64
        %1 = arith.addf %x, %0 : f64
        linalg.yield %1 : f64
    } -> tensor<5x5x5x5xf64, #CSF>

    %l1 = linalg.generic #trait
      ins(%arga: tensor<5x5x5x5xf64, #CSF>) outs(%argx: tensor<5x5x5x5xf64, #CSF>) {
    ^bb0(%in1: f64, %out1: f64):
      %b1 = math.exp %in1 : f64
      linalg.yield %b1 : f64
  } -> tensor<5x5x5x5xf64, #CSF>
    return
    // return %0 : tensor<?x?xf64>
}

// #map0 = affine_map<(d0, d1) -> (d0, d1)>
// // CHECK-LABEL: @add_mul_fusion
// func.func @add_mul_fusion(%arg0: tensor<?x?xf32>, %arg1 : tensor<?x?xf32>, %arg2 : tensor<?x?xf32>) -> tensor<?x?xf32>
// {
//   %c0 = arith.constant 0 : index
//   %c1 = arith.constant 1 : index
//   %0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
//   %1 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
//   %2 = tensor.empty(%0, %1) : tensor<?x?xf32>
//   %3 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel"]}
//       ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
//       outs(%2 : tensor<?x?xf32>) {
//     ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
//       %4 = arith.addf %arg3, %arg4 : f32
//       linalg.yield %4 : f32
//   } -> tensor<?x?xf32>
//   // CHECK: linalg.generic {
//   // CHECK-SAME: indexing_maps = {{\[}}[[$MAP0]], [[$MAP0]], [[$MAP0]], [[$MAP0]]{{\]}}
//   %4 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel"]}
//       ins(%3, %arg2 : tensor<?x?xf32>, tensor<?x?xf32>)
//       outs(%2 : tensor<?x?xf32>) {
//     // CHECK: ^{{[a-zA-Z0-9_]*}}
//     // CHECK-SAME: [[ARG0:%[a-zA-Z0-9_]*]]
//     // CHECK-SAME: [[ARG1:%[a-zA-Z0-9_]*]]
//     // CHECK-SAME: [[ARG2:%[a-zA-Z0-9_]*]]
//     ^bb0(%arg5: f32, %arg6: f32, %arg7: f32):
//       // CHECK: [[T1:%[a-zA-Z0-9_]*]] = arith.addf [[ARG0]], [[ARG1]]
//       // CHECK-NOT: linalg.yield
//       // CHECK: arith.mulf [[T1]], [[ARG2]]
//       // CHECK: linalg.yield
//       %5 = arith.mulf %arg5, %arg6 : f32
//       linalg.yield %5 : f32
//     } -> tensor<?x?xf32>
//   return %4 : tensor<?x?xf32>
// }