#vec_mul = {
  indexing_maps = [
    affine_map<(i) -> (i)>, // A
    affine_map<(i) -> (i)>, // B
    affine_map<(i) -> (i)> // X
  ],
  iterator_types = ["parallel"],
  doc = "X(i) = A(i) * B(i)"
}

#sparse_vec = #sparse_tensor.encoding<{
  map = (i) -> (i : compressed)
}>

func.func @kernel_vec_mul_sd(%arga: tensor<?xf64, #sparse_vec>,
                       %argb: tensor<?xf64>,
                       %argc: tensor<?xf64>,
                       %argd: tensor<?xf64>,
                      //  %argx: tensor<?xf64, #sparse_vec>) {
                       %argx: tensor<?xf64, #sparse_vec>) -> tensor<?xf64, #sparse_vec> {
    %0 = linalg.generic #vec_mul
      ins(%arga, %argb: tensor<?xf64, #sparse_vec>, tensor<?xf64>)
      outs(%argx: tensor<?xf64, #sparse_vec>) {
      ^bb(%a: f64, %b: f64, %x: f64):
        %0 = arith.mulf %a, %b : f64
        // %1 = math.exp %0 : f64
        linalg.yield %0 : f64
    } -> tensor<?xf64, #sparse_vec>
    %1 = linalg.generic #vec_mul
      ins(%0, %argc: tensor<?xf64, #sparse_vec>, tensor<?xf64>)
      outs(%argx: tensor<?xf64, #sparse_vec>) {
      ^bb(%a: f64, %b: f64, %x: f64):
        %1 = arith.mulf %a, %b : f64
        // %2 = math.exp %1 : f64
        linalg.yield %1 : f64
    } -> tensor<?xf64, #sparse_vec>
    %2 = linalg.generic #vec_mul
      ins(%1, %argd: tensor<?xf64, #sparse_vec>, tensor<?xf64>)
      outs(%argx: tensor<?xf64, #sparse_vec>) {
      ^bb(%a: f64, %b: f64, %x: f64):
        %2 = arith.mulf %a, %b : f64
        linalg.yield %2 : f64
    } -> tensor<?xf64, #sparse_vec>
    // return
    return %2 : tensor<?xf64, #sparse_vec>
}

// func.func @kernel_vec_mul_ss(%arga: tensor<?xf64, #sparse_vec>,
//                        %argb: tensor<?xf64, #sparse_vec>,
//                        %argx: tensor<?xf64, #sparse_vec>) {
//                       //  %argx: tensor<?x?xf64>) -> tensor<?x?xf64> {
//     %0 = linalg.generic #vec_mul
//       ins(%arga, %argb: tensor<?xf64, #sparse_vec>, tensor<?xf64, #sparse_vec>)
//       outs(%argx: tensor<?xf64, #sparse_vec>) {
//       ^bb(%a: f64, %b: f64, %x: f64):
//         %0 = arith.mulf %a, %b : f64
//         linalg.yield %0 : f64
//     } -> tensor<?xf64, #sparse_vec>
//     return
//     // return %0 : tensor<?xf64>
// }