#spmm = {
  indexing_maps = [
    affine_map<(i,j,k) -> (i,k)>, // A
    affine_map<(i,j,k) -> (k,j)>, // B
    affine_map<(i,j,k) -> (i,j)>  // X (out)
  ],
  iterator_types = ["parallel", "parallel", "reduction"],
  doc = "X(i,j) += A(i,k) * B(k,j)"
}

#spadd = {
  indexing_maps = [
    affine_map<(i,j) -> (i,j)>, // A
    affine_map<(i,j) -> (i,j)>, // B
    affine_map<(i,j) -> (i,j)>  // X (out)
  ],
  iterator_types = ["parallel", "parallel"],
  doc = "X(i,j) = A(i,j) + B(i,j)"
}

#CSR = #sparse_tensor.encoding<{
  map = (i,j) -> (i : compressed, j : compressed)
}>

func.func @kernel_test(%arga: tensor<?x?xf64, #CSR>,
                       %argb: tensor<?x?xf64, #CSR>,
                       %argc: tensor<?x?xf64, #CSR>,
                       %argd: tensor<?x?xf64, #CSR>,
                       %arge: tensor<?x?xf64, #CSR>,
                      //  %argx: tensor<?x?xf64, #CSR>) {
                       %argx: tensor<?x?xf64, #CSR>) -> tensor<?x?xf64, #CSR> {
    %0 = linalg.generic #spmm
      ins(%arga, %argb: tensor<?x?xf64, #CSR>, tensor<?x?xf64, #CSR>)
      outs(%argx: tensor<?x?xf64, #CSR>) {
      ^bb(%a: f64, %b: f64, %x: f64):
        %0 = arith.mulf %a, %b : f64
        %1 = arith.addf %x, %0 : f64
        linalg.yield %1 : f64
    } -> tensor<?x?xf64, #CSR>

    %1 = linalg.generic #spadd
      ins(%0, %argc: tensor<?x?xf64, #CSR>, tensor<?x?xf64, #CSR>)
      outs(%argx: tensor<?x?xf64, #CSR>) {
      ^bb(%a: f64, %b: f64, %x: f64):
        %1 = arith.addf %a, %b : f64
        linalg.yield %1 : f64
    } -> tensor<?x?xf64, #CSR>

    %2 = linalg.generic #spmm
      ins(%1, %argd: tensor<?x?xf64, #CSR>, tensor<?x?xf64, #CSR>)
      outs(%argx: tensor<?x?xf64, #CSR>) {
      ^bb(%a: f64, %b: f64, %x: f64):
        %2 = arith.mulf %a, %b : f64
        %3 = arith.addf %x, %2 : f64
        linalg.yield %3 : f64
    } -> tensor<?x?xf64, #CSR>

    %3 = linalg.generic #spadd
      ins(%2, %arge: tensor<?x?xf64, #CSR>, tensor<?x?xf64, #CSR>)
      outs(%argx: tensor<?x?xf64, #CSR>) {
      ^bb(%a: f64, %b: f64, %x: f64):
        %3 = arith.addf %a, %b : f64
        linalg.yield %3 : f64
    } -> tensor<?x?xf64, #CSR>
    // return %1 : tensor<?x?xf64, #CSR>
    return %3 : tensor<?x?xf64, #CSR>
    // return
}