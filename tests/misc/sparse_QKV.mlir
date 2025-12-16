#spmm = {
  indexing_maps = [
    affine_map<(i,j,k,l,m,n) -> (i,k,j,m)>, // A
    affine_map<(i,j,k,l,m,n) -> (i,l,j,m)>, // B
    affine_map<(i,j,k,l,m,n) -> (i,l,j,n)>, // C
    affine_map<(i,j,k,l,m,n) -> (i,j,k,n)>  // X (out)
  ],
  iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"],
  doc = "X(i,j,k,n) += A(i,k,j,m) * B(i,l,j,m) * C(i,l,j,n)"
}
// Test mat residual
#CSF = #sparse_tensor.encoding<{
  map = (i,j,k,l) -> (i : compressed, j : compressed, k : compressed, l : compressed)
}>
func.func @kernel_spmm(%arga: tensor<8x8x8x8xf64, #CSF>,
                       %argb: tensor<8x8x8x8xf64, #CSF>,
                       %argc: tensor<8x8x8x8xf64, #CSF>,
                       %argx: tensor<8x8x8x8xf64, #CSF>) {
                      //  %argx: tensor<?x?xf64>) -> tensor<?x?xf64> {
    %5 = tensor.empty() : tensor<8x8x8x8xf64, #CSF>
    %1 = linalg.generic #spmm
      ins(%arga, %argb, %argc: tensor<8x8x8x8xf64, #CSF>, tensor<8x8x8x8xf64, #CSF>, tensor<8x8x8x8xf64, #CSF>)
      outs(%argx: tensor<8x8x8x8xf64, #CSF>) {
      ^bb(%a: f64, %b: f64, %c: f64, %x: f64):
        %0 = arith.mulf %a, %b : f64
        %1 = arith.addf %x, %0 : f64
        %2 = math.exp %1 : f64
        %3 = arith.mulf %2, %c : f64
        %4 = arith.addf %x, %3 : f64
        linalg.yield %4 : f64
    } -> tensor<8x8x8x8xf64, #CSF>
    return
    // return %0 : tensor<?x?xf64>
}