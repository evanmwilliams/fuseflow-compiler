#qk = {
  indexing_maps = [
    affine_map<(i,j,k,l,m) -> (i,k,j,m)>, // A
    affine_map<(i,j,k,l,m) -> (i,l,j,m)>, // B
    affine_map<(i,j,k,l,m) -> (i,j,k,l)>  // X (out)
  ],
  iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"],
  doc = "X(i,j,k,l) += A(i,k,j,m) * B(i,l,j,m)"
}

#qkv = {
  indexing_maps = [
    affine_map<(i,j,k,l,m) -> (i,j,k,l)>, // A
    affine_map<(i,j,k,l,m) -> (i,l,j,m)>, // B
    affine_map<(i,j,k,l,m) -> (i,k,j,m)>  // X (out)
  ],
  iterator_types = ["parallel", "parallel", "parallel", "reduction", "parallel"],
  doc = "Y(i,k,j,m) += X(i,j,k,l) * C(i,l,j,m)"
}

// Test mat residual 
#CSF = #sparse_tensor.encoding<{
  map = (i,j,k,l) -> (i : compressed, j : compressed, k : compressed, l : compressed)
}>

func.func @kernel_test(%arga: tensor<?x?x?x?xf64, #CSF>,
                       %argb: tensor<?x?x?x?xf64, #CSF>,
                       %argc: tensor<?x?x?x?xf64, #CSF>,
                       %argx: tensor<?x?x?x?xf64, #CSF>) -> tensor<?x?x?x?xf64, #CSF> {
                      //  %argx: tensor<?x?xf64>) -> tensor<?x?xf64> {
    %qkt = linalg.generic #qk
      ins(%arga, %argb: tensor<?x?x?x?xf64, #CSF>, tensor<?x?x?x?xf64, #CSF>)
      outs(%argx: tensor<?x?x?x?xf64, #CSF>) {
      ^bb(%a: f64, %b: f64, %x: f64):
        %0 = arith.mulf %a, %b : f64
        %1 = arith.addf %x, %0 : f64
        linalg.yield %1 : f64
    } -> tensor<?x?x?x?xf64, #CSF>

    %1 = linalg.generic #qkv
      ins(%qkt, %argc: tensor<?x?x?x?xf64, #CSF>, tensor<?x?x?x?xf64, #CSF>)
      outs(%argx: tensor<?x?x?x?xf64, #CSF>) {
      ^bb(%a: f64, %b: f64, %x: f64):
        %0 = arith.mulf %a, %b : f64
        %1 = arith.addf %x, %0 : f64
        linalg.yield %1 : f64
    } -> tensor<?x?x?x?xf64, #CSF>
    // return
    return %1 : tensor<?x?x?x?xf64, #CSF>
}