#ikj = {
  indexing_maps = [
    affine_map<(i,k,j) -> (i,k)>, // A
    affine_map<(i,k,j) -> (k,j)>, // B
    affine_map<(i,k,j) -> (i,j)>  // X (out)
  ],
  iterator_types = ["parallel", "reduction", "parallel"],
  doc = "X(i,j) += A(i,k) * B(k,j)"
}

#jik = {
  indexing_maps = [
    affine_map<(j,i,k) -> (i,k)>, // A
    affine_map<(j,i,k) -> (k,j)>, // B
    affine_map<(j,i,k) -> (i,j)>  // X (out)
  ],
  iterator_types = ["parallel", "parallel", "reduction"],
  doc = "X(i,j) += A(i,k) * B(k,j)"
}
#ji = {
  indexing_maps = [
    affine_map<(j,i) -> (i,j)>, // A
    affine_map<(j,i) -> (i,j)>, // B
    affine_map<(j,i) -> (i,j)>  // X (out)
  ],
  iterator_types = ["parallel", "parallel"],
  doc = "X(i,j) += A(i,k) * B(k,j)"
}

#ij = {
  indexing_maps = [
    affine_map<(i,j) -> (i,j)>, // A
    affine_map<(i,j) -> (i,j)>, // B
    affine_map<(i,j) -> (i,j)>  // X (out)
  ],
  iterator_types = ["parallel", "parallel"],
  doc = "X(i,j) += A(i,k) * B(k,j)"
}

#ijk = {
  indexing_maps = [
    affine_map<(i,j,k) -> (i,k)>, // A
    affine_map<(i,j,k) -> (k,j)>, // B
    affine_map<(i,j,k) -> (i,j)>  // X (out)
  ],
  iterator_types = ["parallel", "parallel", "reduction"],
  doc = "X(i,j) += A(i,k) * B(k,j)"
}

#kij = {
  indexing_maps = [
    affine_map<(k,i,j) -> (i,k)>, // A
    affine_map<(k,i,j) -> (k,j)>, // B
    affine_map<(k,i,j) -> (i,j)>  // X (out)
  ],
  iterator_types = ["reduction", "parallel", "parallel"],
  doc = "X(i,j) += A(i,k) * B(k,j)"
}

#jki = {
  indexing_maps = [
    affine_map<(j,k,i) -> (i,k)>, // A
    affine_map<(j,k,i) -> (k,j)>, // B
    affine_map<(j,k,i) -> (i,j)>  // X (out)
  ],
  iterator_types = ["parallel", "reduction", "parallel"],
  doc = "X(i,j) += A(i,k) * B(k,j)"
}

#SPARSE = #sparse_tensor.encoding<{
  map = (i,j) -> (i : compressed, j : compressed)
}>

module {
func.func @kernel_spmm(%arga: tensor<?x?xf64, #SPARSE>,
                       %argb: tensor<?x?xf64, #SPARSE>,
                       %argc: tensor<?x?xf64, #SPARSE>,
                       %argd: tensor<?x?xf64, #SPARSE>,
                       %arge: tensor<?x?xf64, #SPARSE>,
                       %argf: tensor<?x?xf64, #SPARSE>,
                       %argx: tensor<?x?xf64, #SPARSE>) -> tensor<?x?xf64, #SPARSE> {
    %0 = linalg.generic #kij
          ins(%arga, %argb: tensor<?x?xf64, #SPARSE>, tensor<?x?xf64, #SPARSE>)
          outs(%argx: tensor<?x?xf64, #SPARSE>) {
          ^bb(%a: f64, %b: f64, %x: f64):
            %1 = arith.mulf %a, %b : f64
            %2 = arith.addf %x, %1 : f64
            linalg.yield %2 : f64
        } -> tensor<?x?xf64, #SPARSE>
     %7 = linalg.generic #ji
                         ins(%0, %argc: tensor<?x?xf64, #SPARSE>, tensor<?x?xf64, #SPARSE>)
                         outs(%argx: tensor<?x?xf64, #SPARSE>) {
                         ^bb(%a: f64, %b: f64, %x: f64):
                           %7 = arith.mulf %a, %b : f64
                           linalg.yield %7 : f64
                       } -> tensor<?x?xf64, #SPARSE>

     %1 = linalg.generic #jik
               ins(%0, %argc: tensor<?x?xf64, #SPARSE>, tensor<?x?xf64, #SPARSE>)
               outs(%argx: tensor<?x?xf64, #SPARSE>) {
               ^bb(%a: f64, %b: f64, %x: f64):
                 %1 = arith.mulf %a, %b : f64
                 %2 = arith.addf %x, %1 : f64
                 linalg.yield %2 : f64
             } -> tensor<?x?xf64, #SPARSE>

    %5 = linalg.generic #jki
              ins(%0, %argc: tensor<?x?xf64, #SPARSE>, tensor<?x?xf64, #SPARSE>)
              outs(%argx: tensor<?x?xf64, #SPARSE>) {
              ^bb(%a: f64, %b: f64, %x: f64):
                %5 = arith.mulf %a, %b : f64
                %6 = arith.addf %x, %5 : f64
                linalg.yield %6 : f64
            } -> tensor<?x?xf64, #SPARSE>
    %2 = linalg.matmul ins(%1, %argd: tensor<?x?xf64, #SPARSE>, tensor<?x?xf64, #SPARSE>)
                       outs(%argx: tensor<?x?xf64, #SPARSE>) -> tensor<?x?xf64, #SPARSE>
    %3 = linalg.matmul ins(%2, %arge: tensor<?x?xf64, #SPARSE>, tensor<?x?xf64, #SPARSE>)
                       outs(%argx: tensor<?x?xf64, #SPARSE>) -> tensor<?x?xf64, #SPARSE>
    %4 = linalg.matmul ins(%3, %argf: tensor<?x?xf64, #SPARSE>, tensor<?x?xf64, #SPARSE>)
                       outs(%argx: tensor<?x?xf64, #SPARSE>) -> tensor<?x?xf64, #SPARSE>
    return %1 : tensor<?x?xf64, #SPARSE>
}

}