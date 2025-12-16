#trait_scale = {
  indexing_maps = [
    affine_map<(i,j) -> (i,j)>,   // X (out)
    affine_map<(i,j) -> (i,j)>   // Y (out)
  ],
  iterator_types = ["parallel", "parallel"],
  doc = "Y(i,j) = X(i,j)"
}

#CSR = #sparse_tensor.encoding<{
  lvlTypes = [ "dense", "compressed" ],
  dimToLvl = affine_map<(i,j) -> (i,j)>
}>

// func.func @kernel_assign(%arga: tensor<?x?xf32, #CSR>) -> tensor<?x?xf32, #CSR> {
func.func @kernel_assign(%arga: tensor<?x?xf32, #CSR>, 
                         %argx: tensor<?x?xf32, #CSR>) {
    // func.return %0 : tensor<?x?xf64>
    %0 = linalg.generic #trait_scale
      ins(%arga: tensor<?x?xf32, #CSR>) 
      outs(%argx: tensor<?x?xf32, #CSR>) {
        ^bb(%a: f32, %x: f32):
          linalg.yield %a : f32
    } -> tensor<?x?xf32, #CSR>

    // return %0 : tensor<?x?xf32, #CSR>
    return
}

