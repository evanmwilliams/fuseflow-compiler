// RUN: sam-opt %s > %t

#CSR = #sparse_tensor.encoding<{
  lvlTypes = [ "dense", "compressed" ],
  dimToLvl = affine_map<(i,j) -> (i,j)>
}>
// #CSR   = #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>
// #DCSR = #sparse_tensor.encoding<{map = (d0, d1) -> (d0 : compressed, d1 : compressed)}>

// func.func private @getTensorFilename(index) -> (!Filename)

func.func @sparse_new(%arg0: !llvm.ptr<i8>) -> tensor<?x?xf32, #CSR> {
  %0 = sparse_tensor.new %arg0 : !llvm.ptr<i8> to tensor<?x?xf32, #CSR>
  return %0 : tensor<?x?xf32, #CSR>
}