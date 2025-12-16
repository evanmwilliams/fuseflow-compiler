// RUN: sam-opt %s > %t

#CSR = #sparse_tensor.encoding<{
  lvlTypes = [ "dense", "compressed" ],
  dimToLvl = affine_map<(i,j) -> (i,j)>
}>

// func.func @main(%ref: tensor<?xi32, #REF>) -> () {
func.func @main(%ref: tensor<?xi32>) -> (tensor<?xi32>, tensor<?xi32>) {
    %0, %1 = sam.fiber_lookup %ref : tensor<?xi32> -> tensor<?xi32>, tensor<?xi32>
    sam.fiber_write Crd, %0 : tensor<?xi32> 
    return %0, %1: tensor<?xi32>, tensor<?xi32>
}