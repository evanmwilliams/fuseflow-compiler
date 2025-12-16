#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0, 0)>
module {
  // func.func @matmul1(%arg0: tensor<1767x1767xf32>, %arg1: tensor<1767x50xf32>, %arg2: tensor<1767x50xf32>) -> tensor<f32, #sam.encoding< Val, "d0", "t0", "", 0 : i64>> {
  //   %0 = "sam.generator"() : () -> tensor<index, #sam.encoding< Ref, "d0", "t0", "", 0 : i64>>
  //   %output_ref, %output_crd = "sam.fiber_lookup"(%0) <{index_var = 0 : index, tensor_num = 0 : index}> : (tensor<index, #sam.encoding< Ref, "d0", "t0", "", 0 : i64>>) -> (tensor<index, #sam.encoding< Ref, "d0", "t0", "dense", 0 : i64>>, tensor<index, #sam.encoding< Crd, "d0", "t0", "dense", 0 : i64>>)
  //   %1 = "sam.generator"() : () -> tensor<index, #sam.encoding< Ref, "d0", "t1", "root", 0 : i64>>
  //   %2 = "sam.repeat"(%1, %output_ref) : (tensor<index, #sam.encoding< Ref, "d0", "t1", "root", 0 : i64>>, tensor<index, #sam.encoding< Ref, "d0", "t0", "dense", 0 : i64>>) -> tensor<index, #sam.encoding< Ref, "d0", "t1", "dense", 0 : i64>>
  //   %output_ref_0, %output_crd_1 = "sam.fiber_lookup"(%2) <{index_var = 2 : index, tensor_num = 1 : index}> : (tensor<index, #sam.encoding< Ref, "d0", "t1", "dense", 0 : i64>>) -> (tensor<index, #sam.encoding< Ref, "d2", "t1", "dense", 1 : i64>>, tensor<index, #sam.encoding< Crd, "d2", "t1", "dense", 1 : i64>>)
  //   %3 = "sam.repeat"(%output_ref, %output_ref_0) : (tensor<index, #sam.encoding< Ref, "d0", "t0", "dense", 0 : i64>>, tensor<index, #sam.encoding< Ref, "d2", "t1", "dense", 1 : i64>>) -> tensor<index, #sam.encoding< Ref, "d2", "t0", "dense", 0 : i64>>
  //   %output_ref_2, %output_crd_3 = "sam.fiber_lookup"(%3) <{index_var = 1 : index, tensor_num = 0 : index}> : (tensor<index, #sam.encoding< Ref, "d2", "t0", "dense", 0 : i64>>) -> (tensor<index, #sam.encoding< Ref, "d1", "t0", "dense", 1 : i64>>, tensor<index, #sam.encoding< Crd, "d1", "t0", "dense", 1 : i64>>)
  //   %output_ref_4, %output_crd_5 = "sam.fiber_lookup"(%output_ref_0) <{index_var = 1 : index, tensor_num = 1 : index}> : (tensor<index, #sam.encoding< Ref, "d2", "t1", "dense", 1 : i64>>) -> (tensor<index, #sam.encoding< Ref, "d1", "t1", "dense", 0 : i64>>, tensor<index, #sam.encoding< Crd, "d1", "t1", "dense", 0 : i64>>)
  //   %output_crd_6, %output_refs:2 = "sam.joiner"(%output_crd_3, %output_crd_5, %output_ref_2, %output_ref_4) <{joinerOp = #sam<joiner_type Intersect>, operandSegmentSizes = array<i32: 2, 2>}> : (tensor<index, #sam.encoding< Crd, "d1", "t0", "dense", 1 : i64>>, tensor<index, #sam.encoding< Crd, "d1", "t1", "dense", 0 : i64>>, tensor<index, #sam.encoding< Ref, "d1", "t0", "dense", 1 : i64>>, tensor<index, #sam.encoding< Ref, "d1", "t1", "dense", 0 : i64>>) -> (tensor<index, #sam.encoding< Crd, "d1", "t1", "dense", 0 : i64>>, tensor<index, #sam.encoding< Ref, "d1", "t0", "dense", 0 : i64>>, tensor<index, #sam.encoding< Ref, "d1", "t1", "dense", 0 : i64>>)
  //   %4 = "sam.array_val"(%output_refs#0) <{stream_shape = array<i64: 1>}> : (tensor<index, #sam.encoding< Ref, "d1", "t0", "dense", 0 : i64>>) -> tensor<f32, #sam.encoding< Val, "d2", "t0", "", 0 : i64>>
  //   %5 = "sam.array_val"(%output_refs#1) <{stream_shape = array<i64: 1>}> : (tensor<index, #sam.encoding< Ref, "d1", "t1", "dense", 0 : i64>>) -> tensor<f32, #sam.encoding< Val, "d2", "t1", "", 0 : i64>>
  //   %6 = "sam.alu"(%4, %5) <{aluOp = #sam<op_type Mul>}> : (tensor<f32, #sam.encoding< Val, "d2", "t0", "", 0 : i64>>, tensor<f32, #sam.encoding< Val, "d2", "t1", "", 0 : i64>>) -> tensor<f32, #sam.encoding< Val, "d0", "t0", "", 0 : i64>>
  //   %7 = "sam.reduce"(%6) <{reduceType = #sam<reduce_type Add>}> : (tensor<f32, #sam.encoding< Val, "d0", "t0", "", 0 : i64>>) -> tensor<f32, #sam.encoding< Val, "d0", "t0", "", 0 : i64>>
  //   "sam.fiber_write"(%output_crd) <{writeOp = #sam<write_type Crd>}> : (tensor<index, #sam.encoding< Crd, "d0", "t0", "dense", 0 : i64>>) -> ()
  //   "sam.fiber_write"(%output_crd_1) <{writeOp = #sam<write_type Crd>}> : (tensor<index, #sam.encoding< Crd, "d2", "t1", "dense", 1 : i64>>) -> ()
  //   "sam.fiber_write"(%7) <{writeOp = #sam<write_type Val>}> : (tensor<f32, #sam.encoding< Val, "d0", "t0", "", 0 : i64>>) -> ()
  //   %8 = linalg.matmul ins(%arg0, %arg1 : tensor<1767x1767xf32>, tensor<1767x50xf32>) outs(%arg2 : tensor<1767x50xf32>) -> tensor<1767x50xf32>
  //   return %7 : tensor<f32, #sam.encoding< Val, "d0", "t0", "", 0 : i64>>
  // }
  // func.func @matmul3(%arg0: tensor<1767x50xf32>, %arg1: tensor<50x16xf32>, %arg2: tensor<1767x16xf32>) -> tensor<f32, #sam.encoding< Val, "d0", "t0", "", 0 : i64>> {
  //   %0 = "sam.generator"() : () -> tensor<index, #sam.encoding< Ref, "d0", "t0", "", 0 : i64>>
  //   %output_ref, %output_crd = "sam.fiber_lookup"(%0) <{index_var = 0 : index, tensor_num = 0 : index}> : (tensor<index, #sam.encoding< Ref, "d0", "t0", "", 0 : i64>>) -> (tensor<index, #sam.encoding< Ref, "d0", "t0", "dense", 0 : i64>>, tensor<index, #sam.encoding< Crd, "d0", "t0", "dense", 0 : i64>>)
  //   %1 = "sam.generator"() : () -> tensor<index, #sam.encoding< Ref, "d0", "t1", "root", 0 : i64>>
  //   %2 = "sam.repeat"(%1, %output_ref) : (tensor<index, #sam.encoding< Ref, "d0", "t1", "root", 0 : i64>>, tensor<index, #sam.encoding< Ref, "d0", "t0", "dense", 0 : i64>>) -> tensor<index, #sam.encoding< Ref, "d0", "t1", "dense", 0 : i64>>
  //   %output_ref_0, %output_crd_1 = "sam.fiber_lookup"(%2) <{index_var = 2 : index, tensor_num = 1 : index}> : (tensor<index, #sam.encoding< Ref, "d0", "t1", "dense", 0 : i64>>) -> (tensor<index, #sam.encoding< Ref, "d2", "t1", "dense", 1 : i64>>, tensor<index, #sam.encoding< Crd, "d2", "t1", "dense", 1 : i64>>)
  //   %3 = "sam.repeat"(%output_ref, %output_ref_0) : (tensor<index, #sam.encoding< Ref, "d0", "t0", "dense", 0 : i64>>, tensor<index, #sam.encoding< Ref, "d2", "t1", "dense", 1 : i64>>) -> tensor<index, #sam.encoding< Ref, "d2", "t0", "dense", 0 : i64>>
  //   %output_ref_2, %output_crd_3 = "sam.fiber_lookup"(%3) <{index_var = 1 : index, tensor_num = 0 : index}> : (tensor<index, #sam.encoding< Ref, "d2", "t0", "dense", 0 : i64>>) -> (tensor<index, #sam.encoding< Ref, "d1", "t0", "dense", 1 : i64>>, tensor<index, #sam.encoding< Crd, "d1", "t0", "dense", 1 : i64>>)
  //   %output_ref_4, %output_crd_5 = "sam.fiber_lookup"(%output_ref_0) <{index_var = 1 : index, tensor_num = 1 : index}> : (tensor<index, #sam.encoding< Ref, "d2", "t1", "dense", 1 : i64>>) -> (tensor<index, #sam.encoding< Ref, "d1", "t1", "dense", 0 : i64>>, tensor<index, #sam.encoding< Crd, "d1", "t1", "dense", 0 : i64>>)
  //   %output_crd_6, %output_refs:2 = "sam.joiner"(%output_crd_3, %output_crd_5, %output_ref_2, %output_ref_4) <{joinerOp = #sam<joiner_type Intersect>, operandSegmentSizes = array<i32: 2, 2>}> : (tensor<index, #sam.encoding< Crd, "d1", "t0", "dense", 1 : i64>>, tensor<index, #sam.encoding< Crd, "d1", "t1", "dense", 0 : i64>>, tensor<index, #sam.encoding< Ref, "d1", "t0", "dense", 1 : i64>>, tensor<index, #sam.encoding< Ref, "d1", "t1", "dense", 0 : i64>>) -> (tensor<index, #sam.encoding< Crd, "d1", "t1", "dense", 0 : i64>>, tensor<index, #sam.encoding< Ref, "d1", "t0", "dense", 0 : i64>>, tensor<index, #sam.encoding< Ref, "d1", "t1", "dense", 0 : i64>>)
  //   %4 = "sam.array_val"(%output_refs#0) <{stream_shape = array<i64: 1>}> : (tensor<index, #sam.encoding< Ref, "d1", "t0", "dense", 0 : i64>>) -> tensor<f32, #sam.encoding< Val, "d2", "t0", "", 0 : i64>>
  //   %5 = "sam.array_val"(%output_refs#1) <{stream_shape = array<i64: 1>}> : (tensor<index, #sam.encoding< Ref, "d1", "t1", "dense", 0 : i64>>) -> tensor<f32, #sam.encoding< Val, "d2", "t1", "", 0 : i64>>
  //   %6 = "sam.alu"(%4, %5) <{aluOp = #sam<op_type Mul>}> : (tensor<f32, #sam.encoding< Val, "d2", "t0", "", 0 : i64>>, tensor<f32, #sam.encoding< Val, "d2", "t1", "", 0 : i64>>) -> tensor<f32, #sam.encoding< Val, "d0", "t0", "", 0 : i64>>
  //   %7 = "sam.reduce"(%6) <{reduceType = #sam<reduce_type Add>}> : (tensor<f32, #sam.encoding< Val, "d0", "t0", "", 0 : i64>>) -> tensor<f32, #sam.encoding< Val, "d0", "t0", "", 0 : i64>>
  //   "sam.fiber_write"(%output_crd) <{writeOp = #sam<write_type Crd>}> : (tensor<index, #sam.encoding< Crd, "d0", "t0", "dense", 0 : i64>>) -> ()
  //   "sam.fiber_write"(%output_crd_1) <{writeOp = #sam<write_type Crd>}> : (tensor<index, #sam.encoding< Crd, "d2", "t1", "dense", 1 : i64>>) -> ()
  //   "sam.fiber_write"(%7) <{writeOp = #sam<write_type Val>}> : (tensor<f32, #sam.encoding< Val, "d0", "t0", "", 0 : i64>>) -> ()
  //   %8 = linalg.matmul ins(%arg0, %arg1 : tensor<1767x50xf32>, tensor<50x16xf32>) outs(%arg2 : tensor<1767x16xf32>) -> tensor<1767x16xf32>
  //   return %7 : tensor<f32, #sam.encoding< Val, "d0", "t0", "", 0 : i64>>
  // }
  // func.func @matmul5(%arg0: tensor<1767x1767xf32>, %arg1: tensor<1767x16xf32>, %arg2: tensor<1767x16xf32>) -> tensor<f32, #sam.encoding< Val, "d0", "t0", "", 0 : i64>> {
  //   %0 = "sam.generator"() : () -> tensor<index, #sam.encoding< Ref, "d0", "t0", "", 0 : i64>>
  //   %output_ref, %output_crd = "sam.fiber_lookup"(%0) <{index_var = 0 : index, tensor_num = 0 : index}> : (tensor<index, #sam.encoding< Ref, "d0", "t0", "", 0 : i64>>) -> (tensor<index, #sam.encoding< Ref, "d0", "t0", "dense", 0 : i64>>, tensor<index, #sam.encoding< Crd, "d0", "t0", "dense", 0 : i64>>)
  //   %1 = "sam.generator"() : () -> tensor<index, #sam.encoding< Ref, "d0", "t1", "root", 0 : i64>>
  //   %2 = "sam.repeat"(%1, %output_ref) : (tensor<index, #sam.encoding< Ref, "d0", "t1", "root", 0 : i64>>, tensor<index, #sam.encoding< Ref, "d0", "t0", "dense", 0 : i64>>) -> tensor<index, #sam.encoding< Ref, "d0", "t1", "dense", 0 : i64>>
  //   %output_ref_0, %output_crd_1 = "sam.fiber_lookup"(%2) <{index_var = 2 : index, tensor_num = 1 : index}> : (tensor<index, #sam.encoding< Ref, "d0", "t1", "dense", 0 : i64>>) -> (tensor<index, #sam.encoding< Ref, "d2", "t1", "dense", 1 : i64>>, tensor<index, #sam.encoding< Crd, "d2", "t1", "dense", 1 : i64>>)
  //   %3 = "sam.repeat"(%output_ref, %output_ref_0) : (tensor<index, #sam.encoding< Ref, "d0", "t0", "dense", 0 : i64>>, tensor<index, #sam.encoding< Ref, "d2", "t1", "dense", 1 : i64>>) -> tensor<index, #sam.encoding< Ref, "d2", "t0", "dense", 0 : i64>>
  //   %output_ref_2, %output_crd_3 = "sam.fiber_lookup"(%3) <{index_var = 1 : index, tensor_num = 0 : index}> : (tensor<index, #sam.encoding< Ref, "d2", "t0", "dense", 0 : i64>>) -> (tensor<index, #sam.encoding< Ref, "d1", "t0", "dense", 1 : i64>>, tensor<index, #sam.encoding< Crd, "d1", "t0", "dense", 1 : i64>>)
  //   %output_ref_4, %output_crd_5 = "sam.fiber_lookup"(%output_ref_0) <{index_var = 1 : index, tensor_num = 1 : index}> : (tensor<index, #sam.encoding< Ref, "d2", "t1", "dense", 1 : i64>>) -> (tensor<index, #sam.encoding< Ref, "d1", "t1", "dense", 0 : i64>>, tensor<index, #sam.encoding< Crd, "d1", "t1", "dense", 0 : i64>>)
  //   %output_crd_6, %output_refs:2 = "sam.joiner"(%output_crd_3, %output_crd_5, %output_ref_2, %output_ref_4) <{joinerOp = #sam<joiner_type Intersect>, operandSegmentSizes = array<i32: 2, 2>}> : (tensor<index, #sam.encoding< Crd, "d1", "t0", "dense", 1 : i64>>, tensor<index, #sam.encoding< Crd, "d1", "t1", "dense", 0 : i64>>, tensor<index, #sam.encoding< Ref, "d1", "t0", "dense", 1 : i64>>, tensor<index, #sam.encoding< Ref, "d1", "t1", "dense", 0 : i64>>) -> (tensor<index, #sam.encoding< Crd, "d1", "t1", "dense", 0 : i64>>, tensor<index, #sam.encoding< Ref, "d1", "t0", "dense", 0 : i64>>, tensor<index, #sam.encoding< Ref, "d1", "t1", "dense", 0 : i64>>)
  //   %4 = "sam.array_val"(%output_refs#0) <{stream_shape = array<i64: 1>}> : (tensor<index, #sam.encoding< Ref, "d1", "t0", "dense", 0 : i64>>) -> tensor<f32, #sam.encoding< Val, "d2", "t0", "", 0 : i64>>
  //   %5 = "sam.array_val"(%output_refs#1) <{stream_shape = array<i64: 1>}> : (tensor<index, #sam.encoding< Ref, "d1", "t1", "dense", 0 : i64>>) -> tensor<f32, #sam.encoding< Val, "d2", "t1", "", 0 : i64>>
  //   %6 = "sam.alu"(%4, %5) <{aluOp = #sam<op_type Mul>}> : (tensor<f32, #sam.encoding< Val, "d2", "t0", "", 0 : i64>>, tensor<f32, #sam.encoding< Val, "d2", "t1", "", 0 : i64>>) -> tensor<f32, #sam.encoding< Val, "d0", "t0", "", 0 : i64>>
  //   %7 = "sam.reduce"(%6) <{reduceType = #sam<reduce_type Add>}> : (tensor<f32, #sam.encoding< Val, "d0", "t0", "", 0 : i64>>) -> tensor<f32, #sam.encoding< Val, "d0", "t0", "", 0 : i64>>
  //   "sam.fiber_write"(%output_crd) <{writeOp = #sam<write_type Crd>}> : (tensor<index, #sam.encoding< Crd, "d0", "t0", "dense", 0 : i64>>) -> ()
  //   "sam.fiber_write"(%output_crd_1) <{writeOp = #sam<write_type Crd>}> : (tensor<index, #sam.encoding< Crd, "d2", "t1", "dense", 1 : i64>>) -> ()
  //   "sam.fiber_write"(%7) <{writeOp = #sam<write_type Val>}> : (tensor<f32, #sam.encoding< Val, "d0", "t0", "", 0 : i64>>) -> ()
  //   %8 = linalg.matmul ins(%arg0, %arg1 : tensor<1767x1767xf32>, tensor<1767x16xf32>) outs(%arg2 : tensor<1767x16xf32>) -> tensor<1767x16xf32>
  //   return %7 : tensor<f32, #sam.encoding< Val, "d0", "t0", "", 0 : i64>>
  // }
  // func.func @matmul7(%arg0: tensor<1767x16xf32>, %arg1: tensor<16x121xf32>, %arg2: tensor<1767x121xf32>) -> tensor<f32, #sam.encoding< Val, "d0", "t0", "", 0 : i64>> {
  //   %0 = "sam.generator"() : () -> tensor<index, #sam.encoding< Ref, "d0", "t1", "", 0 : i64>>
  //   %output_ref, %output_crd = "sam.fiber_lookup"(%0) <{index_var = 0 : index, tensor_num = 1 : index}> : (tensor<index, #sam.encoding< Ref, "d0", "t1", "", 0 : i64>>) -> (tensor<index, #sam.encoding< Ref, "d0", "t1", "dense", 0 : i64>>, tensor<index, #sam.encoding< Crd, "d0", "t1", "dense", 0 : i64>>)
  //   %1 = "sam.generator"() : () -> tensor<index, #sam.encoding< Ref, "d0", "t0", "root", 0 : i64>>
  //   %2 = "sam.repeat"(%1, %output_ref) : (tensor<index, #sam.encoding< Ref, "d0", "t0", "root", 0 : i64>>, tensor<index, #sam.encoding< Ref, "d0", "t1", "dense", 0 : i64>>) -> tensor<index, #sam.encoding< Ref, "d0", "t0", "dense", 0 : i64>>
  //   %output_ref_0, %output_crd_1 = "sam.fiber_lookup"(%2) <{index_var = 2 : index, tensor_num = 0 : index}> : (tensor<index, #sam.encoding< Ref, "d0", "t0", "dense", 0 : i64>>) -> (tensor<index, #sam.encoding< Ref, "d2", "t0", "dense", 1 : i64>>, tensor<index, #sam.encoding< Crd, "d2", "t0", "dense", 1 : i64>>)
  //   %3 = "sam.repeat"(%output_ref, %output_ref_0) : (tensor<index, #sam.encoding< Ref, "d0", "t1", "dense", 0 : i64>>, tensor<index, #sam.encoding< Ref, "d2", "t0", "dense", 1 : i64>>) -> tensor<index, #sam.encoding< Ref, "d2", "t1", "dense", 0 : i64>>
  //   %output_ref_2, %output_crd_3 = "sam.fiber_lookup"(%output_ref_0) <{index_var = 1 : index, tensor_num = 0 : index}> : (tensor<index, #sam.encoding< Ref, "d2", "t0", "dense", 1 : i64>>) -> (tensor<index, #sam.encoding< Ref, "d1", "t0", "dense", 0 : i64>>, tensor<index, #sam.encoding< Crd, "d1", "t0", "dense", 0 : i64>>)
  //   %output_ref_4, %output_crd_5 = "sam.fiber_lookup"(%3) <{index_var = 1 : index, tensor_num = 1 : index}> : (tensor<index, #sam.encoding< Ref, "d2", "t1", "dense", 0 : i64>>) -> (tensor<index, #sam.encoding< Ref, "d1", "t1", "dense", 1 : i64>>, tensor<index, #sam.encoding< Crd, "d1", "t1", "dense", 1 : i64>>)
  //   %output_crd_6, %output_refs:2 = "sam.joiner"(%output_crd_3, %output_crd_5, %output_ref_2, %output_ref_4) <{joinerOp = #sam<joiner_type Intersect>, operandSegmentSizes = array<i32: 2, 2>}> : (tensor<index, #sam.encoding< Crd, "d1", "t0", "dense", 0 : i64>>, tensor<index, #sam.encoding< Crd, "d1", "t1", "dense", 1 : i64>>, tensor<index, #sam.encoding< Ref, "d1", "t0", "dense", 0 : i64>>, tensor<index, #sam.encoding< Ref, "d1", "t1", "dense", 1 : i64>>) -> (tensor<index, #sam.encoding< Crd, "d1", "t1", "dense", 0 : i64>>, tensor<index, #sam.encoding< Ref, "d1", "t0", "dense", 0 : i64>>, tensor<index, #sam.encoding< Ref, "d1", "t1", "dense", 0 : i64>>)
  //   %4 = "sam.array_val"(%output_refs#0) <{stream_shape = array<i64: 1>}> : (tensor<index, #sam.encoding< Ref, "d1", "t0", "dense", 0 : i64>>) -> tensor<f32, #sam.encoding< Val, "d2", "t0", "", 0 : i64>>
  //   %5 = "sam.array_val"(%output_refs#1) <{stream_shape = array<i64: 1>}> : (tensor<index, #sam.encoding< Ref, "d1", "t1", "dense", 0 : i64>>) -> tensor<f32, #sam.encoding< Val, "d2", "t1", "", 0 : i64>>
  //   %6 = "sam.alu"(%5, %4) <{aluOp = #sam<op_type Mul>}> : (tensor<f32, #sam.encoding< Val, "d2", "t1", "", 0 : i64>>, tensor<f32, #sam.encoding< Val, "d2", "t0", "", 0 : i64>>) -> tensor<f32, #sam.encoding< Val, "d0", "t0", "", 0 : i64>>
  //   %7 = "sam.reduce"(%6) <{reduceType = #sam<reduce_type Add>}> : (tensor<f32, #sam.encoding< Val, "d0", "t0", "", 0 : i64>>) -> tensor<f32, #sam.encoding< Val, "d0", "t0", "", 0 : i64>>
  //   "sam.fiber_write"(%output_crd) <{writeOp = #sam<write_type Crd>}> : (tensor<index, #sam.encoding< Crd, "d0", "t1", "dense", 0 : i64>>) -> ()
  //   "sam.fiber_write"(%output_crd_1) <{writeOp = #sam<write_type Crd>}> : (tensor<index, #sam.encoding< Crd, "d2", "t0", "dense", 1 : i64>>) -> ()
  //   "sam.fiber_write"(%7) <{writeOp = #sam<write_type Val>}> : (tensor<f32, #sam.encoding< Val, "d0", "t0", "", 0 : i64>>) -> ()
  //   %8 = linalg.matmul ins(%arg0, %arg1 : tensor<1767x16xf32>, tensor<16x121xf32>) outs(%arg2 : tensor<1767x121xf32>) -> tensor<1767x121xf32>
  //   return %7 : tensor<f32, #sam.encoding< Val, "d0", "t0", "", 0 : i64>>
  // }
  // func.func @sub9(%arg0: tensor<1767x121xf32>, %arg1: tensor<1767x1xf32>) -> tensor<f32, #sam.encoding< Val, "d0", "t0", "", 0 : i64>> {
  //   %0 = "sam.generator"() : () -> tensor<index, #sam.encoding< Ref, "d0", "t0", "", 0 : i64>>
  //   %output_ref, %output_crd = "sam.fiber_lookup"(%0) <{index_var = 0 : index, tensor_num = 0 : index}> : (tensor<index, #sam.encoding< Ref, "d0", "t0", "", 0 : i64>>) -> (tensor<index, #sam.encoding< Ref, "d0", "t0", "dense", 0 : i64>>, tensor<index, #sam.encoding< Crd, "d0", "t0", "dense", 0 : i64>>)
  //   %1 = "sam.generator"() : () -> tensor<index, #sam.encoding< Ref, "d0", "t1", "", 0 : i64>>
  //   %output_ref_0, %output_crd_1 = "sam.fiber_lookup"(%1) <{index_var = 0 : index, tensor_num = 1 : index}> : (tensor<index, #sam.encoding< Ref, "d0", "t1", "", 0 : i64>>) -> (tensor<index, #sam.encoding< Ref, "d0", "t1", "dense", 0 : i64>>, tensor<index, #sam.encoding< Crd, "d0", "t1", "dense", 0 : i64>>)
  //   %output_crd_2, %output_refs:2 = "sam.joiner"(%output_crd, %output_crd_1, %output_ref, %output_ref_0) <{joinerOp = #sam<joiner_type Intersect>, operandSegmentSizes = array<i32: 2, 2>}> : (tensor<index, #sam.encoding< Crd, "d0", "t0", "dense", 0 : i64>>, tensor<index, #sam.encoding< Crd, "d0", "t1", "dense", 0 : i64>>, tensor<index, #sam.encoding< Ref, "d0", "t0", "dense", 0 : i64>>, tensor<index, #sam.encoding< Ref, "d0", "t1", "dense", 0 : i64>>) -> (tensor<index, #sam.encoding< Crd, "d0", "t1", "dense", 0 : i64>>, tensor<index, #sam.encoding< Ref, "d0", "t0", "dense", 0 : i64>>, tensor<index, #sam.encoding< Ref, "d0", "t1", "dense", 0 : i64>>)
  //   %output_ref_3, %output_crd_4 = "sam.fiber_lookup"(%output_refs#0) <{index_var = 1 : index, tensor_num = 0 : index}> : (tensor<index, #sam.encoding< Ref, "d0", "t0", "dense", 0 : i64>>) -> (tensor<index, #sam.encoding< Ref, "d1", "t0", "dense", 1 : i64>>, tensor<index, #sam.encoding< Crd, "d1", "t0", "dense", 1 : i64>>)
  //   %output_ref_5, %output_crd_6 = "sam.fiber_lookup"(%output_refs#1) <{index_var = 1 : index, tensor_num = 1 : index}> : (tensor<index, #sam.encoding< Ref, "d0", "t1", "dense", 0 : i64>>) -> (tensor<index, #sam.encoding< Ref, "d1", "t1", "dense", 1 : i64>>, tensor<index, #sam.encoding< Crd, "d1", "t1", "dense", 1 : i64>>)
  //   %output_crd_7, %output_refs_8:2 = "sam.joiner"(%output_crd_4, %output_crd_6, %output_ref_3, %output_ref_5) <{joinerOp = #sam<joiner_type Intersect>, operandSegmentSizes = array<i32: 2, 2>}> : (tensor<index, #sam.encoding< Crd, "d1", "t0", "dense", 1 : i64>>, tensor<index, #sam.encoding< Crd, "d1", "t1", "dense", 1 : i64>>, tensor<index, #sam.encoding< Ref, "d1", "t0", "dense", 1 : i64>>, tensor<index, #sam.encoding< Ref, "d1", "t1", "dense", 1 : i64>>) -> (tensor<index, #sam.encoding< Crd, "d1", "t3", "dense", 0 : i64>>, tensor<index, #sam.encoding< Ref, "d1", "t2", "dense", 0 : i64>>, tensor<index, #sam.encoding< Ref, "d1", "t3", "dense", 0 : i64>>)
  //   %2 = "sam.array_val"(%output_refs_8#0) <{stream_shape = array<i64: 1>}> : (tensor<index, #sam.encoding< Ref, "d1", "t2", "dense", 0 : i64>>) -> tensor<f32, #sam.encoding< Val, "d2", "t0", "", 0 : i64>>
  //   %3 = "sam.array_val"(%output_refs_8#1) <{stream_shape = array<i64: 1>}> : (tensor<index, #sam.encoding< Ref, "d1", "t3", "dense", 0 : i64>>) -> tensor<f32, #sam.encoding< Val, "d2", "t1", "", 0 : i64>>
  //   %4 = "sam.alu"(%3, %2) <{aluOp = #sam<op_type Sub>}> : (tensor<f32, #sam.encoding< Val, "d2", "t1", "", 0 : i64>>, tensor<f32, #sam.encoding< Val, "d2", "t0", "", 0 : i64>>) -> tensor<f32, #sam.encoding< Val, "d0", "t0", "", 0 : i64>>
  //   "sam.fiber_write"(%output_crd_2) <{writeOp = #sam<write_type Crd>}> : (tensor<index, #sam.encoding< Crd, "d0", "t1", "dense", 0 : i64>>) -> ()
  //   "sam.fiber_write"(%output_crd_7) <{writeOp = #sam<write_type Crd>}> : (tensor<index, #sam.encoding< Crd, "d1", "t3", "dense", 0 : i64>>) -> ()
  //   "sam.fiber_write"(%4) <{writeOp = #sam<write_type Val>}> : (tensor<f32, #sam.encoding< Val, "d0", "t0", "", 0 : i64>>) -> ()
  //   %5 = tensor.empty() : tensor<1767x121xf32>
  //   %6 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : tensor<1767x121xf32>, tensor<1767x1xf32>) outs(%5 : tensor<1767x121xf32>) {
  //   ^bb0(%in: f32, %in_9: f32, %out: f32):
  //     %7 = arith.subf %in, %in_9 : f32
  //     linalg.yield %7 : f32
  //   } -> tensor<1767x121xf32>
  //   return %4 : tensor<f32, #sam.encoding< Val, "d0", "t0", "", 0 : i64>>
  // }
  // func.func @exp10(%arg0: tensor<1767x121xf32>) -> tensor<f32, #sam.encoding< Val, "d0", "t0", "", 0 : i64>> {
  //   %0 = "sam.generator"() : () -> tensor<index, #sam.encoding< Ref, "d0", "t0", "", 0 : i64>>
  //   %output_ref, %output_crd = "sam.fiber_lookup"(%0) <{index_var = 0 : index, tensor_num = 0 : index}> : (tensor<index, #sam.encoding< Ref, "d0", "t0", "", 0 : i64>>) -> (tensor<index, #sam.encoding< Ref, "d0", "t0", "dense", 0 : i64>>, tensor<index, #sam.encoding< Crd, "d0", "t0", "dense", 0 : i64>>)
  //   %output_ref_0, %output_crd_1 = "sam.fiber_lookup"(%output_ref) <{index_var = 1 : index, tensor_num = 0 : index}> : (tensor<index, #sam.encoding< Ref, "d0", "t0", "dense", 0 : i64>>) -> (tensor<index, #sam.encoding< Ref, "d1", "t0", "dense", 1 : i64>>, tensor<index, #sam.encoding< Crd, "d1", "t0", "dense", 1 : i64>>)
  //   %1 = "sam.array_val"(%output_ref_0) <{stream_shape = array<i64: 1>}> : (tensor<index, #sam.encoding< Ref, "d1", "t0", "dense", 1 : i64>>) -> tensor<f32, #sam.encoding< Val, "d1", "t0", "", 0 : i64>>
  //   %2 = "sam.alu"(%1) <{aluOp = #sam<op_type Exp>}> : (tensor<f32, #sam.encoding< Val, "d1", "t0", "", 0 : i64>>) -> tensor<f32, #sam.encoding< Val, "d0", "t0", "", 0 : i64>>
  //   "sam.fiber_write"(%output_crd) <{writeOp = #sam<write_type Crd>}> : (tensor<index, #sam.encoding< Crd, "d0", "t0", "dense", 0 : i64>>) -> ()
  //   "sam.fiber_write"(%output_crd_1) <{writeOp = #sam<write_type Crd>}> : (tensor<index, #sam.encoding< Crd, "d1", "t0", "dense", 1 : i64>>) -> ()
  //   "sam.fiber_write"(%2) <{writeOp = #sam<write_type Val>}> : (tensor<f32, #sam.encoding< Val, "d0", "t0", "", 0 : i64>>) -> ()
  //   %3 = tensor.empty() : tensor<1767x121xf32>
  //   %4 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<1767x121xf32>) outs(%3 : tensor<1767x121xf32>) {
  //   ^bb0(%in: f32, %out: f32):
  //     %5 = math.exp %in : f32
  //     linalg.yield %5 : f32
  //   } -> tensor<1767x121xf32>
  //   return %2 : tensor<f32, #sam.encoding< Val, "d0", "t0", "", 0 : i64>>
  // }
  func.func @div12(%arg0: tensor<1767x121xf32>, %arg1: tensor<1767x1xf32>) -> tensor<f32, #sam.encoding< Val, "d0", "t0", "", 0 : i64>> {
    %0 = "sam.generator"() : () -> tensor<index, #sam.encoding< Ref, "d0", "t0", "", 0 : i64>>
    %output_ref, %output_crd = "sam.fiber_lookup"(%0) <{index_var = 0 : index, tensor_num = 0 : index}> : (tensor<index, #sam.encoding< Ref, "d0", "t0", "", 0 : i64>>) -> (tensor<index, #sam.encoding< Ref, "d0", "t0", "dense", 0 : i64>>, tensor<index, #sam.encoding< Crd, "d0", "t0", "dense", 0 : i64>>)
    %1 = "sam.generator"() : () -> tensor<index, #sam.encoding< Ref, "d0", "t1", "", 0 : i64>>
    %output_ref_0, %output_crd_1 = "sam.fiber_lookup"(%1) <{index_var = 0 : index, tensor_num = 1 : index}> : (tensor<index, #sam.encoding< Ref, "d0", "t1", "", 0 : i64>>) -> (tensor<index, #sam.encoding< Ref, "d0", "t1", "dense", 0 : i64>>, tensor<index, #sam.encoding< Crd, "d0", "t1", "dense", 0 : i64>>)
    %output_crd_2, %output_refs:2 = "sam.joiner"(%output_crd, %output_crd_1, %output_ref, %output_ref_0) <{joinerOp = #sam<joiner_type Intersect>, operandSegmentSizes = array<i32: 2, 2>}> : (tensor<index, #sam.encoding< Crd, "d0", "t0", "dense", 0 : i64>>, tensor<index, #sam.encoding< Crd, "d0", "t1", "dense", 0 : i64>>, tensor<index, #sam.encoding< Ref, "d0", "t0", "dense", 0 : i64>>, tensor<index, #sam.encoding< Ref, "d0", "t1", "dense", 0 : i64>>) -> (tensor<index, #sam.encoding< Crd, "d0", "t1", "dense", 0 : i64>>, tensor<index, #sam.encoding< Ref, "d0", "t0", "dense", 0 : i64>>, tensor<index, #sam.encoding< Ref, "d0", "t1", "dense", 0 : i64>>)
    %output_ref_3, %output_crd_4 = "sam.fiber_lookup"(%output_refs#0) <{index_var = 1 : index, tensor_num = 0 : index}> : (tensor<index, #sam.encoding< Ref, "d0", "t0", "dense", 0 : i64>>) -> (tensor<index, #sam.encoding< Ref, "d1", "t0", "dense", 1 : i64>>, tensor<index, #sam.encoding< Crd, "d1", "t0", "dense", 1 : i64>>)
    %output_ref_5, %output_crd_6 = "sam.fiber_lookup"(%output_refs#1) <{index_var = 1 : index, tensor_num = 1 : index}> : (tensor<index, #sam.encoding< Ref, "d0", "t1", "dense", 0 : i64>>) -> (tensor<index, #sam.encoding< Ref, "d1", "t1", "dense", 1 : i64>>, tensor<index, #sam.encoding< Crd, "d1", "t1", "dense", 1 : i64>>)
    %output_crd_7, %output_refs_8:2 = "sam.joiner"(%output_crd_4, %output_crd_6, %output_ref_3, %output_ref_5) <{joinerOp = #sam<joiner_type Intersect>, operandSegmentSizes = array<i32: 2, 2>}> : (tensor<index, #sam.encoding< Crd, "d1", "t0", "dense", 1 : i64>>, tensor<index, #sam.encoding< Crd, "d1", "t1", "dense", 1 : i64>>, tensor<index, #sam.encoding< Ref, "d1", "t0", "dense", 1 : i64>>, tensor<index, #sam.encoding< Ref, "d1", "t1", "dense", 1 : i64>>) -> (tensor<index, #sam.encoding< Crd, "d1", "t3", "dense", 0 : i64>>, tensor<index, #sam.encoding< Ref, "d1", "t2", "dense", 0 : i64>>, tensor<index, #sam.encoding< Ref, "d1", "t3", "dense", 0 : i64>>)
    %2 = "sam.array_val"(%output_refs_8#0) <{stream_shape = array<i64: 1>}> : (tensor<index, #sam.encoding< Ref, "d1", "t2", "dense", 0 : i64>>) -> tensor<f32, #sam.encoding< Val, "d2", "t0", "", 0 : i64>>
    %3 = "sam.array_val"(%output_refs_8#1) <{stream_shape = array<i64: 1>}> : (tensor<index, #sam.encoding< Ref, "d1", "t3", "dense", 0 : i64>>) -> tensor<f32, #sam.encoding< Val, "d2", "t1", "", 0 : i64>>
    %4 = "sam.alu"(%3, %2) <{aluOp = #sam<op_type Div>}> : (tensor<f32, #sam.encoding< Val, "d2", "t1", "", 0 : i64>>, tensor<f32, #sam.encoding< Val, "d2", "t0", "", 0 : i64>>) -> tensor<f32, #sam.encoding< Val, "d0", "t0", "", 0 : i64>>
    "sam.fiber_write"(%output_crd_2) <{writeOp = #sam<write_type Crd>}> : (tensor<index, #sam.encoding< Crd, "d0", "t1", "dense", 0 : i64>>) -> ()
    "sam.fiber_write"(%output_crd_7) <{writeOp = #sam<write_type Crd>}> : (tensor<index, #sam.encoding< Crd, "d1", "t3", "dense", 0 : i64>>) -> ()
    "sam.fiber_write"(%4) <{writeOp = #sam<write_type Val>}> : (tensor<f32, #sam.encoding< Val, "d0", "t0", "", 0 : i64>>) -> ()
    %5 = tensor.empty() : tensor<1767x121xf32>
    %6 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : tensor<1767x121xf32>, tensor<1767x1xf32>) outs(%5 : tensor<1767x121xf32>) {
    ^bb0(%in: f32, %in_9: f32, %out: f32):
      %7 = arith.divf %in, %in_9 : f32
      linalg.yield %7 : f32
    } -> tensor<1767x121xf32>
    return %4 : tensor<f32, #sam.encoding< Val, "d0", "t0", "", 0 : i64>>
  }
}