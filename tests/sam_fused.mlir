func.func @forward(%arg0: tensor<1x784xf32>) -> tensor<1x10xf32> {
  %0 = "sam.generator"() : () -> tensor<index, #sam.encoding< Ref, "d0", "t0", "", 0 : i64>>
  %output_ref, %output_crd = "sam.fiber_lookup"(%0) <{index_var = 0 : index, tensor_num = 0 : index}> : (tensor<index, #sam.encoding< Ref, "d0", "t0", "", 0 : i64>>) -> (tensor<index, #sam.encoding< Ref, "d0", "t0", "dense", 0 : i64>>, tensor<index, #sam.encoding< Crd, "d0", "t0", "dense", 0 : i64>>)
  %1 = "sam.generator"() : () -> tensor<index, #sam.encoding< Ref, "d0", "t1", "root", 0 : i64>>
  %2 = "sam.repeat"(%1, %output_ref) : (tensor<index, #sam.encoding< Ref, "d0", "t1", "root", 0 : i64>>, tensor<index, #sam.encoding< Ref, "d0", "t0", "dense", 0 : i64>>) -> tensor<index, #sam.encoding< Ref, "d0", "t1", "dense", 0 : i64>>
  %3 = "sam.generator"() : () -> tensor<index, #sam.encoding< Ref, "d0", "t2", "root", 0 : i64>>
  %4 = "sam.repeat"(%3, %output_ref) : (tensor<index, #sam.encoding< Ref, "d0", "t2", "root", 0 : i64>>, tensor<index, #sam.encoding< Ref, "d0", "t0", "dense", 0 : i64>>) -> tensor<index, #sam.encoding< Ref, "d0", "t2", "dense", 0 : i64>>
  %5 = "sam.generator"() : () -> tensor<index, #sam.encoding< Ref, "d0", "t3", "root", 0 : i64>>
  %6 = "sam.repeat"(%5, %output_ref) : (tensor<index, #sam.encoding< Ref, "d0", "t3", "root", 0 : i64>>, tensor<index, #sam.encoding< Ref, "d0", "t0", "dense", 0 : i64>>) -> tensor<index, #sam.encoding< Ref, "d0", "t3", "dense", 0 : i64>>
  %7 = "sam.generator"() : () -> tensor<index, #sam.encoding< Ref, "d0", "t4", "root", 0 : i64>>
  %8 = "sam.repeat"(%7, %output_ref) : (tensor<index, #sam.encoding< Ref, "d0", "t4", "root", 0 : i64>>, tensor<index, #sam.encoding< Ref, "d0", "t0", "dense", 0 : i64>>) -> tensor<index, #sam.encoding< Ref, "d0", "t4", "dense", 0 : i64>>
  %output_ref_0, %output_crd_1 = "sam.fiber_lookup"(%6) <{index_var = 3 : index, tensor_num = 3 : index}> : (tensor<index, #sam.encoding< Ref, "d0", "t3", "dense", 0 : i64>>) -> (tensor<index, #sam.encoding< Ref, "d3", "t3", "dense", 1 : i64>>, tensor<index, #sam.encoding< Crd, "d3", "t3", "dense", 1 : i64>>)
  %output_ref_2, %output_crd_3 = "sam.fiber_lookup"(%8) <{index_var = 3 : index, tensor_num = 4 : index}> : (tensor<index, #sam.encoding< Ref, "d0", "t4", "dense", 0 : i64>>) -> (tensor<index, #sam.encoding< Ref, "d3", "t4", "dense", 0 : i64>>, tensor<index, #sam.encoding< Crd, "d3", "t4", "dense", 0 : i64>>)
  %output_crd_4, %output_refs:2 = "sam.joiner"(%output_crd_1, %output_crd_3, %output_ref_0, %output_ref_2) <{joinerOp = #sam<joiner_type Intersect>, operandSegmentSizes = array<i32: 2, 2>}> : (tensor<index, #sam.encoding< Crd, "d3", "t3", "dense", 1 : i64>>, tensor<index, #sam.encoding< Crd, "d3", "t4", "dense", 0 : i64>>, tensor<index, #sam.encoding< Ref, "d3", "t3", "dense", 1 : i64>>, tensor<index, #sam.encoding< Ref, "d3", "t4", "dense", 0 : i64>>) -> (tensor<index, #sam.encoding< Crd, "d3", "t1", "dense", 0 : i64>>, tensor<index, #sam.encoding< Ref, "d3", "t0", "dense", 0 : i64>>, tensor<index, #sam.encoding< Ref, "d3", "t1", "dense", 0 : i64>>)
  %9 = "sam.repeat"(%output_ref, %output_refs#1) : (tensor<index, #sam.encoding< Ref, "d0", "t0", "dense", 0 : i64>>, tensor<index, #sam.encoding< Ref, "d3", "t1", "dense", 0 : i64>>) -> tensor<index, #sam.encoding< Ref, "d3", "t0", "dense", 0 : i64>>
  %10 = "sam.repeat"(%2, %output_refs#1) : (tensor<index, #sam.encoding< Ref, "d0", "t1", "dense", 0 : i64>>, tensor<index, #sam.encoding< Ref, "d3", "t1", "dense", 0 : i64>>) -> tensor<index, #sam.encoding< Ref, "d3", "t1", "dense", 0 : i64>>
  %11 = "sam.repeat"(%4, %output_refs#1) : (tensor<index, #sam.encoding< Ref, "d0", "t2", "dense", 0 : i64>>, tensor<index, #sam.encoding< Ref, "d3", "t1", "dense", 0 : i64>>) -> tensor<index, #sam.encoding< Ref, "d3", "t2", "dense", 0 : i64>>
  %output_ref_5, %output_crd_6 = "sam.fiber_lookup"(%10) <{index_var = 1 : index, tensor_num = 1 : index}> : (tensor<index, #sam.encoding< Ref, "d3", "t1", "dense", 0 : i64>>) -> (tensor<index, #sam.encoding< Ref, "d1", "t1", "dense", 1 : i64>>, tensor<index, #sam.encoding< Crd, "d1", "t1", "dense", 1 : i64>>)
  %output_ref_7, %output_crd_8 = "sam.fiber_lookup"(%11) <{index_var = 1 : index, tensor_num = 2 : index}> : (tensor<index, #sam.encoding< Ref, "d3", "t2", "dense", 0 : i64>>) -> (tensor<index, #sam.encoding< Ref, "d1", "t2", "dense", 0 : i64>>, tensor<index, #sam.encoding< Crd, "d1", "t2", "dense", 0 : i64>>)
  %output_ref_9, %output_crd_10 = "sam.fiber_lookup"(%output_refs#0) <{index_var = 1 : index, tensor_num = 3 : index}> : (tensor<index, #sam.encoding< Ref, "d3", "t0", "dense", 0 : i64>>) -> (tensor<index, #sam.encoding< Ref, "d1", "t3", "dense", 0 : i64>>, tensor<index, #sam.encoding< Crd, "d1", "t3", "dense", 0 : i64>>)
  %cst = arith.constant dense<0xFF800000> : tensor<1x1xf32>
  %cst_11 = arith.constant dense<0> : tensor<1x1xi64>
  %cst_12 = arith.constant dense<0.000000e+00> : tensor<1x10xf32>
  %cst_13 = arith.constant dense<0.000000e+00> : tensor<128x10xf32>
  %cst_14 = arith.constant dense<[0.00904431194, -0.0562534407, -0.0540498868, -0.0748723894, -0.0307656825, -0.0435144082, -0.0101534948, 0.008740969, -0.0820712074, 0.0403052047]> : tensor<10xf32>
  %cst_15 = arith.constant dense_resource<__elided__> : tensor<128xf32>
  %cst_16 = arith.constant 0.000000e+00 : f32
  %cst_17 = arith.constant dense_resource<__elided__> : tensor<784x128xf32>
  %cst_18 = arith.constant dense<0.000000e+00> : tensor<1x128xf32>
  %12 = linalg.matmul ins(%arg0, %cst_17 : tensor<1x784xf32>, tensor<784x128xf32>) outs(%cst_18 : tensor<1x128xf32>) -> tensor<1x128xf32>
  %13 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%12, %cst_15 : tensor<1x128xf32>, tensor<128xf32>) outs(%cst_18 : tensor<1x128xf32>) {
  ^bb0(%in: f32, %in_19: f32, %out: f32):
    %22 = arith.addf %in, %in_19 : f32
    linalg.yield %22 : f32
  } -> tensor<1x128xf32>
  %14 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%13 : tensor<1x128xf32>) outs(%cst_18 : tensor<1x128xf32>) {
  ^bb0(%in: f32, %out: f32):
    %22 = arith.cmpf ugt, %in, %cst_16 : f32
    %23 = arith.select %22, %in, %cst_16 : f32
    linalg.yield %23 : f32
  } -> tensor<1x128xf32>
  %15 = linalg.matmul ins(%14, %cst_13 : tensor<1x128xf32>, tensor<128x10xf32>) outs(%cst_12 : tensor<1x10xf32>) -> tensor<1x10xf32>
  %16 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%15, %cst_14 : tensor<1x10xf32>, tensor<10xf32>) outs(%cst_12 : tensor<1x10xf32>) {
  ^bb0(%in: f32, %in_19: f32, %out: f32):
    %22 = arith.addf %in, %in_19 : f32
    linalg.yield %22 : f32
  } -> tensor<1x10xf32>
  %17:2 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, 0)>], iterator_types = ["parallel", "reduction"]} ins(%16 : tensor<1x10xf32>) outs(%cst, %cst_11 : tensor<1x1xf32>, tensor<1x1xi64>) {
  ^bb0(%in: f32, %out: f32, %out_19: i64):
    %22 = linalg.index 1 : index
    %23 = arith.index_cast %22 : index to i64
    %24 = arith.maximumf %in, %out : f32
    %25 = arith.cmpf ogt, %in, %out : f32
    %26 = arith.select %25, %23, %out_19 : i64
    linalg.yield %24, %26 : f32, i64
  } -> (tensor<1x1xf32>, tensor<1x1xi64>)
  %18 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%16, %17#0 : tensor<1x10xf32>, tensor<1x1xf32>) outs(%cst_12 : tensor<1x10xf32>) {
  ^bb0(%in: f32, %in_19: f32, %out: f32):
    %22 = arith.subf %in, %in_19 : f32
    linalg.yield %22 : f32
  } -> tensor<1x10xf32>
  %19 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18 : tensor<1x10xf32>) outs(%cst_12 : tensor<1x10xf32>) {
  ^bb0(%in: f32, %out: f32):
    %22 = math.exp %in : f32
    linalg.yield %22 : f32
  } -> tensor<1x10xf32>
  %20 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, 0)>], iterator_types = ["parallel", "reduction"]} ins(%19 : tensor<1x10xf32>) outs(%cst : tensor<1x1xf32>) {
  ^bb0(%in: f32, %out: f32):
    %22 = arith.addf %in, %out : f32
    linalg.yield %22 : f32
  } -> tensor<1x1xf32>
  %21 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%19, %20 : tensor<1x10xf32>, tensor<1x1xf32>) outs(%cst_12 : tensor<1x10xf32>) {
  ^bb0(%in: f32, %in_19: f32, %out: f32):
    %22 = arith.divf %in, %in_19 : f32
    linalg.yield %22 : f32
  } -> tensor<1x10xf32>
  return %21 : tensor<1x10xf32>
}