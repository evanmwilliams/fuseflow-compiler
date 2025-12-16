#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map3 = affine_map<(d0, d1, d2) -> (0, d1, d2)>
#map4 = affine_map<(d0, d1, d2) -> ()>
#map5 = affine_map<(d0, d1, d2) -> (d0, d1, 0)>
#map6 = affine_map<(d0, d1, d2) -> (0, d1, 0)>
#map7 = affine_map<(d0, d1, d2) -> (d2)>
#map8 = affine_map<(d0, d1) -> ()>
#map9 = affine_map<(d0, d1) -> (d0, d1)>
#map10 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map11 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map12 = affine_map<(d0, d1) -> (d1)>
#map13 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map14 = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>
#map15 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#map16 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map17 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map18 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#map19 = affine_map<() -> ()>
#map20 = affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>
#map21 = affine_map<(d0, d1, d2, d3) -> ()>
#map22 = affine_map<(d0, d1, d2, d3) -> (0, 0, d2, d3)>
#map23 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>
#map24 = affine_map<(d0, d1, d2, d3) -> (0, d1, d2, 0)>
#map25 = affine_map<(d0, d1) -> (d1, d0)>
module attributes {torch.debug_module_name = "_lambda"} {
  ml_program.global private mutable @global_seed(dense<0> : tensor<i64>) : tensor<i64>
  func.func @forward(%arg0: tensor<1x5xi64>) -> tensor<1x5x50257xf32> {
    %cst = arith.constant dense_resource<__elided__> : tensor<768xf32>
    %cst_0 = arith.constant dense_resource<__elided__> : tensor<3072x768xf32>
    %cst_1 = arith.constant dense_resource<__elided__> : tensor<768x3072xf32>
    %cst_2 = arith.constant dense_resource<__elided__> : tensor<3072xf32>
    %cst_3 = arith.constant dense_resource<__elided__> : tensor<768x768xf32>
    %cst_4 = arith.constant dense_resource<__elided__> : tensor<768x2304xf32>
    %cst_5 = arith.constant dense_resource<__elided__> : tensor<2304xf32>
    %cst_6 = arith.constant dense_resource<__elided__> : tensor<1x1x1024x1024xi1>
    %cst_7 = arith.constant dense_resource<__elided__> : tensor<1024x768xf32>
    %cst_8 = arith.constant dense_resource<__elided__> : tensor<50257x768xf32>
    %c50257 = arith.constant 50257 : index
    %c1024 = arith.constant 1024 : index
    %cst_9 = arith.constant 0.000000e+00 : f32
    %cst_10 = arith.constant 0xFF800000 : f32
    %c0_i64 = arith.constant 0 : i64
    %cst_11 = arith.constant -3.4028234663852886E+38 : f64
    %cst_12 = arith.constant 1.000000e-05 : f64
    %cst_13 = arith.constant 8.000000e+00 : f64
    %cst_14 = arith.constant 4.471500e-02 : f64
    %cst_15 = arith.constant 0.79788456080286541 : f64
    %cst_16 = arith.constant 7.680000e+02 : f32
    %cst_17 = arith.constant 5.000000e-01 : f32
    %cst_18 = arith.constant 3.000000e+00 : f32
    %cst_19 = arith.constant 1.000000e+00 : f32
    %0 = tensor.empty() : tensor<5xi64>
    // %1 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%0 : tensor<5xi64>) {
    // ^bb0(%out: i64):
    //   %760 = linalg.index 0 : index
    //   %761 = arith.index_cast %760 : index to i64
    //   linalg.yield %761 : i64
    // } -> tensor<5xi64>
    // %expanded = tensor.expand_shape %1 [[0, 1]] : tensor<5xi64> into tensor<1x5xi64>
    %expanded = tensor.empty() : tensor<1x5xi64>
    %2 = tensor.empty() : tensor<1x5x768xf32>
    %3 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0 : tensor<1x5xi64>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: i64, %out: f32):
      %760 = arith.index_cast %in : i64 to index
      %761 = linalg.index 2 : index
      %762 = arith.cmpi slt, %760, %c50257 : index
      cf.assert %762, "index must be smaller than dim size"
      %763 = arith.cmpi sge, %in, %c0_i64 : i64
      cf.assert %763, "index must be larger or equal to 0"
      %extracted = tensor.extract %cst_8[%760, %761] : tensor<50257x768xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x5x768xf32>
    %4 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded : tensor<1x5xi64>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: i64, %out: f32):
      %760 = arith.index_cast %in : i64 to index
      %761 = linalg.index 2 : index
      %762 = arith.cmpi slt, %760, %c1024 : index
      cf.assert %762, "index must be smaller than dim size"
      %763 = arith.cmpi sge, %in, %c0_i64 : i64
      cf.assert %763, "index must be larger or equal to 0"
      %extracted = tensor.extract %cst_7[%760, %761] : tensor<1024x768xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x5x768xf32>
    %5 = linalg.generic {indexing_maps = [#map3, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3, %4 : tensor<1x5x768xf32>, tensor<1x5x768xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.addf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x768xf32>
    %6 = tensor.empty() : tensor<1x5x1xf32>
    %7 = linalg.generic {indexing_maps = [#map4, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%cst_9 : f32) outs(%6 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x5x1xf32>
    %8 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "reduction"]} ins(%5 : tensor<1x5x768xf32>) outs(%7 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.addf %in, %out : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x1xf32>
    %9 = linalg.generic {indexing_maps = [#map6, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%8 : tensor<1x5x1xf32>) outs(%6 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.divf %in, %cst_16 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x1xf32>
    %10 = linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%9 : tensor<1x5x1xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x5x768xf32>
    %11 = linalg.generic {indexing_maps = [#map3, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%5, %10 : tensor<1x5x768xf32>, tensor<1x5x768xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.subf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x768xf32>
    %12 = linalg.generic {indexing_maps = [#map3, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%11, %11 : tensor<1x5x768xf32>, tensor<1x5x768xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x768xf32>
    %13 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "reduction"]} ins(%12 : tensor<1x5x768xf32>) outs(%7 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.addf %in, %out : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x1xf32>
    %14 = linalg.generic {indexing_maps = [#map6, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%13 : tensor<1x5x1xf32>) outs(%6 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.divf %in, %cst_16 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x1xf32>
    %15 = linalg.generic {indexing_maps = [#map6, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%14 : tensor<1x5x1xf32>) outs(%6 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.truncf %cst_12 : f64 to f32
      %761 = arith.addf %in, %760 : f32
      linalg.yield %761 : f32
    } -> tensor<1x5x1xf32>
    %16 = linalg.generic {indexing_maps = [#map6, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%15 : tensor<1x5x1xf32>) outs(%6 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = math.rsqrt %in : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x1xf32>
    %17 = linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%16 : tensor<1x5x1xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x5x768xf32>
    %18 = linalg.generic {indexing_maps = [#map3, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%11, %17 : tensor<1x5x768xf32>, tensor<1x5x768xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x768xf32>
    %19 = linalg.generic {indexing_maps = [#map3, #map7, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%18, %cst : tensor<1x5x768xf32>, tensor<768xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x768xf32>
    %20 = linalg.generic {indexing_maps = [#map3, #map7, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%19, %cst : tensor<1x5x768xf32>, tensor<768xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.addf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x768xf32>
    %collapsed = tensor.collapse_shape %20 [[0, 1], [2]] : tensor<1x5x768xf32> into tensor<5x768xf32>
    %21 = tensor.empty() : tensor<5x2304xf32>
    %22 = linalg.generic {indexing_maps = [#map8, #map9], iterator_types = ["parallel", "parallel"]} ins(%cst_9 : f32) outs(%21 : tensor<5x2304xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<5x2304xf32>
    %23 = linalg.generic {indexing_maps = [#map10, #map11, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%collapsed, %cst_4 : tensor<5x768xf32>, tensor<768x2304xf32>) outs(%22 : tensor<5x2304xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      %761 = arith.addf %out, %760 : f32
      linalg.yield %761 : f32
    } -> tensor<5x2304xf32>
    %24 = linalg.generic {indexing_maps = [#map12, #map9, #map9], iterator_types = ["parallel", "parallel"]} ins(%cst_5, %23 : tensor<2304xf32>, tensor<5x2304xf32>) outs(%21 : tensor<5x2304xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.addf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<5x2304xf32>
    %expanded_20 = tensor.expand_shape %24 [[0, 1], [2]] : tensor<5x2304xf32> into tensor<1x5x2304xf32>
    %extracted_slice = tensor.extract_slice %expanded_20[0, 0, 0] [1, 5, 768] [1, 1, 1] : tensor<1x5x2304xf32> to tensor<1x5x768xf32>
    %extracted_slice_21 = tensor.extract_slice %expanded_20[0, 0, 768] [1, 5, 768] [1, 1, 1] : tensor<1x5x2304xf32> to tensor<1x5x768xf32>
    %extracted_slice_22 = tensor.extract_slice %expanded_20[0, 0, 1536] [1, 5, 768] [1, 1, 1] : tensor<1x5x2304xf32> to tensor<1x5x768xf32>
    %expanded_23 = tensor.expand_shape %extracted_slice [[0], [1], [2, 3]] : tensor<1x5x768xf32> into tensor<1x5x12x64xf32>
    %25 = tensor.empty() : tensor<1x12x5x64xf32>
    %26 = linalg.generic {indexing_maps = [#map13, #map14], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_23 : tensor<1x5x12x64xf32>) outs(%25 : tensor<1x12x5x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x12x5x64xf32>
    %expanded_24 = tensor.expand_shape %extracted_slice_21 [[0], [1], [2, 3]] : tensor<1x5x768xf32> into tensor<1x5x12x64xf32>
    %27 = linalg.generic {indexing_maps = [#map13, #map14], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_24 : tensor<1x5x12x64xf32>) outs(%25 : tensor<1x12x5x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x12x5x64xf32>
    %expanded_25 = tensor.expand_shape %extracted_slice_22 [[0], [1], [2, 3]] : tensor<1x5x768xf32> into tensor<1x5x12x64xf32>
    %28 = linalg.generic {indexing_maps = [#map13, #map14], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_25 : tensor<1x5x12x64xf32>) outs(%25 : tensor<1x12x5x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x12x5x64xf32>
    %29 = tensor.empty() : tensor<1x12x64x5xf32>
    %30 = linalg.generic {indexing_maps = [#map13, #map15], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%27 : tensor<1x12x5x64xf32>) outs(%29 : tensor<1x12x64x5xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x12x64x5xf32>
    %collapsed_26 = tensor.collapse_shape %26 [[0, 1], [2], [3]] : tensor<1x12x5x64xf32> into tensor<12x5x64xf32>
    %collapsed_27 = tensor.collapse_shape %30 [[0, 1], [2], [3]] : tensor<1x12x64x5xf32> into tensor<12x64x5xf32>
    %31 = tensor.empty() : tensor<12x5x5xf32>
    %32 = linalg.generic {indexing_maps = [#map4, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%cst_9 : f32) outs(%31 : tensor<12x5x5xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<12x5x5xf32>
    %33 = linalg.generic {indexing_maps = [#map16, #map17, #map18], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%collapsed_26, %collapsed_27 : tensor<12x5x64xf32>, tensor<12x64x5xf32>) outs(%32 : tensor<12x5x5xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      %761 = arith.addf %out, %760 : f32
      linalg.yield %761 : f32
    } -> tensor<12x5x5xf32>
    %expanded_28 = tensor.expand_shape %33 [[0, 1], [2], [3]] : tensor<12x5x5xf32> into tensor<1x12x5x5xf32>
    %34 = tensor.empty() : tensor<f64>
    %35 = linalg.generic {indexing_maps = [#map19, #map19], iterator_types = []} ins(%cst_13 : f64) outs(%34 : tensor<f64>) {
    ^bb0(%in: f64, %out: f64):
      linalg.yield %in : f64
    } -> tensor<f64>
    %36 = tensor.empty() : tensor<f32>
    %37 = linalg.generic {indexing_maps = [#map19, #map19], iterator_types = []} ins(%35 : tensor<f64>) outs(%36 : tensor<f32>) {
    ^bb0(%in: f64, %out: f32):
      %760 = arith.truncf %in : f64 to f32
      linalg.yield %760 : f32
    } -> tensor<f32>
    %38 = tensor.empty() : tensor<1x12x5x5xf32>
    %39 = linalg.generic {indexing_maps = [#map20, #map21, #map13], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_28, %37 : tensor<1x12x5x5xf32>, tensor<f32>) outs(%38 : tensor<1x12x5x5xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.divf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x12x5x5xf32>
    %extracted_slice_29 = tensor.extract_slice %cst_6[0, 0, 0, 0] [1, 1, 5, 1024] [1, 1, 1, 1] : tensor<1x1x1024x1024xi1> to tensor<1x1x5x1024xi1>
    %extracted_slice_30 = tensor.extract_slice %extracted_slice_29[0, 0, 0, 0] [1, 1, 5, 5] [1, 1, 1, 1] : tensor<1x1x5x1024xi1> to tensor<1x1x5x5xi1>
    %40 = linalg.generic {indexing_maps = [#map19, #map19], iterator_types = []} ins(%cst_11 : f64) outs(%34 : tensor<f64>) {
    ^bb0(%in: f64, %out: f64):
      linalg.yield %in : f64
    } -> tensor<f64>
    %41 = linalg.generic {indexing_maps = [#map19, #map19], iterator_types = []} ins(%40 : tensor<f64>) outs(%36 : tensor<f32>) {
    ^bb0(%in: f64, %out: f32):
      %760 = arith.truncf %in : f64 to f32
      linalg.yield %760 : f32
    } -> tensor<f32>
    %42 = linalg.generic {indexing_maps = [#map22, #map20, #map21, #map13], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_30, %39, %41 : tensor<1x1x5x5xi1>, tensor<1x12x5x5xf32>, tensor<f32>) outs(%38 : tensor<1x12x5x5xf32>) {
    ^bb0(%in: i1, %in_262: f32, %in_263: f32, %out: f32):
      %760 = arith.select %in, %in_262, %in_263 : f32
      linalg.yield %760 : f32
    } -> tensor<1x12x5x5xf32>
    %43 = tensor.empty() : tensor<1x12x5x1xi64>
    %44 = linalg.generic {indexing_maps = [#map21, #map13], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%c0_i64 : i64) outs(%43 : tensor<1x12x5x1xi64>) {
    ^bb0(%in: i64, %out: i64):
      linalg.yield %in : i64
    } -> tensor<1x12x5x1xi64>
    %45 = tensor.empty() : tensor<1x12x5x1xf32>
    %46 = linalg.generic {indexing_maps = [#map21, #map13], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_10 : f32) outs(%45 : tensor<1x12x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x12x5x1xf32>
    %47:2 = linalg.generic {indexing_maps = [#map13, #map23, #map23], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%42 : tensor<1x12x5x5xf32>) outs(%46, %44 : tensor<1x12x5x1xf32>, tensor<1x12x5x1xi64>) {
    ^bb0(%in: f32, %out: f32, %out_262: i64):
      %760 = linalg.index 3 : index
      %761 = arith.index_cast %760 : index to i64
      %762 = arith.maximumf %in, %out : f32
      %763 = arith.cmpf ogt, %in, %out : f32
      %764 = arith.select %763, %761, %out_262 : i64
      linalg.yield %762, %764 : f32, i64
    } -> (tensor<1x12x5x1xf32>, tensor<1x12x5x1xi64>)
    %48 = linalg.generic {indexing_maps = [#map20, #map24, #map13], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%42, %47#0 : tensor<1x12x5x5xf32>, tensor<1x12x5x1xf32>) outs(%38 : tensor<1x12x5x5xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.subf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x12x5x5xf32>
    %49 = linalg.generic {indexing_maps = [#map20, #map13], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%48 : tensor<1x12x5x5xf32>) outs(%38 : tensor<1x12x5x5xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = math.exp %in : f32
      linalg.yield %760 : f32
    } -> tensor<1x12x5x5xf32>
    %50 = linalg.generic {indexing_maps = [#map21, #map13], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_9 : f32) outs(%45 : tensor<1x12x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x12x5x1xf32>
    %51 = linalg.generic {indexing_maps = [#map13, #map23], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%49 : tensor<1x12x5x5xf32>) outs(%50 : tensor<1x12x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.addf %in, %out : f32
      linalg.yield %760 : f32
    } -> tensor<1x12x5x1xf32>
    %52 = linalg.generic {indexing_maps = [#map20, #map24, #map13], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%49, %51 : tensor<1x12x5x5xf32>, tensor<1x12x5x1xf32>) outs(%38 : tensor<1x12x5x5xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.divf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x12x5x5xf32>
    %collapsed_31 = tensor.collapse_shape %52 [[0, 1], [2], [3]] : tensor<1x12x5x5xf32> into tensor<12x5x5xf32>
    %collapsed_32 = tensor.collapse_shape %28 [[0, 1], [2], [3]] : tensor<1x12x5x64xf32> into tensor<12x5x64xf32>
    %53 = tensor.empty() : tensor<12x5x64xf32>
    %54 = linalg.generic {indexing_maps = [#map4, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%cst_9 : f32) outs(%53 : tensor<12x5x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<12x5x64xf32>
    %55 = linalg.generic {indexing_maps = [#map16, #map17, #map18], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%collapsed_31, %collapsed_32 : tensor<12x5x5xf32>, tensor<12x5x64xf32>) outs(%54 : tensor<12x5x64xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      %761 = arith.addf %out, %760 : f32
      linalg.yield %761 : f32
    } -> tensor<12x5x64xf32>
    %expanded_33 = tensor.expand_shape %55 [[0, 1], [2], [3]] : tensor<12x5x64xf32> into tensor<1x12x5x64xf32>
    %56 = tensor.empty() : tensor<1x5x12x64xf32>
    %57 = linalg.generic {indexing_maps = [#map13, #map14], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_33 : tensor<1x12x5x64xf32>) outs(%56 : tensor<1x5x12x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x5x12x64xf32>
    %58 = linalg.generic {indexing_maps = [#map20, #map13], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%57 : tensor<1x5x12x64xf32>) outs(%56 : tensor<1x5x12x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x5x12x64xf32>
    %collapsed_34 = tensor.collapse_shape %58 [[0, 1], [2, 3]] : tensor<1x5x12x64xf32> into tensor<5x768xf32>
    %59 = tensor.empty() : tensor<5x768xf32>
    %60 = linalg.generic {indexing_maps = [#map8, #map9], iterator_types = ["parallel", "parallel"]} ins(%cst_9 : f32) outs(%59 : tensor<5x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<5x768xf32>
    %61 = linalg.generic {indexing_maps = [#map10, #map11, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%collapsed_34, %cst_3 : tensor<5x768xf32>, tensor<768x768xf32>) outs(%60 : tensor<5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      %761 = arith.addf %out, %760 : f32
      linalg.yield %761 : f32
    } -> tensor<5x768xf32>
    %62 = linalg.generic {indexing_maps = [#map12, #map9, #map9], iterator_types = ["parallel", "parallel"]} ins(%cst, %61 : tensor<768xf32>, tensor<5x768xf32>) outs(%59 : tensor<5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.addf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<5x768xf32>
    %expanded_35 = tensor.expand_shape %62 [[0, 1], [2]] : tensor<5x768xf32> into tensor<1x5x768xf32>
    %63 = linalg.generic {indexing_maps = [#map3, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_35, %5 : tensor<1x5x768xf32>, tensor<1x5x768xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.addf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x768xf32>
    %64 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "reduction"]} ins(%63 : tensor<1x5x768xf32>) outs(%7 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.addf %in, %out : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x1xf32>
    %65 = linalg.generic {indexing_maps = [#map6, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%64 : tensor<1x5x1xf32>) outs(%6 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.divf %in, %cst_16 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x1xf32>
    %66 = linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%65 : tensor<1x5x1xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x5x768xf32>
    %67 = linalg.generic {indexing_maps = [#map3, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%63, %66 : tensor<1x5x768xf32>, tensor<1x5x768xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.subf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x768xf32>
    %68 = linalg.generic {indexing_maps = [#map3, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%67, %67 : tensor<1x5x768xf32>, tensor<1x5x768xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x768xf32>
    %69 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "reduction"]} ins(%68 : tensor<1x5x768xf32>) outs(%7 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.addf %in, %out : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x1xf32>
    %70 = linalg.generic {indexing_maps = [#map6, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%69 : tensor<1x5x1xf32>) outs(%6 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.divf %in, %cst_16 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x1xf32>
    %71 = linalg.generic {indexing_maps = [#map6, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%70 : tensor<1x5x1xf32>) outs(%6 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.truncf %cst_12 : f64 to f32
      %761 = arith.addf %in, %760 : f32
      linalg.yield %761 : f32
    } -> tensor<1x5x1xf32>
    %72 = linalg.generic {indexing_maps = [#map6, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%71 : tensor<1x5x1xf32>) outs(%6 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = math.rsqrt %in : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x1xf32>
    %73 = linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%72 : tensor<1x5x1xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x5x768xf32>
    %74 = linalg.generic {indexing_maps = [#map3, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%67, %73 : tensor<1x5x768xf32>, tensor<1x5x768xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x768xf32>
    %75 = linalg.generic {indexing_maps = [#map3, #map7, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%74, %cst : tensor<1x5x768xf32>, tensor<768xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x768xf32>
    %76 = linalg.generic {indexing_maps = [#map3, #map7, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%75, %cst : tensor<1x5x768xf32>, tensor<768xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.addf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x768xf32>
    %collapsed_36 = tensor.collapse_shape %76 [[0, 1], [2]] : tensor<1x5x768xf32> into tensor<5x768xf32>
    %77 = tensor.empty() : tensor<5x3072xf32>
    %78 = linalg.generic {indexing_maps = [#map8, #map9], iterator_types = ["parallel", "parallel"]} ins(%cst_9 : f32) outs(%77 : tensor<5x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<5x3072xf32>
    %79 = linalg.generic {indexing_maps = [#map10, #map11, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%collapsed_36, %cst_1 : tensor<5x768xf32>, tensor<768x3072xf32>) outs(%78 : tensor<5x3072xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      %761 = arith.addf %out, %760 : f32
      linalg.yield %761 : f32
    } -> tensor<5x3072xf32>
    %80 = linalg.generic {indexing_maps = [#map12, #map9, #map9], iterator_types = ["parallel", "parallel"]} ins(%cst_2, %79 : tensor<3072xf32>, tensor<5x3072xf32>) outs(%77 : tensor<5x3072xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.addf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<5x3072xf32>
    %expanded_37 = tensor.expand_shape %80 [[0, 1], [2]] : tensor<5x3072xf32> into tensor<1x5x3072xf32>
    %81 = tensor.empty() : tensor<1x5x3072xf32>
    %82 = linalg.generic {indexing_maps = [#map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_37 : tensor<1x5x3072xf32>) outs(%81 : tensor<1x5x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.mulf %in, %cst_17 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x3072xf32>
    %83 = linalg.generic {indexing_maps = [#map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_37 : tensor<1x5x3072xf32>) outs(%81 : tensor<1x5x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = math.powf %in, %cst_18 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x3072xf32>
    %84 = linalg.generic {indexing_maps = [#map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%83 : tensor<1x5x3072xf32>) outs(%81 : tensor<1x5x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.truncf %cst_14 : f64 to f32
      %761 = arith.mulf %in, %760 : f32
      linalg.yield %761 : f32
    } -> tensor<1x5x3072xf32>
    %85 = linalg.generic {indexing_maps = [#map3, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_37, %84 : tensor<1x5x3072xf32>, tensor<1x5x3072xf32>) outs(%81 : tensor<1x5x3072xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.addf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x3072xf32>
    %86 = linalg.generic {indexing_maps = [#map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%85 : tensor<1x5x3072xf32>) outs(%81 : tensor<1x5x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.truncf %cst_15 : f64 to f32
      %761 = arith.mulf %in, %760 : f32
      linalg.yield %761 : f32
    } -> tensor<1x5x3072xf32>
    %87 = linalg.generic {indexing_maps = [#map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%86 : tensor<1x5x3072xf32>) outs(%81 : tensor<1x5x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = math.tanh %in : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x3072xf32>
    %88 = linalg.generic {indexing_maps = [#map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%87 : tensor<1x5x3072xf32>) outs(%81 : tensor<1x5x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.addf %in, %cst_19 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x3072xf32>
    %89 = linalg.generic {indexing_maps = [#map3, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%82, %88 : tensor<1x5x3072xf32>, tensor<1x5x3072xf32>) outs(%81 : tensor<1x5x3072xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x3072xf32>
    %collapsed_38 = tensor.collapse_shape %89 [[0, 1], [2]] : tensor<1x5x3072xf32> into tensor<5x3072xf32>
    %90 = linalg.generic {indexing_maps = [#map10, #map11, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%collapsed_38, %cst_0 : tensor<5x3072xf32>, tensor<3072x768xf32>) outs(%60 : tensor<5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      %761 = arith.addf %out, %760 : f32
      linalg.yield %761 : f32
    } -> tensor<5x768xf32>
    %91 = linalg.generic {indexing_maps = [#map12, #map9, #map9], iterator_types = ["parallel", "parallel"]} ins(%cst, %90 : tensor<768xf32>, tensor<5x768xf32>) outs(%59 : tensor<5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.addf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<5x768xf32>
    %expanded_39 = tensor.expand_shape %91 [[0, 1], [2]] : tensor<5x768xf32> into tensor<1x5x768xf32>
    %92 = linalg.generic {indexing_maps = [#map3, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%63, %expanded_39 : tensor<1x5x768xf32>, tensor<1x5x768xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.addf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x768xf32>
    %93 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "reduction"]} ins(%92 : tensor<1x5x768xf32>) outs(%7 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.addf %in, %out : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x1xf32>
    %94 = linalg.generic {indexing_maps = [#map6, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%93 : tensor<1x5x1xf32>) outs(%6 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.divf %in, %cst_16 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x1xf32>
    %95 = linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%94 : tensor<1x5x1xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x5x768xf32>
    %96 = linalg.generic {indexing_maps = [#map3, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%92, %95 : tensor<1x5x768xf32>, tensor<1x5x768xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.subf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x768xf32>
    %97 = linalg.generic {indexing_maps = [#map3, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%96, %96 : tensor<1x5x768xf32>, tensor<1x5x768xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x768xf32>
    %98 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "reduction"]} ins(%97 : tensor<1x5x768xf32>) outs(%7 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.addf %in, %out : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x1xf32>
    %99 = linalg.generic {indexing_maps = [#map6, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%98 : tensor<1x5x1xf32>) outs(%6 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.divf %in, %cst_16 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x1xf32>
    %100 = linalg.generic {indexing_maps = [#map6, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%99 : tensor<1x5x1xf32>) outs(%6 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.truncf %cst_12 : f64 to f32
      %761 = arith.addf %in, %760 : f32
      linalg.yield %761 : f32
    } -> tensor<1x5x1xf32>
    %101 = linalg.generic {indexing_maps = [#map6, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%100 : tensor<1x5x1xf32>) outs(%6 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = math.rsqrt %in : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x1xf32>
    %102 = linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%101 : tensor<1x5x1xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x5x768xf32>
    %103 = linalg.generic {indexing_maps = [#map3, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%96, %102 : tensor<1x5x768xf32>, tensor<1x5x768xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x768xf32>
    %104 = linalg.generic {indexing_maps = [#map3, #map7, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%103, %cst : tensor<1x5x768xf32>, tensor<768xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x768xf32>
    %105 = linalg.generic {indexing_maps = [#map3, #map7, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%104, %cst : tensor<1x5x768xf32>, tensor<768xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.addf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x768xf32>
    %collapsed_40 = tensor.collapse_shape %105 [[0, 1], [2]] : tensor<1x5x768xf32> into tensor<5x768xf32>
    %106 = linalg.generic {indexing_maps = [#map10, #map11, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%collapsed_40, %cst_4 : tensor<5x768xf32>, tensor<768x2304xf32>) outs(%22 : tensor<5x2304xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      %761 = arith.addf %out, %760 : f32
      linalg.yield %761 : f32
    } -> tensor<5x2304xf32>
    %107 = linalg.generic {indexing_maps = [#map12, #map9, #map9], iterator_types = ["parallel", "parallel"]} ins(%cst_5, %106 : tensor<2304xf32>, tensor<5x2304xf32>) outs(%21 : tensor<5x2304xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.addf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<5x2304xf32>
    %expanded_41 = tensor.expand_shape %107 [[0, 1], [2]] : tensor<5x2304xf32> into tensor<1x5x2304xf32>
    %extracted_slice_42 = tensor.extract_slice %expanded_41[0, 0, 0] [1, 5, 768] [1, 1, 1] : tensor<1x5x2304xf32> to tensor<1x5x768xf32>
    %extracted_slice_43 = tensor.extract_slice %expanded_41[0, 0, 768] [1, 5, 768] [1, 1, 1] : tensor<1x5x2304xf32> to tensor<1x5x768xf32>
    %extracted_slice_44 = tensor.extract_slice %expanded_41[0, 0, 1536] [1, 5, 768] [1, 1, 1] : tensor<1x5x2304xf32> to tensor<1x5x768xf32>
    %expanded_45 = tensor.expand_shape %extracted_slice_42 [[0], [1], [2, 3]] : tensor<1x5x768xf32> into tensor<1x5x12x64xf32>
    %108 = linalg.generic {indexing_maps = [#map13, #map14], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_45 : tensor<1x5x12x64xf32>) outs(%25 : tensor<1x12x5x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x12x5x64xf32>
    %expanded_46 = tensor.expand_shape %extracted_slice_43 [[0], [1], [2, 3]] : tensor<1x5x768xf32> into tensor<1x5x12x64xf32>
    %109 = linalg.generic {indexing_maps = [#map13, #map14], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_46 : tensor<1x5x12x64xf32>) outs(%25 : tensor<1x12x5x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x12x5x64xf32>
    %expanded_47 = tensor.expand_shape %extracted_slice_44 [[0], [1], [2, 3]] : tensor<1x5x768xf32> into tensor<1x5x12x64xf32>
    %110 = linalg.generic {indexing_maps = [#map13, #map14], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_47 : tensor<1x5x12x64xf32>) outs(%25 : tensor<1x12x5x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x12x5x64xf32>
    %111 = linalg.generic {indexing_maps = [#map13, #map15], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%109 : tensor<1x12x5x64xf32>) outs(%29 : tensor<1x12x64x5xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x12x64x5xf32>
    %collapsed_48 = tensor.collapse_shape %108 [[0, 1], [2], [3]] : tensor<1x12x5x64xf32> into tensor<12x5x64xf32>
    %collapsed_49 = tensor.collapse_shape %111 [[0, 1], [2], [3]] : tensor<1x12x64x5xf32> into tensor<12x64x5xf32>
    %112 = linalg.generic {indexing_maps = [#map16, #map17, #map18], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%collapsed_48, %collapsed_49 : tensor<12x5x64xf32>, tensor<12x64x5xf32>) outs(%32 : tensor<12x5x5xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      %761 = arith.addf %out, %760 : f32
      linalg.yield %761 : f32
    } -> tensor<12x5x5xf32>
    %expanded_50 = tensor.expand_shape %112 [[0, 1], [2], [3]] : tensor<12x5x5xf32> into tensor<1x12x5x5xf32>
    %113 = linalg.generic {indexing_maps = [#map20, #map21, #map13], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_50, %37 : tensor<1x12x5x5xf32>, tensor<f32>) outs(%38 : tensor<1x12x5x5xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.divf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x12x5x5xf32>
    %114 = linalg.generic {indexing_maps = [#map22, #map20, #map21, #map13], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_30, %113, %41 : tensor<1x1x5x5xi1>, tensor<1x12x5x5xf32>, tensor<f32>) outs(%38 : tensor<1x12x5x5xf32>) {
    ^bb0(%in: i1, %in_262: f32, %in_263: f32, %out: f32):
      %760 = arith.select %in, %in_262, %in_263 : f32
      linalg.yield %760 : f32
    } -> tensor<1x12x5x5xf32>
    %115:2 = linalg.generic {indexing_maps = [#map13, #map23, #map23], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%114 : tensor<1x12x5x5xf32>) outs(%46, %44 : tensor<1x12x5x1xf32>, tensor<1x12x5x1xi64>) {
    ^bb0(%in: f32, %out: f32, %out_262: i64):
      %760 = linalg.index 3 : index
      %761 = arith.index_cast %760 : index to i64
      %762 = arith.maximumf %in, %out : f32
      %763 = arith.cmpf ogt, %in, %out : f32
      %764 = arith.select %763, %761, %out_262 : i64
      linalg.yield %762, %764 : f32, i64
    } -> (tensor<1x12x5x1xf32>, tensor<1x12x5x1xi64>)
    %116 = linalg.generic {indexing_maps = [#map20, #map24, #map13], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%114, %115#0 : tensor<1x12x5x5xf32>, tensor<1x12x5x1xf32>) outs(%38 : tensor<1x12x5x5xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.subf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x12x5x5xf32>
    %117 = linalg.generic {indexing_maps = [#map20, #map13], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%116 : tensor<1x12x5x5xf32>) outs(%38 : tensor<1x12x5x5xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = math.exp %in : f32
      linalg.yield %760 : f32
    } -> tensor<1x12x5x5xf32>
    %118 = linalg.generic {indexing_maps = [#map13, #map23], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%117 : tensor<1x12x5x5xf32>) outs(%50 : tensor<1x12x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.addf %in, %out : f32
      linalg.yield %760 : f32
    } -> tensor<1x12x5x1xf32>
    %119 = linalg.generic {indexing_maps = [#map20, #map24, #map13], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%117, %118 : tensor<1x12x5x5xf32>, tensor<1x12x5x1xf32>) outs(%38 : tensor<1x12x5x5xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.divf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x12x5x5xf32>
    %collapsed_51 = tensor.collapse_shape %119 [[0, 1], [2], [3]] : tensor<1x12x5x5xf32> into tensor<12x5x5xf32>
    %collapsed_52 = tensor.collapse_shape %110 [[0, 1], [2], [3]] : tensor<1x12x5x64xf32> into tensor<12x5x64xf32>
    %120 = linalg.generic {indexing_maps = [#map16, #map17, #map18], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%collapsed_51, %collapsed_52 : tensor<12x5x5xf32>, tensor<12x5x64xf32>) outs(%54 : tensor<12x5x64xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      %761 = arith.addf %out, %760 : f32
      linalg.yield %761 : f32
    } -> tensor<12x5x64xf32>
    %expanded_53 = tensor.expand_shape %120 [[0, 1], [2], [3]] : tensor<12x5x64xf32> into tensor<1x12x5x64xf32>
    %121 = linalg.generic {indexing_maps = [#map13, #map14], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_53 : tensor<1x12x5x64xf32>) outs(%56 : tensor<1x5x12x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x5x12x64xf32>
    %122 = linalg.generic {indexing_maps = [#map20, #map13], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%121 : tensor<1x5x12x64xf32>) outs(%56 : tensor<1x5x12x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x5x12x64xf32>
    %collapsed_54 = tensor.collapse_shape %122 [[0, 1], [2, 3]] : tensor<1x5x12x64xf32> into tensor<5x768xf32>
    %123 = linalg.generic {indexing_maps = [#map10, #map11, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%collapsed_54, %cst_3 : tensor<5x768xf32>, tensor<768x768xf32>) outs(%60 : tensor<5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      %761 = arith.addf %out, %760 : f32
      linalg.yield %761 : f32
    } -> tensor<5x768xf32>
    %124 = linalg.generic {indexing_maps = [#map12, #map9, #map9], iterator_types = ["parallel", "parallel"]} ins(%cst, %123 : tensor<768xf32>, tensor<5x768xf32>) outs(%59 : tensor<5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.addf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<5x768xf32>
    %expanded_55 = tensor.expand_shape %124 [[0, 1], [2]] : tensor<5x768xf32> into tensor<1x5x768xf32>
    %125 = linalg.generic {indexing_maps = [#map3, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_55, %92 : tensor<1x5x768xf32>, tensor<1x5x768xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.addf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x768xf32>
    %126 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "reduction"]} ins(%125 : tensor<1x5x768xf32>) outs(%7 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.addf %in, %out : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x1xf32>
    %127 = linalg.generic {indexing_maps = [#map6, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%126 : tensor<1x5x1xf32>) outs(%6 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.divf %in, %cst_16 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x1xf32>
    %128 = linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%127 : tensor<1x5x1xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x5x768xf32>
    %129 = linalg.generic {indexing_maps = [#map3, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%125, %128 : tensor<1x5x768xf32>, tensor<1x5x768xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.subf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x768xf32>
    %130 = linalg.generic {indexing_maps = [#map3, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%129, %129 : tensor<1x5x768xf32>, tensor<1x5x768xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x768xf32>
    %131 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "reduction"]} ins(%130 : tensor<1x5x768xf32>) outs(%7 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.addf %in, %out : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x1xf32>
    %132 = linalg.generic {indexing_maps = [#map6, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%131 : tensor<1x5x1xf32>) outs(%6 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.divf %in, %cst_16 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x1xf32>
    %133 = linalg.generic {indexing_maps = [#map6, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%132 : tensor<1x5x1xf32>) outs(%6 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.truncf %cst_12 : f64 to f32
      %761 = arith.addf %in, %760 : f32
      linalg.yield %761 : f32
    } -> tensor<1x5x1xf32>
    %134 = linalg.generic {indexing_maps = [#map6, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%133 : tensor<1x5x1xf32>) outs(%6 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = math.rsqrt %in : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x1xf32>
    %135 = linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%134 : tensor<1x5x1xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x5x768xf32>
    %136 = linalg.generic {indexing_maps = [#map3, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%129, %135 : tensor<1x5x768xf32>, tensor<1x5x768xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x768xf32>
    %137 = linalg.generic {indexing_maps = [#map3, #map7, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%136, %cst : tensor<1x5x768xf32>, tensor<768xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x768xf32>
    %138 = linalg.generic {indexing_maps = [#map3, #map7, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%137, %cst : tensor<1x5x768xf32>, tensor<768xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.addf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x768xf32>
    %collapsed_56 = tensor.collapse_shape %138 [[0, 1], [2]] : tensor<1x5x768xf32> into tensor<5x768xf32>
    %139 = linalg.generic {indexing_maps = [#map10, #map11, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%collapsed_56, %cst_1 : tensor<5x768xf32>, tensor<768x3072xf32>) outs(%78 : tensor<5x3072xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      %761 = arith.addf %out, %760 : f32
      linalg.yield %761 : f32
    } -> tensor<5x3072xf32>
    %140 = linalg.generic {indexing_maps = [#map12, #map9, #map9], iterator_types = ["parallel", "parallel"]} ins(%cst_2, %139 : tensor<3072xf32>, tensor<5x3072xf32>) outs(%77 : tensor<5x3072xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.addf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<5x3072xf32>
    %expanded_57 = tensor.expand_shape %140 [[0, 1], [2]] : tensor<5x3072xf32> into tensor<1x5x3072xf32>
    %141 = linalg.generic {indexing_maps = [#map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_57 : tensor<1x5x3072xf32>) outs(%81 : tensor<1x5x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.mulf %in, %cst_17 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x3072xf32>
    %142 = linalg.generic {indexing_maps = [#map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_57 : tensor<1x5x3072xf32>) outs(%81 : tensor<1x5x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = math.powf %in, %cst_18 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x3072xf32>
    %143 = linalg.generic {indexing_maps = [#map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%142 : tensor<1x5x3072xf32>) outs(%81 : tensor<1x5x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.truncf %cst_14 : f64 to f32
      %761 = arith.mulf %in, %760 : f32
      linalg.yield %761 : f32
    } -> tensor<1x5x3072xf32>
    %144 = linalg.generic {indexing_maps = [#map3, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_57, %143 : tensor<1x5x3072xf32>, tensor<1x5x3072xf32>) outs(%81 : tensor<1x5x3072xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.addf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x3072xf32>
    %145 = linalg.generic {indexing_maps = [#map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%144 : tensor<1x5x3072xf32>) outs(%81 : tensor<1x5x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.truncf %cst_15 : f64 to f32
      %761 = arith.mulf %in, %760 : f32
      linalg.yield %761 : f32
    } -> tensor<1x5x3072xf32>
    %146 = linalg.generic {indexing_maps = [#map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%145 : tensor<1x5x3072xf32>) outs(%81 : tensor<1x5x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = math.tanh %in : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x3072xf32>
    %147 = linalg.generic {indexing_maps = [#map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%146 : tensor<1x5x3072xf32>) outs(%81 : tensor<1x5x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.addf %in, %cst_19 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x3072xf32>
    %148 = linalg.generic {indexing_maps = [#map3, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%141, %147 : tensor<1x5x3072xf32>, tensor<1x5x3072xf32>) outs(%81 : tensor<1x5x3072xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x3072xf32>
    %collapsed_58 = tensor.collapse_shape %148 [[0, 1], [2]] : tensor<1x5x3072xf32> into tensor<5x3072xf32>
    %149 = linalg.generic {indexing_maps = [#map10, #map11, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%collapsed_58, %cst_0 : tensor<5x3072xf32>, tensor<3072x768xf32>) outs(%60 : tensor<5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      %761 = arith.addf %out, %760 : f32
      linalg.yield %761 : f32
    } -> tensor<5x768xf32>
    %150 = linalg.generic {indexing_maps = [#map12, #map9, #map9], iterator_types = ["parallel", "parallel"]} ins(%cst, %149 : tensor<768xf32>, tensor<5x768xf32>) outs(%59 : tensor<5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.addf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<5x768xf32>
    %expanded_59 = tensor.expand_shape %150 [[0, 1], [2]] : tensor<5x768xf32> into tensor<1x5x768xf32>
    %151 = linalg.generic {indexing_maps = [#map3, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%125, %expanded_59 : tensor<1x5x768xf32>, tensor<1x5x768xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.addf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x768xf32>
    %152 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "reduction"]} ins(%151 : tensor<1x5x768xf32>) outs(%7 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.addf %in, %out : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x1xf32>
    %153 = linalg.generic {indexing_maps = [#map6, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%152 : tensor<1x5x1xf32>) outs(%6 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.divf %in, %cst_16 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x1xf32>
    %154 = linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%153 : tensor<1x5x1xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x5x768xf32>
    %155 = linalg.generic {indexing_maps = [#map3, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%151, %154 : tensor<1x5x768xf32>, tensor<1x5x768xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.subf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x768xf32>
    %156 = linalg.generic {indexing_maps = [#map3, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%155, %155 : tensor<1x5x768xf32>, tensor<1x5x768xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x768xf32>
    %157 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "reduction"]} ins(%156 : tensor<1x5x768xf32>) outs(%7 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.addf %in, %out : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x1xf32>
    %158 = linalg.generic {indexing_maps = [#map6, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%157 : tensor<1x5x1xf32>) outs(%6 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.divf %in, %cst_16 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x1xf32>
    %159 = linalg.generic {indexing_maps = [#map6, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%158 : tensor<1x5x1xf32>) outs(%6 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.truncf %cst_12 : f64 to f32
      %761 = arith.addf %in, %760 : f32
      linalg.yield %761 : f32
    } -> tensor<1x5x1xf32>
    %160 = linalg.generic {indexing_maps = [#map6, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%159 : tensor<1x5x1xf32>) outs(%6 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = math.rsqrt %in : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x1xf32>
    %161 = linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%160 : tensor<1x5x1xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x5x768xf32>
    %162 = linalg.generic {indexing_maps = [#map3, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%155, %161 : tensor<1x5x768xf32>, tensor<1x5x768xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x768xf32>
    %163 = linalg.generic {indexing_maps = [#map3, #map7, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%162, %cst : tensor<1x5x768xf32>, tensor<768xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x768xf32>
    %164 = linalg.generic {indexing_maps = [#map3, #map7, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%163, %cst : tensor<1x5x768xf32>, tensor<768xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.addf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x768xf32>
    %collapsed_60 = tensor.collapse_shape %164 [[0, 1], [2]] : tensor<1x5x768xf32> into tensor<5x768xf32>
    %165 = linalg.generic {indexing_maps = [#map10, #map11, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%collapsed_60, %cst_4 : tensor<5x768xf32>, tensor<768x2304xf32>) outs(%22 : tensor<5x2304xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      %761 = arith.addf %out, %760 : f32
      linalg.yield %761 : f32
    } -> tensor<5x2304xf32>
    %166 = linalg.generic {indexing_maps = [#map12, #map9, #map9], iterator_types = ["parallel", "parallel"]} ins(%cst_5, %165 : tensor<2304xf32>, tensor<5x2304xf32>) outs(%21 : tensor<5x2304xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.addf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<5x2304xf32>
    %expanded_61 = tensor.expand_shape %166 [[0, 1], [2]] : tensor<5x2304xf32> into tensor<1x5x2304xf32>
    %extracted_slice_62 = tensor.extract_slice %expanded_61[0, 0, 0] [1, 5, 768] [1, 1, 1] : tensor<1x5x2304xf32> to tensor<1x5x768xf32>
    %extracted_slice_63 = tensor.extract_slice %expanded_61[0, 0, 768] [1, 5, 768] [1, 1, 1] : tensor<1x5x2304xf32> to tensor<1x5x768xf32>
    %extracted_slice_64 = tensor.extract_slice %expanded_61[0, 0, 1536] [1, 5, 768] [1, 1, 1] : tensor<1x5x2304xf32> to tensor<1x5x768xf32>
    %expanded_65 = tensor.expand_shape %extracted_slice_62 [[0], [1], [2, 3]] : tensor<1x5x768xf32> into tensor<1x5x12x64xf32>
    %167 = linalg.generic {indexing_maps = [#map13, #map14], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_65 : tensor<1x5x12x64xf32>) outs(%25 : tensor<1x12x5x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x12x5x64xf32>
    %expanded_66 = tensor.expand_shape %extracted_slice_63 [[0], [1], [2, 3]] : tensor<1x5x768xf32> into tensor<1x5x12x64xf32>
    %168 = linalg.generic {indexing_maps = [#map13, #map14], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_66 : tensor<1x5x12x64xf32>) outs(%25 : tensor<1x12x5x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x12x5x64xf32>
    %expanded_67 = tensor.expand_shape %extracted_slice_64 [[0], [1], [2, 3]] : tensor<1x5x768xf32> into tensor<1x5x12x64xf32>
    %169 = linalg.generic {indexing_maps = [#map13, #map14], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_67 : tensor<1x5x12x64xf32>) outs(%25 : tensor<1x12x5x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x12x5x64xf32>
    %170 = linalg.generic {indexing_maps = [#map13, #map15], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%168 : tensor<1x12x5x64xf32>) outs(%29 : tensor<1x12x64x5xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x12x64x5xf32>
    %collapsed_68 = tensor.collapse_shape %167 [[0, 1], [2], [3]] : tensor<1x12x5x64xf32> into tensor<12x5x64xf32>
    %collapsed_69 = tensor.collapse_shape %170 [[0, 1], [2], [3]] : tensor<1x12x64x5xf32> into tensor<12x64x5xf32>
    %171 = linalg.generic {indexing_maps = [#map16, #map17, #map18], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%collapsed_68, %collapsed_69 : tensor<12x5x64xf32>, tensor<12x64x5xf32>) outs(%32 : tensor<12x5x5xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      %761 = arith.addf %out, %760 : f32
      linalg.yield %761 : f32
    } -> tensor<12x5x5xf32>
    %expanded_70 = tensor.expand_shape %171 [[0, 1], [2], [3]] : tensor<12x5x5xf32> into tensor<1x12x5x5xf32>
    %172 = linalg.generic {indexing_maps = [#map20, #map21, #map13], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_70, %37 : tensor<1x12x5x5xf32>, tensor<f32>) outs(%38 : tensor<1x12x5x5xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.divf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x12x5x5xf32>
    %173 = linalg.generic {indexing_maps = [#map22, #map20, #map21, #map13], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_30, %172, %41 : tensor<1x1x5x5xi1>, tensor<1x12x5x5xf32>, tensor<f32>) outs(%38 : tensor<1x12x5x5xf32>) {
    ^bb0(%in: i1, %in_262: f32, %in_263: f32, %out: f32):
      %760 = arith.select %in, %in_262, %in_263 : f32
      linalg.yield %760 : f32
    } -> tensor<1x12x5x5xf32>
    %174:2 = linalg.generic {indexing_maps = [#map13, #map23, #map23], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%173 : tensor<1x12x5x5xf32>) outs(%46, %44 : tensor<1x12x5x1xf32>, tensor<1x12x5x1xi64>) {
    ^bb0(%in: f32, %out: f32, %out_262: i64):
      %760 = linalg.index 3 : index
      %761 = arith.index_cast %760 : index to i64
      %762 = arith.maximumf %in, %out : f32
      %763 = arith.cmpf ogt, %in, %out : f32
      %764 = arith.select %763, %761, %out_262 : i64
      linalg.yield %762, %764 : f32, i64
    } -> (tensor<1x12x5x1xf32>, tensor<1x12x5x1xi64>)
    %175 = linalg.generic {indexing_maps = [#map20, #map24, #map13], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%173, %174#0 : tensor<1x12x5x5xf32>, tensor<1x12x5x1xf32>) outs(%38 : tensor<1x12x5x5xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.subf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x12x5x5xf32>
    %176 = linalg.generic {indexing_maps = [#map20, #map13], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%175 : tensor<1x12x5x5xf32>) outs(%38 : tensor<1x12x5x5xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = math.exp %in : f32
      linalg.yield %760 : f32
    } -> tensor<1x12x5x5xf32>
    %177 = linalg.generic {indexing_maps = [#map13, #map23], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%176 : tensor<1x12x5x5xf32>) outs(%50 : tensor<1x12x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.addf %in, %out : f32
      linalg.yield %760 : f32
    } -> tensor<1x12x5x1xf32>
    %178 = linalg.generic {indexing_maps = [#map20, #map24, #map13], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%176, %177 : tensor<1x12x5x5xf32>, tensor<1x12x5x1xf32>) outs(%38 : tensor<1x12x5x5xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.divf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x12x5x5xf32>
    %collapsed_71 = tensor.collapse_shape %178 [[0, 1], [2], [3]] : tensor<1x12x5x5xf32> into tensor<12x5x5xf32>
    %collapsed_72 = tensor.collapse_shape %169 [[0, 1], [2], [3]] : tensor<1x12x5x64xf32> into tensor<12x5x64xf32>
    %179 = linalg.generic {indexing_maps = [#map16, #map17, #map18], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%collapsed_71, %collapsed_72 : tensor<12x5x5xf32>, tensor<12x5x64xf32>) outs(%54 : tensor<12x5x64xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      %761 = arith.addf %out, %760 : f32
      linalg.yield %761 : f32
    } -> tensor<12x5x64xf32>
    %expanded_73 = tensor.expand_shape %179 [[0, 1], [2], [3]] : tensor<12x5x64xf32> into tensor<1x12x5x64xf32>
    %180 = linalg.generic {indexing_maps = [#map13, #map14], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_73 : tensor<1x12x5x64xf32>) outs(%56 : tensor<1x5x12x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x5x12x64xf32>
    %181 = linalg.generic {indexing_maps = [#map20, #map13], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%180 : tensor<1x5x12x64xf32>) outs(%56 : tensor<1x5x12x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x5x12x64xf32>
    %collapsed_74 = tensor.collapse_shape %181 [[0, 1], [2, 3]] : tensor<1x5x12x64xf32> into tensor<5x768xf32>
    %182 = linalg.generic {indexing_maps = [#map10, #map11, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%collapsed_74, %cst_3 : tensor<5x768xf32>, tensor<768x768xf32>) outs(%60 : tensor<5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      %761 = arith.addf %out, %760 : f32
      linalg.yield %761 : f32
    } -> tensor<5x768xf32>
    %183 = linalg.generic {indexing_maps = [#map12, #map9, #map9], iterator_types = ["parallel", "parallel"]} ins(%cst, %182 : tensor<768xf32>, tensor<5x768xf32>) outs(%59 : tensor<5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.addf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<5x768xf32>
    %expanded_75 = tensor.expand_shape %183 [[0, 1], [2]] : tensor<5x768xf32> into tensor<1x5x768xf32>
    %184 = linalg.generic {indexing_maps = [#map3, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_75, %151 : tensor<1x5x768xf32>, tensor<1x5x768xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.addf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x768xf32>
    %185 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "reduction"]} ins(%184 : tensor<1x5x768xf32>) outs(%7 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.addf %in, %out : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x1xf32>
    %186 = linalg.generic {indexing_maps = [#map6, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%185 : tensor<1x5x1xf32>) outs(%6 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.divf %in, %cst_16 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x1xf32>
    %187 = linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%186 : tensor<1x5x1xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x5x768xf32>
    %188 = linalg.generic {indexing_maps = [#map3, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%184, %187 : tensor<1x5x768xf32>, tensor<1x5x768xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.subf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x768xf32>
    %189 = linalg.generic {indexing_maps = [#map3, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%188, %188 : tensor<1x5x768xf32>, tensor<1x5x768xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x768xf32>
    %190 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "reduction"]} ins(%189 : tensor<1x5x768xf32>) outs(%7 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.addf %in, %out : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x1xf32>
    %191 = linalg.generic {indexing_maps = [#map6, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%190 : tensor<1x5x1xf32>) outs(%6 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.divf %in, %cst_16 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x1xf32>
    %192 = linalg.generic {indexing_maps = [#map6, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%191 : tensor<1x5x1xf32>) outs(%6 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.truncf %cst_12 : f64 to f32
      %761 = arith.addf %in, %760 : f32
      linalg.yield %761 : f32
    } -> tensor<1x5x1xf32>
    %193 = linalg.generic {indexing_maps = [#map6, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%192 : tensor<1x5x1xf32>) outs(%6 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = math.rsqrt %in : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x1xf32>
    %194 = linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%193 : tensor<1x5x1xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x5x768xf32>
    %195 = linalg.generic {indexing_maps = [#map3, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%188, %194 : tensor<1x5x768xf32>, tensor<1x5x768xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x768xf32>
    %196 = linalg.generic {indexing_maps = [#map3, #map7, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%195, %cst : tensor<1x5x768xf32>, tensor<768xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x768xf32>
    %197 = linalg.generic {indexing_maps = [#map3, #map7, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%196, %cst : tensor<1x5x768xf32>, tensor<768xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.addf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x768xf32>
    %collapsed_76 = tensor.collapse_shape %197 [[0, 1], [2]] : tensor<1x5x768xf32> into tensor<5x768xf32>
    %198 = linalg.generic {indexing_maps = [#map10, #map11, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%collapsed_76, %cst_1 : tensor<5x768xf32>, tensor<768x3072xf32>) outs(%78 : tensor<5x3072xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      %761 = arith.addf %out, %760 : f32
      linalg.yield %761 : f32
    } -> tensor<5x3072xf32>
    %199 = linalg.generic {indexing_maps = [#map12, #map9, #map9], iterator_types = ["parallel", "parallel"]} ins(%cst_2, %198 : tensor<3072xf32>, tensor<5x3072xf32>) outs(%77 : tensor<5x3072xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.addf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<5x3072xf32>
    %expanded_77 = tensor.expand_shape %199 [[0, 1], [2]] : tensor<5x3072xf32> into tensor<1x5x3072xf32>
    %200 = linalg.generic {indexing_maps = [#map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_77 : tensor<1x5x3072xf32>) outs(%81 : tensor<1x5x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.mulf %in, %cst_17 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x3072xf32>
    %201 = linalg.generic {indexing_maps = [#map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_77 : tensor<1x5x3072xf32>) outs(%81 : tensor<1x5x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = math.powf %in, %cst_18 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x3072xf32>
    %202 = linalg.generic {indexing_maps = [#map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%201 : tensor<1x5x3072xf32>) outs(%81 : tensor<1x5x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.truncf %cst_14 : f64 to f32
      %761 = arith.mulf %in, %760 : f32
      linalg.yield %761 : f32
    } -> tensor<1x5x3072xf32>
    %203 = linalg.generic {indexing_maps = [#map3, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_77, %202 : tensor<1x5x3072xf32>, tensor<1x5x3072xf32>) outs(%81 : tensor<1x5x3072xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.addf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x3072xf32>
    %204 = linalg.generic {indexing_maps = [#map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%203 : tensor<1x5x3072xf32>) outs(%81 : tensor<1x5x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.truncf %cst_15 : f64 to f32
      %761 = arith.mulf %in, %760 : f32
      linalg.yield %761 : f32
    } -> tensor<1x5x3072xf32>
    %205 = linalg.generic {indexing_maps = [#map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%204 : tensor<1x5x3072xf32>) outs(%81 : tensor<1x5x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = math.tanh %in : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x3072xf32>
    %206 = linalg.generic {indexing_maps = [#map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%205 : tensor<1x5x3072xf32>) outs(%81 : tensor<1x5x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.addf %in, %cst_19 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x3072xf32>
    %207 = linalg.generic {indexing_maps = [#map3, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%200, %206 : tensor<1x5x3072xf32>, tensor<1x5x3072xf32>) outs(%81 : tensor<1x5x3072xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x3072xf32>
    %collapsed_78 = tensor.collapse_shape %207 [[0, 1], [2]] : tensor<1x5x3072xf32> into tensor<5x3072xf32>
    %208 = linalg.generic {indexing_maps = [#map10, #map11, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%collapsed_78, %cst_0 : tensor<5x3072xf32>, tensor<3072x768xf32>) outs(%60 : tensor<5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      %761 = arith.addf %out, %760 : f32
      linalg.yield %761 : f32
    } -> tensor<5x768xf32>
    %209 = linalg.generic {indexing_maps = [#map12, #map9, #map9], iterator_types = ["parallel", "parallel"]} ins(%cst, %208 : tensor<768xf32>, tensor<5x768xf32>) outs(%59 : tensor<5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.addf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<5x768xf32>
    %expanded_79 = tensor.expand_shape %209 [[0, 1], [2]] : tensor<5x768xf32> into tensor<1x5x768xf32>
    %210 = linalg.generic {indexing_maps = [#map3, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%184, %expanded_79 : tensor<1x5x768xf32>, tensor<1x5x768xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.addf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x768xf32>
    %211 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "reduction"]} ins(%210 : tensor<1x5x768xf32>) outs(%7 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.addf %in, %out : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x1xf32>
    %212 = linalg.generic {indexing_maps = [#map6, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%211 : tensor<1x5x1xf32>) outs(%6 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.divf %in, %cst_16 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x1xf32>
    %213 = linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%212 : tensor<1x5x1xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x5x768xf32>
    %214 = linalg.generic {indexing_maps = [#map3, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%210, %213 : tensor<1x5x768xf32>, tensor<1x5x768xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.subf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x768xf32>
    %215 = linalg.generic {indexing_maps = [#map3, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%214, %214 : tensor<1x5x768xf32>, tensor<1x5x768xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x768xf32>
    %216 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "reduction"]} ins(%215 : tensor<1x5x768xf32>) outs(%7 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.addf %in, %out : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x1xf32>
    %217 = linalg.generic {indexing_maps = [#map6, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%216 : tensor<1x5x1xf32>) outs(%6 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.divf %in, %cst_16 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x1xf32>
    %218 = linalg.generic {indexing_maps = [#map6, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%217 : tensor<1x5x1xf32>) outs(%6 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.truncf %cst_12 : f64 to f32
      %761 = arith.addf %in, %760 : f32
      linalg.yield %761 : f32
    } -> tensor<1x5x1xf32>
    %219 = linalg.generic {indexing_maps = [#map6, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%218 : tensor<1x5x1xf32>) outs(%6 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = math.rsqrt %in : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x1xf32>
    %220 = linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%219 : tensor<1x5x1xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x5x768xf32>
    %221 = linalg.generic {indexing_maps = [#map3, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%214, %220 : tensor<1x5x768xf32>, tensor<1x5x768xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x768xf32>
    %222 = linalg.generic {indexing_maps = [#map3, #map7, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%221, %cst : tensor<1x5x768xf32>, tensor<768xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x768xf32>
    %223 = linalg.generic {indexing_maps = [#map3, #map7, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%222, %cst : tensor<1x5x768xf32>, tensor<768xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.addf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x768xf32>
    %collapsed_80 = tensor.collapse_shape %223 [[0, 1], [2]] : tensor<1x5x768xf32> into tensor<5x768xf32>
    %224 = linalg.generic {indexing_maps = [#map10, #map11, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%collapsed_80, %cst_4 : tensor<5x768xf32>, tensor<768x2304xf32>) outs(%22 : tensor<5x2304xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      %761 = arith.addf %out, %760 : f32
      linalg.yield %761 : f32
    } -> tensor<5x2304xf32>
    %225 = linalg.generic {indexing_maps = [#map12, #map9, #map9], iterator_types = ["parallel", "parallel"]} ins(%cst_5, %224 : tensor<2304xf32>, tensor<5x2304xf32>) outs(%21 : tensor<5x2304xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.addf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<5x2304xf32>
    %expanded_81 = tensor.expand_shape %225 [[0, 1], [2]] : tensor<5x2304xf32> into tensor<1x5x2304xf32>
    %extracted_slice_82 = tensor.extract_slice %expanded_81[0, 0, 0] [1, 5, 768] [1, 1, 1] : tensor<1x5x2304xf32> to tensor<1x5x768xf32>
    %extracted_slice_83 = tensor.extract_slice %expanded_81[0, 0, 768] [1, 5, 768] [1, 1, 1] : tensor<1x5x2304xf32> to tensor<1x5x768xf32>
    %extracted_slice_84 = tensor.extract_slice %expanded_81[0, 0, 1536] [1, 5, 768] [1, 1, 1] : tensor<1x5x2304xf32> to tensor<1x5x768xf32>
    %expanded_85 = tensor.expand_shape %extracted_slice_82 [[0], [1], [2, 3]] : tensor<1x5x768xf32> into tensor<1x5x12x64xf32>
    %226 = linalg.generic {indexing_maps = [#map13, #map14], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_85 : tensor<1x5x12x64xf32>) outs(%25 : tensor<1x12x5x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x12x5x64xf32>
    %expanded_86 = tensor.expand_shape %extracted_slice_83 [[0], [1], [2, 3]] : tensor<1x5x768xf32> into tensor<1x5x12x64xf32>
    %227 = linalg.generic {indexing_maps = [#map13, #map14], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_86 : tensor<1x5x12x64xf32>) outs(%25 : tensor<1x12x5x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x12x5x64xf32>
    %expanded_87 = tensor.expand_shape %extracted_slice_84 [[0], [1], [2, 3]] : tensor<1x5x768xf32> into tensor<1x5x12x64xf32>
    %228 = linalg.generic {indexing_maps = [#map13, #map14], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_87 : tensor<1x5x12x64xf32>) outs(%25 : tensor<1x12x5x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x12x5x64xf32>
    %229 = linalg.generic {indexing_maps = [#map13, #map15], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%227 : tensor<1x12x5x64xf32>) outs(%29 : tensor<1x12x64x5xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x12x64x5xf32>
    %collapsed_88 = tensor.collapse_shape %226 [[0, 1], [2], [3]] : tensor<1x12x5x64xf32> into tensor<12x5x64xf32>
    %collapsed_89 = tensor.collapse_shape %229 [[0, 1], [2], [3]] : tensor<1x12x64x5xf32> into tensor<12x64x5xf32>
    %230 = linalg.generic {indexing_maps = [#map16, #map17, #map18], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%collapsed_88, %collapsed_89 : tensor<12x5x64xf32>, tensor<12x64x5xf32>) outs(%32 : tensor<12x5x5xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      %761 = arith.addf %out, %760 : f32
      linalg.yield %761 : f32
    } -> tensor<12x5x5xf32>
    %expanded_90 = tensor.expand_shape %230 [[0, 1], [2], [3]] : tensor<12x5x5xf32> into tensor<1x12x5x5xf32>
    %231 = linalg.generic {indexing_maps = [#map20, #map21, #map13], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_90, %37 : tensor<1x12x5x5xf32>, tensor<f32>) outs(%38 : tensor<1x12x5x5xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.divf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x12x5x5xf32>
    %232 = linalg.generic {indexing_maps = [#map22, #map20, #map21, #map13], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_30, %231, %41 : tensor<1x1x5x5xi1>, tensor<1x12x5x5xf32>, tensor<f32>) outs(%38 : tensor<1x12x5x5xf32>) {
    ^bb0(%in: i1, %in_262: f32, %in_263: f32, %out: f32):
      %760 = arith.select %in, %in_262, %in_263 : f32
      linalg.yield %760 : f32
    } -> tensor<1x12x5x5xf32>
    %233:2 = linalg.generic {indexing_maps = [#map13, #map23, #map23], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%232 : tensor<1x12x5x5xf32>) outs(%46, %44 : tensor<1x12x5x1xf32>, tensor<1x12x5x1xi64>) {
    ^bb0(%in: f32, %out: f32, %out_262: i64):
      %760 = linalg.index 3 : index
      %761 = arith.index_cast %760 : index to i64
      %762 = arith.maximumf %in, %out : f32
      %763 = arith.cmpf ogt, %in, %out : f32
      %764 = arith.select %763, %761, %out_262 : i64
      linalg.yield %762, %764 : f32, i64
    } -> (tensor<1x12x5x1xf32>, tensor<1x12x5x1xi64>)
    %234 = linalg.generic {indexing_maps = [#map20, #map24, #map13], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%232, %233#0 : tensor<1x12x5x5xf32>, tensor<1x12x5x1xf32>) outs(%38 : tensor<1x12x5x5xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.subf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x12x5x5xf32>
    %235 = linalg.generic {indexing_maps = [#map20, #map13], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%234 : tensor<1x12x5x5xf32>) outs(%38 : tensor<1x12x5x5xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = math.exp %in : f32
      linalg.yield %760 : f32
    } -> tensor<1x12x5x5xf32>
    %236 = linalg.generic {indexing_maps = [#map13, #map23], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%235 : tensor<1x12x5x5xf32>) outs(%50 : tensor<1x12x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.addf %in, %out : f32
      linalg.yield %760 : f32
    } -> tensor<1x12x5x1xf32>
    %237 = linalg.generic {indexing_maps = [#map20, #map24, #map13], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%235, %236 : tensor<1x12x5x5xf32>, tensor<1x12x5x1xf32>) outs(%38 : tensor<1x12x5x5xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.divf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x12x5x5xf32>
    %collapsed_91 = tensor.collapse_shape %237 [[0, 1], [2], [3]] : tensor<1x12x5x5xf32> into tensor<12x5x5xf32>
    %collapsed_92 = tensor.collapse_shape %228 [[0, 1], [2], [3]] : tensor<1x12x5x64xf32> into tensor<12x5x64xf32>
    %238 = linalg.generic {indexing_maps = [#map16, #map17, #map18], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%collapsed_91, %collapsed_92 : tensor<12x5x5xf32>, tensor<12x5x64xf32>) outs(%54 : tensor<12x5x64xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      %761 = arith.addf %out, %760 : f32
      linalg.yield %761 : f32
    } -> tensor<12x5x64xf32>
    %expanded_93 = tensor.expand_shape %238 [[0, 1], [2], [3]] : tensor<12x5x64xf32> into tensor<1x12x5x64xf32>
    %239 = linalg.generic {indexing_maps = [#map13, #map14], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_93 : tensor<1x12x5x64xf32>) outs(%56 : tensor<1x5x12x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x5x12x64xf32>
    %240 = linalg.generic {indexing_maps = [#map20, #map13], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%239 : tensor<1x5x12x64xf32>) outs(%56 : tensor<1x5x12x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x5x12x64xf32>
    %collapsed_94 = tensor.collapse_shape %240 [[0, 1], [2, 3]] : tensor<1x5x12x64xf32> into tensor<5x768xf32>
    %241 = linalg.generic {indexing_maps = [#map10, #map11, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%collapsed_94, %cst_3 : tensor<5x768xf32>, tensor<768x768xf32>) outs(%60 : tensor<5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      %761 = arith.addf %out, %760 : f32
      linalg.yield %761 : f32
    } -> tensor<5x768xf32>
    %242 = linalg.generic {indexing_maps = [#map12, #map9, #map9], iterator_types = ["parallel", "parallel"]} ins(%cst, %241 : tensor<768xf32>, tensor<5x768xf32>) outs(%59 : tensor<5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.addf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<5x768xf32>
    %expanded_95 = tensor.expand_shape %242 [[0, 1], [2]] : tensor<5x768xf32> into tensor<1x5x768xf32>
    %243 = linalg.generic {indexing_maps = [#map3, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_95, %210 : tensor<1x5x768xf32>, tensor<1x5x768xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.addf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x768xf32>
    %244 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "reduction"]} ins(%243 : tensor<1x5x768xf32>) outs(%7 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.addf %in, %out : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x1xf32>
    %245 = linalg.generic {indexing_maps = [#map6, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%244 : tensor<1x5x1xf32>) outs(%6 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.divf %in, %cst_16 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x1xf32>
    %246 = linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%245 : tensor<1x5x1xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x5x768xf32>
    %247 = linalg.generic {indexing_maps = [#map3, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%243, %246 : tensor<1x5x768xf32>, tensor<1x5x768xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.subf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x768xf32>
    %248 = linalg.generic {indexing_maps = [#map3, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%247, %247 : tensor<1x5x768xf32>, tensor<1x5x768xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x768xf32>
    %249 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "reduction"]} ins(%248 : tensor<1x5x768xf32>) outs(%7 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.addf %in, %out : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x1xf32>
    %250 = linalg.generic {indexing_maps = [#map6, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%249 : tensor<1x5x1xf32>) outs(%6 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.divf %in, %cst_16 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x1xf32>
    %251 = linalg.generic {indexing_maps = [#map6, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%250 : tensor<1x5x1xf32>) outs(%6 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.truncf %cst_12 : f64 to f32
      %761 = arith.addf %in, %760 : f32
      linalg.yield %761 : f32
    } -> tensor<1x5x1xf32>
    %252 = linalg.generic {indexing_maps = [#map6, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%251 : tensor<1x5x1xf32>) outs(%6 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = math.rsqrt %in : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x1xf32>
    %253 = linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%252 : tensor<1x5x1xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x5x768xf32>
    %254 = linalg.generic {indexing_maps = [#map3, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%247, %253 : tensor<1x5x768xf32>, tensor<1x5x768xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x768xf32>
    %255 = linalg.generic {indexing_maps = [#map3, #map7, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%254, %cst : tensor<1x5x768xf32>, tensor<768xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x768xf32>
    %256 = linalg.generic {indexing_maps = [#map3, #map7, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%255, %cst : tensor<1x5x768xf32>, tensor<768xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.addf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x768xf32>
    %collapsed_96 = tensor.collapse_shape %256 [[0, 1], [2]] : tensor<1x5x768xf32> into tensor<5x768xf32>
    %257 = linalg.generic {indexing_maps = [#map10, #map11, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%collapsed_96, %cst_1 : tensor<5x768xf32>, tensor<768x3072xf32>) outs(%78 : tensor<5x3072xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      %761 = arith.addf %out, %760 : f32
      linalg.yield %761 : f32
    } -> tensor<5x3072xf32>
    %258 = linalg.generic {indexing_maps = [#map12, #map9, #map9], iterator_types = ["parallel", "parallel"]} ins(%cst_2, %257 : tensor<3072xf32>, tensor<5x3072xf32>) outs(%77 : tensor<5x3072xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.addf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<5x3072xf32>
    %expanded_97 = tensor.expand_shape %258 [[0, 1], [2]] : tensor<5x3072xf32> into tensor<1x5x3072xf32>
    %259 = linalg.generic {indexing_maps = [#map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_97 : tensor<1x5x3072xf32>) outs(%81 : tensor<1x5x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.mulf %in, %cst_17 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x3072xf32>
    %260 = linalg.generic {indexing_maps = [#map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_97 : tensor<1x5x3072xf32>) outs(%81 : tensor<1x5x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = math.powf %in, %cst_18 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x3072xf32>
    %261 = linalg.generic {indexing_maps = [#map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%260 : tensor<1x5x3072xf32>) outs(%81 : tensor<1x5x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.truncf %cst_14 : f64 to f32
      %761 = arith.mulf %in, %760 : f32
      linalg.yield %761 : f32
    } -> tensor<1x5x3072xf32>
    %262 = linalg.generic {indexing_maps = [#map3, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_97, %261 : tensor<1x5x3072xf32>, tensor<1x5x3072xf32>) outs(%81 : tensor<1x5x3072xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.addf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x3072xf32>
    %263 = linalg.generic {indexing_maps = [#map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%262 : tensor<1x5x3072xf32>) outs(%81 : tensor<1x5x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.truncf %cst_15 : f64 to f32
      %761 = arith.mulf %in, %760 : f32
      linalg.yield %761 : f32
    } -> tensor<1x5x3072xf32>
    %264 = linalg.generic {indexing_maps = [#map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%263 : tensor<1x5x3072xf32>) outs(%81 : tensor<1x5x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = math.tanh %in : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x3072xf32>
    %265 = linalg.generic {indexing_maps = [#map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%264 : tensor<1x5x3072xf32>) outs(%81 : tensor<1x5x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.addf %in, %cst_19 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x3072xf32>
    %266 = linalg.generic {indexing_maps = [#map3, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%259, %265 : tensor<1x5x3072xf32>, tensor<1x5x3072xf32>) outs(%81 : tensor<1x5x3072xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x3072xf32>
    %collapsed_98 = tensor.collapse_shape %266 [[0, 1], [2]] : tensor<1x5x3072xf32> into tensor<5x3072xf32>
    %267 = linalg.generic {indexing_maps = [#map10, #map11, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%collapsed_98, %cst_0 : tensor<5x3072xf32>, tensor<3072x768xf32>) outs(%60 : tensor<5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      %761 = arith.addf %out, %760 : f32
      linalg.yield %761 : f32
    } -> tensor<5x768xf32>
    %268 = linalg.generic {indexing_maps = [#map12, #map9, #map9], iterator_types = ["parallel", "parallel"]} ins(%cst, %267 : tensor<768xf32>, tensor<5x768xf32>) outs(%59 : tensor<5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.addf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<5x768xf32>
    %expanded_99 = tensor.expand_shape %268 [[0, 1], [2]] : tensor<5x768xf32> into tensor<1x5x768xf32>
    %269 = linalg.generic {indexing_maps = [#map3, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%243, %expanded_99 : tensor<1x5x768xf32>, tensor<1x5x768xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.addf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x768xf32>
    %270 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "reduction"]} ins(%269 : tensor<1x5x768xf32>) outs(%7 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.addf %in, %out : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x1xf32>
    %271 = linalg.generic {indexing_maps = [#map6, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%270 : tensor<1x5x1xf32>) outs(%6 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.divf %in, %cst_16 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x1xf32>
    %272 = linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%271 : tensor<1x5x1xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x5x768xf32>
    %273 = linalg.generic {indexing_maps = [#map3, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%269, %272 : tensor<1x5x768xf32>, tensor<1x5x768xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.subf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x768xf32>
    %274 = linalg.generic {indexing_maps = [#map3, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%273, %273 : tensor<1x5x768xf32>, tensor<1x5x768xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x768xf32>
    %275 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "reduction"]} ins(%274 : tensor<1x5x768xf32>) outs(%7 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.addf %in, %out : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x1xf32>
    %276 = linalg.generic {indexing_maps = [#map6, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%275 : tensor<1x5x1xf32>) outs(%6 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.divf %in, %cst_16 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x1xf32>
    %277 = linalg.generic {indexing_maps = [#map6, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%276 : tensor<1x5x1xf32>) outs(%6 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.truncf %cst_12 : f64 to f32
      %761 = arith.addf %in, %760 : f32
      linalg.yield %761 : f32
    } -> tensor<1x5x1xf32>
    %278 = linalg.generic {indexing_maps = [#map6, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%277 : tensor<1x5x1xf32>) outs(%6 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = math.rsqrt %in : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x1xf32>
    %279 = linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%278 : tensor<1x5x1xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x5x768xf32>
    %280 = linalg.generic {indexing_maps = [#map3, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%273, %279 : tensor<1x5x768xf32>, tensor<1x5x768xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x768xf32>
    %281 = linalg.generic {indexing_maps = [#map3, #map7, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%280, %cst : tensor<1x5x768xf32>, tensor<768xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x768xf32>
    %282 = linalg.generic {indexing_maps = [#map3, #map7, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%281, %cst : tensor<1x5x768xf32>, tensor<768xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.addf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x768xf32>
    %collapsed_100 = tensor.collapse_shape %282 [[0, 1], [2]] : tensor<1x5x768xf32> into tensor<5x768xf32>
    %283 = linalg.generic {indexing_maps = [#map10, #map11, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%collapsed_100, %cst_4 : tensor<5x768xf32>, tensor<768x2304xf32>) outs(%22 : tensor<5x2304xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      %761 = arith.addf %out, %760 : f32
      linalg.yield %761 : f32
    } -> tensor<5x2304xf32>
    %284 = linalg.generic {indexing_maps = [#map12, #map9, #map9], iterator_types = ["parallel", "parallel"]} ins(%cst_5, %283 : tensor<2304xf32>, tensor<5x2304xf32>) outs(%21 : tensor<5x2304xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.addf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<5x2304xf32>
    %expanded_101 = tensor.expand_shape %284 [[0, 1], [2]] : tensor<5x2304xf32> into tensor<1x5x2304xf32>
    %extracted_slice_102 = tensor.extract_slice %expanded_101[0, 0, 0] [1, 5, 768] [1, 1, 1] : tensor<1x5x2304xf32> to tensor<1x5x768xf32>
    %extracted_slice_103 = tensor.extract_slice %expanded_101[0, 0, 768] [1, 5, 768] [1, 1, 1] : tensor<1x5x2304xf32> to tensor<1x5x768xf32>
    %extracted_slice_104 = tensor.extract_slice %expanded_101[0, 0, 1536] [1, 5, 768] [1, 1, 1] : tensor<1x5x2304xf32> to tensor<1x5x768xf32>
    %expanded_105 = tensor.expand_shape %extracted_slice_102 [[0], [1], [2, 3]] : tensor<1x5x768xf32> into tensor<1x5x12x64xf32>
    %285 = linalg.generic {indexing_maps = [#map13, #map14], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_105 : tensor<1x5x12x64xf32>) outs(%25 : tensor<1x12x5x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x12x5x64xf32>
    %expanded_106 = tensor.expand_shape %extracted_slice_103 [[0], [1], [2, 3]] : tensor<1x5x768xf32> into tensor<1x5x12x64xf32>
    %286 = linalg.generic {indexing_maps = [#map13, #map14], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_106 : tensor<1x5x12x64xf32>) outs(%25 : tensor<1x12x5x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x12x5x64xf32>
    %expanded_107 = tensor.expand_shape %extracted_slice_104 [[0], [1], [2, 3]] : tensor<1x5x768xf32> into tensor<1x5x12x64xf32>
    %287 = linalg.generic {indexing_maps = [#map13, #map14], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_107 : tensor<1x5x12x64xf32>) outs(%25 : tensor<1x12x5x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x12x5x64xf32>
    %288 = linalg.generic {indexing_maps = [#map13, #map15], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%286 : tensor<1x12x5x64xf32>) outs(%29 : tensor<1x12x64x5xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x12x64x5xf32>
    %collapsed_108 = tensor.collapse_shape %285 [[0, 1], [2], [3]] : tensor<1x12x5x64xf32> into tensor<12x5x64xf32>
    %collapsed_109 = tensor.collapse_shape %288 [[0, 1], [2], [3]] : tensor<1x12x64x5xf32> into tensor<12x64x5xf32>
    %289 = linalg.generic {indexing_maps = [#map16, #map17, #map18], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%collapsed_108, %collapsed_109 : tensor<12x5x64xf32>, tensor<12x64x5xf32>) outs(%32 : tensor<12x5x5xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      %761 = arith.addf %out, %760 : f32
      linalg.yield %761 : f32
    } -> tensor<12x5x5xf32>
    %expanded_110 = tensor.expand_shape %289 [[0, 1], [2], [3]] : tensor<12x5x5xf32> into tensor<1x12x5x5xf32>
    %290 = linalg.generic {indexing_maps = [#map20, #map21, #map13], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_110, %37 : tensor<1x12x5x5xf32>, tensor<f32>) outs(%38 : tensor<1x12x5x5xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.divf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x12x5x5xf32>
    %291 = linalg.generic {indexing_maps = [#map22, #map20, #map21, #map13], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_30, %290, %41 : tensor<1x1x5x5xi1>, tensor<1x12x5x5xf32>, tensor<f32>) outs(%38 : tensor<1x12x5x5xf32>) {
    ^bb0(%in: i1, %in_262: f32, %in_263: f32, %out: f32):
      %760 = arith.select %in, %in_262, %in_263 : f32
      linalg.yield %760 : f32
    } -> tensor<1x12x5x5xf32>
    %292:2 = linalg.generic {indexing_maps = [#map13, #map23, #map23], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%291 : tensor<1x12x5x5xf32>) outs(%46, %44 : tensor<1x12x5x1xf32>, tensor<1x12x5x1xi64>) {
    ^bb0(%in: f32, %out: f32, %out_262: i64):
      %760 = linalg.index 3 : index
      %761 = arith.index_cast %760 : index to i64
      %762 = arith.maximumf %in, %out : f32
      %763 = arith.cmpf ogt, %in, %out : f32
      %764 = arith.select %763, %761, %out_262 : i64
      linalg.yield %762, %764 : f32, i64
    } -> (tensor<1x12x5x1xf32>, tensor<1x12x5x1xi64>)
    %293 = linalg.generic {indexing_maps = [#map20, #map24, #map13], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%291, %292#0 : tensor<1x12x5x5xf32>, tensor<1x12x5x1xf32>) outs(%38 : tensor<1x12x5x5xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.subf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x12x5x5xf32>
    %294 = linalg.generic {indexing_maps = [#map20, #map13], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%293 : tensor<1x12x5x5xf32>) outs(%38 : tensor<1x12x5x5xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = math.exp %in : f32
      linalg.yield %760 : f32
    } -> tensor<1x12x5x5xf32>
    %295 = linalg.generic {indexing_maps = [#map13, #map23], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%294 : tensor<1x12x5x5xf32>) outs(%50 : tensor<1x12x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.addf %in, %out : f32
      linalg.yield %760 : f32
    } -> tensor<1x12x5x1xf32>
    %296 = linalg.generic {indexing_maps = [#map20, #map24, #map13], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%294, %295 : tensor<1x12x5x5xf32>, tensor<1x12x5x1xf32>) outs(%38 : tensor<1x12x5x5xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.divf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x12x5x5xf32>
    %collapsed_111 = tensor.collapse_shape %296 [[0, 1], [2], [3]] : tensor<1x12x5x5xf32> into tensor<12x5x5xf32>
    %collapsed_112 = tensor.collapse_shape %287 [[0, 1], [2], [3]] : tensor<1x12x5x64xf32> into tensor<12x5x64xf32>
    %297 = linalg.generic {indexing_maps = [#map16, #map17, #map18], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%collapsed_111, %collapsed_112 : tensor<12x5x5xf32>, tensor<12x5x64xf32>) outs(%54 : tensor<12x5x64xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      %761 = arith.addf %out, %760 : f32
      linalg.yield %761 : f32
    } -> tensor<12x5x64xf32>
    %expanded_113 = tensor.expand_shape %297 [[0, 1], [2], [3]] : tensor<12x5x64xf32> into tensor<1x12x5x64xf32>
    %298 = linalg.generic {indexing_maps = [#map13, #map14], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_113 : tensor<1x12x5x64xf32>) outs(%56 : tensor<1x5x12x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x5x12x64xf32>
    %299 = linalg.generic {indexing_maps = [#map20, #map13], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%298 : tensor<1x5x12x64xf32>) outs(%56 : tensor<1x5x12x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x5x12x64xf32>
    %collapsed_114 = tensor.collapse_shape %299 [[0, 1], [2, 3]] : tensor<1x5x12x64xf32> into tensor<5x768xf32>
    %300 = linalg.generic {indexing_maps = [#map10, #map11, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%collapsed_114, %cst_3 : tensor<5x768xf32>, tensor<768x768xf32>) outs(%60 : tensor<5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      %761 = arith.addf %out, %760 : f32
      linalg.yield %761 : f32
    } -> tensor<5x768xf32>
    %301 = linalg.generic {indexing_maps = [#map12, #map9, #map9], iterator_types = ["parallel", "parallel"]} ins(%cst, %300 : tensor<768xf32>, tensor<5x768xf32>) outs(%59 : tensor<5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.addf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<5x768xf32>
    %expanded_115 = tensor.expand_shape %301 [[0, 1], [2]] : tensor<5x768xf32> into tensor<1x5x768xf32>
    %302 = linalg.generic {indexing_maps = [#map3, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_115, %269 : tensor<1x5x768xf32>, tensor<1x5x768xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.addf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x768xf32>
    %303 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "reduction"]} ins(%302 : tensor<1x5x768xf32>) outs(%7 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.addf %in, %out : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x1xf32>
    %304 = linalg.generic {indexing_maps = [#map6, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%303 : tensor<1x5x1xf32>) outs(%6 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.divf %in, %cst_16 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x1xf32>
    %305 = linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%304 : tensor<1x5x1xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x5x768xf32>
    %306 = linalg.generic {indexing_maps = [#map3, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%302, %305 : tensor<1x5x768xf32>, tensor<1x5x768xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.subf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x768xf32>
    %307 = linalg.generic {indexing_maps = [#map3, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%306, %306 : tensor<1x5x768xf32>, tensor<1x5x768xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x768xf32>
    %308 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "reduction"]} ins(%307 : tensor<1x5x768xf32>) outs(%7 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.addf %in, %out : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x1xf32>
    %309 = linalg.generic {indexing_maps = [#map6, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%308 : tensor<1x5x1xf32>) outs(%6 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.divf %in, %cst_16 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x1xf32>
    %310 = linalg.generic {indexing_maps = [#map6, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%309 : tensor<1x5x1xf32>) outs(%6 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.truncf %cst_12 : f64 to f32
      %761 = arith.addf %in, %760 : f32
      linalg.yield %761 : f32
    } -> tensor<1x5x1xf32>
    %311 = linalg.generic {indexing_maps = [#map6, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%310 : tensor<1x5x1xf32>) outs(%6 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = math.rsqrt %in : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x1xf32>
    %312 = linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%311 : tensor<1x5x1xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x5x768xf32>
    %313 = linalg.generic {indexing_maps = [#map3, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%306, %312 : tensor<1x5x768xf32>, tensor<1x5x768xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x768xf32>
    %314 = linalg.generic {indexing_maps = [#map3, #map7, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%313, %cst : tensor<1x5x768xf32>, tensor<768xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x768xf32>
    %315 = linalg.generic {indexing_maps = [#map3, #map7, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%314, %cst : tensor<1x5x768xf32>, tensor<768xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.addf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x768xf32>
    %collapsed_116 = tensor.collapse_shape %315 [[0, 1], [2]] : tensor<1x5x768xf32> into tensor<5x768xf32>
    %316 = linalg.generic {indexing_maps = [#map10, #map11, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%collapsed_116, %cst_1 : tensor<5x768xf32>, tensor<768x3072xf32>) outs(%78 : tensor<5x3072xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      %761 = arith.addf %out, %760 : f32
      linalg.yield %761 : f32
    } -> tensor<5x3072xf32>
    %317 = linalg.generic {indexing_maps = [#map12, #map9, #map9], iterator_types = ["parallel", "parallel"]} ins(%cst_2, %316 : tensor<3072xf32>, tensor<5x3072xf32>) outs(%77 : tensor<5x3072xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.addf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<5x3072xf32>
    %expanded_117 = tensor.expand_shape %317 [[0, 1], [2]] : tensor<5x3072xf32> into tensor<1x5x3072xf32>
    %318 = linalg.generic {indexing_maps = [#map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_117 : tensor<1x5x3072xf32>) outs(%81 : tensor<1x5x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.mulf %in, %cst_17 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x3072xf32>
    %319 = linalg.generic {indexing_maps = [#map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_117 : tensor<1x5x3072xf32>) outs(%81 : tensor<1x5x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = math.powf %in, %cst_18 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x3072xf32>
    %320 = linalg.generic {indexing_maps = [#map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%319 : tensor<1x5x3072xf32>) outs(%81 : tensor<1x5x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.truncf %cst_14 : f64 to f32
      %761 = arith.mulf %in, %760 : f32
      linalg.yield %761 : f32
    } -> tensor<1x5x3072xf32>
    %321 = linalg.generic {indexing_maps = [#map3, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_117, %320 : tensor<1x5x3072xf32>, tensor<1x5x3072xf32>) outs(%81 : tensor<1x5x3072xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.addf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x3072xf32>
    %322 = linalg.generic {indexing_maps = [#map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%321 : tensor<1x5x3072xf32>) outs(%81 : tensor<1x5x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.truncf %cst_15 : f64 to f32
      %761 = arith.mulf %in, %760 : f32
      linalg.yield %761 : f32
    } -> tensor<1x5x3072xf32>
    %323 = linalg.generic {indexing_maps = [#map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%322 : tensor<1x5x3072xf32>) outs(%81 : tensor<1x5x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = math.tanh %in : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x3072xf32>
    %324 = linalg.generic {indexing_maps = [#map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%323 : tensor<1x5x3072xf32>) outs(%81 : tensor<1x5x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.addf %in, %cst_19 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x3072xf32>
    %325 = linalg.generic {indexing_maps = [#map3, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%318, %324 : tensor<1x5x3072xf32>, tensor<1x5x3072xf32>) outs(%81 : tensor<1x5x3072xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x3072xf32>
    %collapsed_118 = tensor.collapse_shape %325 [[0, 1], [2]] : tensor<1x5x3072xf32> into tensor<5x3072xf32>
    %326 = linalg.generic {indexing_maps = [#map10, #map11, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%collapsed_118, %cst_0 : tensor<5x3072xf32>, tensor<3072x768xf32>) outs(%60 : tensor<5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      %761 = arith.addf %out, %760 : f32
      linalg.yield %761 : f32
    } -> tensor<5x768xf32>
    %327 = linalg.generic {indexing_maps = [#map12, #map9, #map9], iterator_types = ["parallel", "parallel"]} ins(%cst, %326 : tensor<768xf32>, tensor<5x768xf32>) outs(%59 : tensor<5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.addf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<5x768xf32>
    %expanded_119 = tensor.expand_shape %327 [[0, 1], [2]] : tensor<5x768xf32> into tensor<1x5x768xf32>
    %328 = linalg.generic {indexing_maps = [#map3, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%302, %expanded_119 : tensor<1x5x768xf32>, tensor<1x5x768xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.addf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x768xf32>
    %329 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "reduction"]} ins(%328 : tensor<1x5x768xf32>) outs(%7 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.addf %in, %out : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x1xf32>
    %330 = linalg.generic {indexing_maps = [#map6, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%329 : tensor<1x5x1xf32>) outs(%6 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.divf %in, %cst_16 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x1xf32>
    %331 = linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%330 : tensor<1x5x1xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x5x768xf32>
    %332 = linalg.generic {indexing_maps = [#map3, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%328, %331 : tensor<1x5x768xf32>, tensor<1x5x768xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.subf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x768xf32>
    %333 = linalg.generic {indexing_maps = [#map3, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%332, %332 : tensor<1x5x768xf32>, tensor<1x5x768xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x768xf32>
    %334 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "reduction"]} ins(%333 : tensor<1x5x768xf32>) outs(%7 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.addf %in, %out : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x1xf32>
    %335 = linalg.generic {indexing_maps = [#map6, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%334 : tensor<1x5x1xf32>) outs(%6 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.divf %in, %cst_16 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x1xf32>
    %336 = linalg.generic {indexing_maps = [#map6, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%335 : tensor<1x5x1xf32>) outs(%6 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.truncf %cst_12 : f64 to f32
      %761 = arith.addf %in, %760 : f32
      linalg.yield %761 : f32
    } -> tensor<1x5x1xf32>
    %337 = linalg.generic {indexing_maps = [#map6, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%336 : tensor<1x5x1xf32>) outs(%6 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = math.rsqrt %in : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x1xf32>
    %338 = linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%337 : tensor<1x5x1xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x5x768xf32>
    %339 = linalg.generic {indexing_maps = [#map3, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%332, %338 : tensor<1x5x768xf32>, tensor<1x5x768xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x768xf32>
    %340 = linalg.generic {indexing_maps = [#map3, #map7, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%339, %cst : tensor<1x5x768xf32>, tensor<768xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x768xf32>
    %341 = linalg.generic {indexing_maps = [#map3, #map7, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%340, %cst : tensor<1x5x768xf32>, tensor<768xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.addf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x768xf32>
    %collapsed_120 = tensor.collapse_shape %341 [[0, 1], [2]] : tensor<1x5x768xf32> into tensor<5x768xf32>
    %342 = linalg.generic {indexing_maps = [#map10, #map11, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%collapsed_120, %cst_4 : tensor<5x768xf32>, tensor<768x2304xf32>) outs(%22 : tensor<5x2304xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      %761 = arith.addf %out, %760 : f32
      linalg.yield %761 : f32
    } -> tensor<5x2304xf32>
    %343 = linalg.generic {indexing_maps = [#map12, #map9, #map9], iterator_types = ["parallel", "parallel"]} ins(%cst_5, %342 : tensor<2304xf32>, tensor<5x2304xf32>) outs(%21 : tensor<5x2304xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.addf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<5x2304xf32>
    %expanded_121 = tensor.expand_shape %343 [[0, 1], [2]] : tensor<5x2304xf32> into tensor<1x5x2304xf32>
    %extracted_slice_122 = tensor.extract_slice %expanded_121[0, 0, 0] [1, 5, 768] [1, 1, 1] : tensor<1x5x2304xf32> to tensor<1x5x768xf32>
    %extracted_slice_123 = tensor.extract_slice %expanded_121[0, 0, 768] [1, 5, 768] [1, 1, 1] : tensor<1x5x2304xf32> to tensor<1x5x768xf32>
    %extracted_slice_124 = tensor.extract_slice %expanded_121[0, 0, 1536] [1, 5, 768] [1, 1, 1] : tensor<1x5x2304xf32> to tensor<1x5x768xf32>
    %expanded_125 = tensor.expand_shape %extracted_slice_122 [[0], [1], [2, 3]] : tensor<1x5x768xf32> into tensor<1x5x12x64xf32>
    %344 = linalg.generic {indexing_maps = [#map13, #map14], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_125 : tensor<1x5x12x64xf32>) outs(%25 : tensor<1x12x5x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x12x5x64xf32>
    %expanded_126 = tensor.expand_shape %extracted_slice_123 [[0], [1], [2, 3]] : tensor<1x5x768xf32> into tensor<1x5x12x64xf32>
    %345 = linalg.generic {indexing_maps = [#map13, #map14], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_126 : tensor<1x5x12x64xf32>) outs(%25 : tensor<1x12x5x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x12x5x64xf32>
    %expanded_127 = tensor.expand_shape %extracted_slice_124 [[0], [1], [2, 3]] : tensor<1x5x768xf32> into tensor<1x5x12x64xf32>
    %346 = linalg.generic {indexing_maps = [#map13, #map14], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_127 : tensor<1x5x12x64xf32>) outs(%25 : tensor<1x12x5x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x12x5x64xf32>
    %347 = linalg.generic {indexing_maps = [#map13, #map15], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%345 : tensor<1x12x5x64xf32>) outs(%29 : tensor<1x12x64x5xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x12x64x5xf32>
    %collapsed_128 = tensor.collapse_shape %344 [[0, 1], [2], [3]] : tensor<1x12x5x64xf32> into tensor<12x5x64xf32>
    %collapsed_129 = tensor.collapse_shape %347 [[0, 1], [2], [3]] : tensor<1x12x64x5xf32> into tensor<12x64x5xf32>
    %348 = linalg.generic {indexing_maps = [#map16, #map17, #map18], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%collapsed_128, %collapsed_129 : tensor<12x5x64xf32>, tensor<12x64x5xf32>) outs(%32 : tensor<12x5x5xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      %761 = arith.addf %out, %760 : f32
      linalg.yield %761 : f32
    } -> tensor<12x5x5xf32>
    %expanded_130 = tensor.expand_shape %348 [[0, 1], [2], [3]] : tensor<12x5x5xf32> into tensor<1x12x5x5xf32>
    %349 = linalg.generic {indexing_maps = [#map20, #map21, #map13], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_130, %37 : tensor<1x12x5x5xf32>, tensor<f32>) outs(%38 : tensor<1x12x5x5xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.divf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x12x5x5xf32>
    %350 = linalg.generic {indexing_maps = [#map22, #map20, #map21, #map13], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_30, %349, %41 : tensor<1x1x5x5xi1>, tensor<1x12x5x5xf32>, tensor<f32>) outs(%38 : tensor<1x12x5x5xf32>) {
    ^bb0(%in: i1, %in_262: f32, %in_263: f32, %out: f32):
      %760 = arith.select %in, %in_262, %in_263 : f32
      linalg.yield %760 : f32
    } -> tensor<1x12x5x5xf32>
    %351:2 = linalg.generic {indexing_maps = [#map13, #map23, #map23], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%350 : tensor<1x12x5x5xf32>) outs(%46, %44 : tensor<1x12x5x1xf32>, tensor<1x12x5x1xi64>) {
    ^bb0(%in: f32, %out: f32, %out_262: i64):
      %760 = linalg.index 3 : index
      %761 = arith.index_cast %760 : index to i64
      %762 = arith.maximumf %in, %out : f32
      %763 = arith.cmpf ogt, %in, %out : f32
      %764 = arith.select %763, %761, %out_262 : i64
      linalg.yield %762, %764 : f32, i64
    } -> (tensor<1x12x5x1xf32>, tensor<1x12x5x1xi64>)
    %352 = linalg.generic {indexing_maps = [#map20, #map24, #map13], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%350, %351#0 : tensor<1x12x5x5xf32>, tensor<1x12x5x1xf32>) outs(%38 : tensor<1x12x5x5xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.subf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x12x5x5xf32>
    %353 = linalg.generic {indexing_maps = [#map20, #map13], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%352 : tensor<1x12x5x5xf32>) outs(%38 : tensor<1x12x5x5xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = math.exp %in : f32
      linalg.yield %760 : f32
    } -> tensor<1x12x5x5xf32>
    %354 = linalg.generic {indexing_maps = [#map13, #map23], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%353 : tensor<1x12x5x5xf32>) outs(%50 : tensor<1x12x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.addf %in, %out : f32
      linalg.yield %760 : f32
    } -> tensor<1x12x5x1xf32>
    %355 = linalg.generic {indexing_maps = [#map20, #map24, #map13], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%353, %354 : tensor<1x12x5x5xf32>, tensor<1x12x5x1xf32>) outs(%38 : tensor<1x12x5x5xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.divf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x12x5x5xf32>
    %collapsed_131 = tensor.collapse_shape %355 [[0, 1], [2], [3]] : tensor<1x12x5x5xf32> into tensor<12x5x5xf32>
    %collapsed_132 = tensor.collapse_shape %346 [[0, 1], [2], [3]] : tensor<1x12x5x64xf32> into tensor<12x5x64xf32>
    %356 = linalg.generic {indexing_maps = [#map16, #map17, #map18], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%collapsed_131, %collapsed_132 : tensor<12x5x5xf32>, tensor<12x5x64xf32>) outs(%54 : tensor<12x5x64xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      %761 = arith.addf %out, %760 : f32
      linalg.yield %761 : f32
    } -> tensor<12x5x64xf32>
    %expanded_133 = tensor.expand_shape %356 [[0, 1], [2], [3]] : tensor<12x5x64xf32> into tensor<1x12x5x64xf32>
    %357 = linalg.generic {indexing_maps = [#map13, #map14], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_133 : tensor<1x12x5x64xf32>) outs(%56 : tensor<1x5x12x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x5x12x64xf32>
    %358 = linalg.generic {indexing_maps = [#map20, #map13], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%357 : tensor<1x5x12x64xf32>) outs(%56 : tensor<1x5x12x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x5x12x64xf32>
    %collapsed_134 = tensor.collapse_shape %358 [[0, 1], [2, 3]] : tensor<1x5x12x64xf32> into tensor<5x768xf32>
    %359 = linalg.generic {indexing_maps = [#map10, #map11, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%collapsed_134, %cst_3 : tensor<5x768xf32>, tensor<768x768xf32>) outs(%60 : tensor<5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      %761 = arith.addf %out, %760 : f32
      linalg.yield %761 : f32
    } -> tensor<5x768xf32>
    %360 = linalg.generic {indexing_maps = [#map12, #map9, #map9], iterator_types = ["parallel", "parallel"]} ins(%cst, %359 : tensor<768xf32>, tensor<5x768xf32>) outs(%59 : tensor<5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.addf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<5x768xf32>
    %expanded_135 = tensor.expand_shape %360 [[0, 1], [2]] : tensor<5x768xf32> into tensor<1x5x768xf32>
    %361 = linalg.generic {indexing_maps = [#map3, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_135, %328 : tensor<1x5x768xf32>, tensor<1x5x768xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.addf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x768xf32>
    %362 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "reduction"]} ins(%361 : tensor<1x5x768xf32>) outs(%7 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.addf %in, %out : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x1xf32>
    %363 = linalg.generic {indexing_maps = [#map6, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%362 : tensor<1x5x1xf32>) outs(%6 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.divf %in, %cst_16 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x1xf32>
    %364 = linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%363 : tensor<1x5x1xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x5x768xf32>
    %365 = linalg.generic {indexing_maps = [#map3, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%361, %364 : tensor<1x5x768xf32>, tensor<1x5x768xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.subf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x768xf32>
    %366 = linalg.generic {indexing_maps = [#map3, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%365, %365 : tensor<1x5x768xf32>, tensor<1x5x768xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x768xf32>
    %367 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "reduction"]} ins(%366 : tensor<1x5x768xf32>) outs(%7 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.addf %in, %out : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x1xf32>
    %368 = linalg.generic {indexing_maps = [#map6, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%367 : tensor<1x5x1xf32>) outs(%6 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.divf %in, %cst_16 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x1xf32>
    %369 = linalg.generic {indexing_maps = [#map6, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%368 : tensor<1x5x1xf32>) outs(%6 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.truncf %cst_12 : f64 to f32
      %761 = arith.addf %in, %760 : f32
      linalg.yield %761 : f32
    } -> tensor<1x5x1xf32>
    %370 = linalg.generic {indexing_maps = [#map6, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%369 : tensor<1x5x1xf32>) outs(%6 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = math.rsqrt %in : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x1xf32>
    %371 = linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%370 : tensor<1x5x1xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x5x768xf32>
    %372 = linalg.generic {indexing_maps = [#map3, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%365, %371 : tensor<1x5x768xf32>, tensor<1x5x768xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x768xf32>
    %373 = linalg.generic {indexing_maps = [#map3, #map7, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%372, %cst : tensor<1x5x768xf32>, tensor<768xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x768xf32>
    %374 = linalg.generic {indexing_maps = [#map3, #map7, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%373, %cst : tensor<1x5x768xf32>, tensor<768xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.addf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x768xf32>
    %collapsed_136 = tensor.collapse_shape %374 [[0, 1], [2]] : tensor<1x5x768xf32> into tensor<5x768xf32>
    %375 = linalg.generic {indexing_maps = [#map10, #map11, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%collapsed_136, %cst_1 : tensor<5x768xf32>, tensor<768x3072xf32>) outs(%78 : tensor<5x3072xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      %761 = arith.addf %out, %760 : f32
      linalg.yield %761 : f32
    } -> tensor<5x3072xf32>
    %376 = linalg.generic {indexing_maps = [#map12, #map9, #map9], iterator_types = ["parallel", "parallel"]} ins(%cst_2, %375 : tensor<3072xf32>, tensor<5x3072xf32>) outs(%77 : tensor<5x3072xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.addf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<5x3072xf32>
    %expanded_137 = tensor.expand_shape %376 [[0, 1], [2]] : tensor<5x3072xf32> into tensor<1x5x3072xf32>
    %377 = linalg.generic {indexing_maps = [#map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_137 : tensor<1x5x3072xf32>) outs(%81 : tensor<1x5x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.mulf %in, %cst_17 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x3072xf32>
    %378 = linalg.generic {indexing_maps = [#map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_137 : tensor<1x5x3072xf32>) outs(%81 : tensor<1x5x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = math.powf %in, %cst_18 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x3072xf32>
    %379 = linalg.generic {indexing_maps = [#map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%378 : tensor<1x5x3072xf32>) outs(%81 : tensor<1x5x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.truncf %cst_14 : f64 to f32
      %761 = arith.mulf %in, %760 : f32
      linalg.yield %761 : f32
    } -> tensor<1x5x3072xf32>
    %380 = linalg.generic {indexing_maps = [#map3, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_137, %379 : tensor<1x5x3072xf32>, tensor<1x5x3072xf32>) outs(%81 : tensor<1x5x3072xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.addf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x3072xf32>
    %381 = linalg.generic {indexing_maps = [#map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%380 : tensor<1x5x3072xf32>) outs(%81 : tensor<1x5x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.truncf %cst_15 : f64 to f32
      %761 = arith.mulf %in, %760 : f32
      linalg.yield %761 : f32
    } -> tensor<1x5x3072xf32>
    %382 = linalg.generic {indexing_maps = [#map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%381 : tensor<1x5x3072xf32>) outs(%81 : tensor<1x5x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = math.tanh %in : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x3072xf32>
    %383 = linalg.generic {indexing_maps = [#map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%382 : tensor<1x5x3072xf32>) outs(%81 : tensor<1x5x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.addf %in, %cst_19 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x3072xf32>
    %384 = linalg.generic {indexing_maps = [#map3, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%377, %383 : tensor<1x5x3072xf32>, tensor<1x5x3072xf32>) outs(%81 : tensor<1x5x3072xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x3072xf32>
    %collapsed_138 = tensor.collapse_shape %384 [[0, 1], [2]] : tensor<1x5x3072xf32> into tensor<5x3072xf32>
    %385 = linalg.generic {indexing_maps = [#map10, #map11, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%collapsed_138, %cst_0 : tensor<5x3072xf32>, tensor<3072x768xf32>) outs(%60 : tensor<5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      %761 = arith.addf %out, %760 : f32
      linalg.yield %761 : f32
    } -> tensor<5x768xf32>
    %386 = linalg.generic {indexing_maps = [#map12, #map9, #map9], iterator_types = ["parallel", "parallel"]} ins(%cst, %385 : tensor<768xf32>, tensor<5x768xf32>) outs(%59 : tensor<5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.addf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<5x768xf32>
    %expanded_139 = tensor.expand_shape %386 [[0, 1], [2]] : tensor<5x768xf32> into tensor<1x5x768xf32>
    %387 = linalg.generic {indexing_maps = [#map3, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%361, %expanded_139 : tensor<1x5x768xf32>, tensor<1x5x768xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.addf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x768xf32>
    %388 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "reduction"]} ins(%387 : tensor<1x5x768xf32>) outs(%7 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.addf %in, %out : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x1xf32>
    %389 = linalg.generic {indexing_maps = [#map6, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%388 : tensor<1x5x1xf32>) outs(%6 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.divf %in, %cst_16 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x1xf32>
    %390 = linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%389 : tensor<1x5x1xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x5x768xf32>
    %391 = linalg.generic {indexing_maps = [#map3, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%387, %390 : tensor<1x5x768xf32>, tensor<1x5x768xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.subf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x768xf32>
    %392 = linalg.generic {indexing_maps = [#map3, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%391, %391 : tensor<1x5x768xf32>, tensor<1x5x768xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x768xf32>
    %393 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "reduction"]} ins(%392 : tensor<1x5x768xf32>) outs(%7 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.addf %in, %out : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x1xf32>
    %394 = linalg.generic {indexing_maps = [#map6, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%393 : tensor<1x5x1xf32>) outs(%6 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.divf %in, %cst_16 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x1xf32>
    %395 = linalg.generic {indexing_maps = [#map6, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%394 : tensor<1x5x1xf32>) outs(%6 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.truncf %cst_12 : f64 to f32
      %761 = arith.addf %in, %760 : f32
      linalg.yield %761 : f32
    } -> tensor<1x5x1xf32>
    %396 = linalg.generic {indexing_maps = [#map6, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%395 : tensor<1x5x1xf32>) outs(%6 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = math.rsqrt %in : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x1xf32>
    %397 = linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%396 : tensor<1x5x1xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x5x768xf32>
    %398 = linalg.generic {indexing_maps = [#map3, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%391, %397 : tensor<1x5x768xf32>, tensor<1x5x768xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x768xf32>
    %399 = linalg.generic {indexing_maps = [#map3, #map7, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%398, %cst : tensor<1x5x768xf32>, tensor<768xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x768xf32>
    %400 = linalg.generic {indexing_maps = [#map3, #map7, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%399, %cst : tensor<1x5x768xf32>, tensor<768xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.addf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x768xf32>
    %collapsed_140 = tensor.collapse_shape %400 [[0, 1], [2]] : tensor<1x5x768xf32> into tensor<5x768xf32>
    %401 = linalg.generic {indexing_maps = [#map10, #map11, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%collapsed_140, %cst_4 : tensor<5x768xf32>, tensor<768x2304xf32>) outs(%22 : tensor<5x2304xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      %761 = arith.addf %out, %760 : f32
      linalg.yield %761 : f32
    } -> tensor<5x2304xf32>
    %402 = linalg.generic {indexing_maps = [#map12, #map9, #map9], iterator_types = ["parallel", "parallel"]} ins(%cst_5, %401 : tensor<2304xf32>, tensor<5x2304xf32>) outs(%21 : tensor<5x2304xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.addf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<5x2304xf32>
    %expanded_141 = tensor.expand_shape %402 [[0, 1], [2]] : tensor<5x2304xf32> into tensor<1x5x2304xf32>
    %extracted_slice_142 = tensor.extract_slice %expanded_141[0, 0, 0] [1, 5, 768] [1, 1, 1] : tensor<1x5x2304xf32> to tensor<1x5x768xf32>
    %extracted_slice_143 = tensor.extract_slice %expanded_141[0, 0, 768] [1, 5, 768] [1, 1, 1] : tensor<1x5x2304xf32> to tensor<1x5x768xf32>
    %extracted_slice_144 = tensor.extract_slice %expanded_141[0, 0, 1536] [1, 5, 768] [1, 1, 1] : tensor<1x5x2304xf32> to tensor<1x5x768xf32>
    %expanded_145 = tensor.expand_shape %extracted_slice_142 [[0], [1], [2, 3]] : tensor<1x5x768xf32> into tensor<1x5x12x64xf32>
    %403 = linalg.generic {indexing_maps = [#map13, #map14], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_145 : tensor<1x5x12x64xf32>) outs(%25 : tensor<1x12x5x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x12x5x64xf32>
    %expanded_146 = tensor.expand_shape %extracted_slice_143 [[0], [1], [2, 3]] : tensor<1x5x768xf32> into tensor<1x5x12x64xf32>
    %404 = linalg.generic {indexing_maps = [#map13, #map14], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_146 : tensor<1x5x12x64xf32>) outs(%25 : tensor<1x12x5x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x12x5x64xf32>
    %expanded_147 = tensor.expand_shape %extracted_slice_144 [[0], [1], [2, 3]] : tensor<1x5x768xf32> into tensor<1x5x12x64xf32>
    %405 = linalg.generic {indexing_maps = [#map13, #map14], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_147 : tensor<1x5x12x64xf32>) outs(%25 : tensor<1x12x5x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x12x5x64xf32>
    %406 = linalg.generic {indexing_maps = [#map13, #map15], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%404 : tensor<1x12x5x64xf32>) outs(%29 : tensor<1x12x64x5xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x12x64x5xf32>
    %collapsed_148 = tensor.collapse_shape %403 [[0, 1], [2], [3]] : tensor<1x12x5x64xf32> into tensor<12x5x64xf32>
    %collapsed_149 = tensor.collapse_shape %406 [[0, 1], [2], [3]] : tensor<1x12x64x5xf32> into tensor<12x64x5xf32>
    %407 = linalg.generic {indexing_maps = [#map16, #map17, #map18], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%collapsed_148, %collapsed_149 : tensor<12x5x64xf32>, tensor<12x64x5xf32>) outs(%32 : tensor<12x5x5xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      %761 = arith.addf %out, %760 : f32
      linalg.yield %761 : f32
    } -> tensor<12x5x5xf32>
    %expanded_150 = tensor.expand_shape %407 [[0, 1], [2], [3]] : tensor<12x5x5xf32> into tensor<1x12x5x5xf32>
    %408 = linalg.generic {indexing_maps = [#map20, #map21, #map13], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_150, %37 : tensor<1x12x5x5xf32>, tensor<f32>) outs(%38 : tensor<1x12x5x5xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.divf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x12x5x5xf32>
    %409 = linalg.generic {indexing_maps = [#map22, #map20, #map21, #map13], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_30, %408, %41 : tensor<1x1x5x5xi1>, tensor<1x12x5x5xf32>, tensor<f32>) outs(%38 : tensor<1x12x5x5xf32>) {
    ^bb0(%in: i1, %in_262: f32, %in_263: f32, %out: f32):
      %760 = arith.select %in, %in_262, %in_263 : f32
      linalg.yield %760 : f32
    } -> tensor<1x12x5x5xf32>
    %410:2 = linalg.generic {indexing_maps = [#map13, #map23, #map23], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%409 : tensor<1x12x5x5xf32>) outs(%46, %44 : tensor<1x12x5x1xf32>, tensor<1x12x5x1xi64>) {
    ^bb0(%in: f32, %out: f32, %out_262: i64):
      %760 = linalg.index 3 : index
      %761 = arith.index_cast %760 : index to i64
      %762 = arith.maximumf %in, %out : f32
      %763 = arith.cmpf ogt, %in, %out : f32
      %764 = arith.select %763, %761, %out_262 : i64
      linalg.yield %762, %764 : f32, i64
    } -> (tensor<1x12x5x1xf32>, tensor<1x12x5x1xi64>)
    %411 = linalg.generic {indexing_maps = [#map20, #map24, #map13], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%409, %410#0 : tensor<1x12x5x5xf32>, tensor<1x12x5x1xf32>) outs(%38 : tensor<1x12x5x5xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.subf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x12x5x5xf32>
    %412 = linalg.generic {indexing_maps = [#map20, #map13], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%411 : tensor<1x12x5x5xf32>) outs(%38 : tensor<1x12x5x5xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = math.exp %in : f32
      linalg.yield %760 : f32
    } -> tensor<1x12x5x5xf32>
    %413 = linalg.generic {indexing_maps = [#map13, #map23], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%412 : tensor<1x12x5x5xf32>) outs(%50 : tensor<1x12x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.addf %in, %out : f32
      linalg.yield %760 : f32
    } -> tensor<1x12x5x1xf32>
    %414 = linalg.generic {indexing_maps = [#map20, #map24, #map13], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%412, %413 : tensor<1x12x5x5xf32>, tensor<1x12x5x1xf32>) outs(%38 : tensor<1x12x5x5xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.divf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x12x5x5xf32>
    %collapsed_151 = tensor.collapse_shape %414 [[0, 1], [2], [3]] : tensor<1x12x5x5xf32> into tensor<12x5x5xf32>
    %collapsed_152 = tensor.collapse_shape %405 [[0, 1], [2], [3]] : tensor<1x12x5x64xf32> into tensor<12x5x64xf32>
    %415 = linalg.generic {indexing_maps = [#map16, #map17, #map18], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%collapsed_151, %collapsed_152 : tensor<12x5x5xf32>, tensor<12x5x64xf32>) outs(%54 : tensor<12x5x64xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      %761 = arith.addf %out, %760 : f32
      linalg.yield %761 : f32
    } -> tensor<12x5x64xf32>
    %expanded_153 = tensor.expand_shape %415 [[0, 1], [2], [3]] : tensor<12x5x64xf32> into tensor<1x12x5x64xf32>
    %416 = linalg.generic {indexing_maps = [#map13, #map14], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_153 : tensor<1x12x5x64xf32>) outs(%56 : tensor<1x5x12x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x5x12x64xf32>
    %417 = linalg.generic {indexing_maps = [#map20, #map13], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%416 : tensor<1x5x12x64xf32>) outs(%56 : tensor<1x5x12x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x5x12x64xf32>
    %collapsed_154 = tensor.collapse_shape %417 [[0, 1], [2, 3]] : tensor<1x5x12x64xf32> into tensor<5x768xf32>
    %418 = linalg.generic {indexing_maps = [#map10, #map11, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%collapsed_154, %cst_3 : tensor<5x768xf32>, tensor<768x768xf32>) outs(%60 : tensor<5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      %761 = arith.addf %out, %760 : f32
      linalg.yield %761 : f32
    } -> tensor<5x768xf32>
    %419 = linalg.generic {indexing_maps = [#map12, #map9, #map9], iterator_types = ["parallel", "parallel"]} ins(%cst, %418 : tensor<768xf32>, tensor<5x768xf32>) outs(%59 : tensor<5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.addf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<5x768xf32>
    %expanded_155 = tensor.expand_shape %419 [[0, 1], [2]] : tensor<5x768xf32> into tensor<1x5x768xf32>
    %420 = linalg.generic {indexing_maps = [#map3, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_155, %387 : tensor<1x5x768xf32>, tensor<1x5x768xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.addf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x768xf32>
    %421 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "reduction"]} ins(%420 : tensor<1x5x768xf32>) outs(%7 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.addf %in, %out : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x1xf32>
    %422 = linalg.generic {indexing_maps = [#map6, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%421 : tensor<1x5x1xf32>) outs(%6 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.divf %in, %cst_16 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x1xf32>
    %423 = linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%422 : tensor<1x5x1xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x5x768xf32>
    %424 = linalg.generic {indexing_maps = [#map3, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%420, %423 : tensor<1x5x768xf32>, tensor<1x5x768xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.subf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x768xf32>
    %425 = linalg.generic {indexing_maps = [#map3, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%424, %424 : tensor<1x5x768xf32>, tensor<1x5x768xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x768xf32>
    %426 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "reduction"]} ins(%425 : tensor<1x5x768xf32>) outs(%7 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.addf %in, %out : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x1xf32>
    %427 = linalg.generic {indexing_maps = [#map6, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%426 : tensor<1x5x1xf32>) outs(%6 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.divf %in, %cst_16 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x1xf32>
    %428 = linalg.generic {indexing_maps = [#map6, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%427 : tensor<1x5x1xf32>) outs(%6 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.truncf %cst_12 : f64 to f32
      %761 = arith.addf %in, %760 : f32
      linalg.yield %761 : f32
    } -> tensor<1x5x1xf32>
    %429 = linalg.generic {indexing_maps = [#map6, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%428 : tensor<1x5x1xf32>) outs(%6 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = math.rsqrt %in : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x1xf32>
    %430 = linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%429 : tensor<1x5x1xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x5x768xf32>
    %431 = linalg.generic {indexing_maps = [#map3, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%424, %430 : tensor<1x5x768xf32>, tensor<1x5x768xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x768xf32>
    %432 = linalg.generic {indexing_maps = [#map3, #map7, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%431, %cst : tensor<1x5x768xf32>, tensor<768xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x768xf32>
    %433 = linalg.generic {indexing_maps = [#map3, #map7, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%432, %cst : tensor<1x5x768xf32>, tensor<768xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.addf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x768xf32>
    %collapsed_156 = tensor.collapse_shape %433 [[0, 1], [2]] : tensor<1x5x768xf32> into tensor<5x768xf32>
    %434 = linalg.generic {indexing_maps = [#map10, #map11, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%collapsed_156, %cst_1 : tensor<5x768xf32>, tensor<768x3072xf32>) outs(%78 : tensor<5x3072xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      %761 = arith.addf %out, %760 : f32
      linalg.yield %761 : f32
    } -> tensor<5x3072xf32>
    %435 = linalg.generic {indexing_maps = [#map12, #map9, #map9], iterator_types = ["parallel", "parallel"]} ins(%cst_2, %434 : tensor<3072xf32>, tensor<5x3072xf32>) outs(%77 : tensor<5x3072xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.addf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<5x3072xf32>
    %expanded_157 = tensor.expand_shape %435 [[0, 1], [2]] : tensor<5x3072xf32> into tensor<1x5x3072xf32>
    %436 = linalg.generic {indexing_maps = [#map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_157 : tensor<1x5x3072xf32>) outs(%81 : tensor<1x5x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.mulf %in, %cst_17 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x3072xf32>
    %437 = linalg.generic {indexing_maps = [#map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_157 : tensor<1x5x3072xf32>) outs(%81 : tensor<1x5x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = math.powf %in, %cst_18 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x3072xf32>
    %438 = linalg.generic {indexing_maps = [#map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%437 : tensor<1x5x3072xf32>) outs(%81 : tensor<1x5x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.truncf %cst_14 : f64 to f32
      %761 = arith.mulf %in, %760 : f32
      linalg.yield %761 : f32
    } -> tensor<1x5x3072xf32>
    %439 = linalg.generic {indexing_maps = [#map3, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_157, %438 : tensor<1x5x3072xf32>, tensor<1x5x3072xf32>) outs(%81 : tensor<1x5x3072xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.addf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x3072xf32>
    %440 = linalg.generic {indexing_maps = [#map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%439 : tensor<1x5x3072xf32>) outs(%81 : tensor<1x5x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.truncf %cst_15 : f64 to f32
      %761 = arith.mulf %in, %760 : f32
      linalg.yield %761 : f32
    } -> tensor<1x5x3072xf32>
    %441 = linalg.generic {indexing_maps = [#map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%440 : tensor<1x5x3072xf32>) outs(%81 : tensor<1x5x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = math.tanh %in : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x3072xf32>
    %442 = linalg.generic {indexing_maps = [#map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%441 : tensor<1x5x3072xf32>) outs(%81 : tensor<1x5x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.addf %in, %cst_19 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x3072xf32>
    %443 = linalg.generic {indexing_maps = [#map3, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%436, %442 : tensor<1x5x3072xf32>, tensor<1x5x3072xf32>) outs(%81 : tensor<1x5x3072xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x3072xf32>
    %collapsed_158 = tensor.collapse_shape %443 [[0, 1], [2]] : tensor<1x5x3072xf32> into tensor<5x3072xf32>
    %444 = linalg.generic {indexing_maps = [#map10, #map11, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%collapsed_158, %cst_0 : tensor<5x3072xf32>, tensor<3072x768xf32>) outs(%60 : tensor<5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      %761 = arith.addf %out, %760 : f32
      linalg.yield %761 : f32
    } -> tensor<5x768xf32>
    %445 = linalg.generic {indexing_maps = [#map12, #map9, #map9], iterator_types = ["parallel", "parallel"]} ins(%cst, %444 : tensor<768xf32>, tensor<5x768xf32>) outs(%59 : tensor<5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.addf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<5x768xf32>
    %expanded_159 = tensor.expand_shape %445 [[0, 1], [2]] : tensor<5x768xf32> into tensor<1x5x768xf32>
    %446 = linalg.generic {indexing_maps = [#map3, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%420, %expanded_159 : tensor<1x5x768xf32>, tensor<1x5x768xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.addf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x768xf32>
    %447 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "reduction"]} ins(%446 : tensor<1x5x768xf32>) outs(%7 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.addf %in, %out : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x1xf32>
    %448 = linalg.generic {indexing_maps = [#map6, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%447 : tensor<1x5x1xf32>) outs(%6 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.divf %in, %cst_16 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x1xf32>
    %449 = linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%448 : tensor<1x5x1xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x5x768xf32>
    %450 = linalg.generic {indexing_maps = [#map3, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%446, %449 : tensor<1x5x768xf32>, tensor<1x5x768xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.subf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x768xf32>
    %451 = linalg.generic {indexing_maps = [#map3, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%450, %450 : tensor<1x5x768xf32>, tensor<1x5x768xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x768xf32>
    %452 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "reduction"]} ins(%451 : tensor<1x5x768xf32>) outs(%7 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.addf %in, %out : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x1xf32>
    %453 = linalg.generic {indexing_maps = [#map6, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%452 : tensor<1x5x1xf32>) outs(%6 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.divf %in, %cst_16 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x1xf32>
    %454 = linalg.generic {indexing_maps = [#map6, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%453 : tensor<1x5x1xf32>) outs(%6 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.truncf %cst_12 : f64 to f32
      %761 = arith.addf %in, %760 : f32
      linalg.yield %761 : f32
    } -> tensor<1x5x1xf32>
    %455 = linalg.generic {indexing_maps = [#map6, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%454 : tensor<1x5x1xf32>) outs(%6 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = math.rsqrt %in : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x1xf32>
    %456 = linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%455 : tensor<1x5x1xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x5x768xf32>
    %457 = linalg.generic {indexing_maps = [#map3, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%450, %456 : tensor<1x5x768xf32>, tensor<1x5x768xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x768xf32>
    %458 = linalg.generic {indexing_maps = [#map3, #map7, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%457, %cst : tensor<1x5x768xf32>, tensor<768xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x768xf32>
    %459 = linalg.generic {indexing_maps = [#map3, #map7, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%458, %cst : tensor<1x5x768xf32>, tensor<768xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.addf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x768xf32>
    %collapsed_160 = tensor.collapse_shape %459 [[0, 1], [2]] : tensor<1x5x768xf32> into tensor<5x768xf32>
    %460 = linalg.generic {indexing_maps = [#map10, #map11, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%collapsed_160, %cst_4 : tensor<5x768xf32>, tensor<768x2304xf32>) outs(%22 : tensor<5x2304xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      %761 = arith.addf %out, %760 : f32
      linalg.yield %761 : f32
    } -> tensor<5x2304xf32>
    %461 = linalg.generic {indexing_maps = [#map12, #map9, #map9], iterator_types = ["parallel", "parallel"]} ins(%cst_5, %460 : tensor<2304xf32>, tensor<5x2304xf32>) outs(%21 : tensor<5x2304xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.addf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<5x2304xf32>
    %expanded_161 = tensor.expand_shape %461 [[0, 1], [2]] : tensor<5x2304xf32> into tensor<1x5x2304xf32>
    %extracted_slice_162 = tensor.extract_slice %expanded_161[0, 0, 0] [1, 5, 768] [1, 1, 1] : tensor<1x5x2304xf32> to tensor<1x5x768xf32>
    %extracted_slice_163 = tensor.extract_slice %expanded_161[0, 0, 768] [1, 5, 768] [1, 1, 1] : tensor<1x5x2304xf32> to tensor<1x5x768xf32>
    %extracted_slice_164 = tensor.extract_slice %expanded_161[0, 0, 1536] [1, 5, 768] [1, 1, 1] : tensor<1x5x2304xf32> to tensor<1x5x768xf32>
    %expanded_165 = tensor.expand_shape %extracted_slice_162 [[0], [1], [2, 3]] : tensor<1x5x768xf32> into tensor<1x5x12x64xf32>
    %462 = linalg.generic {indexing_maps = [#map13, #map14], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_165 : tensor<1x5x12x64xf32>) outs(%25 : tensor<1x12x5x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x12x5x64xf32>
    %expanded_166 = tensor.expand_shape %extracted_slice_163 [[0], [1], [2, 3]] : tensor<1x5x768xf32> into tensor<1x5x12x64xf32>
    %463 = linalg.generic {indexing_maps = [#map13, #map14], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_166 : tensor<1x5x12x64xf32>) outs(%25 : tensor<1x12x5x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x12x5x64xf32>
    %expanded_167 = tensor.expand_shape %extracted_slice_164 [[0], [1], [2, 3]] : tensor<1x5x768xf32> into tensor<1x5x12x64xf32>
    %464 = linalg.generic {indexing_maps = [#map13, #map14], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_167 : tensor<1x5x12x64xf32>) outs(%25 : tensor<1x12x5x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x12x5x64xf32>
    %465 = linalg.generic {indexing_maps = [#map13, #map15], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%463 : tensor<1x12x5x64xf32>) outs(%29 : tensor<1x12x64x5xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x12x64x5xf32>
    %collapsed_168 = tensor.collapse_shape %462 [[0, 1], [2], [3]] : tensor<1x12x5x64xf32> into tensor<12x5x64xf32>
    %collapsed_169 = tensor.collapse_shape %465 [[0, 1], [2], [3]] : tensor<1x12x64x5xf32> into tensor<12x64x5xf32>
    %466 = linalg.generic {indexing_maps = [#map16, #map17, #map18], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%collapsed_168, %collapsed_169 : tensor<12x5x64xf32>, tensor<12x64x5xf32>) outs(%32 : tensor<12x5x5xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      %761 = arith.addf %out, %760 : f32
      linalg.yield %761 : f32
    } -> tensor<12x5x5xf32>
    %expanded_170 = tensor.expand_shape %466 [[0, 1], [2], [3]] : tensor<12x5x5xf32> into tensor<1x12x5x5xf32>
    %467 = linalg.generic {indexing_maps = [#map20, #map21, #map13], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_170, %37 : tensor<1x12x5x5xf32>, tensor<f32>) outs(%38 : tensor<1x12x5x5xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.divf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x12x5x5xf32>
    %468 = linalg.generic {indexing_maps = [#map22, #map20, #map21, #map13], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_30, %467, %41 : tensor<1x1x5x5xi1>, tensor<1x12x5x5xf32>, tensor<f32>) outs(%38 : tensor<1x12x5x5xf32>) {
    ^bb0(%in: i1, %in_262: f32, %in_263: f32, %out: f32):
      %760 = arith.select %in, %in_262, %in_263 : f32
      linalg.yield %760 : f32
    } -> tensor<1x12x5x5xf32>
    %469:2 = linalg.generic {indexing_maps = [#map13, #map23, #map23], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%468 : tensor<1x12x5x5xf32>) outs(%46, %44 : tensor<1x12x5x1xf32>, tensor<1x12x5x1xi64>) {
    ^bb0(%in: f32, %out: f32, %out_262: i64):
      %760 = linalg.index 3 : index
      %761 = arith.index_cast %760 : index to i64
      %762 = arith.maximumf %in, %out : f32
      %763 = arith.cmpf ogt, %in, %out : f32
      %764 = arith.select %763, %761, %out_262 : i64
      linalg.yield %762, %764 : f32, i64
    } -> (tensor<1x12x5x1xf32>, tensor<1x12x5x1xi64>)
    %470 = linalg.generic {indexing_maps = [#map20, #map24, #map13], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%468, %469#0 : tensor<1x12x5x5xf32>, tensor<1x12x5x1xf32>) outs(%38 : tensor<1x12x5x5xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.subf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x12x5x5xf32>
    %471 = linalg.generic {indexing_maps = [#map20, #map13], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%470 : tensor<1x12x5x5xf32>) outs(%38 : tensor<1x12x5x5xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = math.exp %in : f32
      linalg.yield %760 : f32
    } -> tensor<1x12x5x5xf32>
    %472 = linalg.generic {indexing_maps = [#map13, #map23], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%471 : tensor<1x12x5x5xf32>) outs(%50 : tensor<1x12x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.addf %in, %out : f32
      linalg.yield %760 : f32
    } -> tensor<1x12x5x1xf32>
    %473 = linalg.generic {indexing_maps = [#map20, #map24, #map13], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%471, %472 : tensor<1x12x5x5xf32>, tensor<1x12x5x1xf32>) outs(%38 : tensor<1x12x5x5xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.divf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x12x5x5xf32>
    %collapsed_171 = tensor.collapse_shape %473 [[0, 1], [2], [3]] : tensor<1x12x5x5xf32> into tensor<12x5x5xf32>
    %collapsed_172 = tensor.collapse_shape %464 [[0, 1], [2], [3]] : tensor<1x12x5x64xf32> into tensor<12x5x64xf32>
    %474 = linalg.generic {indexing_maps = [#map16, #map17, #map18], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%collapsed_171, %collapsed_172 : tensor<12x5x5xf32>, tensor<12x5x64xf32>) outs(%54 : tensor<12x5x64xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      %761 = arith.addf %out, %760 : f32
      linalg.yield %761 : f32
    } -> tensor<12x5x64xf32>
    %expanded_173 = tensor.expand_shape %474 [[0, 1], [2], [3]] : tensor<12x5x64xf32> into tensor<1x12x5x64xf32>
    %475 = linalg.generic {indexing_maps = [#map13, #map14], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_173 : tensor<1x12x5x64xf32>) outs(%56 : tensor<1x5x12x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x5x12x64xf32>
    %476 = linalg.generic {indexing_maps = [#map20, #map13], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%475 : tensor<1x5x12x64xf32>) outs(%56 : tensor<1x5x12x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x5x12x64xf32>
    %collapsed_174 = tensor.collapse_shape %476 [[0, 1], [2, 3]] : tensor<1x5x12x64xf32> into tensor<5x768xf32>
    %477 = linalg.generic {indexing_maps = [#map10, #map11, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%collapsed_174, %cst_3 : tensor<5x768xf32>, tensor<768x768xf32>) outs(%60 : tensor<5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      %761 = arith.addf %out, %760 : f32
      linalg.yield %761 : f32
    } -> tensor<5x768xf32>
    %478 = linalg.generic {indexing_maps = [#map12, #map9, #map9], iterator_types = ["parallel", "parallel"]} ins(%cst, %477 : tensor<768xf32>, tensor<5x768xf32>) outs(%59 : tensor<5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.addf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<5x768xf32>
    %expanded_175 = tensor.expand_shape %478 [[0, 1], [2]] : tensor<5x768xf32> into tensor<1x5x768xf32>
    %479 = linalg.generic {indexing_maps = [#map3, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_175, %446 : tensor<1x5x768xf32>, tensor<1x5x768xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.addf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x768xf32>
    %480 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "reduction"]} ins(%479 : tensor<1x5x768xf32>) outs(%7 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.addf %in, %out : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x1xf32>
    %481 = linalg.generic {indexing_maps = [#map6, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%480 : tensor<1x5x1xf32>) outs(%6 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.divf %in, %cst_16 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x1xf32>
    %482 = linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%481 : tensor<1x5x1xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x5x768xf32>
    %483 = linalg.generic {indexing_maps = [#map3, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%479, %482 : tensor<1x5x768xf32>, tensor<1x5x768xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.subf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x768xf32>
    %484 = linalg.generic {indexing_maps = [#map3, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%483, %483 : tensor<1x5x768xf32>, tensor<1x5x768xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x768xf32>
    %485 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "reduction"]} ins(%484 : tensor<1x5x768xf32>) outs(%7 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.addf %in, %out : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x1xf32>
    %486 = linalg.generic {indexing_maps = [#map6, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%485 : tensor<1x5x1xf32>) outs(%6 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.divf %in, %cst_16 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x1xf32>
    %487 = linalg.generic {indexing_maps = [#map6, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%486 : tensor<1x5x1xf32>) outs(%6 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.truncf %cst_12 : f64 to f32
      %761 = arith.addf %in, %760 : f32
      linalg.yield %761 : f32
    } -> tensor<1x5x1xf32>
    %488 = linalg.generic {indexing_maps = [#map6, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%487 : tensor<1x5x1xf32>) outs(%6 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = math.rsqrt %in : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x1xf32>
    %489 = linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%488 : tensor<1x5x1xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x5x768xf32>
    %490 = linalg.generic {indexing_maps = [#map3, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%483, %489 : tensor<1x5x768xf32>, tensor<1x5x768xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x768xf32>
    %491 = linalg.generic {indexing_maps = [#map3, #map7, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%490, %cst : tensor<1x5x768xf32>, tensor<768xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x768xf32>
    %492 = linalg.generic {indexing_maps = [#map3, #map7, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%491, %cst : tensor<1x5x768xf32>, tensor<768xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.addf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x768xf32>
    %collapsed_176 = tensor.collapse_shape %492 [[0, 1], [2]] : tensor<1x5x768xf32> into tensor<5x768xf32>
    %493 = linalg.generic {indexing_maps = [#map10, #map11, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%collapsed_176, %cst_1 : tensor<5x768xf32>, tensor<768x3072xf32>) outs(%78 : tensor<5x3072xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      %761 = arith.addf %out, %760 : f32
      linalg.yield %761 : f32
    } -> tensor<5x3072xf32>
    %494 = linalg.generic {indexing_maps = [#map12, #map9, #map9], iterator_types = ["parallel", "parallel"]} ins(%cst_2, %493 : tensor<3072xf32>, tensor<5x3072xf32>) outs(%77 : tensor<5x3072xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.addf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<5x3072xf32>
    %expanded_177 = tensor.expand_shape %494 [[0, 1], [2]] : tensor<5x3072xf32> into tensor<1x5x3072xf32>
    %495 = linalg.generic {indexing_maps = [#map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_177 : tensor<1x5x3072xf32>) outs(%81 : tensor<1x5x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.mulf %in, %cst_17 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x3072xf32>
    %496 = linalg.generic {indexing_maps = [#map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_177 : tensor<1x5x3072xf32>) outs(%81 : tensor<1x5x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = math.powf %in, %cst_18 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x3072xf32>
    %497 = linalg.generic {indexing_maps = [#map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%496 : tensor<1x5x3072xf32>) outs(%81 : tensor<1x5x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.truncf %cst_14 : f64 to f32
      %761 = arith.mulf %in, %760 : f32
      linalg.yield %761 : f32
    } -> tensor<1x5x3072xf32>
    %498 = linalg.generic {indexing_maps = [#map3, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_177, %497 : tensor<1x5x3072xf32>, tensor<1x5x3072xf32>) outs(%81 : tensor<1x5x3072xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.addf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x3072xf32>
    %499 = linalg.generic {indexing_maps = [#map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%498 : tensor<1x5x3072xf32>) outs(%81 : tensor<1x5x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.truncf %cst_15 : f64 to f32
      %761 = arith.mulf %in, %760 : f32
      linalg.yield %761 : f32
    } -> tensor<1x5x3072xf32>
    %500 = linalg.generic {indexing_maps = [#map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%499 : tensor<1x5x3072xf32>) outs(%81 : tensor<1x5x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = math.tanh %in : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x3072xf32>
    %501 = linalg.generic {indexing_maps = [#map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%500 : tensor<1x5x3072xf32>) outs(%81 : tensor<1x5x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.addf %in, %cst_19 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x3072xf32>
    %502 = linalg.generic {indexing_maps = [#map3, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%495, %501 : tensor<1x5x3072xf32>, tensor<1x5x3072xf32>) outs(%81 : tensor<1x5x3072xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x3072xf32>
    %collapsed_178 = tensor.collapse_shape %502 [[0, 1], [2]] : tensor<1x5x3072xf32> into tensor<5x3072xf32>
    %503 = linalg.generic {indexing_maps = [#map10, #map11, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%collapsed_178, %cst_0 : tensor<5x3072xf32>, tensor<3072x768xf32>) outs(%60 : tensor<5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      %761 = arith.addf %out, %760 : f32
      linalg.yield %761 : f32
    } -> tensor<5x768xf32>
    %504 = linalg.generic {indexing_maps = [#map12, #map9, #map9], iterator_types = ["parallel", "parallel"]} ins(%cst, %503 : tensor<768xf32>, tensor<5x768xf32>) outs(%59 : tensor<5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.addf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<5x768xf32>
    %expanded_179 = tensor.expand_shape %504 [[0, 1], [2]] : tensor<5x768xf32> into tensor<1x5x768xf32>
    %505 = linalg.generic {indexing_maps = [#map3, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%479, %expanded_179 : tensor<1x5x768xf32>, tensor<1x5x768xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.addf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x768xf32>
    %506 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "reduction"]} ins(%505 : tensor<1x5x768xf32>) outs(%7 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.addf %in, %out : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x1xf32>
    %507 = linalg.generic {indexing_maps = [#map6, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%506 : tensor<1x5x1xf32>) outs(%6 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.divf %in, %cst_16 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x1xf32>
    %508 = linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%507 : tensor<1x5x1xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x5x768xf32>
    %509 = linalg.generic {indexing_maps = [#map3, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%505, %508 : tensor<1x5x768xf32>, tensor<1x5x768xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.subf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x768xf32>
    %510 = linalg.generic {indexing_maps = [#map3, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%509, %509 : tensor<1x5x768xf32>, tensor<1x5x768xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x768xf32>
    %511 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "reduction"]} ins(%510 : tensor<1x5x768xf32>) outs(%7 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.addf %in, %out : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x1xf32>
    %512 = linalg.generic {indexing_maps = [#map6, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%511 : tensor<1x5x1xf32>) outs(%6 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.divf %in, %cst_16 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x1xf32>
    %513 = linalg.generic {indexing_maps = [#map6, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%512 : tensor<1x5x1xf32>) outs(%6 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.truncf %cst_12 : f64 to f32
      %761 = arith.addf %in, %760 : f32
      linalg.yield %761 : f32
    } -> tensor<1x5x1xf32>
    %514 = linalg.generic {indexing_maps = [#map6, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%513 : tensor<1x5x1xf32>) outs(%6 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = math.rsqrt %in : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x1xf32>
    %515 = linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%514 : tensor<1x5x1xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x5x768xf32>
    %516 = linalg.generic {indexing_maps = [#map3, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%509, %515 : tensor<1x5x768xf32>, tensor<1x5x768xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x768xf32>
    %517 = linalg.generic {indexing_maps = [#map3, #map7, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%516, %cst : tensor<1x5x768xf32>, tensor<768xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x768xf32>
    %518 = linalg.generic {indexing_maps = [#map3, #map7, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%517, %cst : tensor<1x5x768xf32>, tensor<768xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.addf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x768xf32>
    %collapsed_180 = tensor.collapse_shape %518 [[0, 1], [2]] : tensor<1x5x768xf32> into tensor<5x768xf32>
    %519 = linalg.generic {indexing_maps = [#map10, #map11, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%collapsed_180, %cst_4 : tensor<5x768xf32>, tensor<768x2304xf32>) outs(%22 : tensor<5x2304xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      %761 = arith.addf %out, %760 : f32
      linalg.yield %761 : f32
    } -> tensor<5x2304xf32>
    %520 = linalg.generic {indexing_maps = [#map12, #map9, #map9], iterator_types = ["parallel", "parallel"]} ins(%cst_5, %519 : tensor<2304xf32>, tensor<5x2304xf32>) outs(%21 : tensor<5x2304xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.addf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<5x2304xf32>
    %expanded_181 = tensor.expand_shape %520 [[0, 1], [2]] : tensor<5x2304xf32> into tensor<1x5x2304xf32>
    %extracted_slice_182 = tensor.extract_slice %expanded_181[0, 0, 0] [1, 5, 768] [1, 1, 1] : tensor<1x5x2304xf32> to tensor<1x5x768xf32>
    %extracted_slice_183 = tensor.extract_slice %expanded_181[0, 0, 768] [1, 5, 768] [1, 1, 1] : tensor<1x5x2304xf32> to tensor<1x5x768xf32>
    %extracted_slice_184 = tensor.extract_slice %expanded_181[0, 0, 1536] [1, 5, 768] [1, 1, 1] : tensor<1x5x2304xf32> to tensor<1x5x768xf32>
    %expanded_185 = tensor.expand_shape %extracted_slice_182 [[0], [1], [2, 3]] : tensor<1x5x768xf32> into tensor<1x5x12x64xf32>
    %521 = linalg.generic {indexing_maps = [#map13, #map14], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_185 : tensor<1x5x12x64xf32>) outs(%25 : tensor<1x12x5x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x12x5x64xf32>
    %expanded_186 = tensor.expand_shape %extracted_slice_183 [[0], [1], [2, 3]] : tensor<1x5x768xf32> into tensor<1x5x12x64xf32>
    %522 = linalg.generic {indexing_maps = [#map13, #map14], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_186 : tensor<1x5x12x64xf32>) outs(%25 : tensor<1x12x5x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x12x5x64xf32>
    %expanded_187 = tensor.expand_shape %extracted_slice_184 [[0], [1], [2, 3]] : tensor<1x5x768xf32> into tensor<1x5x12x64xf32>
    %523 = linalg.generic {indexing_maps = [#map13, #map14], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_187 : tensor<1x5x12x64xf32>) outs(%25 : tensor<1x12x5x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x12x5x64xf32>
    %524 = linalg.generic {indexing_maps = [#map13, #map15], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%522 : tensor<1x12x5x64xf32>) outs(%29 : tensor<1x12x64x5xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x12x64x5xf32>
    %collapsed_188 = tensor.collapse_shape %521 [[0, 1], [2], [3]] : tensor<1x12x5x64xf32> into tensor<12x5x64xf32>
    %collapsed_189 = tensor.collapse_shape %524 [[0, 1], [2], [3]] : tensor<1x12x64x5xf32> into tensor<12x64x5xf32>
    %525 = linalg.generic {indexing_maps = [#map16, #map17, #map18], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%collapsed_188, %collapsed_189 : tensor<12x5x64xf32>, tensor<12x64x5xf32>) outs(%32 : tensor<12x5x5xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      %761 = arith.addf %out, %760 : f32
      linalg.yield %761 : f32
    } -> tensor<12x5x5xf32>
    %expanded_190 = tensor.expand_shape %525 [[0, 1], [2], [3]] : tensor<12x5x5xf32> into tensor<1x12x5x5xf32>
    %526 = linalg.generic {indexing_maps = [#map20, #map21, #map13], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_190, %37 : tensor<1x12x5x5xf32>, tensor<f32>) outs(%38 : tensor<1x12x5x5xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.divf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x12x5x5xf32>
    %527 = linalg.generic {indexing_maps = [#map22, #map20, #map21, #map13], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_30, %526, %41 : tensor<1x1x5x5xi1>, tensor<1x12x5x5xf32>, tensor<f32>) outs(%38 : tensor<1x12x5x5xf32>) {
    ^bb0(%in: i1, %in_262: f32, %in_263: f32, %out: f32):
      %760 = arith.select %in, %in_262, %in_263 : f32
      linalg.yield %760 : f32
    } -> tensor<1x12x5x5xf32>
    %528:2 = linalg.generic {indexing_maps = [#map13, #map23, #map23], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%527 : tensor<1x12x5x5xf32>) outs(%46, %44 : tensor<1x12x5x1xf32>, tensor<1x12x5x1xi64>) {
    ^bb0(%in: f32, %out: f32, %out_262: i64):
      %760 = linalg.index 3 : index
      %761 = arith.index_cast %760 : index to i64
      %762 = arith.maximumf %in, %out : f32
      %763 = arith.cmpf ogt, %in, %out : f32
      %764 = arith.select %763, %761, %out_262 : i64
      linalg.yield %762, %764 : f32, i64
    } -> (tensor<1x12x5x1xf32>, tensor<1x12x5x1xi64>)
    %529 = linalg.generic {indexing_maps = [#map20, #map24, #map13], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%527, %528#0 : tensor<1x12x5x5xf32>, tensor<1x12x5x1xf32>) outs(%38 : tensor<1x12x5x5xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.subf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x12x5x5xf32>
    %530 = linalg.generic {indexing_maps = [#map20, #map13], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%529 : tensor<1x12x5x5xf32>) outs(%38 : tensor<1x12x5x5xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = math.exp %in : f32
      linalg.yield %760 : f32
    } -> tensor<1x12x5x5xf32>
    %531 = linalg.generic {indexing_maps = [#map13, #map23], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%530 : tensor<1x12x5x5xf32>) outs(%50 : tensor<1x12x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.addf %in, %out : f32
      linalg.yield %760 : f32
    } -> tensor<1x12x5x1xf32>
    %532 = linalg.generic {indexing_maps = [#map20, #map24, #map13], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%530, %531 : tensor<1x12x5x5xf32>, tensor<1x12x5x1xf32>) outs(%38 : tensor<1x12x5x5xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.divf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x12x5x5xf32>
    %collapsed_191 = tensor.collapse_shape %532 [[0, 1], [2], [3]] : tensor<1x12x5x5xf32> into tensor<12x5x5xf32>
    %collapsed_192 = tensor.collapse_shape %523 [[0, 1], [2], [3]] : tensor<1x12x5x64xf32> into tensor<12x5x64xf32>
    %533 = linalg.generic {indexing_maps = [#map16, #map17, #map18], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%collapsed_191, %collapsed_192 : tensor<12x5x5xf32>, tensor<12x5x64xf32>) outs(%54 : tensor<12x5x64xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      %761 = arith.addf %out, %760 : f32
      linalg.yield %761 : f32
    } -> tensor<12x5x64xf32>
    %expanded_193 = tensor.expand_shape %533 [[0, 1], [2], [3]] : tensor<12x5x64xf32> into tensor<1x12x5x64xf32>
    %534 = linalg.generic {indexing_maps = [#map13, #map14], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_193 : tensor<1x12x5x64xf32>) outs(%56 : tensor<1x5x12x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x5x12x64xf32>
    %535 = linalg.generic {indexing_maps = [#map20, #map13], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%534 : tensor<1x5x12x64xf32>) outs(%56 : tensor<1x5x12x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x5x12x64xf32>
    %collapsed_194 = tensor.collapse_shape %535 [[0, 1], [2, 3]] : tensor<1x5x12x64xf32> into tensor<5x768xf32>
    %536 = linalg.generic {indexing_maps = [#map10, #map11, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%collapsed_194, %cst_3 : tensor<5x768xf32>, tensor<768x768xf32>) outs(%60 : tensor<5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      %761 = arith.addf %out, %760 : f32
      linalg.yield %761 : f32
    } -> tensor<5x768xf32>
    %537 = linalg.generic {indexing_maps = [#map12, #map9, #map9], iterator_types = ["parallel", "parallel"]} ins(%cst, %536 : tensor<768xf32>, tensor<5x768xf32>) outs(%59 : tensor<5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.addf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<5x768xf32>
    %expanded_195 = tensor.expand_shape %537 [[0, 1], [2]] : tensor<5x768xf32> into tensor<1x5x768xf32>
    %538 = linalg.generic {indexing_maps = [#map3, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_195, %505 : tensor<1x5x768xf32>, tensor<1x5x768xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.addf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x768xf32>
    %539 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "reduction"]} ins(%538 : tensor<1x5x768xf32>) outs(%7 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.addf %in, %out : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x1xf32>
    %540 = linalg.generic {indexing_maps = [#map6, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%539 : tensor<1x5x1xf32>) outs(%6 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.divf %in, %cst_16 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x1xf32>
    %541 = linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%540 : tensor<1x5x1xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x5x768xf32>
    %542 = linalg.generic {indexing_maps = [#map3, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%538, %541 : tensor<1x5x768xf32>, tensor<1x5x768xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.subf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x768xf32>
    %543 = linalg.generic {indexing_maps = [#map3, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%542, %542 : tensor<1x5x768xf32>, tensor<1x5x768xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x768xf32>
    %544 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "reduction"]} ins(%543 : tensor<1x5x768xf32>) outs(%7 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.addf %in, %out : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x1xf32>
    %545 = linalg.generic {indexing_maps = [#map6, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%544 : tensor<1x5x1xf32>) outs(%6 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.divf %in, %cst_16 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x1xf32>
    %546 = linalg.generic {indexing_maps = [#map6, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%545 : tensor<1x5x1xf32>) outs(%6 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.truncf %cst_12 : f64 to f32
      %761 = arith.addf %in, %760 : f32
      linalg.yield %761 : f32
    } -> tensor<1x5x1xf32>
    %547 = linalg.generic {indexing_maps = [#map6, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%546 : tensor<1x5x1xf32>) outs(%6 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = math.rsqrt %in : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x1xf32>
    %548 = linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%547 : tensor<1x5x1xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x5x768xf32>
    %549 = linalg.generic {indexing_maps = [#map3, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%542, %548 : tensor<1x5x768xf32>, tensor<1x5x768xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x768xf32>
    %550 = linalg.generic {indexing_maps = [#map3, #map7, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%549, %cst : tensor<1x5x768xf32>, tensor<768xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x768xf32>
    %551 = linalg.generic {indexing_maps = [#map3, #map7, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%550, %cst : tensor<1x5x768xf32>, tensor<768xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.addf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x768xf32>
    %collapsed_196 = tensor.collapse_shape %551 [[0, 1], [2]] : tensor<1x5x768xf32> into tensor<5x768xf32>
    %552 = linalg.generic {indexing_maps = [#map10, #map11, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%collapsed_196, %cst_1 : tensor<5x768xf32>, tensor<768x3072xf32>) outs(%78 : tensor<5x3072xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      %761 = arith.addf %out, %760 : f32
      linalg.yield %761 : f32
    } -> tensor<5x3072xf32>
    %553 = linalg.generic {indexing_maps = [#map12, #map9, #map9], iterator_types = ["parallel", "parallel"]} ins(%cst_2, %552 : tensor<3072xf32>, tensor<5x3072xf32>) outs(%77 : tensor<5x3072xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.addf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<5x3072xf32>
    %expanded_197 = tensor.expand_shape %553 [[0, 1], [2]] : tensor<5x3072xf32> into tensor<1x5x3072xf32>
    %554 = linalg.generic {indexing_maps = [#map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_197 : tensor<1x5x3072xf32>) outs(%81 : tensor<1x5x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.mulf %in, %cst_17 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x3072xf32>
    %555 = linalg.generic {indexing_maps = [#map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_197 : tensor<1x5x3072xf32>) outs(%81 : tensor<1x5x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = math.powf %in, %cst_18 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x3072xf32>
    %556 = linalg.generic {indexing_maps = [#map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%555 : tensor<1x5x3072xf32>) outs(%81 : tensor<1x5x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.truncf %cst_14 : f64 to f32
      %761 = arith.mulf %in, %760 : f32
      linalg.yield %761 : f32
    } -> tensor<1x5x3072xf32>
    %557 = linalg.generic {indexing_maps = [#map3, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_197, %556 : tensor<1x5x3072xf32>, tensor<1x5x3072xf32>) outs(%81 : tensor<1x5x3072xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.addf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x3072xf32>
    %558 = linalg.generic {indexing_maps = [#map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%557 : tensor<1x5x3072xf32>) outs(%81 : tensor<1x5x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.truncf %cst_15 : f64 to f32
      %761 = arith.mulf %in, %760 : f32
      linalg.yield %761 : f32
    } -> tensor<1x5x3072xf32>
    %559 = linalg.generic {indexing_maps = [#map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%558 : tensor<1x5x3072xf32>) outs(%81 : tensor<1x5x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = math.tanh %in : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x3072xf32>
    %560 = linalg.generic {indexing_maps = [#map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%559 : tensor<1x5x3072xf32>) outs(%81 : tensor<1x5x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.addf %in, %cst_19 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x3072xf32>
    %561 = linalg.generic {indexing_maps = [#map3, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%554, %560 : tensor<1x5x3072xf32>, tensor<1x5x3072xf32>) outs(%81 : tensor<1x5x3072xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x3072xf32>
    %collapsed_198 = tensor.collapse_shape %561 [[0, 1], [2]] : tensor<1x5x3072xf32> into tensor<5x3072xf32>
    %562 = linalg.generic {indexing_maps = [#map10, #map11, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%collapsed_198, %cst_0 : tensor<5x3072xf32>, tensor<3072x768xf32>) outs(%60 : tensor<5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      %761 = arith.addf %out, %760 : f32
      linalg.yield %761 : f32
    } -> tensor<5x768xf32>
    %563 = linalg.generic {indexing_maps = [#map12, #map9, #map9], iterator_types = ["parallel", "parallel"]} ins(%cst, %562 : tensor<768xf32>, tensor<5x768xf32>) outs(%59 : tensor<5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.addf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<5x768xf32>
    %expanded_199 = tensor.expand_shape %563 [[0, 1], [2]] : tensor<5x768xf32> into tensor<1x5x768xf32>
    %564 = linalg.generic {indexing_maps = [#map3, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%538, %expanded_199 : tensor<1x5x768xf32>, tensor<1x5x768xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.addf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x768xf32>
    %565 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "reduction"]} ins(%564 : tensor<1x5x768xf32>) outs(%7 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.addf %in, %out : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x1xf32>
    %566 = linalg.generic {indexing_maps = [#map6, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%565 : tensor<1x5x1xf32>) outs(%6 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.divf %in, %cst_16 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x1xf32>
    %567 = linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%566 : tensor<1x5x1xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x5x768xf32>
    %568 = linalg.generic {indexing_maps = [#map3, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%564, %567 : tensor<1x5x768xf32>, tensor<1x5x768xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.subf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x768xf32>
    %569 = linalg.generic {indexing_maps = [#map3, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%568, %568 : tensor<1x5x768xf32>, tensor<1x5x768xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x768xf32>
    %570 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "reduction"]} ins(%569 : tensor<1x5x768xf32>) outs(%7 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.addf %in, %out : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x1xf32>
    %571 = linalg.generic {indexing_maps = [#map6, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%570 : tensor<1x5x1xf32>) outs(%6 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.divf %in, %cst_16 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x1xf32>
    %572 = linalg.generic {indexing_maps = [#map6, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%571 : tensor<1x5x1xf32>) outs(%6 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.truncf %cst_12 : f64 to f32
      %761 = arith.addf %in, %760 : f32
      linalg.yield %761 : f32
    } -> tensor<1x5x1xf32>
    %573 = linalg.generic {indexing_maps = [#map6, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%572 : tensor<1x5x1xf32>) outs(%6 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = math.rsqrt %in : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x1xf32>
    %574 = linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%573 : tensor<1x5x1xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x5x768xf32>
    %575 = linalg.generic {indexing_maps = [#map3, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%568, %574 : tensor<1x5x768xf32>, tensor<1x5x768xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x768xf32>
    %576 = linalg.generic {indexing_maps = [#map3, #map7, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%575, %cst : tensor<1x5x768xf32>, tensor<768xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x768xf32>
    %577 = linalg.generic {indexing_maps = [#map3, #map7, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%576, %cst : tensor<1x5x768xf32>, tensor<768xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.addf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x768xf32>
    %collapsed_200 = tensor.collapse_shape %577 [[0, 1], [2]] : tensor<1x5x768xf32> into tensor<5x768xf32>
    %578 = linalg.generic {indexing_maps = [#map10, #map11, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%collapsed_200, %cst_4 : tensor<5x768xf32>, tensor<768x2304xf32>) outs(%22 : tensor<5x2304xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      %761 = arith.addf %out, %760 : f32
      linalg.yield %761 : f32
    } -> tensor<5x2304xf32>
    %579 = linalg.generic {indexing_maps = [#map12, #map9, #map9], iterator_types = ["parallel", "parallel"]} ins(%cst_5, %578 : tensor<2304xf32>, tensor<5x2304xf32>) outs(%21 : tensor<5x2304xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.addf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<5x2304xf32>
    %expanded_201 = tensor.expand_shape %579 [[0, 1], [2]] : tensor<5x2304xf32> into tensor<1x5x2304xf32>
    %extracted_slice_202 = tensor.extract_slice %expanded_201[0, 0, 0] [1, 5, 768] [1, 1, 1] : tensor<1x5x2304xf32> to tensor<1x5x768xf32>
    %extracted_slice_203 = tensor.extract_slice %expanded_201[0, 0, 768] [1, 5, 768] [1, 1, 1] : tensor<1x5x2304xf32> to tensor<1x5x768xf32>
    %extracted_slice_204 = tensor.extract_slice %expanded_201[0, 0, 1536] [1, 5, 768] [1, 1, 1] : tensor<1x5x2304xf32> to tensor<1x5x768xf32>
    %expanded_205 = tensor.expand_shape %extracted_slice_202 [[0], [1], [2, 3]] : tensor<1x5x768xf32> into tensor<1x5x12x64xf32>
    %580 = linalg.generic {indexing_maps = [#map13, #map14], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_205 : tensor<1x5x12x64xf32>) outs(%25 : tensor<1x12x5x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x12x5x64xf32>
    %expanded_206 = tensor.expand_shape %extracted_slice_203 [[0], [1], [2, 3]] : tensor<1x5x768xf32> into tensor<1x5x12x64xf32>
    %581 = linalg.generic {indexing_maps = [#map13, #map14], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_206 : tensor<1x5x12x64xf32>) outs(%25 : tensor<1x12x5x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x12x5x64xf32>
    %expanded_207 = tensor.expand_shape %extracted_slice_204 [[0], [1], [2, 3]] : tensor<1x5x768xf32> into tensor<1x5x12x64xf32>
    %582 = linalg.generic {indexing_maps = [#map13, #map14], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_207 : tensor<1x5x12x64xf32>) outs(%25 : tensor<1x12x5x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x12x5x64xf32>
    %583 = linalg.generic {indexing_maps = [#map13, #map15], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%581 : tensor<1x12x5x64xf32>) outs(%29 : tensor<1x12x64x5xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x12x64x5xf32>
    %collapsed_208 = tensor.collapse_shape %580 [[0, 1], [2], [3]] : tensor<1x12x5x64xf32> into tensor<12x5x64xf32>
    %collapsed_209 = tensor.collapse_shape %583 [[0, 1], [2], [3]] : tensor<1x12x64x5xf32> into tensor<12x64x5xf32>
    %584 = linalg.generic {indexing_maps = [#map16, #map17, #map18], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%collapsed_208, %collapsed_209 : tensor<12x5x64xf32>, tensor<12x64x5xf32>) outs(%32 : tensor<12x5x5xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      %761 = arith.addf %out, %760 : f32
      linalg.yield %761 : f32
    } -> tensor<12x5x5xf32>
    %expanded_210 = tensor.expand_shape %584 [[0, 1], [2], [3]] : tensor<12x5x5xf32> into tensor<1x12x5x5xf32>
    %585 = linalg.generic {indexing_maps = [#map20, #map21, #map13], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_210, %37 : tensor<1x12x5x5xf32>, tensor<f32>) outs(%38 : tensor<1x12x5x5xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.divf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x12x5x5xf32>
    %586 = linalg.generic {indexing_maps = [#map22, #map20, #map21, #map13], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_30, %585, %41 : tensor<1x1x5x5xi1>, tensor<1x12x5x5xf32>, tensor<f32>) outs(%38 : tensor<1x12x5x5xf32>) {
    ^bb0(%in: i1, %in_262: f32, %in_263: f32, %out: f32):
      %760 = arith.select %in, %in_262, %in_263 : f32
      linalg.yield %760 : f32
    } -> tensor<1x12x5x5xf32>
    %587:2 = linalg.generic {indexing_maps = [#map13, #map23, #map23], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%586 : tensor<1x12x5x5xf32>) outs(%46, %44 : tensor<1x12x5x1xf32>, tensor<1x12x5x1xi64>) {
    ^bb0(%in: f32, %out: f32, %out_262: i64):
      %760 = linalg.index 3 : index
      %761 = arith.index_cast %760 : index to i64
      %762 = arith.maximumf %in, %out : f32
      %763 = arith.cmpf ogt, %in, %out : f32
      %764 = arith.select %763, %761, %out_262 : i64
      linalg.yield %762, %764 : f32, i64
    } -> (tensor<1x12x5x1xf32>, tensor<1x12x5x1xi64>)
    %588 = linalg.generic {indexing_maps = [#map20, #map24, #map13], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%586, %587#0 : tensor<1x12x5x5xf32>, tensor<1x12x5x1xf32>) outs(%38 : tensor<1x12x5x5xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.subf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x12x5x5xf32>
    %589 = linalg.generic {indexing_maps = [#map20, #map13], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%588 : tensor<1x12x5x5xf32>) outs(%38 : tensor<1x12x5x5xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = math.exp %in : f32
      linalg.yield %760 : f32
    } -> tensor<1x12x5x5xf32>
    %590 = linalg.generic {indexing_maps = [#map13, #map23], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%589 : tensor<1x12x5x5xf32>) outs(%50 : tensor<1x12x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.addf %in, %out : f32
      linalg.yield %760 : f32
    } -> tensor<1x12x5x1xf32>
    %591 = linalg.generic {indexing_maps = [#map20, #map24, #map13], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%589, %590 : tensor<1x12x5x5xf32>, tensor<1x12x5x1xf32>) outs(%38 : tensor<1x12x5x5xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.divf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x12x5x5xf32>
    %collapsed_211 = tensor.collapse_shape %591 [[0, 1], [2], [3]] : tensor<1x12x5x5xf32> into tensor<12x5x5xf32>
    %collapsed_212 = tensor.collapse_shape %582 [[0, 1], [2], [3]] : tensor<1x12x5x64xf32> into tensor<12x5x64xf32>
    %592 = linalg.generic {indexing_maps = [#map16, #map17, #map18], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%collapsed_211, %collapsed_212 : tensor<12x5x5xf32>, tensor<12x5x64xf32>) outs(%54 : tensor<12x5x64xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      %761 = arith.addf %out, %760 : f32
      linalg.yield %761 : f32
    } -> tensor<12x5x64xf32>
    %expanded_213 = tensor.expand_shape %592 [[0, 1], [2], [3]] : tensor<12x5x64xf32> into tensor<1x12x5x64xf32>
    %593 = linalg.generic {indexing_maps = [#map13, #map14], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_213 : tensor<1x12x5x64xf32>) outs(%56 : tensor<1x5x12x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x5x12x64xf32>
    %594 = linalg.generic {indexing_maps = [#map20, #map13], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%593 : tensor<1x5x12x64xf32>) outs(%56 : tensor<1x5x12x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x5x12x64xf32>
    %collapsed_214 = tensor.collapse_shape %594 [[0, 1], [2, 3]] : tensor<1x5x12x64xf32> into tensor<5x768xf32>
    %595 = linalg.generic {indexing_maps = [#map10, #map11, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%collapsed_214, %cst_3 : tensor<5x768xf32>, tensor<768x768xf32>) outs(%60 : tensor<5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      %761 = arith.addf %out, %760 : f32
      linalg.yield %761 : f32
    } -> tensor<5x768xf32>
    %596 = linalg.generic {indexing_maps = [#map12, #map9, #map9], iterator_types = ["parallel", "parallel"]} ins(%cst, %595 : tensor<768xf32>, tensor<5x768xf32>) outs(%59 : tensor<5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.addf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<5x768xf32>
    %expanded_215 = tensor.expand_shape %596 [[0, 1], [2]] : tensor<5x768xf32> into tensor<1x5x768xf32>
    %597 = linalg.generic {indexing_maps = [#map3, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_215, %564 : tensor<1x5x768xf32>, tensor<1x5x768xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.addf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x768xf32>
    %598 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "reduction"]} ins(%597 : tensor<1x5x768xf32>) outs(%7 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.addf %in, %out : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x1xf32>
    %599 = linalg.generic {indexing_maps = [#map6, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%598 : tensor<1x5x1xf32>) outs(%6 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.divf %in, %cst_16 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x1xf32>
    %600 = linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%599 : tensor<1x5x1xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x5x768xf32>
    %601 = linalg.generic {indexing_maps = [#map3, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%597, %600 : tensor<1x5x768xf32>, tensor<1x5x768xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.subf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x768xf32>
    %602 = linalg.generic {indexing_maps = [#map3, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%601, %601 : tensor<1x5x768xf32>, tensor<1x5x768xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x768xf32>
    %603 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "reduction"]} ins(%602 : tensor<1x5x768xf32>) outs(%7 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.addf %in, %out : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x1xf32>
    %604 = linalg.generic {indexing_maps = [#map6, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%603 : tensor<1x5x1xf32>) outs(%6 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.divf %in, %cst_16 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x1xf32>
    %605 = linalg.generic {indexing_maps = [#map6, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%604 : tensor<1x5x1xf32>) outs(%6 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.truncf %cst_12 : f64 to f32
      %761 = arith.addf %in, %760 : f32
      linalg.yield %761 : f32
    } -> tensor<1x5x1xf32>
    %606 = linalg.generic {indexing_maps = [#map6, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%605 : tensor<1x5x1xf32>) outs(%6 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = math.rsqrt %in : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x1xf32>
    %607 = linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%606 : tensor<1x5x1xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x5x768xf32>
    %608 = linalg.generic {indexing_maps = [#map3, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%601, %607 : tensor<1x5x768xf32>, tensor<1x5x768xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x768xf32>
    %609 = linalg.generic {indexing_maps = [#map3, #map7, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%608, %cst : tensor<1x5x768xf32>, tensor<768xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x768xf32>
    %610 = linalg.generic {indexing_maps = [#map3, #map7, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%609, %cst : tensor<1x5x768xf32>, tensor<768xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.addf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x768xf32>
    %collapsed_216 = tensor.collapse_shape %610 [[0, 1], [2]] : tensor<1x5x768xf32> into tensor<5x768xf32>
    %611 = linalg.generic {indexing_maps = [#map10, #map11, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%collapsed_216, %cst_1 : tensor<5x768xf32>, tensor<768x3072xf32>) outs(%78 : tensor<5x3072xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      %761 = arith.addf %out, %760 : f32
      linalg.yield %761 : f32
    } -> tensor<5x3072xf32>
    %612 = linalg.generic {indexing_maps = [#map12, #map9, #map9], iterator_types = ["parallel", "parallel"]} ins(%cst_2, %611 : tensor<3072xf32>, tensor<5x3072xf32>) outs(%77 : tensor<5x3072xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.addf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<5x3072xf32>
    %expanded_217 = tensor.expand_shape %612 [[0, 1], [2]] : tensor<5x3072xf32> into tensor<1x5x3072xf32>
    %613 = linalg.generic {indexing_maps = [#map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_217 : tensor<1x5x3072xf32>) outs(%81 : tensor<1x5x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.mulf %in, %cst_17 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x3072xf32>
    %614 = linalg.generic {indexing_maps = [#map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_217 : tensor<1x5x3072xf32>) outs(%81 : tensor<1x5x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = math.powf %in, %cst_18 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x3072xf32>
    %615 = linalg.generic {indexing_maps = [#map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%614 : tensor<1x5x3072xf32>) outs(%81 : tensor<1x5x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.truncf %cst_14 : f64 to f32
      %761 = arith.mulf %in, %760 : f32
      linalg.yield %761 : f32
    } -> tensor<1x5x3072xf32>
    %616 = linalg.generic {indexing_maps = [#map3, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_217, %615 : tensor<1x5x3072xf32>, tensor<1x5x3072xf32>) outs(%81 : tensor<1x5x3072xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.addf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x3072xf32>
    %617 = linalg.generic {indexing_maps = [#map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%616 : tensor<1x5x3072xf32>) outs(%81 : tensor<1x5x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.truncf %cst_15 : f64 to f32
      %761 = arith.mulf %in, %760 : f32
      linalg.yield %761 : f32
    } -> tensor<1x5x3072xf32>
    %618 = linalg.generic {indexing_maps = [#map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%617 : tensor<1x5x3072xf32>) outs(%81 : tensor<1x5x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = math.tanh %in : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x3072xf32>
    %619 = linalg.generic {indexing_maps = [#map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%618 : tensor<1x5x3072xf32>) outs(%81 : tensor<1x5x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.addf %in, %cst_19 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x3072xf32>
    %620 = linalg.generic {indexing_maps = [#map3, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%613, %619 : tensor<1x5x3072xf32>, tensor<1x5x3072xf32>) outs(%81 : tensor<1x5x3072xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x3072xf32>
    %collapsed_218 = tensor.collapse_shape %620 [[0, 1], [2]] : tensor<1x5x3072xf32> into tensor<5x3072xf32>
    %621 = linalg.generic {indexing_maps = [#map10, #map11, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%collapsed_218, %cst_0 : tensor<5x3072xf32>, tensor<3072x768xf32>) outs(%60 : tensor<5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      %761 = arith.addf %out, %760 : f32
      linalg.yield %761 : f32
    } -> tensor<5x768xf32>
    %622 = linalg.generic {indexing_maps = [#map12, #map9, #map9], iterator_types = ["parallel", "parallel"]} ins(%cst, %621 : tensor<768xf32>, tensor<5x768xf32>) outs(%59 : tensor<5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.addf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<5x768xf32>
    %expanded_219 = tensor.expand_shape %622 [[0, 1], [2]] : tensor<5x768xf32> into tensor<1x5x768xf32>
    %623 = linalg.generic {indexing_maps = [#map3, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%597, %expanded_219 : tensor<1x5x768xf32>, tensor<1x5x768xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.addf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x768xf32>
    %624 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "reduction"]} ins(%623 : tensor<1x5x768xf32>) outs(%7 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.addf %in, %out : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x1xf32>
    %625 = linalg.generic {indexing_maps = [#map6, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%624 : tensor<1x5x1xf32>) outs(%6 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.divf %in, %cst_16 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x1xf32>
    %626 = linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%625 : tensor<1x5x1xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x5x768xf32>
    %627 = linalg.generic {indexing_maps = [#map3, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%623, %626 : tensor<1x5x768xf32>, tensor<1x5x768xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.subf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x768xf32>
    %628 = linalg.generic {indexing_maps = [#map3, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%627, %627 : tensor<1x5x768xf32>, tensor<1x5x768xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x768xf32>
    %629 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "reduction"]} ins(%628 : tensor<1x5x768xf32>) outs(%7 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.addf %in, %out : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x1xf32>
    %630 = linalg.generic {indexing_maps = [#map6, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%629 : tensor<1x5x1xf32>) outs(%6 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.divf %in, %cst_16 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x1xf32>
    %631 = linalg.generic {indexing_maps = [#map6, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%630 : tensor<1x5x1xf32>) outs(%6 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.truncf %cst_12 : f64 to f32
      %761 = arith.addf %in, %760 : f32
      linalg.yield %761 : f32
    } -> tensor<1x5x1xf32>
    %632 = linalg.generic {indexing_maps = [#map6, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%631 : tensor<1x5x1xf32>) outs(%6 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = math.rsqrt %in : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x1xf32>
    %633 = linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%632 : tensor<1x5x1xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x5x768xf32>
    %634 = linalg.generic {indexing_maps = [#map3, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%627, %633 : tensor<1x5x768xf32>, tensor<1x5x768xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x768xf32>
    %635 = linalg.generic {indexing_maps = [#map3, #map7, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%634, %cst : tensor<1x5x768xf32>, tensor<768xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x768xf32>
    %636 = linalg.generic {indexing_maps = [#map3, #map7, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%635, %cst : tensor<1x5x768xf32>, tensor<768xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.addf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x768xf32>
    %collapsed_220 = tensor.collapse_shape %636 [[0, 1], [2]] : tensor<1x5x768xf32> into tensor<5x768xf32>
    %637 = linalg.generic {indexing_maps = [#map10, #map11, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%collapsed_220, %cst_4 : tensor<5x768xf32>, tensor<768x2304xf32>) outs(%22 : tensor<5x2304xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      %761 = arith.addf %out, %760 : f32
      linalg.yield %761 : f32
    } -> tensor<5x2304xf32>
    %638 = linalg.generic {indexing_maps = [#map12, #map9, #map9], iterator_types = ["parallel", "parallel"]} ins(%cst_5, %637 : tensor<2304xf32>, tensor<5x2304xf32>) outs(%21 : tensor<5x2304xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.addf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<5x2304xf32>
    %expanded_221 = tensor.expand_shape %638 [[0, 1], [2]] : tensor<5x2304xf32> into tensor<1x5x2304xf32>
    %extracted_slice_222 = tensor.extract_slice %expanded_221[0, 0, 0] [1, 5, 768] [1, 1, 1] : tensor<1x5x2304xf32> to tensor<1x5x768xf32>
    %extracted_slice_223 = tensor.extract_slice %expanded_221[0, 0, 768] [1, 5, 768] [1, 1, 1] : tensor<1x5x2304xf32> to tensor<1x5x768xf32>
    %extracted_slice_224 = tensor.extract_slice %expanded_221[0, 0, 1536] [1, 5, 768] [1, 1, 1] : tensor<1x5x2304xf32> to tensor<1x5x768xf32>
    %expanded_225 = tensor.expand_shape %extracted_slice_222 [[0], [1], [2, 3]] : tensor<1x5x768xf32> into tensor<1x5x12x64xf32>
    %639 = linalg.generic {indexing_maps = [#map13, #map14], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_225 : tensor<1x5x12x64xf32>) outs(%25 : tensor<1x12x5x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x12x5x64xf32>
    %expanded_226 = tensor.expand_shape %extracted_slice_223 [[0], [1], [2, 3]] : tensor<1x5x768xf32> into tensor<1x5x12x64xf32>
    %640 = linalg.generic {indexing_maps = [#map13, #map14], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_226 : tensor<1x5x12x64xf32>) outs(%25 : tensor<1x12x5x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x12x5x64xf32>
    %expanded_227 = tensor.expand_shape %extracted_slice_224 [[0], [1], [2, 3]] : tensor<1x5x768xf32> into tensor<1x5x12x64xf32>
    %641 = linalg.generic {indexing_maps = [#map13, #map14], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_227 : tensor<1x5x12x64xf32>) outs(%25 : tensor<1x12x5x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x12x5x64xf32>
    %642 = linalg.generic {indexing_maps = [#map13, #map15], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%640 : tensor<1x12x5x64xf32>) outs(%29 : tensor<1x12x64x5xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x12x64x5xf32>
    %collapsed_228 = tensor.collapse_shape %639 [[0, 1], [2], [3]] : tensor<1x12x5x64xf32> into tensor<12x5x64xf32>
    %collapsed_229 = tensor.collapse_shape %642 [[0, 1], [2], [3]] : tensor<1x12x64x5xf32> into tensor<12x64x5xf32>
    %643 = linalg.generic {indexing_maps = [#map16, #map17, #map18], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%collapsed_228, %collapsed_229 : tensor<12x5x64xf32>, tensor<12x64x5xf32>) outs(%32 : tensor<12x5x5xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      %761 = arith.addf %out, %760 : f32
      linalg.yield %761 : f32
    } -> tensor<12x5x5xf32>
    %expanded_230 = tensor.expand_shape %643 [[0, 1], [2], [3]] : tensor<12x5x5xf32> into tensor<1x12x5x5xf32>
    %644 = linalg.generic {indexing_maps = [#map20, #map21, #map13], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_230, %37 : tensor<1x12x5x5xf32>, tensor<f32>) outs(%38 : tensor<1x12x5x5xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.divf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x12x5x5xf32>
    %645 = linalg.generic {indexing_maps = [#map22, #map20, #map21, #map13], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_30, %644, %41 : tensor<1x1x5x5xi1>, tensor<1x12x5x5xf32>, tensor<f32>) outs(%38 : tensor<1x12x5x5xf32>) {
    ^bb0(%in: i1, %in_262: f32, %in_263: f32, %out: f32):
      %760 = arith.select %in, %in_262, %in_263 : f32
      linalg.yield %760 : f32
    } -> tensor<1x12x5x5xf32>
    %646:2 = linalg.generic {indexing_maps = [#map13, #map23, #map23], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%645 : tensor<1x12x5x5xf32>) outs(%46, %44 : tensor<1x12x5x1xf32>, tensor<1x12x5x1xi64>) {
    ^bb0(%in: f32, %out: f32, %out_262: i64):
      %760 = linalg.index 3 : index
      %761 = arith.index_cast %760 : index to i64
      %762 = arith.maximumf %in, %out : f32
      %763 = arith.cmpf ogt, %in, %out : f32
      %764 = arith.select %763, %761, %out_262 : i64
      linalg.yield %762, %764 : f32, i64
    } -> (tensor<1x12x5x1xf32>, tensor<1x12x5x1xi64>)
    %647 = linalg.generic {indexing_maps = [#map20, #map24, #map13], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%645, %646#0 : tensor<1x12x5x5xf32>, tensor<1x12x5x1xf32>) outs(%38 : tensor<1x12x5x5xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.subf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x12x5x5xf32>
    %648 = linalg.generic {indexing_maps = [#map20, #map13], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%647 : tensor<1x12x5x5xf32>) outs(%38 : tensor<1x12x5x5xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = math.exp %in : f32
      linalg.yield %760 : f32
    } -> tensor<1x12x5x5xf32>
    %649 = linalg.generic {indexing_maps = [#map13, #map23], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%648 : tensor<1x12x5x5xf32>) outs(%50 : tensor<1x12x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.addf %in, %out : f32
      linalg.yield %760 : f32
    } -> tensor<1x12x5x1xf32>
    %650 = linalg.generic {indexing_maps = [#map20, #map24, #map13], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%648, %649 : tensor<1x12x5x5xf32>, tensor<1x12x5x1xf32>) outs(%38 : tensor<1x12x5x5xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.divf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x12x5x5xf32>
    %collapsed_231 = tensor.collapse_shape %650 [[0, 1], [2], [3]] : tensor<1x12x5x5xf32> into tensor<12x5x5xf32>
    %collapsed_232 = tensor.collapse_shape %641 [[0, 1], [2], [3]] : tensor<1x12x5x64xf32> into tensor<12x5x64xf32>
    %651 = linalg.generic {indexing_maps = [#map16, #map17, #map18], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%collapsed_231, %collapsed_232 : tensor<12x5x5xf32>, tensor<12x5x64xf32>) outs(%54 : tensor<12x5x64xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      %761 = arith.addf %out, %760 : f32
      linalg.yield %761 : f32
    } -> tensor<12x5x64xf32>
    %expanded_233 = tensor.expand_shape %651 [[0, 1], [2], [3]] : tensor<12x5x64xf32> into tensor<1x12x5x64xf32>
    %652 = linalg.generic {indexing_maps = [#map13, #map14], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_233 : tensor<1x12x5x64xf32>) outs(%56 : tensor<1x5x12x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x5x12x64xf32>
    %653 = linalg.generic {indexing_maps = [#map20, #map13], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%652 : tensor<1x5x12x64xf32>) outs(%56 : tensor<1x5x12x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x5x12x64xf32>
    %collapsed_234 = tensor.collapse_shape %653 [[0, 1], [2, 3]] : tensor<1x5x12x64xf32> into tensor<5x768xf32>
    %654 = linalg.generic {indexing_maps = [#map10, #map11, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%collapsed_234, %cst_3 : tensor<5x768xf32>, tensor<768x768xf32>) outs(%60 : tensor<5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      %761 = arith.addf %out, %760 : f32
      linalg.yield %761 : f32
    } -> tensor<5x768xf32>
    %655 = linalg.generic {indexing_maps = [#map12, #map9, #map9], iterator_types = ["parallel", "parallel"]} ins(%cst, %654 : tensor<768xf32>, tensor<5x768xf32>) outs(%59 : tensor<5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.addf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<5x768xf32>
    %expanded_235 = tensor.expand_shape %655 [[0, 1], [2]] : tensor<5x768xf32> into tensor<1x5x768xf32>
    %656 = linalg.generic {indexing_maps = [#map3, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_235, %623 : tensor<1x5x768xf32>, tensor<1x5x768xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.addf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x768xf32>
    %657 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "reduction"]} ins(%656 : tensor<1x5x768xf32>) outs(%7 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.addf %in, %out : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x1xf32>
    %658 = linalg.generic {indexing_maps = [#map6, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%657 : tensor<1x5x1xf32>) outs(%6 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.divf %in, %cst_16 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x1xf32>
    %659 = linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%658 : tensor<1x5x1xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x5x768xf32>
    %660 = linalg.generic {indexing_maps = [#map3, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%656, %659 : tensor<1x5x768xf32>, tensor<1x5x768xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.subf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x768xf32>
    %661 = linalg.generic {indexing_maps = [#map3, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%660, %660 : tensor<1x5x768xf32>, tensor<1x5x768xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x768xf32>
    %662 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "reduction"]} ins(%661 : tensor<1x5x768xf32>) outs(%7 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.addf %in, %out : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x1xf32>
    %663 = linalg.generic {indexing_maps = [#map6, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%662 : tensor<1x5x1xf32>) outs(%6 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.divf %in, %cst_16 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x1xf32>
    %664 = linalg.generic {indexing_maps = [#map6, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%663 : tensor<1x5x1xf32>) outs(%6 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.truncf %cst_12 : f64 to f32
      %761 = arith.addf %in, %760 : f32
      linalg.yield %761 : f32
    } -> tensor<1x5x1xf32>
    %665 = linalg.generic {indexing_maps = [#map6, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%664 : tensor<1x5x1xf32>) outs(%6 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = math.rsqrt %in : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x1xf32>
    %666 = linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%665 : tensor<1x5x1xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x5x768xf32>
    %667 = linalg.generic {indexing_maps = [#map3, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%660, %666 : tensor<1x5x768xf32>, tensor<1x5x768xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x768xf32>
    %668 = linalg.generic {indexing_maps = [#map3, #map7, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%667, %cst : tensor<1x5x768xf32>, tensor<768xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x768xf32>
    %669 = linalg.generic {indexing_maps = [#map3, #map7, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%668, %cst : tensor<1x5x768xf32>, tensor<768xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.addf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x768xf32>
    %collapsed_236 = tensor.collapse_shape %669 [[0, 1], [2]] : tensor<1x5x768xf32> into tensor<5x768xf32>
    %670 = linalg.generic {indexing_maps = [#map10, #map11, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%collapsed_236, %cst_1 : tensor<5x768xf32>, tensor<768x3072xf32>) outs(%78 : tensor<5x3072xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      %761 = arith.addf %out, %760 : f32
      linalg.yield %761 : f32
    } -> tensor<5x3072xf32>
    %671 = linalg.generic {indexing_maps = [#map12, #map9, #map9], iterator_types = ["parallel", "parallel"]} ins(%cst_2, %670 : tensor<3072xf32>, tensor<5x3072xf32>) outs(%77 : tensor<5x3072xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.addf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<5x3072xf32>
    %expanded_237 = tensor.expand_shape %671 [[0, 1], [2]] : tensor<5x3072xf32> into tensor<1x5x3072xf32>
    %672 = linalg.generic {indexing_maps = [#map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_237 : tensor<1x5x3072xf32>) outs(%81 : tensor<1x5x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.mulf %in, %cst_17 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x3072xf32>
    %673 = linalg.generic {indexing_maps = [#map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_237 : tensor<1x5x3072xf32>) outs(%81 : tensor<1x5x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = math.powf %in, %cst_18 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x3072xf32>
    %674 = linalg.generic {indexing_maps = [#map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%673 : tensor<1x5x3072xf32>) outs(%81 : tensor<1x5x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.truncf %cst_14 : f64 to f32
      %761 = arith.mulf %in, %760 : f32
      linalg.yield %761 : f32
    } -> tensor<1x5x3072xf32>
    %675 = linalg.generic {indexing_maps = [#map3, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_237, %674 : tensor<1x5x3072xf32>, tensor<1x5x3072xf32>) outs(%81 : tensor<1x5x3072xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.addf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x3072xf32>
    %676 = linalg.generic {indexing_maps = [#map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%675 : tensor<1x5x3072xf32>) outs(%81 : tensor<1x5x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.truncf %cst_15 : f64 to f32
      %761 = arith.mulf %in, %760 : f32
      linalg.yield %761 : f32
    } -> tensor<1x5x3072xf32>
    %677 = linalg.generic {indexing_maps = [#map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%676 : tensor<1x5x3072xf32>) outs(%81 : tensor<1x5x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = math.tanh %in : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x3072xf32>
    %678 = linalg.generic {indexing_maps = [#map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%677 : tensor<1x5x3072xf32>) outs(%81 : tensor<1x5x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.addf %in, %cst_19 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x3072xf32>
    %679 = linalg.generic {indexing_maps = [#map3, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%672, %678 : tensor<1x5x3072xf32>, tensor<1x5x3072xf32>) outs(%81 : tensor<1x5x3072xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x3072xf32>
    %collapsed_238 = tensor.collapse_shape %679 [[0, 1], [2]] : tensor<1x5x3072xf32> into tensor<5x3072xf32>
    %680 = linalg.generic {indexing_maps = [#map10, #map11, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%collapsed_238, %cst_0 : tensor<5x3072xf32>, tensor<3072x768xf32>) outs(%60 : tensor<5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      %761 = arith.addf %out, %760 : f32
      linalg.yield %761 : f32
    } -> tensor<5x768xf32>
    %681 = linalg.generic {indexing_maps = [#map12, #map9, #map9], iterator_types = ["parallel", "parallel"]} ins(%cst, %680 : tensor<768xf32>, tensor<5x768xf32>) outs(%59 : tensor<5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.addf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<5x768xf32>
    %expanded_239 = tensor.expand_shape %681 [[0, 1], [2]] : tensor<5x768xf32> into tensor<1x5x768xf32>
    %682 = linalg.generic {indexing_maps = [#map3, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%656, %expanded_239 : tensor<1x5x768xf32>, tensor<1x5x768xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.addf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x768xf32>
    %683 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "reduction"]} ins(%682 : tensor<1x5x768xf32>) outs(%7 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.addf %in, %out : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x1xf32>
    %684 = linalg.generic {indexing_maps = [#map6, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%683 : tensor<1x5x1xf32>) outs(%6 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.divf %in, %cst_16 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x1xf32>
    %685 = linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%684 : tensor<1x5x1xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x5x768xf32>
    %686 = linalg.generic {indexing_maps = [#map3, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%682, %685 : tensor<1x5x768xf32>, tensor<1x5x768xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.subf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x768xf32>
    %687 = linalg.generic {indexing_maps = [#map3, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%686, %686 : tensor<1x5x768xf32>, tensor<1x5x768xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x768xf32>
    %688 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "reduction"]} ins(%687 : tensor<1x5x768xf32>) outs(%7 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.addf %in, %out : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x1xf32>
    %689 = linalg.generic {indexing_maps = [#map6, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%688 : tensor<1x5x1xf32>) outs(%6 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.divf %in, %cst_16 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x1xf32>
    %690 = linalg.generic {indexing_maps = [#map6, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%689 : tensor<1x5x1xf32>) outs(%6 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.truncf %cst_12 : f64 to f32
      %761 = arith.addf %in, %760 : f32
      linalg.yield %761 : f32
    } -> tensor<1x5x1xf32>
    %691 = linalg.generic {indexing_maps = [#map6, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%690 : tensor<1x5x1xf32>) outs(%6 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = math.rsqrt %in : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x1xf32>
    %692 = linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%691 : tensor<1x5x1xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x5x768xf32>
    %693 = linalg.generic {indexing_maps = [#map3, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%686, %692 : tensor<1x5x768xf32>, tensor<1x5x768xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x768xf32>
    %694 = linalg.generic {indexing_maps = [#map3, #map7, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%693, %cst : tensor<1x5x768xf32>, tensor<768xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x768xf32>
    %695 = linalg.generic {indexing_maps = [#map3, #map7, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%694, %cst : tensor<1x5x768xf32>, tensor<768xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.addf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x768xf32>
    %collapsed_240 = tensor.collapse_shape %695 [[0, 1], [2]] : tensor<1x5x768xf32> into tensor<5x768xf32>
    %696 = linalg.generic {indexing_maps = [#map10, #map11, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%collapsed_240, %cst_4 : tensor<5x768xf32>, tensor<768x2304xf32>) outs(%22 : tensor<5x2304xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      %761 = arith.addf %out, %760 : f32
      linalg.yield %761 : f32
    } -> tensor<5x2304xf32>
    %697 = linalg.generic {indexing_maps = [#map12, #map9, #map9], iterator_types = ["parallel", "parallel"]} ins(%cst_5, %696 : tensor<2304xf32>, tensor<5x2304xf32>) outs(%21 : tensor<5x2304xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.addf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<5x2304xf32>
    %expanded_241 = tensor.expand_shape %697 [[0, 1], [2]] : tensor<5x2304xf32> into tensor<1x5x2304xf32>
    %extracted_slice_242 = tensor.extract_slice %expanded_241[0, 0, 0] [1, 5, 768] [1, 1, 1] : tensor<1x5x2304xf32> to tensor<1x5x768xf32>
    %extracted_slice_243 = tensor.extract_slice %expanded_241[0, 0, 768] [1, 5, 768] [1, 1, 1] : tensor<1x5x2304xf32> to tensor<1x5x768xf32>
    %extracted_slice_244 = tensor.extract_slice %expanded_241[0, 0, 1536] [1, 5, 768] [1, 1, 1] : tensor<1x5x2304xf32> to tensor<1x5x768xf32>
    %expanded_245 = tensor.expand_shape %extracted_slice_242 [[0], [1], [2, 3]] : tensor<1x5x768xf32> into tensor<1x5x12x64xf32>
    %698 = linalg.generic {indexing_maps = [#map13, #map14], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_245 : tensor<1x5x12x64xf32>) outs(%25 : tensor<1x12x5x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x12x5x64xf32>
    %expanded_246 = tensor.expand_shape %extracted_slice_243 [[0], [1], [2, 3]] : tensor<1x5x768xf32> into tensor<1x5x12x64xf32>
    %699 = linalg.generic {indexing_maps = [#map13, #map14], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_246 : tensor<1x5x12x64xf32>) outs(%25 : tensor<1x12x5x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x12x5x64xf32>
    %expanded_247 = tensor.expand_shape %extracted_slice_244 [[0], [1], [2, 3]] : tensor<1x5x768xf32> into tensor<1x5x12x64xf32>
    %700 = linalg.generic {indexing_maps = [#map13, #map14], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_247 : tensor<1x5x12x64xf32>) outs(%25 : tensor<1x12x5x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x12x5x64xf32>
    %701 = linalg.generic {indexing_maps = [#map13, #map15], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%699 : tensor<1x12x5x64xf32>) outs(%29 : tensor<1x12x64x5xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x12x64x5xf32>
    %collapsed_248 = tensor.collapse_shape %698 [[0, 1], [2], [3]] : tensor<1x12x5x64xf32> into tensor<12x5x64xf32>
    %collapsed_249 = tensor.collapse_shape %701 [[0, 1], [2], [3]] : tensor<1x12x64x5xf32> into tensor<12x64x5xf32>
    %702 = linalg.generic {indexing_maps = [#map16, #map17, #map18], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%collapsed_248, %collapsed_249 : tensor<12x5x64xf32>, tensor<12x64x5xf32>) outs(%32 : tensor<12x5x5xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      %761 = arith.addf %out, %760 : f32
      linalg.yield %761 : f32
    } -> tensor<12x5x5xf32>
    %expanded_250 = tensor.expand_shape %702 [[0, 1], [2], [3]] : tensor<12x5x5xf32> into tensor<1x12x5x5xf32>
    %703 = linalg.generic {indexing_maps = [#map20, #map21, #map13], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_250, %37 : tensor<1x12x5x5xf32>, tensor<f32>) outs(%38 : tensor<1x12x5x5xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.divf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x12x5x5xf32>
    %704 = linalg.generic {indexing_maps = [#map22, #map20, #map21, #map13], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_30, %703, %41 : tensor<1x1x5x5xi1>, tensor<1x12x5x5xf32>, tensor<f32>) outs(%38 : tensor<1x12x5x5xf32>) {
    ^bb0(%in: i1, %in_262: f32, %in_263: f32, %out: f32):
      %760 = arith.select %in, %in_262, %in_263 : f32
      linalg.yield %760 : f32
    } -> tensor<1x12x5x5xf32>
    %705:2 = linalg.generic {indexing_maps = [#map13, #map23, #map23], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%704 : tensor<1x12x5x5xf32>) outs(%46, %44 : tensor<1x12x5x1xf32>, tensor<1x12x5x1xi64>) {
    ^bb0(%in: f32, %out: f32, %out_262: i64):
      %760 = linalg.index 3 : index
      %761 = arith.index_cast %760 : index to i64
      %762 = arith.maximumf %in, %out : f32
      %763 = arith.cmpf ogt, %in, %out : f32
      %764 = arith.select %763, %761, %out_262 : i64
      linalg.yield %762, %764 : f32, i64
    } -> (tensor<1x12x5x1xf32>, tensor<1x12x5x1xi64>)
    %706 = linalg.generic {indexing_maps = [#map20, #map24, #map13], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%704, %705#0 : tensor<1x12x5x5xf32>, tensor<1x12x5x1xf32>) outs(%38 : tensor<1x12x5x5xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.subf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x12x5x5xf32>
    %707 = linalg.generic {indexing_maps = [#map20, #map13], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%706 : tensor<1x12x5x5xf32>) outs(%38 : tensor<1x12x5x5xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = math.exp %in : f32
      linalg.yield %760 : f32
    } -> tensor<1x12x5x5xf32>
    %708 = linalg.generic {indexing_maps = [#map13, #map23], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%707 : tensor<1x12x5x5xf32>) outs(%50 : tensor<1x12x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.addf %in, %out : f32
      linalg.yield %760 : f32
    } -> tensor<1x12x5x1xf32>
    %709 = linalg.generic {indexing_maps = [#map20, #map24, #map13], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%707, %708 : tensor<1x12x5x5xf32>, tensor<1x12x5x1xf32>) outs(%38 : tensor<1x12x5x5xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.divf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x12x5x5xf32>
    %collapsed_251 = tensor.collapse_shape %709 [[0, 1], [2], [3]] : tensor<1x12x5x5xf32> into tensor<12x5x5xf32>
    %collapsed_252 = tensor.collapse_shape %700 [[0, 1], [2], [3]] : tensor<1x12x5x64xf32> into tensor<12x5x64xf32>
    %710 = linalg.generic {indexing_maps = [#map16, #map17, #map18], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%collapsed_251, %collapsed_252 : tensor<12x5x5xf32>, tensor<12x5x64xf32>) outs(%54 : tensor<12x5x64xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      %761 = arith.addf %out, %760 : f32
      linalg.yield %761 : f32
    } -> tensor<12x5x64xf32>
    %expanded_253 = tensor.expand_shape %710 [[0, 1], [2], [3]] : tensor<12x5x64xf32> into tensor<1x12x5x64xf32>
    %711 = linalg.generic {indexing_maps = [#map13, #map14], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_253 : tensor<1x12x5x64xf32>) outs(%56 : tensor<1x5x12x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x5x12x64xf32>
    %712 = linalg.generic {indexing_maps = [#map20, #map13], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%711 : tensor<1x5x12x64xf32>) outs(%56 : tensor<1x5x12x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x5x12x64xf32>
    %collapsed_254 = tensor.collapse_shape %712 [[0, 1], [2, 3]] : tensor<1x5x12x64xf32> into tensor<5x768xf32>
    %713 = linalg.generic {indexing_maps = [#map10, #map11, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%collapsed_254, %cst_3 : tensor<5x768xf32>, tensor<768x768xf32>) outs(%60 : tensor<5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      %761 = arith.addf %out, %760 : f32
      linalg.yield %761 : f32
    } -> tensor<5x768xf32>
    %714 = linalg.generic {indexing_maps = [#map12, #map9, #map9], iterator_types = ["parallel", "parallel"]} ins(%cst, %713 : tensor<768xf32>, tensor<5x768xf32>) outs(%59 : tensor<5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.addf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<5x768xf32>
    %expanded_255 = tensor.expand_shape %714 [[0, 1], [2]] : tensor<5x768xf32> into tensor<1x5x768xf32>
    %715 = linalg.generic {indexing_maps = [#map3, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_255, %682 : tensor<1x5x768xf32>, tensor<1x5x768xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.addf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x768xf32>
    %716 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "reduction"]} ins(%715 : tensor<1x5x768xf32>) outs(%7 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.addf %in, %out : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x1xf32>
    %717 = linalg.generic {indexing_maps = [#map6, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%716 : tensor<1x5x1xf32>) outs(%6 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.divf %in, %cst_16 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x1xf32>
    %718 = linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%717 : tensor<1x5x1xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x5x768xf32>
    %719 = linalg.generic {indexing_maps = [#map3, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%715, %718 : tensor<1x5x768xf32>, tensor<1x5x768xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.subf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x768xf32>
    %720 = linalg.generic {indexing_maps = [#map3, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%719, %719 : tensor<1x5x768xf32>, tensor<1x5x768xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x768xf32>
    %721 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "reduction"]} ins(%720 : tensor<1x5x768xf32>) outs(%7 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.addf %in, %out : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x1xf32>
    %722 = linalg.generic {indexing_maps = [#map6, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%721 : tensor<1x5x1xf32>) outs(%6 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.divf %in, %cst_16 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x1xf32>
    %723 = linalg.generic {indexing_maps = [#map6, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%722 : tensor<1x5x1xf32>) outs(%6 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.truncf %cst_12 : f64 to f32
      %761 = arith.addf %in, %760 : f32
      linalg.yield %761 : f32
    } -> tensor<1x5x1xf32>
    %724 = linalg.generic {indexing_maps = [#map6, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%723 : tensor<1x5x1xf32>) outs(%6 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = math.rsqrt %in : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x1xf32>
    %725 = linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%724 : tensor<1x5x1xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x5x768xf32>
    %726 = linalg.generic {indexing_maps = [#map3, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%719, %725 : tensor<1x5x768xf32>, tensor<1x5x768xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x768xf32>
    %727 = linalg.generic {indexing_maps = [#map3, #map7, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%726, %cst : tensor<1x5x768xf32>, tensor<768xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x768xf32>
    %728 = linalg.generic {indexing_maps = [#map3, #map7, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%727, %cst : tensor<1x5x768xf32>, tensor<768xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.addf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x768xf32>
    %collapsed_256 = tensor.collapse_shape %728 [[0, 1], [2]] : tensor<1x5x768xf32> into tensor<5x768xf32>
    %729 = linalg.generic {indexing_maps = [#map10, #map11, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%collapsed_256, %cst_1 : tensor<5x768xf32>, tensor<768x3072xf32>) outs(%78 : tensor<5x3072xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      %761 = arith.addf %out, %760 : f32
      linalg.yield %761 : f32
    } -> tensor<5x3072xf32>
    %730 = linalg.generic {indexing_maps = [#map12, #map9, #map9], iterator_types = ["parallel", "parallel"]} ins(%cst_2, %729 : tensor<3072xf32>, tensor<5x3072xf32>) outs(%77 : tensor<5x3072xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.addf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<5x3072xf32>
    %expanded_257 = tensor.expand_shape %730 [[0, 1], [2]] : tensor<5x3072xf32> into tensor<1x5x3072xf32>
    %731 = linalg.generic {indexing_maps = [#map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_257 : tensor<1x5x3072xf32>) outs(%81 : tensor<1x5x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.mulf %in, %cst_17 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x3072xf32>
    %732 = linalg.generic {indexing_maps = [#map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_257 : tensor<1x5x3072xf32>) outs(%81 : tensor<1x5x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = math.powf %in, %cst_18 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x3072xf32>
    %733 = linalg.generic {indexing_maps = [#map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%732 : tensor<1x5x3072xf32>) outs(%81 : tensor<1x5x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.truncf %cst_14 : f64 to f32
      %761 = arith.mulf %in, %760 : f32
      linalg.yield %761 : f32
    } -> tensor<1x5x3072xf32>
    %734 = linalg.generic {indexing_maps = [#map3, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_257, %733 : tensor<1x5x3072xf32>, tensor<1x5x3072xf32>) outs(%81 : tensor<1x5x3072xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.addf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x3072xf32>
    %735 = linalg.generic {indexing_maps = [#map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%734 : tensor<1x5x3072xf32>) outs(%81 : tensor<1x5x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.truncf %cst_15 : f64 to f32
      %761 = arith.mulf %in, %760 : f32
      linalg.yield %761 : f32
    } -> tensor<1x5x3072xf32>
    %736 = linalg.generic {indexing_maps = [#map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%735 : tensor<1x5x3072xf32>) outs(%81 : tensor<1x5x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = math.tanh %in : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x3072xf32>
    %737 = linalg.generic {indexing_maps = [#map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%736 : tensor<1x5x3072xf32>) outs(%81 : tensor<1x5x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.addf %in, %cst_19 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x3072xf32>
    %738 = linalg.generic {indexing_maps = [#map3, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%731, %737 : tensor<1x5x3072xf32>, tensor<1x5x3072xf32>) outs(%81 : tensor<1x5x3072xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x3072xf32>
    %collapsed_258 = tensor.collapse_shape %738 [[0, 1], [2]] : tensor<1x5x3072xf32> into tensor<5x3072xf32>
    %739 = linalg.generic {indexing_maps = [#map10, #map11, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%collapsed_258, %cst_0 : tensor<5x3072xf32>, tensor<3072x768xf32>) outs(%60 : tensor<5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      %761 = arith.addf %out, %760 : f32
      linalg.yield %761 : f32
    } -> tensor<5x768xf32>
    %740 = linalg.generic {indexing_maps = [#map12, #map9, #map9], iterator_types = ["parallel", "parallel"]} ins(%cst, %739 : tensor<768xf32>, tensor<5x768xf32>) outs(%59 : tensor<5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.addf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<5x768xf32>
    %expanded_259 = tensor.expand_shape %740 [[0, 1], [2]] : tensor<5x768xf32> into tensor<1x5x768xf32>
    %741 = linalg.generic {indexing_maps = [#map3, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%715, %expanded_259 : tensor<1x5x768xf32>, tensor<1x5x768xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.addf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x768xf32>
    %742 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "reduction"]} ins(%741 : tensor<1x5x768xf32>) outs(%7 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.addf %in, %out : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x1xf32>
    %743 = linalg.generic {indexing_maps = [#map6, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%742 : tensor<1x5x1xf32>) outs(%6 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.divf %in, %cst_16 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x1xf32>
    %744 = linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%743 : tensor<1x5x1xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x5x768xf32>
    %745 = linalg.generic {indexing_maps = [#map3, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%741, %744 : tensor<1x5x768xf32>, tensor<1x5x768xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.subf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x768xf32>
    %746 = linalg.generic {indexing_maps = [#map3, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%745, %745 : tensor<1x5x768xf32>, tensor<1x5x768xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x768xf32>
    %747 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "reduction"]} ins(%746 : tensor<1x5x768xf32>) outs(%7 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.addf %in, %out : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x1xf32>
    %748 = linalg.generic {indexing_maps = [#map6, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%747 : tensor<1x5x1xf32>) outs(%6 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.divf %in, %cst_16 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x1xf32>
    %749 = linalg.generic {indexing_maps = [#map6, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%748 : tensor<1x5x1xf32>) outs(%6 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = arith.truncf %cst_12 : f64 to f32
      %761 = arith.addf %in, %760 : f32
      linalg.yield %761 : f32
    } -> tensor<1x5x1xf32>
    %750 = linalg.generic {indexing_maps = [#map6, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%749 : tensor<1x5x1xf32>) outs(%6 : tensor<1x5x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %760 = math.rsqrt %in : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x1xf32>
    %751 = linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%750 : tensor<1x5x1xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x5x768xf32>
    %752 = linalg.generic {indexing_maps = [#map3, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%745, %751 : tensor<1x5x768xf32>, tensor<1x5x768xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x768xf32>
    %753 = linalg.generic {indexing_maps = [#map3, #map7, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%752, %cst : tensor<1x5x768xf32>, tensor<768xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x768xf32>
    %754 = linalg.generic {indexing_maps = [#map3, #map7, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%753, %cst : tensor<1x5x768xf32>, tensor<768xf32>) outs(%2 : tensor<1x5x768xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.addf %in, %in_262 : f32
      linalg.yield %760 : f32
    } -> tensor<1x5x768xf32>
    %755 = tensor.empty() : tensor<768x50257xf32>
    %756 = linalg.generic {indexing_maps = [#map9, #map25], iterator_types = ["parallel", "parallel"]} ins(%cst_8 : tensor<50257x768xf32>) outs(%755 : tensor<768x50257xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<768x50257xf32>
    %collapsed_260 = tensor.collapse_shape %754 [[0, 1], [2]] : tensor<1x5x768xf32> into tensor<5x768xf32>
    %757 = tensor.empty() : tensor<5x50257xf32>
    %758 = linalg.generic {indexing_maps = [#map8, #map9], iterator_types = ["parallel", "parallel"]} ins(%cst_9 : f32) outs(%757 : tensor<5x50257xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<5x50257xf32>
    %759 = linalg.generic {indexing_maps = [#map10, #map11, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%collapsed_260, %756 : tensor<5x768xf32>, tensor<768x50257xf32>) outs(%758 : tensor<5x50257xf32>) {
    ^bb0(%in: f32, %in_262: f32, %out: f32):
      %760 = arith.mulf %in, %in_262 : f32
      %761 = arith.addf %out, %760 : f32
      linalg.yield %761 : f32
    } -> tensor<5x50257xf32>
    %expanded_261 = tensor.expand_shape %759 [[0, 1], [2]] : tensor<5x50257xf32> into tensor<1x5x50257xf32>
    return %expanded_261 : tensor<1x5x50257xf32>
  }
}

