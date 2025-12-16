load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")

package(
    default_visibility = ["//visibility:public"],
)

td_library(
    name = "td_files",
    srcs = [
        "SamDialect.td",
        "SamAttr.td",
        "SamTypes.td",
        "SamEnums.td",
    ],
    includes = ["@sam//include"],
    deps = [
        "@llvm-project//mlir:BuiltinDialectTdFiles",
        "@llvm-project//mlir:FunctionInterfacesTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:SideEffectInterfacesTdFiles",
        "@llvm-project//mlir:ViewLikeInterfaceTdFiles",
        "@llvm-project//mlir:InferTypeOpInterfaceTdFiles",
        "@llvm-project//mlir:ControlFlowInterfacesTdFiles",
        "@llvm-project//mlir:DestinationStyleOpInterfaceTdFiles",
        "@llvm-project//mlir:LoopLikeInterfaceTdFiles",
        "@llvm-project//mlir:TilingInterfaceTdFiles",
    ],
)

gentbl_cc_library(
    name = "dialect_inc_gen",
    tbl_outs = [
        (
            ["-gen-dialect-decls"],
            "SamDialect.h.inc",
        ),
        (
            ["-gen-dialect-defs"],
            "SamDialect.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "SamDialect.td",
    deps = [
        ":td_files",
        # "@llvm-project//mlir:SparseTensorDialect",
        # "@llvm-project//mlir:SparseTensorEnums",
        # "@llvm-project//mlir:SparseTensorPipelines",
        # "@llvm-project//mlir:SparseTensorTransforms",
        # "@llvm-project//mlir:SparseTensorTypesIncGen",
        # "@llvm-project//mlir:SparseTensorUtils",
    ],
)

gentbl_cc_library(
    name = "types_inc_gen",
    tbl_outs = [
        (
            ["-gen-typedef-decls"],
            "SamTypes.h.inc",
        ),
        (
            ["-gen-typedef-defs"],
            "SamTypes.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "SamTypes.td",
    deps = [
        ":dialect_inc_gen",
        ":td_files",
    ],
)

gentbl_cc_library(
    name = "attrs_inc_gen",
    tbl_outs = [
        (
            ["-gen-attrdef-decls"],
            "SamAttrDefs.h.inc",
        ),
        (
            ["-gen-attrdef-defs"],
            "SamAttrDefs.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "SamAttr.td",
    deps = [
        ":dialect_inc_gen",
        ":td_files",
    ],
)

gentbl_cc_library(
    name = "ops_inc_gen",
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "SamOps.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "SamOps.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "SamOps.td",
    deps = [
        ":dialect_inc_gen",
        ":td_files",
        ":types_inc_gen",
    ],
)

gentbl_cc_library(
    name = "enums_inc_gen",
    tbl_outs = [
        (
            ["-gen-enum-decls"],
            "SamEnumDefs.h.inc",
        ),
        (
            ["-gen-enum-defs"],
            "SamEnumDefs.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "SamEnums.td",
    deps = [
        ":dialect_inc_gen",
        ":td_files",
    ],
)

cc_library(
    name = "Sam",
    srcs = ["SamDialect.cpp"],
    hdrs = [
        "SamDialect.h",
        "SamOps.h",
        "SamTypes.h",
    ],
    deps = [
        ":attrs_inc_gen",
        ":dialect_inc_gen",
        ":enums_inc_gen",
        ":ops_inc_gen",
        ":types_inc_gen",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:SparseTensorDialect",
        "@llvm-project//mlir:SparseTensorEnums",
        "@llvm-project//mlir:SparseTensorPipelines",
        "@llvm-project//mlir:SparseTensorTransforms",
        "@llvm-project//mlir:SparseTensorTypesIncGen",
        "@llvm-project//mlir:SparseTensorUtils",
    ],
)
