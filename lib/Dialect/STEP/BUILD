load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")

package(
    default_visibility = ["//visibility:public"],
)

td_library(
    name = "td_files",
    srcs = [
        "StepAttr.td",
        "StepDialect.td",
        "StepEnums.td",
        "StepTypes.td",
    ],
    includes = ["@step//include"],
    deps = [
        "@llvm-project//mlir:BuiltinDialectTdFiles",
        "@llvm-project//mlir:FunctionInterfacesTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:SideEffectInterfacesTdFiles",
    ],
)

gentbl_cc_library(
    name = "dialect_inc_gen",
    tbl_outs = [
        (
            ["-gen-dialect-decls"],
            "StepDialect.h.inc",
        ),
        (
            ["-gen-dialect-defs"],
            "StepDialect.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "StepDialect.td",
    deps = [
        ":td_files",
    ],
)

gentbl_cc_library(
    name = "types_inc_gen",
    tbl_outs = [
        (
            ["-gen-typedef-decls"],
            "StepTypes.h.inc",
        ),
        (
            ["-gen-typedef-defs"],
            "StepTypes.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "StepTypes.td",
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
            "StepAttrDefs.h.inc",
        ),
        (
            ["-gen-attrdef-defs"],
            "StepAttrDefs.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "StepAttr.td",
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
            "StepOps.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "StepOps.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "StepOps.td",
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
            "StepEnumDefs.h.inc",
        ),
        (
            ["-gen-enum-defs"],
            "StepEnumDefs.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "StepEnums.td",
    deps = [
        ":dialect_inc_gen",
        ":td_files",
    ],
)

cc_library(
    name = "Step",
    srcs = ["StepDialect.cpp"],
    hdrs = [
        "StepDialect.h",
        "StepOps.h",
        "StepTypes.h",
    ],
    deps = [
        ":attrs_inc_gen",
        ":dialect_inc_gen",
        ":enums_inc_gen",
        ":ops_inc_gen",
        ":types_inc_gen",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)
