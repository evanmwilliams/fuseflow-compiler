# Inlining `add_mlir_dialect(Sam sam)` commands so that
# we can custom name `*.inc` generated files.
set(LLVM_TARGET_DEFINITIONS SamOps.td)
mlir_tablegen(SamOps.h.inc -gen-op-decls)
mlir_tablegen(SamOps.cpp.inc -gen-op-defs)

mlir_tablegen(SamTypes.h.inc -gen-typedef-decls -typedefs-dialect=sam)
mlir_tablegen(SamTypes.cpp.inc -gen-typedef-defs -typedefs-dialect=sam)

mlir_tablegen(SamEnumDefs.h.inc -gen-enum-decls -typedefs-dialect=sam)
mlir_tablegen(SamEnumDefs.cpp.inc -gen-enum-defs -typedefs-dialect=sam)

mlir_tablegen(SamAttrDefs.h.inc -gen-attrdef-decls -typedefs-dialect=sam)
mlir_tablegen(SamAttrDefs.cpp.inc -gen-attrdef-defs -typedefs-dialect=sam)

mlir_tablegen(SamDialect.h.inc -gen-dialect-decls -dialect=sam)
mlir_tablegen(SamDialect.cpp.inc -gen-dialect-defs -dialect=sam)
add_public_tablegen_target(MLIRSamOpsIncGen)
add_dependencies(mlir-headers MLIRSamOpsIncGen)

add_mlir_doc(SamDialect SamDialect Sam/ -gen-dialect-doc)

#set(LLVM_TARGET_DEFINITIONS SamPatterns.td)
#mlir_tablegen(SamCanonicalize.cpp.inc -gen-rewriters)
#add_public_tablegen_target(MLIRSamCanonicalizationIncGen)

add_mlir_dialect_library(MLIRSam
        SamDialect.cpp
               # SamOps.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/lib/Dialect/Sam

        LINK_LIBS PUBLIC
)
