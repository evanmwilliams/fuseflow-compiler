#add_subdirectory(${CMAKE_SOURCE_DIR}/external/or-tools)
find_package(ortools CONFIG REQUIRED)
add_mlir_library(LinalgToSam
        LinalgToSam.cpp
        FusedCIN.cpp
        FusionGroupRewrite.cpp
        ConvertFillOps.cpp
        RemoveJoinerDuplicateStreams.cpp
        Vars.cpp
        ConstructEinsum.cpp
        InsertIterateLocate.cpp

        ${PROJECT_SOURCE_DIR}/lib/Conversion/SparseToSam/
        ADDITIONAL_HEADER_DIRS

        DEPENDS
        PassesIncGen

        LINK_COMPONENTS
        Core

        LINK_LIBS PUBLIC
        MLIRSam
        MLIRArithDialect
        MLIRLinalgDialect
        MLIRLinalgTransforms
        MLIRFuncDialect
        MLIRFuncTransforms
        MLIRIR
        MLIRPass
        MLIRSCFDialect
        MLIRTensorDialect
        MLIRTransforms
        ortools::ortools
)

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name SparseToSam)
add_dependencies(mlir-headers MLIRSamOpsIncGen)
add_public_tablegen_target(PassesIncGen)
add_mlir_doc(LinalgToSam LinalgToSam SparseToSam/ -gen-pass-doc)