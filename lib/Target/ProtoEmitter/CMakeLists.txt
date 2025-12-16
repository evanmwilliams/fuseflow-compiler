#set(Protobuf_USE_STATIC_LIBS ON)
include(FindProtobuf)
find_package(Protobuf)
include_directories(${Protobuf_INCLUDE_DIRS})

set(PROTO_FILES
    tortilla/proto/tortilla.proto
    tortilla/proto/comal.proto
    tortilla/proto/ops.proto
    tortilla/proto/stream.proto
)

protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS ${PROTO_FILES})
add_library(proto ${PROTO_SRCS} ${PROTO_HDRS}
        ProtoEmitter.h)
add_mlir_library(ProtoEmitter
        ProtoEmitter.cpp

        ${PROJECT_SOURCE_DIR}/lib/Target/ProtoEmitter
        ADDITIONAL_HEADER_DIRS

        LINK_COMPONENTS
        Core

        LINK_LIBS PUBLIC
        proto
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
        protobuf::libprotobuf
)

add_mlir_doc(ProtoEmitter ProtoEmitter ProtoEmitter/ -gen-pass-doc)
