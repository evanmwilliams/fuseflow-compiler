#ifndef LIB_TARGET_PROTO_PROTOEMITTER
#define LIB_TARGET_PROTO_PROTOEMITTER

#include "lib/Dialect/SAM/SamOps.h"
#include "llvm/ADT/DenseMap.h" // from @llvm-project
#include "llvm/ADT/TypeSwitch.h" // from @llvm-project
#include "mlir/Dialect/Affine/IR/AffineOps.h" // from @llvm-project
#include "mlir/Dialect/Arith/IR/Arith.h" // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h" // from @llvm-project
#include "mlir/Dialect/Index/IR/IndexDialect.h" // from @llvm-project
#include "mlir/Dialect/Index/IR/IndexOps.h" // from @llvm-project
#include "mlir/Dialect/MLProgram/IR/MLProgram.h"
#include "mlir/Dialect/Math/IR/Math.h" // from @llvm-project
#include "mlir/Dialect/MemRef/IR/MemRef.h" // from @llvm-project
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h" // from @llvm-project
#include "mlir/IR/BuiltinOps.h" // from @llvm-project
#include "mlir/IR/Operation.h" // from @llvm-project
#include "mlir/Support/IndentedOstream.h" // from @llvm-project
#include "mlir/Support/LogicalResult.h"

#include "lib/Dialect/SAM/SamDialect.h"
// #include "tortilla/comal.pb.h"
#include "lib/Target/ProtoEmitter/comal.pb.h"
#include "lib/Target/ProtoEmitter/tortilla.pb.h"

// ComalGraph test;

namespace mlir::sam {

    void registerToProtoPass();

    LogicalResult translateToProto(Operation *op, llvm::raw_ostream &os);

    class ChannelTrack {
    public:
        ChannelTrack() = default;

        // int get_create_channel(std::string label, int id) {
        //   auto key = std::make_pair(label, id);
        unsigned int get_create_channel(const Value key) {
            if (const auto ref = chan_map.find(key); ref != chan_map.end()) {
                return ref->second;
            }
            const auto new_ctr = counter++;
            chan_map[key] = new_ctr;
            return new_ctr;
        }

        unsigned int create_channel(const Value key) {
            const auto new_ctr = counter++;
            chan_map[key] = new_ctr;
            return new_ctr;
        }

        unsigned int create_channel(Value val, unsigned id) {
            auto key = std::make_pair(val, id);
            auto new_ctr = counter++;
            chan_map_fork[key] = new_ctr;
            return new_ctr;
        }

    private:
        DenseMap<Value, unsigned int> chan_map;
        DenseMap<std::pair<Value, unsigned>, unsigned int> chan_map_fork;
        unsigned int counter = 1;
    };

    class ProtoEmitter { //: CommonRustEmitter {
    public:
        // ProtoEmitter() : os_(llvm::raw_ostream){};
        ProtoEmitter(llvm::raw_ostream &os) : os_(os) {}; //:
        // CommonRustEmitter(os){};

        virtual LogicalResult translate(Operation &operation);

    protected:
        // struct insertBroadcastVisitor;
        // struct insertBroadcastInputVisitor;
        ChannelTrack channel_tracker;
        raw_indented_ostream os_;

    private:
        int64_t context_count_ = 0;
        int64_t channel_count_ = 0;

        std::vector<unsigned int> input_lst;
        tortilla::ProgramGraph pg;

        llvm::DenseMap<Value, std::string> value_to_channel_name_;
        llvm::DenseMap<Value, std::string> value_to_variable_name_;

        llvm::DenseMap<std::pair<Value, unsigned>,
                std::vector<std::variant<tortilla::RefStream *, tortilla::CrdStream *, tortilla::ValStream *>>>
                map_channel_fork;
        llvm::DenseMap<Value,
                std::vector<std::variant<tortilla::RefStream *, tortilla::CrdStream *, tortilla::ValStream *>>>
                map_channel;

        llvm::StringRef getOrCreateSender(Value value);

        llvm::StringRef getOrCreateReceiver(Value value);

        int64_t getNodeID() { return ++context_count_; }

        // Functions for emitting ops
        LogicalResult printOperation(ModuleOp);

        LogicalResult printOperation(func::FuncOp);

        LogicalResult printOperation(func::ReturnOp);

        LogicalResult printOperation(tensor::ExtractOp);

        LogicalResult printOperation(tensor::InsertOp);

        LogicalResult printOperation(arith::ConstantOp);

        LogicalResult printOperation(tensor::GenerateOp);

        template<typename CastType>
        LogicalResult printCast(CastType);

        LogicalResult printYield(tensor::YieldOp);

        LogicalResult printBinaryOp(mlir::Operation *, llvm::StringRef);

        // Functions for emitting Sam Patterns
        LogicalResult printOperation(sam::SamFiberLookup);

        LogicalResult printOperation(sam::SamFiberWrite);

        LogicalResult printOperation(sam::SamGenerator);

        LogicalResult printOperation(sam::SamArrayVal);

        LogicalResult printOperation(sam::SamALU);

        LogicalResult printOperation(sam::SamJoiner);

        LogicalResult printOperation(sam::SamRepeat);

        LogicalResult printOperation(sam::SamReduce);

        LogicalResult printOperation(sam::SamSpacc);

        LogicalResult printOperation(sam::SamScatter);

        LogicalResult printOperation(sam::SamGather);

        LogicalResult printOperation(sam::SamYield);

        LogicalResult printOperation(sam::SamLocate);

        static LogicalResult printOperation(sam::SamOut);

        LogicalResult printOperation(sam::SamCrdDrop);

        LogicalResult printOperation(sam::SamGenRef);
        LogicalResult printOperation(sam::SamConcat);

        void insertBroadcast();

        void insertVoidChannels();
    };

} // namespace mlir::sam

#endif /* LIB_TARGET_PROTO_PROTOEMITTER */
