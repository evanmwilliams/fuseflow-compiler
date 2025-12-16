#include "lib/Target/ProtoEmitter/ProtoEmitter.h"

#include <fstream>
#include "llvm/Support/FormatVariadic.h" // from @llvm-project

#include "lib/Dialect/SAM/SamDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/include/mlir/IR/BuiltinAttributes.h" // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h" // from @llvm-project
// #include "mlir/include/mlir/IR/BuiltinTypes.h"       // from @llvm-project
// #include "mlir/include/mlir/IR/Visitors.h"           // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h" // from @llvm-project
#include "mlir/include/mlir/Tools/mlir-translate/Translation.h" // from @llvm-project

#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>

using namespace mlir::sparse_tensor;

namespace mlir::sam
{
    struct appendIDVisitor
    {
        std::vector<unsigned int> *input_lst;

        void operator()(const tortilla::RefStream *stream) const { input_lst->push_back(stream->id().id()); }

        void operator()(const tortilla::CrdStream *stream) const { input_lst->push_back(stream->id().id()); }

        void operator()(const tortilla::ValStream *stream) const { input_lst->push_back(stream->id().id()); }
    };

    struct insertBroadcastInputVisitor
    {
        tortilla::Broadcast *new_broadcast;
        std::vector<unsigned int> *input_lst;

        void operator()(const tortilla::RefStream *stream) const
        {
            new_broadcast->mutable_ref()->mutable_input()->mutable_id()->set_id(stream->id().id());
            input_lst->push_back(stream->id().id());
        }

        void operator()(const tortilla::CrdStream *stream) const
        {
            new_broadcast->mutable_crd()->mutable_input()->mutable_id()->set_id(stream->id().id());
            input_lst->push_back(stream->id().id());
        }

        void operator()(const tortilla::ValStream *stream) const
        {
            new_broadcast->mutable_val()->mutable_input()->mutable_id()->set_id(stream->id().id());
            input_lst->push_back(stream->id().id());
        }
    };

    struct insertBroadcastVisitor
    {
        tortilla::Broadcast *new_broadcast;
        unsigned int new_channel_id;

        void operator()(tortilla::RefStream *stream) const
        {
            new_broadcast->mutable_ref()->mutable_outputs()->Add()->mutable_id()->set_id(new_channel_id);
            stream->mutable_id()->set_id(new_channel_id);
        }

        void operator()(tortilla::CrdStream *stream) const
        {
            new_broadcast->mutable_crd()->mutable_outputs()->Add()->mutable_id()->set_id(new_channel_id);
            stream->mutable_id()->set_id(new_channel_id);
        }

        void operator()(tortilla::ValStream *stream) const
        {
            new_broadcast->mutable_val()->mutable_outputs()->Add()->mutable_id()->set_id(new_channel_id);
            stream->mutable_id()->set_id(new_channel_id);
        }
    };


    void ProtoEmitter::insertVoidChannels()
    {
        for (int i = 0; i < pg.operators_size(); ++i)
        {
            if (const auto operation = pg.mutable_operators(i); operation->name() == "fiberlookup")
            {
                if (auto out1 = operation->fiber_lookup().output_ref().id().id();
                    std::find(input_lst.begin(), input_lst.end(), out1) == input_lst.end())
                {
                    operation->mutable_fiber_lookup()->mutable_output_ref()->mutable_id()->set_id(0);
                }
                if (auto out2 = operation->fiber_lookup().output_crd().id().id();
                    std::find(input_lst.begin(), input_lst.end(), out2) == input_lst.end())
                {
                    operation->mutable_fiber_lookup()->mutable_output_crd()->mutable_id()->set_id(0);
                }
            }
            else if (operation->name() == "repeat")
            {
//                if (auto out_ref = operation->repeat().output_ref().id().id();
//                    std::find(input_lst.begin(), input_lst.end(), out_ref) == input_lst.end())
//                {
//                    operation->mutable_repeat()->mutable_output_ref()->mutable_id()->set_id(0);
//                }
            }
            else if (operation->name() == "joiner")
            {
                for (int j = 0; j < operation->joiner().output_refs_size(); ++j)
                {
                    //                    auto out_ref = operation->joiner().output_refs()[i].id().id();
                    unsigned int out_ref;
                    if (operation->joiner().output_refs()[j].has_ref_stream())
                    {
                        out_ref = operation->joiner().output_refs()[j].ref_stream().id().id();
                    }
                    else
                    {
                        out_ref = operation->joiner().output_refs()[j].val_stream().id().id();
                    }
                    if (std::find(input_lst.begin(), input_lst.end(), out_ref) == input_lst.end())
                    {
                        //                        operation->mutable_joiner()->mutable_output_refs(i)->mutable_id()->set_id(0);
                        if (operation->mutable_joiner()->mutable_output_refs(j)->has_ref_stream())
                        {
                            operation->mutable_joiner()
                                ->mutable_output_refs(j)
                                ->mutable_ref_stream()
                                ->mutable_id()
                                ->set_id(0);
                        }
                        else
                        {
                            operation->mutable_joiner()
                                ->mutable_output_refs(j)
                                ->mutable_val_stream()
                                ->mutable_id()
                                ->set_id(0);
                        }
                    }
                }
                auto out_crd = operation->joiner().output_crd().id().id();
                if (std::find(input_lst.begin(), input_lst.end(), out_crd) == input_lst.end())
                {
                    operation->mutable_joiner()->mutable_output_crd()->mutable_id()->set_id(0);
                }
            }
            else if (operation->name() == "reduce")
            {
                if (const unsigned int out_val = operation->reduce().output_val().id().id();
                    std::find(input_lst.begin(), input_lst.end(), out_val) == input_lst.end())
                {
                    operation->mutable_reduce()->mutable_output_val()->mutable_id()->set_id(0);
                }
            }
            else if (operation->name() == "locate")
            {
                if (const unsigned int out_ref1 = operation->locate().output_ref1().id().id();
                    std::find(input_lst.begin(), input_lst.end(), out_ref1) == input_lst.end())
                {
                    operation->mutable_locate()->mutable_output_ref1()->mutable_id()->set_id(0);
                }
                if (const unsigned int out_ref2 = operation->locate().output_ref2().id().id();
                    std::find(input_lst.begin(), input_lst.end(), out_ref2) == input_lst.end())
                {
                    operation->mutable_locate()->mutable_output_ref2()->mutable_id()->set_id(0);
                }
                if (const unsigned int out_crd = operation->locate().output_crd().id().id();
                    std::find(input_lst.begin(), input_lst.end(), out_crd) == input_lst.end())
                {
                    operation->mutable_locate()->mutable_output_crd()->mutable_id()->set_id(0);
                }
            }
            else if (operation->name() == "concat")
            {
                // Check all out_crds
                for (int j = 0; j < operation->concat().out_crds_size(); ++j)
                {
                    if (const unsigned int out_crd_id = operation->concat().out_crds(j).id().id();
                        std::find(input_lst.begin(), input_lst.end(), out_crd_id) == input_lst.end())
                    {
                        operation->mutable_concat()->mutable_out_crds(j)->mutable_id()->set_id(0);
                    }
                }
                // Check all out_refs
                for (int j = 0; j < operation->concat().out_refs_size(); ++j)
                {
                    if (const unsigned int out_ref_id = operation->concat().out_refs(j).id().id();
                        std::find(input_lst.begin(), input_lst.end(), out_ref_id) == input_lst.end())
                    {
                        operation->mutable_concat()->mutable_out_refs(j)->mutable_id()->set_id(0);
                    }
                }
                // Check out_val
                if (operation->concat().has_out_val())
                {
                    if (const unsigned int out_val_id = operation->concat().out_val().id().id();
                        std::find(input_lst.begin(), input_lst.end(), out_val_id) == input_lst.end())
                    {
                        operation->mutable_concat()->mutable_out_val()->mutable_id()->set_id(0);
                    }
                }
            }
            else if (operation->name() == "spacc")
            {
                if (const unsigned int out_val = operation->spacc().output_val().id().id();
                    std::find(input_lst.begin(), input_lst.end(), out_val) == input_lst.end())
                {
                    operation->mutable_spacc()->mutable_output_val()->mutable_id()->set_id(0);
                }
                for (int outer_id = 0; outer_id < operation->spacc().output_outer_crds_size(); outer_id++)
                {
                    const unsigned int output_outer_id = operation->spacc().output_outer_crds(outer_id).id().id();
                    if (const auto outer = operation->mutable_spacc()->mutable_output_outer_crds(outer_id);
                        std::find(input_lst.begin(), input_lst.end(), output_outer_id) == input_lst.end())
                    {
                        outer->mutable_id()->set_id(0);
                    }
                }
                if (const unsigned int inner_crd_id = operation->spacc().output_inner_crd().id().id();
                    std::find(input_lst.begin(), input_lst.end(), inner_crd_id) == input_lst.end())
                {
                    operation->mutable_spacc()->mutable_output_inner_crd()->mutable_id()->set_id(0);
                }
            }
        }
    }

    void ProtoEmitter::insertBroadcast()
    {
        llvm::DenseMap<
            Operation *,
            std::vector<std::vector<std::variant<tortilla::RefStream *, tortilla::CrdStream *, tortilla::ValStream *>>>>
            fork_map;
        llvm::DenseMap<Operation *, Value> op_to_val;

        for (auto key : map_channel)
        {
            tortilla::Broadcast *new_broadcast;

            if (key.second.size() <= 1)
            {
                std::visit(appendIDVisitor{&input_lst}, key.second[0]);
                continue;
            }
            else
            {
                auto new_op = pg.add_operators();
                new_op->set_name("broadcast");
                new_op->set_id(getNodeID());
                new_broadcast = new_op->mutable_broadcast();
                std::visit(insertBroadcastInputVisitor{new_broadcast, &input_lst}, key.second[1]);
            }
            for (auto val : key.second)
            {
                std::visit(insertBroadcastVisitor{new_broadcast, channel_tracker.create_channel(key.first)}, val);
                std::visit(appendIDVisitor{&input_lst}, val);
            }
        }
    }

    class TensorNameRegistry
    {
    public:
        TensorNameRegistry() : indexCounter(1) {}

        std::string getTensorName(mlir::Value tensor)
        {
            std::string alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
            if (!tensorMap.contains(tensor))
            {
                tensorMap[tensor] = alphabet[indexCounter++];
            }
            return tensorMap[tensor];
        }

    private:
        DenseMap<mlir::Value, std::string> tensorMap;
        int indexCounter;
    };

    TensorNameRegistry tensorRegistry;

    LogicalResult ProtoEmitter::translate(Operation &operation)
    {
        LogicalResult status =
            llvm::TypeSwitch<Operation &, LogicalResult>(operation)
                .Case<ModuleOp>([&](auto op) { return printOperation(op); })
                .Case<func::FuncOp>([&](auto op) { return printOperation(op); })
                .Case<func::ReturnOp>([&](auto op) { return printOperation(op); })
                .Case<sam::SamFiberLookup, sam::SamFiberWrite, sam::SamGenerator, sam::SamArrayVal, sam::SamALU,
                      sam::SamJoiner, sam::SamRepeat, sam::SamReduce, sam::SamSpacc, sam::SamScatter, sam::SamGather,
                      sam::SamOut, sam::SamCrdDrop, sam::SamGenRef, sam::SamLocate, sam::SamConcat>(
                    [&](auto op) { return printOperation(op); })
                .Case<linalg::MatmulOp, arith::ConstantOp, tensor::EmptyOp, linalg::GenericOp>([&](auto op)
                                                                                               { return success(); })
                .Case<linalg::AddOp>([&](auto op) { return success(); })
                .Case<ml_program::GlobalOp>([&](auto op) { return success(); })
                .Case<UnrealizedConversionCastOp>([&](auto op) { return success(); })
                .Default([&](Operation &) { return operation.emitError("Unable to find printer for op."); });

        if (failed(status))
        {
            operation.emitOpError(llvm::formatv("Failed to translate op {0}", operation.getName()));
            return failure();
        }
        return success();
    };

    LogicalResult ProtoEmitter::printOperation(func::FuncOp funcOp)
    {
        for (auto &blockOp : funcOp)
        {
            for (auto &op : blockOp)
            {
                auto result = translate(op);
                if (failed(result))
                {
                    return result;
                }
            }
        }

        return success();
    };

    LogicalResult ProtoEmitter::printOperation(func::ReturnOp retOp)
    {
        insertBroadcast();
        insertVoidChannels();
        auto comal_graph = tortilla::ComalGraph();
        comal_graph.set_name("comal graph");
        comal_graph.set_channel_size(1024);
        comal_graph.mutable_graph()->CopyFrom(pg);
        std::string str;
        google::protobuf::TextFormat::PrintToString(comal_graph, &str);
        os_ << str << "\n";

        // TODO: Figure out how to parametrize the output file name
        std::ofstream ofs("/tmp/op.bin", std::ios_base::out | std::ios_base::binary);
        comal_graph.SerializeToOstream(&ofs);

        return success();
    };

    LogicalResult ProtoEmitter::printOperation(ModuleOp moduleOp)
    {
        for (Operation &op : moduleOp)
        {
            if (failed(translate(op)))
            {
                return failure();
            }
        }

        return success();
    };

    // TODO: Placeholder emitter for now, this op will have already been deleted by
    // fusion pass
    LogicalResult ProtoEmitter::printOperation(sam::SamOut samOp) { return success(); };

    LogicalResult ProtoEmitter::printOperation(sam::SamFiberLookup lookupOp)
    {
        tortilla::Operation *new_op = pg.add_operators();
        std::string name = "fiberlookup";
        new_op->set_name(name);
        new_op->set_id(getNodeID());

        const SamStreamEncodingAttr enc = getSamStreamEncoding(lookupOp.getOutputRef().getType());
        std::string format = enc.getFormat().str();
        std::string tensor = enc.getTensor().str();

        tensor = tensor.substr(0, tensor.find('_'));
        std::string final_tensor = tensor;

        std::string var = enc.getVar().str();
        std::string label = name + " " + var;

        new_op->mutable_fiber_lookup()->set_format(format);
        new_op->mutable_fiber_lookup()->set_tensor(final_tensor);
        new_op->mutable_fiber_lookup()->set_mode(enc.getMode().getInt());
        new_op->mutable_fiber_lookup()->set_index(var);
        new_op->mutable_fiber_lookup()->set_label(label);

        const auto in_ref = lookupOp.getInputRef();
        const auto out_ref = lookupOp.getOutputRef();
        const auto out_crd = lookupOp.getOutputCrd();

        new_op->mutable_fiber_lookup()->mutable_input_ref()->mutable_id()->set_id(
            channel_tracker.get_create_channel(in_ref));
        new_op->mutable_fiber_lookup()->mutable_output_ref()->mutable_id()->set_id(
            channel_tracker.get_create_channel(out_ref));
        new_op->mutable_fiber_lookup()->mutable_output_crd()->mutable_id()->set_id(
            channel_tracker.get_create_channel(out_crd));

        map_channel[in_ref].emplace_back(new_op->mutable_fiber_lookup()->mutable_input_ref());

        return success();
    };

    LogicalResult ProtoEmitter::printOperation(sam::SamFiberWrite writeOp)
    {
        tortilla::Operation *new_op = pg.add_operators();
        auto writeType = writeOp.getWriteOp();

        std::string name = writeType == sam::WriteType::Crd ? "fiberwrite" : "valwrite";

        new_op->set_name(name);
        new_op->set_id(getNodeID());

        SamStreamEncodingAttr enc = getSamStreamEncoding(writeOp.getInputVal().getType());

        std::string tensor = enc.getTensor().str();
        std::string var = enc.getVar().str();
        std::string format = enc.getFormat().str();
        std::string label = name + " " + var;

        auto in_val = writeOp.getInputVal();

        if (writeType == sam::WriteType::Crd)
        {
            new_op->mutable_fiber_write()->set_format(format);
            new_op->mutable_fiber_write()->set_tensor(tensor);
            new_op->mutable_fiber_write()->set_index(var);
            new_op->mutable_fiber_write()->set_label(name + " " + var);

            new_op->mutable_fiber_write()->mutable_input_crd()->mutable_id()->set_id(
                channel_tracker.get_create_channel(in_val));
            map_channel[in_val].emplace_back(new_op->mutable_fiber_write()->mutable_input_crd());
        }
        else
        {
            // TODO: Might need to add in crdsize/segsize for old sam code gen
            new_op->mutable_val_write()->set_label(name);
            new_op->mutable_val_write()->mutable_input_val()->mutable_id()->set_id(
                channel_tracker.get_create_channel(in_val));
            map_channel[in_val].emplace_back(new_op->mutable_val_write()->mutable_input_val());
        }
        return success();
    };

    LogicalResult ProtoEmitter::printOperation(sam::SamGenerator genOp)
    {
        tortilla::Operation *new_op = pg.add_operators();
        std::string name = "root";
        const SamStreamEncodingAttr enc = getSamStreamEncoding(genOp.getOutputRef().getType());
        const std::string tensor = enc.getTensor().str();

        new_op->set_name(name);

        new_op->set_id(getNodeID());

        new_op->mutable_root()->set_label(name + " " + tensor);

        auto out_ref = genOp.getOutputRef();
        new_op->mutable_root()->mutable_output_ref()->mutable_id()->set_id(channel_tracker.get_create_channel(out_ref));
        return success();
    };

    LogicalResult ProtoEmitter::printOperation(sam::SamArrayVal arrayOp)
    {
        tortilla::Operation *new_op = pg.add_operators();
        std::string name = "arrayval";
        const SamStreamEncodingAttr enc = getSamStreamEncoding(arrayOp.getOutputVal().getType());
        std::string tensor = enc.getTensor().str();

        // TODO: Need to switch to using variadic input to arrayop instead
        // uint64_t stream_shape = enc.getStreamShape().getUInt();
        // bool blocked = enc.getBlocked().getValue();
        // We assume all the lengths are the same even for blocked case
        const uint64_t stream_shape = arrayOp.getStreamShape()[0];
        const bool blocked = arrayOp.getStreamShape().size() > 1 ? true : false;

        tensor = tensor.substr(0, tensor.find('_'));
        std::string final_tensor = tensor;

        new_op->set_name(name);
        new_op->set_id(getNodeID());
        // new_op->mutable_array()->set_tensor(tensor);
        new_op->mutable_array()->set_tensor(final_tensor);
        new_op->mutable_array()->set_label(name + " " + final_tensor);
        new_op->mutable_array()->set_blocked(blocked);
        new_op->mutable_array()->set_stream_shape(stream_shape);

        auto in_ref = arrayOp.getInputRef();
        auto out_val = arrayOp.getOutputVal();
        new_op->mutable_array()->mutable_input_ref()->mutable_id()->set_id(channel_tracker.get_create_channel(in_ref));
        new_op->mutable_array()->mutable_output_val()->mutable_id()->set_id(
            channel_tracker.get_create_channel(out_val));

        map_channel[in_ref].emplace_back(new_op->mutable_array()->mutable_input_ref());
        return success();
    };

    LogicalResult ProtoEmitter::printOperation(sam::SamALU aluOp)
    {
        tortilla::Operation *new_op = pg.add_operators();
        const auto opType = aluOp.getAluOp();
        new_op->set_id(getNodeID());

        tortilla::ALU_Stage *new_stage = new_op->mutable_alu()->add_stages();

        auto scalar = aluOp.getScalarAttr().getValueAsDouble();


        std::string name;
        if (opType == sam::OpType::Mul)
        {
            if (!llvm::isa<sam::SamReduce>(*(aluOp->getUsers().begin())) &&
                !llvm::isa<sam::SamSpacc>(*(aluOp->getUsers().begin())))
            {
                new_stage->set_op(tortilla::ALU_ALUOp_ELEMMUL);
            }
            else
            {
                new_stage->set_op(tortilla::ALU_ALUOp_MUL);
            }
            name = "mul";
        }
        else if (opType == sam::OpType::Add)
        {
            new_stage->set_op(tortilla::ALU_ALUOp_ADD);
            name = "add";
        }
        else if (opType == sam::OpType::Div)
        {
            new_stage->set_op(tortilla::ALU_ALUOp_DIV);
            name = "div";
        }
        else if (opType == sam::OpType::Exp)
        {
            new_stage->set_op(tortilla::ALU_ALUOp_EXP);
            name = "exp";
        }
        else if (opType == sam::OpType::Sub)
        {
            new_stage->set_op(tortilla::ALU_ALUOp_SUB);
            name = "sub";
        }
        else if (opType == sam::OpType::Max)
        {
            new_stage->set_op(tortilla::ALU_ALUOp_MAX);
            name = "max";
        }
        else if (opType == sam::OpType::ScalarAdd)
        {
            new_stage->set_op(tortilla::ALU_ALUOp_SCALARADD);
            name = "scalarAdd";
        }
        else if (opType == sam::OpType::ScalarMul)
        {
            new_stage->set_op(tortilla::ALU_ALUOp_SCALARMUL);
            name = "scalarMul";
        }
        else if (opType == sam::OpType::ScalarDiv)
        {
            new_stage->set_op(tortilla::ALU_ALUOp_SCALARDIV);
            name = "scalarDiv";
        }
        else if (opType == sam::OpType::RSqrt)
        {
            new_stage->set_op(tortilla::ALU_ALUOp_RSQRT);
            name = "rsqrt";
        }
        new_op->set_name(name);

        new_stage->add_inputs(0);
        new_op->mutable_alu()->mutable_vals()->add_inputs();
        if (opType == sam::OpType::Add || opType == sam::OpType::Sub || opType == sam::OpType::Mul ||
            opType == sam::OpType::Div)
        {
            new_stage->add_inputs(1);
            new_op->mutable_alu()->mutable_vals()->add_inputs();
        }

        if (opType == sam::OpType::ScalarMul || opType == sam::OpType::ScalarAdd || opType == sam::OpType::ScalarDiv)
        {
            new_op->mutable_alu()->set_scalar(scalar);
        }

        new_stage->set_output(0);
        new_op->mutable_alu()->set_output_val(0);
        new_op->mutable_alu()->set_label(name);

        unsigned int num_inputs = new_op->mutable_alu()->mutable_vals()->inputs_size();

        for (int i = 0; i < num_inputs; ++i)
        {
            auto in_val = aluOp.getInputVal()[i];
            new_op->mutable_alu()->mutable_vals()->mutable_inputs(i)->mutable_id()->set_id(
                channel_tracker.get_create_channel(in_val));
            map_channel[in_val].emplace_back(new_op->mutable_alu()->mutable_vals()->mutable_inputs(i));
        }

        const Value out_val = aluOp.getOutputVal()[0];
        new_op->mutable_alu()->mutable_vals()->mutable_output()->mutable_id()->set_id(
            channel_tracker.get_create_channel(out_val));

        return success();
    };

    LogicalResult ProtoEmitter::printOperation(sam::SamJoiner joinerOp)
    {
        tortilla::Operation *new_op = pg.add_operators();
        new_op->set_id(getNodeID());

        auto joinerType = joinerOp.getJoinerOp();
        std::string name = "joiner";
        // joinerType == sam::JoinerType::Intersect ? "intersect" : "union";
        SamStreamEncodingAttr enc = getSamStreamEncoding(joinerOp.getOutputCrd().getType());
        std::string var = enc.getVar().str();
        new_op->set_name(name);
        std::string label_prefix = joinerType == sam::JoinerType::Intersect ? "Intersect" : "Union";
        new_op->mutable_joiner()->set_label(label_prefix + " " + var);
        auto tortillaJoinerType =
            joinerType == sam::JoinerType::Intersect ? tortilla::Joiner_Type_INTERSECT : tortilla::Joiner_Type_UNION;


        new_op->mutable_joiner()->set_join_type(tortillaJoinerType);
        new_op->mutable_joiner()->set_index(var);

        auto in_crds = joinerOp.getInputCrd();
        auto in_refs = joinerOp.getInputRef();
        auto out_crd = joinerOp.getOutputCrd();
        auto out_refs = joinerOp.getOutputRefs();

        for (int i = 0; i < in_crds.size(); ++i)
        {
            SamStreamEncodingAttr enc1 = getSamStreamEncoding(out_refs[i].getType());
            new_op->mutable_joiner()->add_input_pairs();
            new_op->mutable_joiner()->mutable_input_pairs(i)->mutable_crd()->mutable_id()->set_id(
                channel_tracker.get_create_channel(in_crds[i]));
            new_op->mutable_joiner()->add_output_refs();
            // Check ref type to set joiner ref crd bundle type to either val (in case of joiners after spaccs) or ref
            if (const SamStreamEncodingAttr input_enc = getSamStreamEncoding(in_refs[i].getType()))
            {
                const auto stream_kind = input_enc.getStreamKind().getValue();
                if (stream_kind == sam::StreamKind::Val)
                {
                    new_op->mutable_joiner()
                        ->mutable_input_pairs(i)
                        ->mutable_in_ref()
                        ->mutable_val_stream()
                        ->mutable_id()
                        ->set_id(channel_tracker.get_create_channel(in_refs[i]));
                    new_op->mutable_joiner()->mutable_output_refs(i)->mutable_val_stream()->mutable_id()->set_id(
                        channel_tracker.get_create_channel(out_refs[i]));
                    new_op->mutable_joiner()->mutable_output_refs(i)->mutable_val_stream()->set_name(
                        "out_ref-" + enc1.getTensor().str());
                    map_channel[in_refs[i]].emplace_back(
                        new_op->mutable_joiner()->mutable_input_pairs(i)->mutable_in_ref()->mutable_val_stream());
                }
                else if (stream_kind == sam::StreamKind::Ref)
                {
                    new_op->mutable_joiner()
                        ->mutable_input_pairs(i)
                        ->mutable_in_ref()
                        ->mutable_ref_stream()
                        ->mutable_id()
                        ->set_id(channel_tracker.get_create_channel(in_refs[i]));
                    new_op->mutable_joiner()->mutable_output_refs(i)->mutable_ref_stream()->mutable_id()->set_id(
                        channel_tracker.get_create_channel(out_refs[i]));
                    new_op->mutable_joiner()->mutable_output_refs(i)->mutable_ref_stream()->set_name(
                        "out_ref-" + enc1.getTensor().str());
                    map_channel[in_refs[i]].emplace_back(
                        new_op->mutable_joiner()->mutable_input_pairs(i)->mutable_in_ref()->mutable_ref_stream());
                }
                // const auto ref_type = stream_kind == sam::StreamKind::Val
                // ? tortilla::Joiner_RefType::Joiner_RefType_VAL
                // : tortilla::Joiner_RefType::Joiner_RefType_REF;
                // new_op->mutable_joiner()->mutable_input_pairs(i)->set_ref_type(ref_type);
            }

            map_channel[in_crds[i]].emplace_back(new_op->mutable_joiner()->mutable_input_pairs(i)->mutable_crd());
        }

        new_op->mutable_joiner()->mutable_output_crd()->mutable_id()->set_id(
            channel_tracker.get_create_channel(out_crd));

        return success();
    };

    LogicalResult ProtoEmitter::printOperation(sam::SamReduce reduceOp)
    {
        tortilla::Operation *new_op = pg.add_operators();
        std::string name = "reduce";
        new_op->set_name(name);
        new_op->set_id(getNodeID());
        new_op->mutable_reduce()->set_label(name);
        auto reduceType = reduceOp.getReduceType();
        if (reduceType == sam::ReduceType::Add)
        {
            new_op->mutable_reduce()->set_reduce_type(tortilla::Reduce_Type_ADD);
        }
        else if (reduceType == sam::ReduceType::AddSum)
        {
            new_op->mutable_reduce()->set_reduce_type(tortilla::Reduce_Type_ADDSUM);
        }
        else if (reduceType == sam::ReduceType::Max)
        {
            new_op->mutable_reduce()->set_reduce_type(tortilla::Reduce_Type_MAX);
        }

        auto in_val = reduceOp.getInputVal();
        auto out_val = reduceOp.getOutputVal();
        new_op->mutable_reduce()->mutable_input_val()->mutable_id()->set_id(channel_tracker.get_create_channel(in_val));
        new_op->mutable_reduce()->mutable_output_val()->mutable_id()->set_id(
            channel_tracker.get_create_channel(out_val));

        map_channel[in_val].emplace_back(new_op->mutable_reduce()->mutable_input_val());
        return success();
    };

    LogicalResult ProtoEmitter::printOperation(sam::SamLocate locateOp)
    {
        tortilla::Operation *new_op = pg.add_operators();
        std::string name = "locate";
        std::string label_pref = "Locate";
        new_op->set_name(name);
        new_op->set_id(getNodeID());
        new_op->mutable_locate()->set_label(label_pref);

        auto input_crd = locateOp.getInputCrd();
        auto input_ref = locateOp.getInputRef();
        auto output_ref1 = locateOp.getOutputRef1();
        auto output_ref2 = locateOp.getOutputRef2();
        auto output_crd = locateOp.getOutputCrd();

        new_op->mutable_locate()->mutable_input_crd()->mutable_id()->set_id(
            channel_tracker.get_create_channel(input_crd));
        map_channel[input_crd].emplace_back(new_op->mutable_locate()->mutable_input_crd());

        new_op->mutable_locate()->mutable_input_ref()->mutable_id()->set_id(
            channel_tracker.get_create_channel(input_ref));
        map_channel[input_ref].emplace_back(new_op->mutable_locate()->mutable_input_ref());

        new_op->mutable_locate()->mutable_output_ref1()->mutable_id()->set_id(
            channel_tracker.get_create_channel(output_ref1));
        new_op->mutable_locate()->mutable_output_ref2()->mutable_id()->set_id(
            channel_tracker.get_create_channel(output_ref2));
        new_op->mutable_locate()->mutable_output_crd()->mutable_id()->set_id(
            channel_tracker.get_create_channel(output_crd));

        return success();
    };

    LogicalResult ProtoEmitter::printOperation(sam::SamGenRef genRefOp)
    {
        tortilla::Operation *new_op = pg.add_operators();
        std::string name = "gen_ref";
        new_op->set_name(name);
        new_op->set_id(getNodeID());
        new_op->mutable_gen_ref()->set_label(name);

        auto in_crd = genRefOp.getInputCrd();
        auto out_ref = genRefOp.getOutputRef();
        new_op->mutable_gen_ref()->mutable_input_crd()->mutable_id()->set_id(
            channel_tracker.get_create_channel(in_crd));
        new_op->mutable_gen_ref()->mutable_output_ref()->mutable_id()->set_id(
            channel_tracker.get_create_channel(out_ref));

        map_channel[in_crd].emplace_back(new_op->mutable_gen_ref()->mutable_input_crd());
        return success();
    };

    LogicalResult ProtoEmitter::printOperation(sam::SamConcat concatOp)
    {
        tortilla::Operation *new_op = pg.add_operators();
        std::string name = "concat";
        new_op->set_name(name);
        new_op->set_id(getNodeID());

        auto *concat = new_op->mutable_concat();
        concat->set_label(name);
        concat->set_axis(concatOp.getAxisAttr().getInt());
        concat->set_rank(concatOp.getRankAttr().getInt());
        // Emit dense length of concatenated dimension
        concat->set_dim_len(concatOp.getDimLenAttr().getInt());

        // Inputs: flattened in_crds and matched in_refs/in_vals (per-level)
        auto inCrds = concatOp.getInCrds();
        for (auto crd : inCrds)
        {
            concat->add_in_crds()->mutable_id()->set_id(channel_tracker.get_create_channel(crd));
            map_channel[crd].emplace_back(concat->mutable_in_crds(concat->in_crds_size() - 1));
        }
        // Note: MLIR op provides per-level streams in getInVals(); here we split to repeated in_refs and repeated
        // in_vals.
        for (auto s : concatOp.getInVals())
        {
            auto enc = getSamStreamEncoding(s.getType());
            if (enc.getStreamKind().getValue() == sam::StreamKind::Ref)
            {
                concat->add_in_refs()->mutable_id()->set_id(channel_tracker.get_create_channel(s));
                map_channel[s].emplace_back(concat->mutable_in_refs(concat->in_refs_size() - 1));
            }
            else
            {
                concat->add_in_vals()->mutable_id()->set_id(channel_tracker.get_create_channel(s));
                map_channel[s].emplace_back(concat->mutable_in_vals(concat->in_vals_size() - 1));
            }
        }

        // Outputs: out_crds (rank outputs) and out_refs/out_val per level
        for (auto out_crd : concatOp.getOutCrds())
        {
            concat->add_out_crds()->mutable_id()->set_id(channel_tracker.get_create_channel(out_crd));
        }
        for (auto s : concatOp.getOutVals())
        {
            auto enc = getSamStreamEncoding(s.getType());
            if (enc.getStreamKind().getValue() == sam::StreamKind::Ref)
            {
                concat->add_out_refs()->mutable_id()->set_id(channel_tracker.get_create_channel(s));
            }
            else
            {
                concat->mutable_out_val()->mutable_id()->set_id(channel_tracker.get_create_channel(s));
            }
        }

        return success();
    };

    LogicalResult ProtoEmitter::printOperation(sam::SamRepeat repeatOp)
    {
        // tortilla::Operation *new_rsig = pg.add_operators();

        SamStreamEncodingAttr enc = getSamStreamEncoding(repeatOp.getOutputRef().getType());
        std::string var = enc.getVar().str();
        std::string tensor = enc.getTensor().str();

        const auto in_ref = repeatOp.getInputRef();
        const auto out_ref = repeatOp.getOutputRef();
        const auto repeat_ref = repeatOp.getRepeatRef();

        tortilla::Operation *new_op = pg.add_operators();
        std::string name = "repeat";
        new_op->set_name(name);
        new_op->set_id(getNodeID());

        new_op->mutable_repeat()->set_label(name + " " + var + ": " + tensor);
        new_op->mutable_repeat()->set_index(var);
        new_op->mutable_repeat()->set_tensor(tensor);

        if (const SamStreamEncodingAttr rep_enc = getSamStreamEncoding(repeat_ref.getType());
            rep_enc.getStreamKind().getValue() == sam::StreamKind::Val)
        {
            new_op->mutable_repeat()->mutable_rep_val()->mutable_id()->set_id(
                channel_tracker.get_create_channel(repeat_ref));
            map_channel[repeat_ref].emplace_back(new_op->mutable_repeat()->mutable_rep_val());
        }
        else
        {
            new_op->mutable_repeat()->mutable_rep_ref()->mutable_id()->set_id(
                channel_tracker.get_create_channel(repeat_ref));
            map_channel[repeat_ref].emplace_back(new_op->mutable_repeat()->mutable_rep_ref());
        }
        if (const SamStreamEncodingAttr in_ref_enc = getSamStreamEncoding(in_ref.getType());
            in_ref_enc.getStreamKind().getValue() == sam::StreamKind::Val)
        {
            new_op->mutable_repeat()->mutable_in_val()->mutable_id()->set_id(
                channel_tracker.get_create_channel(in_ref));
            new_op->mutable_repeat()->mutable_out_val()->mutable_id()->set_id(
                channel_tracker.get_create_channel(out_ref));
            map_channel[in_ref].emplace_back(new_op->mutable_repeat()->mutable_in_val());
        }
        else
        {
            new_op->mutable_repeat()->mutable_in_ref()->mutable_id()->set_id(
                channel_tracker.get_create_channel(in_ref));
            new_op->mutable_repeat()->mutable_out_ref()->mutable_id()->set_id(
                channel_tracker.get_create_channel(out_ref));
            map_channel[in_ref].emplace_back(new_op->mutable_repeat()->mutable_in_ref());
        }

        return success();
    };

    LogicalResult ProtoEmitter::printOperation(sam::SamSpacc spaccOp)
    {
        auto comal_graph = tortilla::ComalGraph();
        comal_graph.set_name("comal graph");
        comal_graph.set_channel_size(1024);
        comal_graph.mutable_graph()->CopyFrom(pg);
        std::string str;
        google::protobuf::TextFormat::PrintToString(comal_graph, &str);
        // os_ << str << "\n";

        tortilla::Operation *new_op = pg.add_operators();
        std::string name = "spacc";
        const std::string label_pref = "SparseAccumulator";
        auto orderNum = spaccOp.getOrderAttr().getInt();
        const std::string order = std::to_string(orderNum);
        new_op->set_name(name);
        new_op->set_id(getNodeID());
        new_op->mutable_spacc()->set_label(label_pref + " " + order);
        new_op->mutable_spacc()->set_order(orderNum);

        const auto input_inner_crd = spaccOp.getInputInnerCrd();
        const auto input_outer_crds = spaccOp.getInputOuterCrds();
        const auto input_val = spaccOp.getInputVal();
        const auto output_inner_crd = spaccOp.getOutputInnerCrd();
        const auto output_outer_crds = spaccOp.getOutputOuterCrds();
        const auto output_val = spaccOp.getOutputVal();

        new_op->mutable_spacc()->mutable_input_inner_crd()->mutable_id()->set_id(
            channel_tracker.get_create_channel(input_inner_crd));
        if (const SamStreamEncodingAttr inner_enc = getSamStreamEncoding(input_inner_crd.getType()))
        {
            new_op->mutable_spacc()->set_inner_crd(inner_enc.getVar().str());
        }

        for (int i = 0; i < input_outer_crds.size(); ++i)
        {
            new_op->mutable_spacc()->add_input_outer_crds();
            new_op->mutable_spacc()->mutable_input_outer_crds(i)->mutable_id()->set_id(
                channel_tracker.get_create_channel(input_outer_crds[i]));
            map_channel[input_outer_crds[i]].emplace_back(new_op->mutable_spacc()->mutable_input_outer_crds(i));
        }
        new_op->mutable_spacc()->mutable_input_val()->mutable_id()->set_id(
            channel_tracker.get_create_channel(input_val));
        new_op->mutable_spacc()->mutable_output_inner_crd()->mutable_id()->set_id(
            channel_tracker.get_create_channel(output_inner_crd));
        new_op->mutable_spacc()->mutable_output_val()->mutable_id()->set_id(
            channel_tracker.get_create_channel(output_val));
        for (int i = 0; i < output_outer_crds.size(); ++i)
        {
            new_op->mutable_spacc()->add_output_outer_crds();
            new_op->mutable_spacc()->mutable_output_outer_crds(i)->mutable_id()->set_id(
                channel_tracker.get_create_channel(output_outer_crds[i]));
            if (SamStreamEncodingAttr outer_enc = getSamStreamEncoding(output_outer_crds[i].getType()))
            {
                new_op->mutable_spacc()->add_outer_crds();
                new_op->mutable_spacc()->set_outer_crds(i, outer_enc.getVar().str());
            }
        }

        map_channel[input_inner_crd].emplace_back(new_op->mutable_spacc()->mutable_input_inner_crd());
        map_channel[input_val].emplace_back(new_op->mutable_spacc()->mutable_input_val());
        return success();
    };

    LogicalResult ProtoEmitter::printOperation(sam::SamCrdDrop dropOp)
    {
        tortilla::Operation *new_op = pg.add_operators();
        std::string name = "crddrop";
        std::string label_pref = "CrdDrop";
        new_op->set_name(name);
        new_op->set_id(getNodeID());
        new_op->mutable_coord_drop()->set_label(label_pref);

        auto input_inner_crd = dropOp.getInputInnerCrd();
        auto input_outer_crd = dropOp.getInputOuterCrd();
        auto output_inner_crd = dropOp.getOutputInnerCrd();
        auto output_outer_crd = dropOp.getOutputOuterCrd();

        new_op->mutable_coord_drop()->mutable_input_inner_crd()->mutable_id()->set_id(
            channel_tracker.get_create_channel(input_inner_crd));
        SamStreamEncodingAttr inner_enc = getSamStreamEncoding(input_inner_crd.getType());
        if (inner_enc)
        {
            new_op->mutable_coord_drop()->set_inner_crd(inner_enc.getVar().str());
        }

        new_op->mutable_coord_drop()->mutable_input_outer_crd()->mutable_id()->set_id(
            channel_tracker.get_create_channel(input_outer_crd));
        map_channel[input_outer_crd].emplace_back(new_op->mutable_coord_drop()->mutable_input_outer_crd());

        new_op->mutable_coord_drop()->mutable_output_inner_crd()->mutable_id()->set_id(
            channel_tracker.get_create_channel(output_inner_crd));
        new_op->mutable_coord_drop()->mutable_output_outer_crd()->mutable_id()->set_id(
            channel_tracker.get_create_channel(output_outer_crd));
        SamStreamEncodingAttr outer_enc = getSamStreamEncoding(output_outer_crd.getType());
        // if (outer_enc) {
        new_op->mutable_coord_drop()->set_outer_crd(outer_enc.getVar().str());
        // }

        map_channel[input_inner_crd].emplace_back(new_op->mutable_coord_drop()->mutable_input_inner_crd());
        return success();
    };


    LogicalResult ProtoEmitter::printOperation(sam::SamScatter scatOp)
    {
        int64_t par_factor = scatOp.getParFactorAttr().getInt();
        auto input_stream = scatOp.getInputStream();
        auto output_stream = scatOp.getScatteredStreams();

        SamStreamEncodingAttr input_enc = getSamStreamEncoding(input_stream.getType());
        auto stream_type = input_enc.getStreamKind().getValue();

        tortilla::Fork *new_fork;
        auto new_op = pg.add_operators();
        new_op->set_name("fork");
        new_op->set_id(getNodeID());
        new_fork = new_op->mutable_fork();

        if (stream_type == sam::StreamKind::Crd)
        {
            new_fork->mutable_crd()->mutable_input()->mutable_id()->set_id(
                channel_tracker.get_create_channel(input_stream));
            map_channel[input_stream].emplace_back(new_fork->mutable_crd()->mutable_input());
        }
        else if (stream_type == sam::StreamKind::Ref)
        {
            new_fork->mutable_ref()->mutable_input()->mutable_id()->set_id(
                channel_tracker.get_create_channel(input_stream));
            map_channel[input_stream].emplace_back(new_fork->mutable_ref()->mutable_input());
        }
        else if (stream_type == sam::StreamKind::Val)
        {
            new_fork->mutable_val()->mutable_input()->mutable_id()->set_id(
                channel_tracker.get_create_channel(input_stream));
            map_channel[input_stream].emplace_back(new_fork->mutable_val()->mutable_input());
        }

        for (int i = 0; i < par_factor; i++)
        {
            if (stream_type == sam::StreamKind::Crd)
            {
                new_fork->mutable_crd()->mutable_outputs()->Add()->mutable_id()->set_id(
                    channel_tracker.get_create_channel(output_stream[i]));
            }
            else if (stream_type == sam::StreamKind::Ref)
            {
                new_fork->mutable_ref()->mutable_outputs()->Add()->mutable_id()->set_id(
                    channel_tracker.get_create_channel(output_stream[i]));
            }
            else if (stream_type == sam::StreamKind::Val)
            {
                new_fork->mutable_val()->mutable_outputs()->Add()->mutable_id()->set_id(
                    channel_tracker.create_channel(output_stream[i]));
            }
        }

        return success();
    }

    LogicalResult ProtoEmitter::printOperation(sam::SamGather gatOp)
    {
        int64_t par_factor = gatOp.getParFactorAttr().getInt();
        auto input_stream = gatOp.getScatterStreams();
        auto output_stream = gatOp.getJoinedStream();
        SamStreamEncodingAttr input_enc = getSamStreamEncoding(output_stream.getType());
        auto stream_type = input_enc.getStreamKind().getValue();

        tortilla::Join *new_join;
        auto new_op = pg.add_operators();
        new_op->set_name("join");
        new_op->set_id(getNodeID());
        new_join = new_op->mutable_join();

        for (int i = 0; i < par_factor; i++)
        {
            if (stream_type == sam::StreamKind::Crd)
            {
                new_join->mutable_crd()->mutable_inputs()->Add()->mutable_id()->set_id(
                    channel_tracker.get_create_channel(input_stream[i]));
                map_channel[input_stream[i]].push_back(new_join->mutable_crd()->mutable_inputs(i));
            }
            else if (stream_type == sam::StreamKind::Ref)
            {
                new_join->mutable_ref()->mutable_inputs()->Add()->mutable_id()->set_id(
                    channel_tracker.get_create_channel(input_stream[i]));
                map_channel[input_stream[i]].push_back(new_join->mutable_ref()->mutable_inputs(i));
            }
            else if (stream_type == sam::StreamKind::Val)
            {
                new_join->mutable_val()->mutable_inputs()->Add()->mutable_id()->set_id(
                    channel_tracker.get_create_channel(input_stream[i]));
                map_channel[input_stream[i]].push_back(new_join->mutable_val()->mutable_inputs(i));
            }
        }

        if (stream_type == sam::StreamKind::Crd)
        {
            new_join->mutable_crd()->mutable_output()->mutable_id()->set_id(
                channel_tracker.get_create_channel(output_stream));
        }
        else if (stream_type == sam::StreamKind::Ref)
        {
            new_join->mutable_ref()->mutable_output()->mutable_id()->set_id(
                channel_tracker.get_create_channel(output_stream));
        }
        else if (stream_type == sam::StreamKind::Val)
        {
            new_join->mutable_val()->mutable_output()->mutable_id()->set_id(
                channel_tracker.get_create_channel(output_stream));
        }

        return success();
    }

    void registerToProtoPass()
    {
        TranslateFromMLIRRegistration reg(
            "emit-proto", "translate from generic MLIR to Comal proto representation",
            [](Operation *op, llvm::raw_ostream &output) { return translateToProto(op, output); },
            [](DialectRegistry &registry)
            {
                registry
                    .insert<arith::ArithDialect, func::FuncDialect, mlir::linalg::LinalgDialect, memref::MemRefDialect,
                            affine::AffineDialect, math::MathDialect, tensor::TensorDialect, index::IndexDialect,
                            sam::SamDialect, sparse_tensor::SparseTensorDialect, ml_program::MLProgramDialect>();
            });
    };

    LogicalResult translateToProto(Operation *op, llvm::raw_ostream &os)
    {
        ProtoEmitter translator(os);
        return translator.translate(*op);
    };
} // namespace mlir::sam
