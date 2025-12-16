//
// Created by rubensl on 11/3/24.
//

#ifndef SAMML_CONSTRUCTEINSUM_H
#define SAMML_CONSTRUCTEINSUM_H

#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include "FusedCIN.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/SparseTensor/Utils/Merger.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"

class ASTVisitor;

class TensorNode;

class BinaryOpNode;

class UnaryOpNode;

class FuncOpNode;

static std::string indexVocab = "ijklmnopqrstuvwxyzabcdefgh";

class ASTNode;

static std::string formatIndexVar(const IndexVar &var, const bool useIndexVocab = false)
{
    if (useIndexVocab)
    {
        return std::string(1, indexVocab[var.getId()]);
    }
    else
    {
        return "i" + std::to_string(var.getId());
    }
}

// Forward declare FuncOpNode before the other node classes
class FuncOpNode : public ASTNode
{
public:
    std::string opType;
    std::vector<std::shared_ptr<ASTNode>> inputs;
    std::optional<int> concatDim; // Optional, used for concat operations
    int opCount{0};
    int bytesCount{0};
    TensorView intermediateView;
    bool useIndexVocab{false};

    std::map<int, std::vector<std::string>> dimensionMapping;

    FuncOpNode(const std::string &opType, const std::vector<std::shared_ptr<ASTNode>> &inputs, const TensorView &view,
               std::optional<int> concatDim = std::nullopt, const bool useIndexVocab = false) :
        opType(opType), inputs(inputs), concatDim(concatDim), intermediateView(view), useIndexVocab(useIndexVocab)
    {
        if (opType == "concat" && concatDim.has_value())
        {
            calculateConcatDimensionMapping();
        }
    }

    TensorView getView() override { return intermediateView; }
    TensorView getView() const override { return intermediateView; }
    NodeType nodeType() override { return NodeType::FuncOp; }
    NodeType nodeType() const override { return NodeType::FuncOp; }
    void accept(ASTVisitor &visitor) override;
    double getBytesCount() override { return bytesCount; }
    double getBytesCount() const override { return bytesCount; }
    double getOpsCount() override { return opCount; }
    double getOpsCount() const override { return opCount; }

    // ------------------------- FusedCodegenEnv Integration Methods ------------------------------
    unsigned getNumTensors() const override
    {
        unsigned total = 0;
        for (const auto &input : inputs)
        {
            total += input->getNumTensors();
        }
        return total;
    }

    unsigned getNumLoops() const override { return intermediateView.getUniqueVars().size(); }

    unsigned getMaxRank() const override
    {
        unsigned maxRank = intermediateView.getUniqueVars().size();
        for (const auto &input : inputs)
        {
            maxRank = std::max(maxRank, input->getMaxRank());
        }
        return maxRank;
    }

    llvm::SmallVector<mlir::Value> getTensorOperands() const override
    {
        llvm::SmallVector<mlir::Value> operands;
        for (const auto &input : inputs)
        {
            auto inputOps = input->getTensorOperands();
            operands.insert(operands.end(), inputOps.begin(), inputOps.end());
        }
        return operands;
    }

    llvm::SmallVector<mlir::Type> getTensorTypes() const override
    {
        llvm::SmallVector<mlir::Type> types;
        for (const auto &input : inputs)
        {
            auto inputTypes = input->getTensorTypes();
            types.insert(types.end(), inputTypes.begin(), inputTypes.end());
        }
        return types;
    }

    llvm::SmallVector<mlir::AffineMap> getIndexingMaps() const override
    {
        llvm::SmallVector<mlir::AffineMap> maps;
        for (const auto &input : inputs)
        {
            auto inputMaps = input->getIndexingMaps();
            maps.insert(maps.end(), inputMaps.begin(), inputMaps.end());
        }
        return maps;
    }

    llvm::SmallVector<mlir::Value> getLoopVars() const override { return llvm::SmallVector<mlir::Value>(); }

    mlir::Value getLoopVar(unsigned i) const override { return mlir::Value(); }

    mlir::Value getRootExpression() const override { return intermediateView.getValue(); }

    mlir::Operation *getYieldOp() const override { return nullptr; }

    llvm::SmallVector<mlir::sparse_tensor::SparseTensorEncodingAttr> getSparseEncodings() const override
    {
        llvm::SmallVector<mlir::sparse_tensor::SparseTensorEncodingAttr> encodings;
        for (const auto &input : inputs)
        {
            auto inputEncodings = input->getSparseEncodings();
            encodings.insert(encodings.end(), inputEncodings.begin(), inputEncodings.end());
        }
        return encodings;
    }

    bool hasSparseOutput() const override { return !getSparseEncodings().empty(); }

    mlir::OpOperand *getSparseOutput() const override { return nullptr; }

    bool isScheduled() const override { return true; }
    void setScheduled(bool scheduled) override { /* No-op for now */ }

    std::string getDimensionMappingString() const
    {
        if (opType != "concat" || !concatDim.has_value() || dimensionMapping.empty())
        {
            return "";
        }

        const int dim = concatDim.value();
        if (dimensionMapping.find(dim) == dimensionMapping.end())
        {
            return "";
        }

        const auto &inputDims = dimensionMapping.at(dim);
        if (inputDims.empty())
        {
            return "";
        }

        // Build the mapping string: "i+l->o"
        std::string mapping = "";
        for (size_t i = 0; i < inputDims.size(); ++i)
        {
            if (i > 0)
                mapping += "+";
            mapping += inputDims[i];
        }

        // Get output dimension name
        auto outputVars = intermediateView.getUniqueVars();
        if (dim < outputVars.size())
        {
            const auto outputIndexVar = outputVars[dim].getLogicalIndexVar();
            const std::string outputDim = formatIndexVar(outputIndexVar, useIndexVocab);
            mapping += "->" + outputDim;
        }

        return mapping;
    }

private:
    void calculateConcatDimensionMapping()
    {
        if (opType != "concat" || !concatDim.has_value() || inputs.empty())
        {
            return;
        }

        int dim = concatDim.value();
        std::vector<std::string> inputDims;

        // Get dimension names for each input tensor at the concat dimension
        for (size_t i = 0; i < inputs.size(); ++i)
        {
            if (auto inputView = inputs[i]->getView(); !inputView.empty())
            {
                if (auto uniqueVars = inputView.getUniqueVars(); dim < uniqueVars.size())
                {
                    const auto indexVar = uniqueVars[dim].getLogicalIndexVar();
                    const std::string dimName = formatIndexVar(indexVar, useIndexVocab);
                    inputDims.push_back(dimName);
                }
            }
        }

        if (!inputDims.empty())
        {
            dimensionMapping[dim] = inputDims;
        }
    }
};

class TensorNode : public ASTNode
{
public:
    TensorView view;
    std::string tensor_name;
    int opCount{0};
    int bytesCount{0};

    explicit TensorNode(const TensorView &tensorView) :
        view(tensorView), tensor_name(tensorView.getName().substr(0, tensorView.getName().find('-')))
    {
    }

    TensorView getView() override { return view; }

    TensorView getView() const override { return view; }

    NodeType nodeType() override { return NodeType::Tensor; }

    NodeType nodeType() const override { return NodeType::Tensor; }

    void accept(ASTVisitor &visitor) override;

    double getBytesCount() override { return bytesCount; }

    double getBytesCount() const override { return bytesCount; }

    double getOpsCount() override { return opCount; }

    double getOpsCount() const override { return opCount; }

    // ------------------------- FusedCodegenEnv Integration Methods ------------------------------

    unsigned getNumTensors() const override { return 1; } // This node represents one tensor
    unsigned getNumLoops() const override { return view.getUniqueVars().size(); }
    unsigned getMaxRank() const override { return view.getUniqueVars().size(); }

    llvm::SmallVector<mlir::Value> getTensorOperands() const override
    {
        llvm::SmallVector<mlir::Value> operands;
        if (!view.empty())
        {
            operands.push_back(view.getValue());
        }
        return operands;
    }

    llvm::SmallVector<mlir::Type> getTensorTypes() const override
    {
        llvm::SmallVector<mlir::Type> types;
        if (!view.empty())
        {
            types.push_back(view.getValue().getType());
        }
        return types;
    }

    llvm::SmallVector<mlir::AffineMap> getIndexingMaps() const override
    {
        // For tensor nodes, create identity maps for each dimension
        llvm::SmallVector<mlir::AffineMap> maps;
        if (!view.empty())
        {
            auto type = view.getValue().getType();
            if (mlir::isa<mlir::RankedTensorType>(type))
            {
                auto tensorType = mlir::cast<mlir::RankedTensorType>(type);
                for (unsigned i = 0; i < tensorType.getRank(); ++i)
                {
                    maps.push_back(mlir::AffineMap::get(1, 0, mlir::getAffineDimExpr(0, view.getValue().getContext())));
                }
            }
        }
        return maps;
    }

    llvm::SmallVector<mlir::Value> getLoopVars() const override
    {
        // Return empty for now - loop vars are typically created during codegen
        return llvm::SmallVector<mlir::Value>();
    }

    mlir::Value getLoopVar(unsigned i) const override
    {
        // Return null value - loop vars are typically created during codegen
        return mlir::Value();
    }

    mlir::Value getRootExpression() const override { return view.getValue(); }

    mlir::Operation *getYieldOp() const override
    {
        // Tensor nodes don't have yield ops
        return nullptr;
    }

    llvm::SmallVector<mlir::sparse_tensor::SparseTensorEncodingAttr> getSparseEncodings() const override
    {
        llvm::SmallVector<mlir::sparse_tensor::SparseTensorEncodingAttr> encodings;
        if (!view.empty())
        {
            auto type = view.getValue().getType();
            if (mlir::isa<mlir::RankedTensorType>(type))
            {
                auto tensorType = mlir::cast<mlir::RankedTensorType>(type);
                auto encoding = tensorType.getEncoding();
                if (mlir::isa<mlir::sparse_tensor::SparseTensorEncodingAttr>(encoding))
                {
                    encodings.push_back(mlir::cast<mlir::sparse_tensor::SparseTensorEncodingAttr>(encoding));
                }
            }
        }
        return encodings;
    }

    bool hasSparseOutput() const override { return !getSparseEncodings().empty(); }

    mlir::OpOperand *getSparseOutput() const override
    {
        // Tensor nodes don't have OpOperands directly
        return nullptr;
    }

    bool isScheduled() const override { return true; } // Assume scheduled by default
    void setScheduled(bool scheduled) override { /* No-op for now */ }
};

class BinaryOpNode : public ASTNode
{
public:
    std::shared_ptr<ASTNode> left;
    std::shared_ptr<ASTNode> right;
    std::string op; // "+" for addition, "*" for multiplication, etc.
    int opCount{0};
    int bytesCount{0};
    std::vector<IndexVar> loopOrder;
    std::optional<IndexVar> redVar;
    TensorNode intermediateView;

    BinaryOpNode(std::shared_ptr<ASTNode> left, std::shared_ptr<ASTNode> right, std::string op, const TensorView &view,
                 const std::vector<IndexVar> &loop, std::optional<IndexVar> red = {}) :
        left(std::move(left)), right(std::move(right)), op(std::move(op)), loopOrder(loop), intermediateView(view),
        redVar(red)
    {
    }

    TensorView getView() override { return intermediateView.getView(); }

    TensorView getView() const override { return intermediateView.getView(); }

    NodeType nodeType() override { return NodeType::BinaryOp; }

    NodeType nodeType() const override { return NodeType::BinaryOp; }

    void accept(ASTVisitor &visitor) override;

    double getBytesCount() override { return bytesCount; }

    double getBytesCount() const override { return bytesCount; }

    double getOpsCount() override { return opCount; }

    double getOpsCount() const override { return opCount; }

    // ------------------------- FusedCodegenEnv Integration Methods ------------------------------

    unsigned getNumTensors() const override
    {
        // Binary op combines tensors from left and right children
        return left->getNumTensors() + right->getNumTensors();
    }

    unsigned getNumLoops() const override
    {
        // Use the loop order from this node
        return loopOrder.size();
    }

    unsigned getMaxRank() const override
    {
        // Maximum rank from children and this node
        return std::max({left->getMaxRank(), right->getMaxRank(),
                         static_cast<unsigned>(intermediateView.getView().getUniqueVars().size())});
    }

    llvm::SmallVector<mlir::Value> getTensorOperands() const override
    {
        llvm::SmallVector<mlir::Value> operands;
        auto leftOps = left->getTensorOperands();
        auto rightOps = right->getTensorOperands();
        operands.insert(operands.end(), leftOps.begin(), leftOps.end());
        operands.insert(operands.end(), rightOps.begin(), rightOps.end());
        return operands;
    }

    llvm::SmallVector<mlir::Type> getTensorTypes() const override
    {
        llvm::SmallVector<mlir::Type> types;
        auto leftTypes = left->getTensorTypes();
        auto rightTypes = right->getTensorTypes();
        types.insert(types.end(), leftTypes.begin(), leftTypes.end());
        types.insert(types.end(), rightTypes.begin(), rightTypes.end());
        return types;
    }

    llvm::SmallVector<mlir::AffineMap> getIndexingMaps() const override
    {
        llvm::SmallVector<mlir::AffineMap> maps;
        auto leftMaps = left->getIndexingMaps();
        auto rightMaps = right->getIndexingMaps();
        maps.insert(maps.end(), leftMaps.begin(), leftMaps.end());
        maps.insert(maps.end(), rightMaps.begin(), rightMaps.end());
        return maps;
    }

    llvm::SmallVector<mlir::Value> getLoopVars() const override
    {
        // Return empty for now - loop vars are typically created during codegen
        return llvm::SmallVector<mlir::Value>();
    }

    mlir::Value getLoopVar(unsigned i) const override
    {
        // Return null value - loop vars are typically created during codegen
        return mlir::Value();
    }

    mlir::Value getRootExpression() const override { return intermediateView.getView().getValue(); }

    mlir::Operation *getYieldOp() const override
    {
        // Binary op nodes don't have yield ops directly
        return nullptr;
    }

    llvm::SmallVector<mlir::sparse_tensor::SparseTensorEncodingAttr> getSparseEncodings() const override
    {
        llvm::SmallVector<mlir::sparse_tensor::SparseTensorEncodingAttr> encodings;
        auto leftEncodings = left->getSparseEncodings();
        auto rightEncodings = right->getSparseEncodings();
        encodings.insert(encodings.end(), leftEncodings.begin(), leftEncodings.end());
        encodings.insert(encodings.end(), rightEncodings.begin(), rightEncodings.end());
        return encodings;
    }

    bool hasSparseOutput() const override { return !getSparseEncodings().empty(); }

    mlir::OpOperand *getSparseOutput() const override
    {
        // Binary op nodes don't have OpOperands directly
        return nullptr;
    }

    bool isScheduled() const override { return true; } // Assume scheduled by default
    void setScheduled(bool scheduled) override { /* No-op for now */ }
};

class UnaryOpNode : public ASTNode
{
public:
    std::shared_ptr<ASTNode> input;
    std::string op; // "+" for addition, "*" for multiplication, etc.
    int opCount{0};
    int bytesCount{0};
    std::vector<IndexVar> loopOrder;
    TensorNode intermediateView;

    UnaryOpNode(std::shared_ptr<ASTNode> input, const TensorView &view, std::string op) :
        input(std::move(input)), op(std::move(op)), intermediateView(view)
    {
        // Set loop order to the index variables in the intermediateView
        auto uniqueVars = view.getUniqueVars();
        for (auto &var : uniqueVars)
        {
            loopOrder.push_back(var.getLogicalIndexVar());
        }
    }

    TensorView getView() override { return intermediateView.getView(); }

    TensorView getView() const override { return intermediateView.getView(); }

    NodeType nodeType() override { return NodeType::UnaryOp; }

    NodeType nodeType() const override { return NodeType::UnaryOp; }

    void accept(ASTVisitor &visitor) override;

    double getBytesCount() override { return bytesCount; }

    double getBytesCount() const override { return bytesCount; }

    double getOpsCount() override { return opCount; }

    double getOpsCount() const override { return opCount; }

    // ------------------------- FusedCodegenEnv Integration Methods ------------------------------

    unsigned getNumTensors() const override
    {
        // Unary op uses tensors from input child
        return input->getNumTensors();
    }

    unsigned getNumLoops() const override
    {
        // Use the loop order from this node
        return loopOrder.size();
    }

    unsigned getMaxRank() const override
    {
        // Maximum rank from input child and this node
        return std::max(input->getMaxRank(), static_cast<unsigned>(intermediateView.getView().getUniqueVars().size()));
    }

    llvm::SmallVector<mlir::Value> getTensorOperands() const override { return input->getTensorOperands(); }

    llvm::SmallVector<mlir::Type> getTensorTypes() const override { return input->getTensorTypes(); }

    llvm::SmallVector<mlir::AffineMap> getIndexingMaps() const override { return input->getIndexingMaps(); }

    llvm::SmallVector<mlir::Value> getLoopVars() const override
    {
        // Return empty for now - loop vars are typically created during codegen
        return llvm::SmallVector<mlir::Value>();
    }

    mlir::Value getLoopVar(unsigned i) const override
    {
        // Return null value - loop vars are typically created during codegen
        return mlir::Value();
    }

    mlir::Value getRootExpression() const override { return intermediateView.getView().getValue(); }

    mlir::Operation *getYieldOp() const override
    {
        // Unary op nodes don't have yield ops directly
        return nullptr;
    }

    llvm::SmallVector<mlir::sparse_tensor::SparseTensorEncodingAttr> getSparseEncodings() const override
    {
        return input->getSparseEncodings();
    }

    bool hasSparseOutput() const override { return !getSparseEncodings().empty(); }

    mlir::OpOperand *getSparseOutput() const override
    {
        // Unary op nodes don't have OpOperands directly
        return nullptr;
    }

    bool isScheduled() const override { return true; } // Assume scheduled by default
    void setScheduled(bool scheduled) override { /* No-op for now */ }
};

class ScalarNode : public ASTNode
{
public:
    double value;

    explicit ScalarNode(double value) : value(value) {}

    TensorView getView() override { return TensorView{}; }

    TensorView getView() const override { return TensorView{}; }

    NodeType nodeType() override { return NodeType::Scalar; }

    NodeType nodeType() const override { return NodeType::Scalar; }

    void accept(ASTVisitor &visitor) override;

    double getBytesCount() override { return 0.0; }

    double getBytesCount() const override { return 0.0; }

    double getOpsCount() override { return 0.0; }

    double getOpsCount() const override { return 0.0; }

    // ------------------------- FusedCodegenEnv Integration Methods ------------------------------

    unsigned getNumTensors() const override { return 0; } // Scalars don't represent tensors
    unsigned getNumLoops() const override { return 0; } // Scalars don't have loops
    unsigned getMaxRank() const override { return 0; } // Scalars don't have rank

    llvm::SmallVector<mlir::Value> getTensorOperands() const override
    {
        return llvm::SmallVector<mlir::Value>(); // Scalars don't have tensor operands
    }

    llvm::SmallVector<mlir::Type> getTensorTypes() const override
    {
        return llvm::SmallVector<mlir::Type>(); // Scalars don't have tensor types
    }

    llvm::SmallVector<mlir::AffineMap> getIndexingMaps() const override
    {
        return llvm::SmallVector<mlir::AffineMap>(); // Scalars don't have indexing maps
    }

    llvm::SmallVector<mlir::Value> getLoopVars() const override
    {
        return llvm::SmallVector<mlir::Value>(); // Scalars don't have loop vars
    }

    mlir::Value getLoopVar(unsigned i) const override
    {
        return mlir::Value(); // Scalars don't have loop vars
    }

    mlir::Value getRootExpression() const override
    {
        return mlir::Value(); // Scalars don't have root expressions
    }

    mlir::Operation *getYieldOp() const override
    {
        return nullptr; // Scalars don't have yield ops
    }

    llvm::SmallVector<mlir::sparse_tensor::SparseTensorEncodingAttr> getSparseEncodings() const override
    {
        return llvm::SmallVector<mlir::sparse_tensor::SparseTensorEncodingAttr>(); // Scalars don't have sparse
                                                                                   // encodings
    }

    bool hasSparseOutput() const override
    {
        return false; // Scalars don't have sparse output
    }

    mlir::OpOperand *getSparseOutput() const override
    {
        return nullptr; // Scalars don't have sparse output
    }

    bool isScheduled() const override { return true; } // Scalars are always "scheduled"
    void setScheduled(bool scheduled) override { /* No-op for scalars */ }
};

class ASTVisitor
{
public:
    virtual void visit(BinaryOpNode &node) = 0;

    virtual void visit(UnaryOpNode &node) = 0;

    virtual void visit(TensorNode &node) = 0;

    virtual void visit(ScalarNode &node) = 0;

    virtual void visit(FuncOpNode &node) = 0;
};

class CSEVisitor : public ASTVisitor
{
public:
    std::shared_ptr<ASTNode> optimizedRoot;

    void visit(BinaryOpNode &node) override
    {
        node.left->accept(*this);
        node.left = optimizedRoot;

        node.right->accept(*this);
        node.right = optimizedRoot;

        auto hash = binaryHash(node.op, node.left, node.right);
        auto it = cseMap.find(hash);
        if (it != cseMap.end())
        {
            optimizedRoot = it->second;
        }
        else
        {
            optimizedRoot = std::make_shared<BinaryOpNode>(node.left, node.right, node.op, node.getView(),
                                                           node.loopOrder, node.redVar);
            cseMap[hash] = optimizedRoot;
        }
    }

    void visit(UnaryOpNode &node) override
    {
        node.input->accept(*this);
        node.input = optimizedRoot;

        auto hash = unaryHash(node.op, node.input);
        auto it = cseMap.find(hash);
        if (it != cseMap.end())
        {
            optimizedRoot = it->second;
        }
        else
        {
            optimizedRoot = std::make_shared<UnaryOpNode>(node.input, node.getView(), node.op);
            cseMap[hash] = optimizedRoot;
        }
    }

    void visit(TensorNode &node) override
    {
        auto hash = tensorHash(node);
        auto it = cseMap.find(hash);
        if (it != cseMap.end())
        {
            optimizedRoot = it->second;
        }
        else
        {
            optimizedRoot = std::make_shared<TensorNode>(node.getView());
            cseMap[hash] = optimizedRoot;
        }
    }

    void visit(ScalarNode &node) override
    {
        auto hash = scalarHash(node);
        auto it = cseMap.find(hash);
        if (it != cseMap.end())
        {
            optimizedRoot = it->second;
        }
        else
        {
            optimizedRoot = std::make_shared<ScalarNode>(node.value);
            cseMap[hash] = optimizedRoot;
        }
    }

    void visit(FuncOpNode &node) override
    {
        // Visit all inputs
        for (auto &input : node.inputs)
        {
            input->accept(*this);
            input = optimizedRoot;
        }

        // Create hash for function operation
        auto hash = funcHash(node.opType, node.inputs, node.concatDim);
        auto it = cseMap.find(hash);
        if (it != cseMap.end())
        {
            optimizedRoot = it->second;
        }
        else
        {
            optimizedRoot = std::make_shared<FuncOpNode>(node.opType, node.inputs, node.getView(), node.concatDim);
            cseMap[hash] = optimizedRoot;
        }
    }

private:
    std::unordered_map<std::string, std::shared_ptr<ASTNode>> cseMap;

    std::string binaryHash(const std::string &op, const std::shared_ptr<ASTNode> &left,
                           const std::shared_ptr<ASTNode> &right)
    {
        return "(" + op + "," + nodeHash(left) + "," + nodeHash(right) + ")";
    }

    std::string unaryHash(const std::string &op, const std::shared_ptr<ASTNode> &input)
    {
        return "(" + op + "," + nodeHash(input) + ")";
    }

    std::string tensorHash(const TensorNode &node) { return "Tensor(" + node.tensor_name + ")"; }

    std::string scalarHash(const ScalarNode &node) { return "Scalar(" + std::to_string(node.value) + ")"; }

    std::string funcHash(const std::string &opType, const std::vector<std::shared_ptr<ASTNode>> &inputs,
                         std::optional<int> concatDim)
    {
        std::string hash = "Func(" + opType;
        for (const auto &input : inputs)
        {
            hash += "," + nodeHash(input);
        }
        if (concatDim.has_value())
        {
            hash += ",dim=" + std::to_string(concatDim.value());
        }
        hash += ")";
        return hash;
    }

    std::string nodeHash(const std::shared_ptr<ASTNode> &node)
    {
        if (node->nodeType() == NodeType::BinaryOp)
        {
            auto bin = static_cast<BinaryOpNode *>(node.get());
            return binaryHash(bin->op, bin->left, bin->right);
        }
        if (node->nodeType() == NodeType::UnaryOp)
        {
            auto un = static_cast<UnaryOpNode *>(node.get());
            return unaryHash(un->op, un->input);
        }
        if (node->nodeType() == NodeType::Tensor)
        {
            auto tensor = static_cast<TensorNode *>(node.get());
            return tensorHash(*tensor);
        }
        if (node->nodeType() == NodeType::Scalar)
        {
            auto scalar = static_cast<ScalarNode *>(node.get());
            return scalarHash(*scalar);
        }
        if (node->nodeType() == NodeType::FuncOp)
        {
            auto func = static_cast<FuncOpNode *>(node.get());
            return funcHash(func->opType, func->inputs, func->concatDim);
        }
        return "Unknown";
    }
};

class EinsumTreeVisitor : public ASTVisitor
{
public:
    std::map<std::shared_ptr<ASTNode>, std::string> nodeIds;
    std::map<std::shared_ptr<ASTNode>, std::string> tempNames;
    std::vector<std::pair<std::shared_ptr<ASTNode>, std::string>> orderedNodes;
    std::vector<std::pair<std::string, std::string>> edges;
    int nodeCounter{0};
    int tempCounter{0};
    std::string currentNodeId;
    bool isRoot{true};
    bool useIndexVocab{false}; // Toggle for index variable formatting

    EinsumTreeVisitor() {}
    explicit EinsumTreeVisitor(bool useIndexVocab) : useIndexVocab(useIndexVocab) {}

    void printTree()
    {
        std::cout << "\n// Einsum Expression Tree:\n";
        std::cout << "// =======================\n";

        // Print all nodes in order
        for (const auto &[node, id] : orderedNodes)
        {
            std::cout << "// Node " << id << ": ";
            printNodeExpression(node);
            std::cout << "\n";
        }

        // Print edges/connections
        if (!edges.empty())
        {
            std::cout << "\n// Connections:\n";
            for (const auto &[parent, child] : edges)
            {
                std::cout << "//   " << parent << " -> " << child << "\n";
            }
        }
        std::cout << "\n";
    }

    void visit(BinaryOpNode &node) override
    {
        // Create node ID for this binary operation
        std::string nodeId = "N" + std::to_string(nodeCounter++);
        std::string tempName = "temp" + std::to_string(tempCounter++);
        currentNodeId = nodeId;
        nodeIds[std::shared_ptr<ASTNode>(&node, [](ASTNode *) {})] = nodeId;
        tempNames[std::shared_ptr<ASTNode>(&node, [](ASTNode *) {})] = tempName;
        orderedNodes.emplace_back(std::shared_ptr<ASTNode>(&node, [](ASTNode *) {}), nodeId);

        // Visit children and establish parent-child relationships
        if (node.left->nodeType() == NodeType::BinaryOp || node.left->nodeType() == NodeType::UnaryOp ||
            node.left->nodeType() == NodeType::FuncOp)
        {
            std::string parentId = currentNodeId;
            node.left->accept(*this);
            edges.emplace_back(parentId, currentNodeId);
        }

        if (node.right->nodeType() == NodeType::BinaryOp || node.right->nodeType() == NodeType::UnaryOp ||
            node.right->nodeType() == NodeType::FuncOp)
        {
            std::string parentId = currentNodeId;
            node.right->accept(*this);
            edges.emplace_back(parentId, currentNodeId);
        }

        currentNodeId = nodeId;
    }

    void visit(UnaryOpNode &node) override
    {
        // Create node ID for this unary operation
        std::string nodeId = "N" + std::to_string(nodeCounter++);
        std::string tempName = "temp" + std::to_string(tempCounter++);
        currentNodeId = nodeId;
        nodeIds[std::shared_ptr<ASTNode>(&node, [](ASTNode *) {})] = nodeId;
        tempNames[std::shared_ptr<ASTNode>(&node, [](ASTNode *) {})] = tempName;
        orderedNodes.emplace_back(std::shared_ptr<ASTNode>(&node, [](ASTNode *) {}), nodeId);

        // Visit child and establish parent-child relationship
        if (node.input->nodeType() == NodeType::BinaryOp || node.input->nodeType() == NodeType::UnaryOp ||
            node.input->nodeType() == NodeType::FuncOp)
        {
            std::string parentId = currentNodeId;
            node.input->accept(*this);
            edges.emplace_back(parentId, currentNodeId);
        }

        currentNodeId = nodeId;
    }

    void visit(TensorNode &node) override
    {
        // Tensor nodes don't create separate expressions, they're just operands
        // So we don't create node IDs for them
    }

    void visit(ScalarNode &node) override
    {
        // Scalar nodes don't create separate expressions
    }

    void visit(FuncOpNode &node) override
    {
        std::string nodeId = "N" + std::to_string(nodeCounter++);
        std::string tempName = "temp" + std::to_string(tempCounter++);
        currentNodeId = nodeId;
        nodeIds[std::shared_ptr<ASTNode>(&node, [](ASTNode *) {})] = nodeId;
        tempNames[std::shared_ptr<ASTNode>(&node, [](ASTNode *) {})] = tempName;
        orderedNodes.emplace_back(std::shared_ptr<ASTNode>(&node, [](ASTNode *) {}), nodeId);

        for (size_t i = 0; i < node.inputs.size(); ++i)
        {
            auto &input = node.inputs[i];
            if (input->nodeType() == NodeType::BinaryOp || input->nodeType() == NodeType::UnaryOp ||
                input->nodeType() == NodeType::FuncOp)
            {
                std::string parentId = currentNodeId;
                input->accept(*this);
                edges.emplace_back(parentId, currentNodeId);
            }
        }
        currentNodeId = nodeId;
    }

private:
    void printNodeExpression(std::shared_ptr<ASTNode> node)
    {
        if (node->nodeType() == NodeType::BinaryOp)
        {
            auto binNode = static_cast<BinaryOpNode *>(node.get());
            printTensorViewWithTempName(binNode->getView(), node);
            std::cout << " = ";
            printOperand(binNode->left);
            std::cout << " " << binNode->op << " ";
            printOperand(binNode->right);

            // Print loop order
            std::cout << " [Loop Order: ";
            for (size_t i = 0; i < binNode->loopOrder.size(); ++i)
            {
                std::cout << formatIndexVar(binNode->loopOrder[i], useIndexVocab);
                if (i != binNode->loopOrder.size() - 1)
                {
                    std::cout << ",";
                }
            }
            std::cout << "]";

            // Print reduction variable if present
            if (binNode->redVar.has_value())
            {
                std::cout << " [Reduction: " << formatIndexVar(binNode->redVar.value(), useIndexVocab) << "]";
            }
        }
        else if (node->nodeType() == NodeType::UnaryOp)
        {
            auto unaryNode = static_cast<UnaryOpNode *>(node.get());
            printTensorViewWithTempName(unaryNode->getView(), node);
            std::cout << " = " << unaryNode->op << "(";
            printOperand(unaryNode->input);
            std::cout << ")";

            // Print loop order
            std::cout << " [Loop Order: ";
            for (size_t i = 0; i < unaryNode->loopOrder.size(); ++i)
            {
                std::cout << formatIndexVar(unaryNode->loopOrder[i], useIndexVocab);
                if (i != unaryNode->loopOrder.size() - 1)
                {
                    std::cout << ",";
                }
            }
            std::cout << "]";
        }
        else if (node->nodeType() == NodeType::FuncOp)
        {
            auto funcNode = static_cast<FuncOpNode *>(node.get());
            printTensorViewWithTempName(funcNode->getView(), node);
            std::cout << " = " << funcNode->opType << "(";
            for (size_t i = 0; i < funcNode->inputs.size(); ++i)
            {
                printOperand(funcNode->inputs[i]);
                if (i < funcNode->inputs.size() - 1)
                {
                    std::cout << ", ";
                }
            }
            if (funcNode->opType == "concat" && funcNode->concatDim.has_value())
            {
                std::cout << ")[dim=" << funcNode->concatDim.value();

                // Add dimension mapping information
                std::string dimMapping = funcNode->getDimensionMappingString();
                if (!dimMapping.empty())
                {
                    std::cout << ", " << dimMapping;
                }

                std::cout << "]";
            }
            else
            {
                std::cout << ")";
            }
        }
    }

    void printOperand(std::shared_ptr<ASTNode> operand)
    {
        if (operand->nodeType() == NodeType::Tensor)
        {
            auto tensorNode = static_cast<TensorNode *>(operand.get());
            printTensorAccess(*tensorNode);
        }
        else if (operand->nodeType() == NodeType::Scalar)
        {
            auto scalarNode = static_cast<ScalarNode *>(operand.get());
            std::cout << scalarNode->value;
        }
        else if (operand->nodeType() == NodeType::BinaryOp || operand->nodeType() == NodeType::UnaryOp ||
                 operand->nodeType() == NodeType::FuncOp)
        {
            // For nested operations, print the temp name if available, otherwise use the tensor view
            auto it = tempNames.find(operand);
            if (it != tempNames.end())
            {
                // Print temp name with indices
                printTensorViewWithTempName(operand->getView(), operand);
            }
            else
            {
                printTensorView(operand->getView());
            }
        }
    }

    void printTensorViewWithTempName(const TensorView &view, std::shared_ptr<ASTNode> node)
    {
        auto uniqueVars = view.getUniqueVars();
        std::vector<IndexVar> vars;
        for (auto &var : uniqueVars)
        {
            vars.push_back(var.getLogicalIndexVar());
        }

        // Always use temp name for intermediate results
        std::string tensorName = "temp";
        auto tempIt = tempNames.find(node);
        if (tempIt != tempNames.end())
        {
            tensorName = tempIt->second;
        }
        else
        {
            // Fallback for nodes without temp names (shouldn't happen for intermediate results)
            tensorName = "temp" + std::to_string(tempCounter++);
        }

        std::cout << tensorName << "(";
        for (size_t i = 0; i < vars.size(); ++i)
        {
            std::cout << formatIndexVar(vars[i], useIndexVocab);
            if (i != vars.size() - 1)
            {
                std::cout << ",";
            }
        }
        std::cout << ")";
    }

    void printTensorAccess(const TensorNode &node)
    {
        auto uniqueVars = node.view.getUniqueVars();
        std::vector<IndexVar> vars;
        for (auto &var : uniqueVars)
        {
            vars.push_back(var.getLogicalIndexVar());
        }
        std::cout << node.tensor_name << "(";
        for (size_t i = 0; i < vars.size(); ++i)
        {
            std::cout << formatIndexVar(vars[i], useIndexVocab);
            if (i != vars.size() - 1)
            {
                std::cout << ",";
            }
        }
        std::cout << ")";
    }

    void printTensorView(const TensorView &view)
    {
        auto uniqueVars = view.getUniqueVars();
        std::vector<IndexVar> vars;
        for (auto &var : uniqueVars)
        {
            vars.push_back(var.getLogicalIndexVar());
        }

        // Generate a name based on the view or use "temp" for intermediate results
        std::string tensorName = "temp";
        // Note: The LLVM/MLIR code is commented out due to namespace issues in the header
        // if (auto blockArg = llvm::dyn_cast<mlir::BlockArgument>(view.getValue())) {
        //     tensorName = "arg" + std::to_string(blockArg.getArgNumber());
        // }

        std::cout << tensorName << "(";
        for (size_t i = 0; i < vars.size(); ++i)
        {
            std::cout << formatIndexVar(vars[i], useIndexVocab);
            if (i != vars.size() - 1)
            {
                std::cout << ",";
            }
        }
        std::cout << ")";
    }
};

class GetExprVisitor : public ASTVisitor
{
public:
    std::optional<TensorNode> finalView;
    bool lhsEmitted{false};
    bool useIndexVocab{false}; // Toggle for index variable formatting

    GetExprVisitor() = default;

    // Store left hand side tensor for returned tensor view
    explicit GetExprVisitor(const TensorView &view) : finalView(TensorNode(view))
    {
        std::cout << "\n// Fused Einsum Expression:\n";
        std::cout << "// ========================\n";
    }
    explicit GetExprVisitor(bool useIndexVocab) : useIndexVocab(useIndexVocab)
    {
        std::cout << "\n// Fused Einsum Expression:\n";
        std::cout << "// =======================\n";
    }
    GetExprVisitor(const TensorView &view, bool useIndexVocab) :
        finalView(TensorNode(view)), useIndexVocab(useIndexVocab)
    {
        std::cout << "\n// Fused Einsum Expression:\n";
        std::cout << "// =======================\n";
    }

    void visit(BinaryOpNode &node) override
    {
        // Generate code for binary operation
        if (!lhsEmitted)
        {
            std::cout << "// ";
            finalView.value().accept(*this);
            std::cout << "=";
            lhsEmitted = true;
        }
        node.left->accept(*this);
        std::cout << node.op;
        node.right->accept(*this);
    }

    void visit(UnaryOpNode &node) override
    {
        // Generate code for binary operation
        if (!lhsEmitted)
        {
            finalView.value().accept(*this);
            std::cout << "=";
            lhsEmitted = true;
        }
        std::cout << node.op << "(";
        node.input->accept(*this);
        std::cout << ")";
    }

    void visit(TensorNode &node) override
    {
        // Generate code for tensor access
        auto uniqueVars = node.view.getUniqueVars();
        std::vector<IndexVar> vars;
        for (auto &var : uniqueVars)
        {
            vars.push_back(var.getLogicalIndexVar());
        }
        std::cout << node.tensor_name << "(";
        for (auto varIter : llvm::enumerate(vars))
        {
            auto var = varIter.value();
            unsigned int index = varIter.index();
            std::cout << formatIndexVar(var, useIndexVocab);
            if (index != vars.size() - 1)
            {
                std::cout << ",";
            }
        }
        std::cout << ")";
    }

    void visit(ScalarNode &node) override
    {
        // Generate code for scalar values
        llvm::outs() << "Inside scalar: Not implemented yet";
    }

    void visit(FuncOpNode &node) override
    {
        if (!lhsEmitted)
        {
            finalView.value().accept(*this);
            std::cout << "=";
            lhsEmitted = true;
        }
        std::cout << node.opType << "(";
        for (size_t i = 0; i < node.inputs.size(); ++i)
        {
            node.inputs[i]->accept(*this);
            if (i < node.inputs.size() - 1)
            {
                std::cout << ", ";
            }
        }
        if (node.opType == "concat" && node.concatDim.has_value())
        {
            std::cout << ")[dim=" << node.concatDim.value();

            // Add dimension mapping information
            std::string dimMapping = node.getDimensionMappingString();
            if (!dimMapping.empty())
            {
                std::cout << ", " << dimMapping;
            }

            std::cout << "]";
        }
        else
        {
            std::cout << ")";
        }
    }
};

class CollectIndicesVisitor : public ASTVisitor
{
public:
    std::map<IndexVar, std::vector<std::pair<std::string, std::string>>> indexData;

    void visit(BinaryOpNode &node) override
    {
        node.left->accept(*this);
        node.right->accept(*this);
    }

    void visit(UnaryOpNode &node) override { node.input->accept(*this); }

    void visit(TensorNode &node) override
    {
        auto uniqueVars = node.view.getUniqueVars();
        std::vector<IndexVar> vars;
        for (auto varIter : llvm::enumerate(uniqueVars))
        {
            const auto &var = varIter.value();
            unsigned int idx = varIter.index();
            auto indexVar = var.getLogicalIndexVar();
            std::string format;
            if (const auto enc = mlir::sparse_tensor::getSparseTensorEncoding(node.view.getValue().getType()))
            {
                const auto lvlType = enc.getLvlTypes();
                format = toMLIRString(lvlType[idx]);
            }
            else
            {
                format = "dense";
            }
            indexData[indexVar].emplace_back(node.tensor_name, format);
            vars.push_back(indexVar);
        }
    }

    void visit(ScalarNode &node) override
    {
        // Generate code for scalar values
        llvm::outs() << "Inside scalar: Not implemented yet";
    }

    void visit(FuncOpNode &node) override
    {
        for (auto &input : node.inputs)
        {
            input->accept(*this);
        }
    }
};

class StatVisitor : public ASTVisitor
{
public:
    // Index vars along with the estimated number of coordinates read
    // For intersected vars, user input for estimated for percentage nnz's is used to estimate
    std::map<IndexVar, double> indexVarDim;
    std::map<IndexVar, double> indexVarSparsity;
    bool returnedTensorAdded{false};
    int opsCount{0};
    int bytesCount{0};

    explicit StatVisitor(std::map<IndexVar, double> &indices, std::map<IndexVar, double> &sparsity) :
        indexVarDim(indices), indexVarSparsity(sparsity)
    {
    }

    void visit(BinaryOpNode &node) override
    {
        // Traverse children to update ops count and bytes count
        // TensorNode traversals just update the bytes count while op nodes update the ops count
        // TODO: Might be incorrect since I don't factor previous counts

        if (!returnedTensorAdded)
        {
            double interBytes = 1.0;
            for (auto &var : node.intermediateView.getView().getUniqueVars())
            {
                auto indexVar = var.getLogicalIndexVar();
                // Write seg writes for current level
                node.bytesCount += interBytes;
                auto currCount = indexVarDim.at(indexVar);
                if (indexVarSparsity.count(indexVar))
                {
                    // currCount /= indexVarSparsity.at(indexVar);
                }
                interBytes *= currCount;
                // Write crd writes for current level
                node.bytesCount += interBytes;
            }
            // Write value writes, should be same as last level
            node.bytesCount += interBytes;
            returnedTensorAdded = true;
        }

        node.left->accept(*this);
        node.right->accept(*this);

        // Get bytes and ops count from traversing input nodes of binary op
        // auto child1_bytes = node.left->getBytesCount();
        auto child1_bytes = node.left->getBytesCount();
        auto child2_bytes = node.right->getBytesCount();

        auto child1_ops = node.left->getOpsCount();
        auto child2_ops = node.right->getOpsCount();

        //        std::cout << "Operand 1 bytes: " << child1_bytes << std::endl;
        //        std::cout << "Operand 2 bytes: " << child2_bytes << std::endl;

        node.opCount = child1_ops + child2_ops;
        double currOpCount = 1.0;

        // Get loop order involved in current op
        auto loops = node.intermediateView.getView().getLoops();
        auto localOrderBack = node.loopOrder.back();
        std::vector scopedLoops(loops.begin(), loops.end());
        unsigned int lastLoopIdx = llvm::find(scopedLoops, localOrderBack) - scopedLoops.begin();
        auto node1IndexVars = llvm::map_to_vector(node.left->getView().getUniqueVars(),
                                                  [&](UniqueVar &a) { return a.getLogicalIndexVar(); });
        auto node2IndexVars = llvm::map_to_vector(node.right->getView().getUniqueVars(),
                                                  [&](UniqueVar &a) { return a.getLogicalIndexVar(); });
        for (int i = 0; i <= lastLoopIdx; i++)
        {
            auto globalLoop = scopedLoops[i];
            auto indexVarSize = indexVarDim.at(globalLoop);

            if (node.op == "*" || node.op == "/")
            {
                currOpCount *= indexVarSize;
                //            if (llvm::dyn_cast<TensorNode>(node.left)) {}
                //            auto t = std::dynamic_pointer_cast<TensorNode>(node.left.get());
                if (llvm::dyn_cast<mlir::BlockArgument>(node.left->getView().getValue()) ||
                    llvm::dyn_cast<mlir::arith::ConstantOp>(node.left->getView().getValue().getDefiningOp()))
                {
                    if (llvm::find(node1IndexVars, globalLoop) == node1IndexVars.end())
                    {
                        //                    node.bytesCount += child1_bytes;
                        child1_bytes *= indexVarSize;
                    }
                    if (i == 0)
                    {
                        child1_bytes *= 2;
                    }
                }
                if (llvm::dyn_cast<mlir::BlockArgument>(node.right->getView().getValue()) ||
                    llvm::dyn_cast<mlir::arith::ConstantOp>(node.right->getView().getValue().getDefiningOp()))
                {
                    if (llvm::find(node2IndexVars, globalLoop) == node2IndexVars.end())
                    {
                        //                    node.bytesCount += child2_bytes;
                        child2_bytes *= indexVarSize;
                    }
                    if (i == 0)
                    {
                        child2_bytes *= 2;
                    }
                }
            }
            else if (node.op == "+" || node.op == "-")
            {
                if (llvm::find(node1IndexVars, globalLoop) != node1IndexVars.end())
                {
                    currOpCount *= indexVarSize;
                }
            }
        }

        //        std::cout << "Operand 1 bytes: " << child1_bytes << std::endl;
        //        std::cout << "Operand 2 bytes: " << child2_bytes << std::endl;

        // Should generalize to all binary ops since loop order and existence of reduction variable would dictate what
        // gets added
        if (node.redVar.has_value())
        {
            // TODO: Figure out if this applies for all dataflows
            currOpCount *= 2;
        }
        node.opCount += currOpCount;
        node.bytesCount += child1_bytes + child2_bytes;

        //         if (!returnedTensorAdded) {
        //             node.intermediateView.accept(*this);
        //             bytesCount += node.intermediateView.bytesCount;
        //             returnedTensorAdded = true;
        //         }

        //         node.left->accept(*this);
        //         node.right->accept(*this);
        // //        if (node.op == "*") {
        //         int currOpCount = 1;
        //         for (auto &var: node.loopOrder) {
        //             currOpCount *= indexVarDim.at(var);
        //         }
        //         // Should generalize to all binary ops since loop order and existence of reduction variable would
        //         dictate what gets added if (node.redVar.has_value()) {
        //             int order = (int) (llvm::find(node.loopOrder, node.redVar.value()) - node.loopOrder.begin());
        //             int currRedCount = 1;
        //             for (int i = 0; i <= order; i++) {
        //                 int loopIdx = (int) node.loopOrder.size() - 1 - i;
        //                 std::cout << "Adding " << node.loopOrder.at(loopIdx) << " to reduction count" << std::endl;
        //                 currRedCount *= indexVarDim.at(node.loopOrder.at(loopIdx));
        //             }
        //             currOpCount += currRedCount;
        //         }
        //         node.opCount += currOpCount;
        //         opsCount += currOpCount;
        // //        }
    }

    // void visit(UnaryOpNode &node) override {
    //     if (!returnedTensorAdded) {
    //         node.intermediateView.accept(*this);
    //         bytesCount += node.intermediateView.bytesCount;
    //         returnedTensorAdded = true;
    //     }
    //     node.input->accept(*this);
    //     node.intermediateView.accept(*this);
    //     opsCount += node.intermediateView.opCount;
    // }

    void visit(UnaryOpNode &node) override
    {
        if (!returnedTensorAdded)
        {
            double interBytes = 1.0;
            for (auto &var : node.input->getView().getUniqueVars())
            {
                auto indexVar = var.getLogicalIndexVar();
                // Write seg writes for current level
                node.bytesCount += interBytes;
                interBytes *= indexVarDim.at(indexVar);
                // Write crd writes for current level
                node.bytesCount += interBytes;
            }
            // Write value writes, should be same as last level
            node.bytesCount += interBytes;
            returnedTensorAdded = true;
        }
        node.input->accept(*this);
        node.opCount += node.input->getOpsCount();

        std::cout << "Op Count before: " << node.opCount << std::endl;

        double localCount = 1.0;
        auto loops = node.input->getView().getLoops();
        std::vector loopVec(loops.begin(), loops.end());
        unsigned int pos =
            llvm::find(loopVec, node.input->getView().getUniqueVars().back().getLogicalIndexVar()) - loopVec.begin();

        for (auto &loop : loopVec)
        {
            std::cout << "Loop: " << indexVocab[loop.getId()] << std::endl;
        }
        // for (auto &loop : node.input->getView().getUniqueVars()) {
        //     auto index = loop.getLogicalIndexVar();
        //     std::cout << "Unique loop: " << indexVocab[index.getId()] << std::endl;
        // }
        // for (auto &loop : node.intermediateView.getView().getUniqueVars()) {
        //     auto index = loop.getLogicalIndexVar();
        //     std::cout << "Other loop: " << indexVocab[index.getId()] << std::endl;
        // }

        // Add in local unary op count
        for (unsigned int i = 0; i <= pos; i++)
        {
            // for (auto &var: node.input->getView().getUniqueVars()) {
            // auto indexVar = var.getLogicalIndexVar();
            auto indexVar = loopVec[i];
            localCount *= indexVarDim.at(indexVar);
        }
        std::cout << "Op Count after: " << localCount << std::endl;

        if (node.op == "softmax")
        {
            localCount *= 4;
        }
        double localMemCount = node.input->getBytesCount();
        if (llvm::dyn_cast<mlir::BlockArgument>(node.input->getView().getValue()) ||
            llvm::dyn_cast<mlir::arith::ConstantOp>(node.input->getView().getValue().getDefiningOp()))
        {
            localMemCount *= 2;
        }

        node.opCount += localCount;
        node.bytesCount += localMemCount;
        //        std::cout << "Bytes from input: " << node.opCount << std::endl;
    }

    void visit(TensorNode &node) override
    {
        double currOpCount = 1;
        double currMemCount = 1;
        auto indexVars =
            llvm::map_to_vector(node.getView().getUniqueVars(), [&](UniqueVar &a) { return a.getLogicalIndexVar(); });
        for (auto &indexVar : indexVars)
        {
            currMemCount *= indexVarDim.at(indexVar);
        }

        node.bytesCount += currMemCount;
        // Add value access count
        node.opCount = 0.0;
        // int currOpCount = 1;
        // for (auto &var: node.getView().getUniqueVars()) {
        //     auto indexVar = var.getLogicalIndexVar();
        //     currOpCount *= indexVarDim.at(indexVar);
        // }
        // node.opCount += currOpCount;
        // node.bytesCount = currOpCount * 3;
        // bytesCount += currOpCount * 3;
    }

    void visit(ScalarNode &node) override
    {
        // Generate code for scalar values
        llvm::outs() << "Inside scalar: Not implemented yet";
    }

    void visit(FuncOpNode &node) override
    {
        if (!returnedTensorAdded)
        {
            double interBytes = 1.0;
            for (auto &var : node.intermediateView.getUniqueVars())
            {
                auto indexVar = var.getLogicalIndexVar();
                node.bytesCount += interBytes;
                auto currCount = indexVarDim.at(indexVar);
                if (indexVarSparsity.count(indexVar))
                {
                    // currCount /= indexVarSparsity.at(indexVar);
                }
                interBytes *= currCount;
                node.bytesCount += interBytes;
            }
            node.bytesCount += interBytes;
            returnedTensorAdded = true;
        }

        // Visit all inputs
        for (auto &input : node.inputs)
        {
            input->accept(*this);
            node.opCount += input->getOpsCount();
            node.bytesCount += input->getBytesCount();
        }

        // Add operation-specific costs
        double localOpCount = 1.0;
        for (auto &var : node.intermediateView.getUniqueVars())
        {
            auto indexVar = var.getLogicalIndexVar();
            localOpCount *= indexVarDim.at(indexVar);
        }

        // Different operations may have different cost multipliers
        if (node.opType == "transpose")
        {
            // Transpose is essentially a copy operation
            localOpCount *= 1.0;
        }
        else if (node.opType == "concat")
        {
            // Concatenation involves copying data
            localOpCount *= 1.0;
        }
        else if (node.opType == "split")
        {
            // Split involves copying/indexing data
            localOpCount *= 1.0;
        }

        node.opCount += localOpCount;
    }
};

class SoftmaxPatternMatcher : public ASTVisitor
{
public:
    std::shared_ptr<ASTNode> optimizedRoot;
    bool patternMatched{false};
    std::shared_ptr<AnalysisScope> scope; // Add scope to access einsumMap

    SoftmaxPatternMatcher() = default;
    explicit SoftmaxPatternMatcher(std::shared_ptr<AnalysisScope> scope) : scope(scope) {}

    void visit(BinaryOpNode &node) override
    {
        // First, recursively apply pattern matching to children
        node.left->accept(*this);
        if (patternMatched && scope)
        {
            // Update einsumMap if this node was in the map
            for (auto &[view, ast] : scope->einsumMap)
            {
                if (ast.get() == node.left.get())
                {
                    ast = optimizedRoot;
                    break;
                }
            }
            node.left = optimizedRoot;
            patternMatched = false; // Reset for next pattern
        }

        node.right->accept(*this);
        if (patternMatched && scope)
        {
            // Update einsumMap if this node was in the map
            for (auto &[view, ast] : scope->einsumMap)
            {
                if (ast.get() == node.right.get())
                {
                    ast = optimizedRoot;
                    break;
                }
            }
            node.right = optimizedRoot;
            patternMatched = false; // Reset for next pattern
        }

        // Now check if this node itself is a softmax pattern
        if (node.op == "/" && !patternMatched)
        {
            // Check if left side is exp(tensor - maxReduce(tensor))
            // and right side is reduce(exp(tensor - maxReduce(tensor)))
            bool leftMatches = isExpMinusMaxPattern(node.left);
            bool rightMatches = isReduceExpMinusMaxPattern(node.right);

            if (leftMatches && rightMatches)
            {
                // Extract the input tensor from the pattern
                auto inputTensor = extractInputTensor(node.left);
                if (inputTensor)
                {
                    // Create a softmax UnaryOpNode
                    optimizedRoot = std::make_shared<UnaryOpNode>(inputTensor, node.getView(), "softmax");
                    patternMatched = true;

                    // Update einsumMap if this node was in the map
                    if (scope)
                    {
                        for (auto &[view, ast] : scope->einsumMap)
                        {
                            if (ast.get() == &node)
                            {
                                ast = optimizedRoot;
                                break;
                            }
                        }
                    }
                    return;
                }
            }
        }

        // If no pattern matched at this level, create the binary op
        if (!patternMatched)
        {
            optimizedRoot = std::make_shared<BinaryOpNode>(node.left, node.right, node.op, node.getView(),
                                                           node.loopOrder, node.redVar);
        }
    }

    void visit(UnaryOpNode &node) override
    {
        // If this is already a softmax node, keep it
        if (node.op == "softmax" && !patternMatched)
        {
            optimizedRoot = std::make_shared<UnaryOpNode>(node.input, node.getView(), node.op);
            return;
        }

        // Recursively apply pattern matching to input
        node.input->accept(*this);
        if (patternMatched && scope)
        {
            // Update einsumMap if this node was in the map
            for (auto &[view, ast] : scope->einsumMap)
            {
                if (ast.get() == node.input.get())
                {
                    ast = optimizedRoot;
                    break;
                }
            }
            node.input = optimizedRoot;
            patternMatched = false; // Reset for next pattern
        }

        if (!patternMatched)
        {
            optimizedRoot = std::make_shared<UnaryOpNode>(node.input, node.getView(), node.op);
        }
    }

    void visit(TensorNode &node) override
    {
        if (!patternMatched)
        {
            optimizedRoot = std::make_shared<TensorNode>(node.getView());
        }
    }

    void visit(ScalarNode &node) override
    {
        if (!patternMatched)
        {
            optimizedRoot = std::make_shared<ScalarNode>(node.value);
        }
    }

    void visit(FuncOpNode &node) override
    {
        // Visit all inputs
        for (auto &input : node.inputs)
        {
            input->accept(*this);
            if (patternMatched && scope)
            {
                for (auto &[view, ast] : scope->einsumMap)
                {
                    if (ast.get() == input.get())
                    {
                        ast = optimizedRoot;
                        break;
                    }
                }
                input = optimizedRoot;
                patternMatched = false;
            }
        }

        if (!patternMatched)
        {
            optimizedRoot = std::make_shared<FuncOpNode>(node.opType, node.inputs, node.getView(), node.concatDim);
        }
    }

private:
    bool isExpMinusMaxPattern(const std::shared_ptr<ASTNode> &node)
    {
        if (node->nodeType() != NodeType::UnaryOp)
        {
            return false;
        }
        auto unary = static_cast<UnaryOpNode *>(node.get());

        if (unary->op != "exp")
        {
            return false;
        }

        // Check if input is (tensor - maxReduce(tensor))
        if (unary->input->nodeType() != NodeType::BinaryOp)
        {
            return false;
        }
        auto bin = static_cast<BinaryOpNode *>(unary->input.get());

        if (bin->op != "-")
        {
            return false;
        }

        // Check if right is maxReduce(expression)
        if (bin->right->nodeType() != NodeType::UnaryOp)
        {
            return false;
        }

        auto maxReduce = static_cast<UnaryOpNode *>(bin->right.get());

        if (maxReduce->op != "maxReduce")
        {
            return false;
        }

        // Check if maxReduce input matches the left side
        // For now, we'll check if they're the same node or have the same structure
        // This is a simplified check - you might want to implement a more sophisticated comparison
        bool sameExpression = (bin->left == maxReduce->input);

        return sameExpression;
    }

    bool isReduceExpMinusMaxPattern(const std::shared_ptr<ASTNode> &node)
    {
        if (node->nodeType() != NodeType::UnaryOp)
        {
            return false;
        }
        auto unary = static_cast<UnaryOpNode *>(node.get());

        if (unary->op != "reduce")
        {
            return false;
        }

        // Check if input is exp(tensor - maxReduce(tensor))
        return isExpMinusMaxPattern(unary->input);
    }

    std::shared_ptr<ASTNode> extractInputTensor(const std::shared_ptr<ASTNode> &node)
    {
        if (node->nodeType() != NodeType::UnaryOp)
            return nullptr;
        auto unary = static_cast<UnaryOpNode *>(node.get());
        if (unary->op != "exp")
            return nullptr;

        if (unary->input->nodeType() != NodeType::BinaryOp)
            return nullptr;
        auto bin = static_cast<BinaryOpNode *>(unary->input.get());
        if (bin->op != "-")
            return nullptr;

        // Return the left side of the subtraction (the input expression)
        // This can be any ASTNode, not just a TensorNode
        return bin->left;
    }
};

class LayernormPatternMatcher : public ASTVisitor
{
public:
    std::shared_ptr<ASTNode> optimizedRoot;
    bool patternMatched{false};
    std::shared_ptr<AnalysisScope> scope; // Add scope to access einsumMap

    LayernormPatternMatcher() = default;
    explicit LayernormPatternMatcher(std::shared_ptr<AnalysisScope> scope) : scope(scope) {}

    void visit(BinaryOpNode &node) override
    {
        // First, recursively apply pattern matching to children
        node.left->accept(*this);
        if (patternMatched && scope)
        {
            // Update einsumMap if this node was in the map
            for (auto &[view, ast] : scope->einsumMap)
            {
                if (ast.get() == node.left.get())
                {
                    ast = optimizedRoot;
                    break;
                }
            }
            node.left = optimizedRoot;
            patternMatched = false; // Reset for next pattern
        }

        node.right->accept(*this);
        if (patternMatched && scope)
        {
            // Update einsumMap if this node was in the map
            for (auto &[view, ast] : scope->einsumMap)
            {
                if (ast.get() == node.right.get())
                {
                    ast = optimizedRoot;
                    break;
                }
            }
            node.right = optimizedRoot;
            patternMatched = false; // Reset for next pattern
        }

        // Now check if this node itself is a layernorm pattern
        if (node.op == "*" && !patternMatched)
        {
            // Check if this is the final multiplication in layernorm pattern
            // Left should be (tensor - mean)
            // Right should be rsqrt(variance)
            bool leftMatches = isSubtractMeanPattern(node.left);
            bool rightMatches = isRsqrtVariancePattern(node.right);

            // Alternative: try a simpler approach - just check for rsqrt on the right
            if (!rightMatches)
            {
                rightMatches = isSimpleRsqrtPattern(node.right);
            }

            if (leftMatches && rightMatches)
            {
                // Extract the input tensor from the pattern
                auto inputTensor = extractInputTensorFromSubtractMean(node.left);
                if (inputTensor)
                {
                    // Create a layernorm UnaryOpNode
                    optimizedRoot = std::make_shared<UnaryOpNode>(inputTensor, node.getView(), "layernorm");
                    patternMatched = true;

                    // Update einsumMap if this node was in the map
                    if (scope)
                    {
                        for (auto &[view, ast] : scope->einsumMap)
                        {
                            if (ast.get() == &node)
                            {
                                ast = optimizedRoot;
                                break;
                            }
                        }
                    }
                    return;
                }
            }
        }

        // If no pattern matched at this level, create the binary op
        if (!patternMatched)
        {
            optimizedRoot = std::make_shared<BinaryOpNode>(node.left, node.right, node.op, node.getView(),
                                                           node.loopOrder, node.redVar);
        }
    }

    void visit(UnaryOpNode &node) override
    {
        // If this is already a layernorm node, keep it
        if (node.op == "layernorm" && !patternMatched)
        {
            optimizedRoot = std::make_shared<UnaryOpNode>(node.input, node.getView(), node.op);
            return;
        }

        // Recursively apply pattern matching to input
        node.input->accept(*this);
        if (patternMatched && scope)
        {
            // Update einsumMap if this node was in the map
            for (auto &[view, ast] : scope->einsumMap)
            {
                if (ast.get() == node.input.get())
                {
                    ast = optimizedRoot;
                    break;
                }
            }
            node.input = optimizedRoot;
            patternMatched = false; // Reset for next pattern
        }

        if (!patternMatched)
        {
            optimizedRoot = std::make_shared<UnaryOpNode>(node.input, node.getView(), node.op);
        }
    }

    void visit(TensorNode &node) override
    {
        if (!patternMatched)
        {
            optimizedRoot = std::make_shared<TensorNode>(node.getView());
        }
    }

    void visit(ScalarNode &node) override
    {
        if (!patternMatched)
        {
            optimizedRoot = std::make_shared<ScalarNode>(node.value);
        }
    }

    void visit(FuncOpNode &node) override
    {
        // Visit all inputs
        for (auto &input : node.inputs)
        {
            input->accept(*this);
            if (patternMatched && scope)
            {
                for (auto &[view, ast] : scope->einsumMap)
                {
                    if (ast.get() == input.get())
                    {
                        ast = optimizedRoot;
                        break;
                    }
                }
                input = optimizedRoot;
                patternMatched = false;
            }
        }

        if (!patternMatched)
        {
            optimizedRoot = std::make_shared<FuncOpNode>(node.opType, node.inputs, node.getView(), node.concatDim);
        }
    }

private:
    bool isSubtractMeanPattern(const std::shared_ptr<ASTNode> &node)
    {
        if (node->nodeType() != NodeType::BinaryOp)
        {
            return false;
        }
        auto bin = static_cast<BinaryOpNode *>(node.get());
        if (bin->op != "-")
        {
            return false;
        }

        // Check if right side is scalarDiv(reduce(tensor))
        bool result = isScalarDivReducePattern(bin->right);
        return result;
    }

    bool isScalarDivReducePattern(const std::shared_ptr<ASTNode> &node)
    {
        if (node->nodeType() != NodeType::UnaryOp)
        {
            return false;
        }
        auto unary = static_cast<UnaryOpNode *>(node.get());
        if (unary->op != "scalarDiv")
        {
            return false;
        }

        // Check if input is reduce(tensor)
        bool result = isReducePattern(unary->input);
        return result;
    }

    bool isReducePattern(const std::shared_ptr<ASTNode> &node)
    {
        if (node->nodeType() != NodeType::UnaryOp)
        {
            return false;
        }
        auto unary = static_cast<UnaryOpNode *>(node.get());
        bool result = (unary->op == "reduce");
        return result;
    }

    bool isRsqrtVariancePattern(const std::shared_ptr<ASTNode> &node)
    {
        if (node->nodeType() != NodeType::UnaryOp)
        {
            return false;
        }
        auto unary = static_cast<UnaryOpNode *>(node.get());
        if (unary->op != "rsqrt")
        {
            return false;
        }

        // Check if input is scalarAdd(downcast(...))
        bool result = isScalarAddDowncastPattern(unary->input);
        return result;
    }

    bool isScalarAddDowncastPattern(const std::shared_ptr<ASTNode> &node)
    {
        if (node->nodeType() != NodeType::UnaryOp)
        {
            return false;
        }
        auto unary = static_cast<UnaryOpNode *>(node.get());
        if (unary->op != "scalarAdd")
        {
            return false;
        }

        // Check if input is downcast(...)
        bool result = isDowncastPattern(unary->input);
        return result;
    }

    bool isDowncastPattern(const std::shared_ptr<ASTNode> &node)
    {
        if (node->nodeType() != NodeType::UnaryOp)
        {
            return false;
        }
        auto unary = static_cast<UnaryOpNode *>(node.get());
        if (unary->op != "downcast")
        {
            return false;
        }

        // Check if input is scalarDiv(reduce(upcast(...)))
        bool result = isScalarDivReduceUpcastPattern(unary->input);
        return result;
    }

    bool isScalarDivReduceUpcastPattern(const std::shared_ptr<ASTNode> &node)
    {
        if (node->nodeType() != NodeType::UnaryOp)
        {
            return false;
        }
        auto unary = static_cast<UnaryOpNode *>(node.get());
        if (unary->op != "scalarDiv")
        {
            return false;
        }

        // Check if input is reduce(upcast(...))
        bool result = isReduceUpcastPattern(unary->input);
        return result;
    }

    bool isReduceUpcastPattern(const std::shared_ptr<ASTNode> &node)
    {
        if (node->nodeType() != NodeType::UnaryOp)
        {
            return false;
        }
        auto unary = static_cast<UnaryOpNode *>(node.get());
        if (unary->op != "reduce")
        {
            return false;
        }

        // Check if input is upcast(...)
        bool result = isUpcastPattern(unary->input);
        return result;
    }

    bool isUpcastPattern(const std::shared_ptr<ASTNode> &node)
    {
        if (node->nodeType() != NodeType::UnaryOp)
        {
            return false;
        }
        auto unary = static_cast<UnaryOpNode *>(node.get());
        bool result = (unary->op == "upcast");
        return result;
    }

    std::shared_ptr<ASTNode> extractInputTensorFromSubtractMean(const std::shared_ptr<ASTNode> &node)
    {
        if (node->nodeType() != NodeType::BinaryOp)
            return nullptr;
        auto bin = static_cast<BinaryOpNode *>(node.get());
        if (bin->op != "-")
            return nullptr;

        // Return the left side of the subtraction (the input tensor)
        return bin->left;
    }

    bool isSimpleRsqrtPattern(const std::shared_ptr<ASTNode> &node)
    {
        if (node->nodeType() != NodeType::UnaryOp)
        {
            return false;
        }
        auto unary = static_cast<UnaryOpNode *>(node.get());
        if (unary->op != "rsqrt")
        {
            return false;
        }
        // For now, just check if it's an rsqrt operation
        // We can make this more sophisticated later
        return true;
    }
};

#endif // SAMML_CONSTRUCTEINSUM_H
