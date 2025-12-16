#ifndef VARS_H
#define VARS_H

#include "lib/Dialect/SAM/SamDialect.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"

#include "ortools/graph/connected_components.h"
#include "ortools/graph/graph.h"
#include "ortools/graph/topologicalsorter.h"
#include <functional>
#include <iostream>
#include <memory>
#include <llvm/include/llvm/ADT/DenseMapInfo.h>
#include <llvm/include/llvm/ADT/DenseMap.h>

class FusedCIN;

class TensorView;

class UniqueVar;

class IndexVar;

// Tensor Native Index Var
// TODO: Figure out name, TensorRankRef/UniqueIndexVar TBD
// TODO: Need to incorporate format in fused CIN
// TODO: Incorporate transposed views of same tensor
// TODO: Incorporate feature for ignoring partial orders for some expressions
// through a flag
class UniqueVar
{
public:
    UniqueVar() = default;

    UniqueVar(const std::shared_ptr<FusedCIN>& _tensor, unsigned int _id);

    explicit UniqueVar(const unsigned int _id) : tensor(nullptr), id(_id)
    {
    }

    [[nodiscard]] unsigned int getId() const { return id; };

    friend std::ostream& operator<<(std::ostream& os, const UniqueVar& var);

    [[nodiscard]] IndexVar getLogicalIndexVar() const;

    std::shared_ptr<FusedCIN> getTensor() { return tensor; }

    UniqueVar& operator=(const UniqueVar& var)
    {
        this->id = var.id;
        this->tensor = var.tensor;
        return *this;
    }

    bool operator<(const UniqueVar& var) const
    {
        return this->getId() < var.getId();
    }

    bool operator==(UniqueVar& var) const
    {
        return this->getId() == var.getId();
    }

    //    [[nodiscard]] std::shared_ptr<TensorView> getView() const { return view; }

    //    void setView(const TensorView &_view) { view = std::make_shared<TensorView>(_view); }

    mlir::Operation* op{};

    bool operator==(const UniqueVar& var) const { return this->getId() == var.getId(); }

private:
    std::shared_ptr<FusedCIN> tensor;
    std::shared_ptr<TensorView> view;

    //    std::shared_ptr<TensorView>;
    unsigned int id;
};

class IndexVar
{
public:
    IndexVar() = default;

    explicit IndexVar(const unsigned int _id) : id(_id)
    {
    }

    [[nodiscard]] unsigned int getId() const { return id; };

    friend std::ostream& operator<<(std::ostream& os, const IndexVar& var);

    bool operator==(const IndexVar& var) const { return this->getId() == var.getId(); }

    bool operator<(const IndexVar& var) const { return this->getId() < var.getId(); }

private:
    unsigned int id;
};

struct [[maybe_unused]] UniqueVarCompare
{
    bool operator()(const UniqueVar& lhs, const UniqueVar& rhs) const;
};

struct IndexVarCompare
{
    bool operator()(const IndexVar& lhs, const IndexVar& rhs) const;
};

struct TensorViewCompare
{
    bool operator()(const TensorView& lhs, const TensorView& rhs) const;
};

template <>
struct std::hash<TensorView>
{
    std::size_t operator()(const TensorView& view) const noexcept;
};

namespace llvm
{
    template <>
    struct DenseMapInfo<TensorView>
    {
        static TensorView getEmptyKey();

        static TensorView getTombstoneKey();

        static unsigned getHashValue(const TensorView& val); //{ return Val * 37U; }
        static bool isEqual(const TensorView& LHS, const TensorView& RHS);
    };
}

#endif //VARS_H
