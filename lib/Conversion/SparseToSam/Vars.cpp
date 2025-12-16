#include "Vars.h"
#include "FusedCIN.h"

bool TensorViewCompare::operator()(const TensorView &lhs, const TensorView &rhs) const
{
    //    return lhs.getView() != rhs.getView();
    // std::cout << "LHS: " << lhs << std::endl;
    // std::cout << "RHS: " << rhs << std::endl;
    if (lhs.getView() != rhs.getView())
    {
        if (lhs.getId() < rhs.getId())
        {
            return true;
        }
    }
    return false;
    // else
    // {
    // return lhs.getView() != rhs.getView();
    // }
    // return lhs < rhs;
}

// struct std::equal_to<TensorView>
// {
//     bool operator()(const TensorView& lhs, const TensorView& rhs) const
//     {
//         return lhs == rhs;
//     }
// };

std::size_t std::hash<TensorView>::operator()(const TensorView &view) const noexcept
{
    return (std::hash<unsigned int>{}(view.getId()));
}

// TensorView llvm::DenseMapInfo<TensorView, void>::getEmptyKey()
// {
// }
//
// TensorView llvm::DenseMapInfo<TensorView, void>::getTombstoneKey()
// {
// }
//

UniqueVar::UniqueVar(const std::shared_ptr<FusedCIN> &_tensor, const unsigned int _id) : tensor(_tensor), id(_id) {}

bool UniqueVarCompare::operator()(const UniqueVar &lhs, const UniqueVar &rhs) const
{
    return lhs.getId() < rhs.getId();
}

bool IndexVarCompare::operator()(const IndexVar &lhs, const IndexVar &rhs) const { return lhs.getId() < rhs.getId(); }
