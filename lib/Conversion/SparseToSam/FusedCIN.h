#ifndef LIB_CONVERSION_FUSEDCIN_H_
#define LIB_CONVERSION_FUSEDCIN_H_

#include <algorithm>
#include <functional>
#include <memory>
#include <set>
#include "Table.h"
#include "TopoSort.h"
#include "Vars.h"
#include "lib/Dialect/SAM/SamDialect.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "ortools/graph/connected_components.h"
#include "ortools/graph/graph.h"

enum class NodeType
{
    BinaryOp,
    UnaryOp,
    Tensor,
    Scalar,
    FuncOp,
};

class FusedCIN;

class TensorView;

// class UniqueVar;
// class IndexVar;
class AnalysisScope;

class TensorViewImpl;

class VarASTNode;

// class ASTNode;
class ASTNode
{
public:
    virtual ~ASTNode() = default;
    virtual void accept(class ASTVisitor &visitor) = 0;
    // Non-const access (legacy)
    virtual TensorView getView() = 0;
    // Const-qualified access so callers with const ASTNode* work.
    virtual TensorView getView() const = 0;
    // Node kind accessors
    virtual NodeType nodeType() = 0;
    virtual NodeType nodeType() const = 0;
    // Stats
    virtual double getBytesCount() = 0;
    virtual double getBytesCount() const = 0;
    virtual double getOpsCount() = 0;
    virtual double getOpsCount() const = 0;

    // ------------------------- Methods for FusedCodegenEnv Integration ------------------------------
    // These methods provide the same information that linalg::GenericOp provides to CodegenEnv

    /// Core information (similar to linalg::GenericOp)
    virtual unsigned getNumTensors() const = 0;
    virtual unsigned getNumLoops() const = 0;
    virtual unsigned getMaxRank() const = 0;

    /// Tensor information
    virtual llvm::SmallVector<mlir::Value> getTensorOperands() const = 0;
    virtual llvm::SmallVector<mlir::Type> getTensorTypes() const = 0;
    virtual llvm::SmallVector<mlir::AffineMap> getIndexingMaps() const = 0;

    /// Loop information
    virtual llvm::SmallVector<mlir::Value> getLoopVars() const = 0;
    virtual mlir::Value getLoopVar(unsigned i) const = 0;

    /// Expression tree (similar to linalg region)
    virtual mlir::Value getRootExpression() const = 0;
    virtual mlir::Operation *getYieldOp() const = 0;

    /// Sparse tensor metadata
    virtual llvm::SmallVector<mlir::sparse_tensor::SparseTensorEncodingAttr> getSparseEncodings() const = 0;
    virtual bool hasSparseOutput() const = 0;
    virtual mlir::OpOperand *getSparseOutput() const = 0;

    /// For Merger integration (similar to buildTensorExpFromLinalg)
    // virtual std::optional<mlir::sparse_tensor::ExprId>
    // buildTensorExpFromAST(mlir::sparse_tensor::Merger &merger, mlir::OpBuilder &builder, mlir::Location loc) const =
    // 0;

    /// Loop scheduling (similar to linalg's "sorted" attribute)
    virtual bool isScheduled() const = 0;
    virtual void setScheduled(bool scheduled) = 0;
};


using compFunc = std::function<mlir::Value(const TensorView &)>;

class JoinNode
{
public:
    JoinNode(const std::vector<std::shared_ptr<VarASTNode>> &_children, const mlir::sam::JoinerType _joinType) :
        children(_children), joinType(_joinType)
    {
    }

    explicit JoinNode(const mlir::sam::JoinerType _joinType) : joinType(_joinType) {}

    [[nodiscard]] mlir::sam::JoinerType getType() const { return this->joinType; }

    void add_children(const std::shared_ptr<VarASTNode> &child) { children.push_back(child); }

private:
    std::vector<std::shared_ptr<VarASTNode>> children;
    mlir::sam::JoinerType joinType;
};

class Leaf
{
public:
    // explicit Leaf(const unsigned int id) : id(id)
    explicit Leaf(const UniqueVar &var) : id(var) {}

    // [[nodiscard]] unsigned int get() const { return this->id; }
    [[nodiscard]] UniqueVar get() const { return this->id; }

private:
    // std::vector<unsigned int> ids;
    // unsigned int id;
    UniqueVar id;
    // TensorView view;
};

class VarASTNode
{
public:
    explicit VarASTNode(const UniqueVar &leafId) : node(Leaf(leafId)) {}

    explicit VarASTNode(Leaf leaf) : node(leaf) {}

    explicit VarASTNode(JoinNode node) : node(node) {}

    [[nodiscard]] bool isLeaf() const { return std::holds_alternative<Leaf>(this->node); }
    // unsigned int getLeaf() { return std::get<Leaf>(this->node).get(); }

    [[nodiscard]] Leaf getLeaf() const
    {
        assert(std::holds_alternative<Leaf>(this->node));
        return std::get<Leaf>(this->node);
    }

private:
    std::variant<Leaf, JoinNode> node;
};

class TensorViewImpl
{
public:
    TensorViewImpl(const std::vector<UniqueVar> &_uniqueVars, const mlir::Value _value,
                   const compFunc &comp = nullptr) : uniqueVars(_uniqueVars), value(_value)
    {
        id = counter++;
    }

    std::vector<UniqueVar> &getUniqueVars() { return uniqueVars; }

    bool operator!=(const TensorViewImpl &impl) const { return !(*this == impl); }

    bool operator<(const TensorViewImpl &impl) const
    {
        if (this->getId() != impl.getId())
        {
            return this->getId() < impl.getId();
        }
        return false;
    }

    compFunc comp;

    [[nodiscard]] unsigned int getId() const { return id; }

    std::function<Table(const std::vector<IndexVar> &)> constructTable;

    std::function<std::set<IndexVar>(void)> getScopedLoops;

    void setContractionFunction(
        // const std::function<std::vector<VarASTNode>(void)>& contractionFunc)
        const std::function<void()> &contractionFunc)
    {
        this->getContractions = contractionFunc;
    }

    void setConstructTableFunction(const std::function<Table(const std::vector<IndexVar> &loopOrder)> &constructFunc)
    {
        this->constructTable = constructFunc;
    }

    void setScopedLoopFunction(const std::function<std::set<IndexVar>(void)> &reductionFunc)
    {
        this->getScopedLoops = reductionFunc;
    }

    [[nodiscard]] bool empty() const { return this->uniqueVars.empty(); }

    // std::function<std::vector<VarASTNode>(void)> getContractions;
    std::function<void()> getContractions;

    // private:
    static unsigned int counter;
    unsigned int id;
    std::vector<UniqueVar> uniqueVars{};
    mlir::Value value;
    llvm::DenseMap<mlir::AffineExpr, UniqueVar> exprToUniqueVar;
    std::vector<UniqueVar> loopOrder;
    std::set<IndexVar> localLoops{};
    std::string name;

    [[nodiscard]] mlir::Value getValue() const { return value; }

    bool operator==(const TensorViewImpl &view) const { return this->getId() == view.getId(); }

    friend std::ostream &operator<<(std::ostream &os, const TensorViewImpl &view);

    bool completed{false};
};

class TensorView
{
public:
    TensorView(const std::vector<UniqueVar> &_uniqueVars, mlir::Value value) : uniqueVars(_uniqueVars)
    {
        tensorView = std::make_shared<TensorViewImpl>(_uniqueVars, value);
    }

    TensorView() = default;

    [[nodiscard]] auto &getUniqueVars() const { return tensorView->getUniqueVars(); };

    [[nodiscard]] auto getValue() const { return tensorView->getValue(); }

    [[nodiscard]] bool empty() const { return this->tensorView == nullptr; }

    friend std::ostream &operator<<(std::ostream &os, const TensorView &view);

    std::vector<UniqueVar> uniqueVars;

    std::string name;

    bool operator==(const TensorView &view) const { return *this->tensorView == *view.tensorView; }

    bool operator<(const TensorView &view) const { return *this->tensorView < *view.tensorView; }

    TensorView &operator=(const TensorView &view) = default;

    void setName(const std::string &tensorName) const { tensorView->name = tensorName; }

    [[nodiscard]] std::string getName() const { return tensorView->name; }
    // {
    //// this->comp = view.comp;
    // this->uniqueVars = view.uniqueVars;
    // this->tensorView = view.tensorView;
    // return *this;
    // }

    bool operator==(TensorView &view) const
    {
        // TODO: Might need to check if vars are equal instead, in cases where reduction indexvars vary when they are
        // equivalent
        return this->tensorView == view.tensorView;
    }

    bool operator!=(TensorView &view) const { return !(*this == view); }

    [[nodiscard]] mlir::Value comp() const { return this->tensorView->comp(*this); }

    // void constructTable() { this->constructTable(); }
    [[nodiscard]] unsigned int getId() const { return this->tensorView->getId(); }

    void setContractionFunction(
        // const std::function<std::vector<VarASTNode>(void)>& contractionFunc) const
        const std::function<void(void)> &contractionFunc) const
    {
        this->tensorView->setContractionFunction(contractionFunc);
    }

    void
    setConstructTableFunction(const std::function<Table(const std::vector<IndexVar> &loopOrder)> &constructFunc) const
    {
        this->tensorView->setConstructTableFunction(constructFunc);
    }

    void setScopedLoopFunction(const std::function<std::set<IndexVar>(void)> &reductionFunc) const
    {
        this->tensorView->setScopedLoopFunction(reductionFunc);
    }

    void setCompleted() const { this->tensorView->completed = true; }

    [[nodiscard]] bool getCompleted() const { return this->tensorView->completed; }

    [[nodiscard]] Table constructTable(const std::vector<IndexVar> &vars) const
    {
        return this->tensorView->constructTable(vars);
    }


    [[nodiscard]] std::set<IndexVar> getScopedLoops() const { return this->tensorView->getScopedLoops(); }

    // std::vector<VarASTNode> getContractions() const
    void getContractions() const { this->tensorView->getContractions(); }

    [[nodiscard]] auto getView() const { return this->tensorView; }

    void insertLoop(const IndexVar &var) const { this->tensorView->localLoops.insert(var); }

    [[nodiscard]] std::set<IndexVar> getLoops() const { return this->tensorView->localLoops; }

private:
    std::shared_ptr<TensorViewImpl> tensorView;
};

class FusedCIN
{
public:
    FusedCIN() = default;

    explicit FusedCIN(const std::shared_ptr<AnalysisScope> &_scope) : op(nullptr), scope(_scope) {}

    explicit FusedCIN(const std::shared_ptr<AnalysisScope> &_scope, mlir::Operation *operation) :
        op(operation), scope(_scope)
    {
    }

    std::function<TensorView()> makeInterface;

    std::vector<UniqueVar> &getUniqueVars(const unsigned int index) { return uniqueVars[index]; }

    std::vector<TensorView> &getTensorViews() { return tensorViews; }

    std::vector<std::vector<UniqueVar>> uniqueVars;
    std::vector<TensorView> tensorViews;
    std::vector<mlir::utils::IteratorType> iteratorTypes;
    std::vector<UniqueVar> loopOrder;
    mlir::Operation *op{};
    mlir::Value val;
    std::shared_ptr<AnalysisScope> scope;
    mlir::Value tensorValue;
};

class JoinerAST
{
public:
    explicit JoinerAST(const IndexVar &var) : indexVar(var) {}

private:
    IndexVar indexVar;
    std::map<mlir::sam::JoinerType, std::set<TensorView>> node;
};

class AnalysisScope
{
public:
    AnalysisScope() : indexCounter(0), logicalIndexCounter(0), useUserInput(false) {};

    explicit AnalysisScope(bool useUserInput) : indexCounter(0), logicalIndexCounter(0), useUserInput(useUserInput) {};

    [[nodiscard]] int getId() const { return indexCounter; }

    UniqueVar getNewUniqueVar(const std::shared_ptr<FusedCIN> &tensor);

    void markEqual(const UniqueVar &a, const UniqueVar &b);

    void addPartialOrder(const UniqueVar &a, const UniqueVar &b);

    void addNode(const UniqueVar &a);

    void printConnectedComponents();

    IndexVar getLogicalIndexVar(const UniqueVar *var);

    void setIndexingMaps(const llvm::SmallVector<mlir::AffineMap> &_indexingMaps) { indexingMaps = _indexingMaps; }

    [[nodiscard]] int getNumSets() const { return equalityGraphComponents.GetNumberOfComponents(); }

    std::vector<IndexVar> getLoopOrder();

    std::vector<std::vector<IndexVar>> getAllLoopOrders();

    static void set_joiner_type(std::pair<unsigned int, unsigned int> varPair, mlir::sam::JoinerType type)
    {
        // joinerTypes[varPair] = type;
    }

    // llvm::DenseMap<std::pair<unsigned int, unsigned int>,
    // mlir::sam::JoinerType>
    //     joinerTypes;
    bool firstTime{true};
    std::map<TensorView, mlir::Value, TensorViewCompare> tensorStreamMap;
    std::map<TensorView, std::vector<UniqueVar>> tensorLoopOrders;
    std::map<TensorView, UniqueVar, TensorViewCompare> tensorReductions;
    llvm::DenseMap<mlir::Value, mlir::Value> regionOpToStream;
    llvm::DenseMap<mlir::AffineExpr, llvm::SmallVector<mlir::Value>> indexVarMap;
    std::vector<TensorView> operands;
    std::map<unsigned int, mlir::sam::JoinerType> joinerTypes;

    std::map<IndexVar, std::map<mlir::sam::JoinerType, std::set<TensorView>>, IndexVarCompare> contractionType;
    std::map<IndexVar, std::vector<mlir::sam::JoinerType>> contractionOrder;
    // TODO: Potentially could lead to collisions
    std::map<std::pair<IndexVar, mlir::sam::JoinerType>, std::set<std::pair<std::string, std::vector<IndexVar>>>>
        tensorContractionSet;
    //    std::map<std::pair<IndexVar, mlir::sam::JoinerType>, std::set<std::pair<std::string, std::vector<IndexVar>>>>
    //        tensorContractionSet;
    std::map<IndexVar, llvm::DenseSet<mlir::Value>> valContractionSet;

    std::set<IndexVar> reducedVars;
    std::map<TensorView, std::set<IndexVar>> outReduceMap;
    //    std::map<TensorView, std::set<IndexVar>> excludedReduceMap;
    std::map<std::pair<IndexVar, mlir::sam::JoinerType>, std::vector<std::pair<TensorView, TensorView>>> joinerPatches;
    std::map<IndexVar, bool> multipleSpacc;
    std::map<IndexVar, std::vector<TensorView>> innerCrds;
    std::map<UniqueVar, UniqueVar, UniqueVarCompare> equalVars;
    std::map<IndexVar, std::unordered_set<TensorView>> contractionMap;
    std::map<IndexVar, TensorView> indexToView;
    std::map<std::pair<std::string, std::vector<IndexVar>>, std::vector<IndexVar>> updatedEffOrder;
    std::map<std::pair<std::string, std::vector<IndexVar>>, Column> columnCache;
    std::map<IndexVar, llvm::SmallVector<mlir::Value>> currVals;
    std::map<unsigned int, TensorView> varToView;
    llvm::DenseMap<mlir::Value, std::string> tensorNames;
    std::map<IndexVar, TensorView> seenInner;
    std::map<TensorView, std::string> tensorViewNames;
    llvm::DenseMap<mlir::Value, unsigned int> tensorIds;
    std::map<TensorView, mlir::AffineExpr> prevIndex;
    std::vector<IndexVar> reductionVars;
    std::vector<UniqueVar> uniqueRedVars;
    std::vector<TensorView> views;
    std::vector<IndexVar> effLoopOrder;
    std::map<IndexVar, TensorView> varMap;
    std::map<TensorView, std::vector<unsigned int>> modeOrders;
    // Attempting to set all reduction variables of intermediate ops with same order equal
    llvm::DenseMap<mlir::Value, UniqueVar> ignoredDims;

    std::map<std::pair<std::string, std::vector<IndexVar>>, TensorView> tableCache;
    std::map<IndexVar, std::map<TensorView, TensorView>> cellIndirectionMap;
    std::map<TensorView, std::shared_ptr<ASTNode>> einsumMap;
    bool potentialSoftmax{false};

    // Tracking for concat operations and their dimension variables
    struct ConcatInfo {
        TensorView outputView;
        uint64_t concatDim;
        UniqueVar concatDimUniqueVar;  // The unique variable for the concatenated dimension
        std::vector<TensorView> inputViews;
    };
    std::vector<ConcatInfo> concatOperations;

private:
    util::Graph equalityGraph;
    DenseConnectedComponentsFinder equalityGraphComponents;
    util::Graph partialOrderings;
    int indexCounter;
    int logicalIndexCounter;
    llvm::DenseMap<int, int> logicalIndexMap;
    llvm::SmallVector<mlir::AffineMap> indexingMaps;
    llvm::SmallVector<UniqueVar> vars;
    llvm::SmallVector<mlir::AffineExpr> loopOrder;
    bool useUserInput;
};

template <typename T>
void printAllOrders(std::vector<std::vector<T>> allOrders);

#endif
